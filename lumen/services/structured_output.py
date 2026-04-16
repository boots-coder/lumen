"""
Structured Output — force model to output valid JSON matching a schema.

Supports:
1. OpenAI response_format: {"type": "json_schema", "json_schema": {...}}
2. Anthropic tool_use trick: define a single "output" tool with the desired schema
3. Universal: JSON mode + post-validation with Pydantic
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Type

from ..providers.model_profiles import ModelProfile

logger = logging.getLogger(__name__)

# We do a runtime check for pydantic availability so the module stays
# importable even when pydantic is not installed.
try:
    from pydantic import BaseModel as _PydanticBaseModel, ValidationError as _PydanticValidationError

    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _PydanticBaseModel = None  # type: ignore[assignment,misc]
    _PydanticValidationError = None  # type: ignore[assignment,misc]
    _HAS_PYDANTIC = False


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Desired output format from the model."""
    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StructuredOutputConfig:
    """
    Configuration for structured output enforcement.

    Parameters
    ----------
    format : OutputFormat
        The desired output format.
    schema : dict | type[BaseModel] | None
        The target JSON schema.  Can be a raw dict schema or a Pydantic
        model class whose ``.model_json_schema()`` will be called.
    strict : bool
        If True, OpenAI strict mode is used (all properties required,
        no additionalProperties).
    schema_name : str
        Name for the schema (used in OpenAI ``json_schema`` wrapper).
    """
    format: OutputFormat = OutputFormat.TEXT
    schema: dict[str, Any] | type | None = None  # type: ignore[type-arg]
    strict: bool = True
    schema_name: str = "output"

    def get_json_schema(self) -> dict[str, Any] | None:
        """
        Resolve *schema* to a plain dict JSON schema.

        Returns ``None`` when no schema is configured.
        """
        if self.schema is None:
            return None

        if isinstance(self.schema, dict):
            return self.schema

        # Pydantic model class
        if _HAS_PYDANTIC and isinstance(self.schema, type) and issubclass(self.schema, _PydanticBaseModel):
            return self.schema.model_json_schema()

        raise TypeError(
            f"Unsupported schema type: {type(self.schema)}. "
            "Expected dict or pydantic.BaseModel subclass."
        )


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class StructuredResult:
    """Result of parsing + validating structured output."""
    data: dict[str, Any] | Any  # Parsed dict, or Pydantic model instance
    raw_text: str
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Payload preparation
# ---------------------------------------------------------------------------

def prepare_structured_request(
    config: StructuredOutputConfig,
    profile: ModelProfile,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Modify an API request *payload* to enforce structured output.

    Dispatches to the right strategy based on the model profile:

    - **OpenAI**: ``response_format: {"type": "json_schema", ...}``
    - **Anthropic**: wraps the schema as a constrained ``tool_use`` tool
    - **Generic / fallback**: ``response_format: {"type": "json_object"}``

    Returns
    -------
    dict
        The mutated payload (also modified in-place for convenience).
    """
    if config.format == OutputFormat.TEXT:
        return payload

    json_schema = config.get_json_schema()

    # --- OpenAI & OpenAI-compat providers ---
    if profile.family in ("openai", "openai_reasoning", "deepseek", "qwen",
                          "llama", "mistral", "generic"):
        if config.format == OutputFormat.JSON_SCHEMA and json_schema is not None:
            schema_body: dict[str, Any] = dict(json_schema)
            if config.strict:
                # Ensure strict-mode constraints
                if "properties" in schema_body:
                    schema_body.setdefault("required", list(schema_body["properties"].keys()))
                    schema_body["additionalProperties"] = False
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": config.schema_name,
                    "strict": config.strict,
                    "schema": schema_body,
                },
            }
        else:
            # Plain JSON mode (no schema enforcement on API side)
            payload["response_format"] = {"type": "json_object"}
        return payload

    # --- Anthropic ---
    if profile.family == "anthropic":
        if config.format == OutputFormat.JSON_SCHEMA and json_schema is not None:
            # Wrap as a single-tool constraint: the model is forced to call
            # an "output" tool whose input_schema matches the desired shape.
            output_tool: dict[str, Any] = {
                "name": config.schema_name,
                "description": (
                    "Return your answer by calling this tool with the data "
                    "structured according to the schema."
                ),
                "input_schema": json_schema,
            }
            existing_tools = payload.get("tools") or []
            payload["tools"] = existing_tools + [output_tool]
            payload["tool_choice"] = {"type": "tool", "name": config.schema_name}
        else:
            # Anthropic has no generic "json mode" — instruct via system
            # prompt hint and rely on post-validation.
            logger.debug(
                "Anthropic has no native JSON mode; relying on prompt + validation."
            )
        return payload

    # --- Gemini ---
    if profile.family == "gemini":
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": config.schema_name,
                    "schema": json_schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        return payload

    # Fallback: best-effort JSON mode
    payload["response_format"] = {"type": "json_object"}
    return payload


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_output(
    raw_text: str,
    config: StructuredOutputConfig,
) -> StructuredResult:
    """
    Parse and optionally validate raw model output against a schema.

    Steps:
      1. Parse the text as JSON.
      2. If a Pydantic model schema was provided, validate via the model.
      3. If a dict schema was provided, do a lightweight type check.
      4. Return a :class:`StructuredResult`.
    """
    if config.format == OutputFormat.TEXT:
        return StructuredResult(data={}, raw_text=raw_text, is_valid=True)

    # Step 1: JSON parse
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return StructuredResult(
            data={},
            raw_text=raw_text,
            is_valid=False,
            validation_errors=[f"Invalid JSON: {exc}"],
        )

    # Step 2/3: Schema validation
    errors: list[str] = []

    if config.schema is not None:
        # Pydantic model class
        if (
            _HAS_PYDANTIC
            and isinstance(config.schema, type)
            and issubclass(config.schema, _PydanticBaseModel)
        ):
            try:
                instance = config.schema.model_validate(parsed)
                return StructuredResult(
                    data=instance,
                    raw_text=raw_text,
                    is_valid=True,
                )
            except _PydanticValidationError as exc:
                errors = [str(e) for e in exc.errors()]
                return StructuredResult(
                    data=parsed,
                    raw_text=raw_text,
                    is_valid=False,
                    validation_errors=errors,
                )

        # Dict schema — lightweight required-field check
        if isinstance(config.schema, dict):
            json_schema = config.schema
            required = json_schema.get("required", [])
            if isinstance(parsed, dict):
                missing = [f for f in required if f not in parsed]
                if missing:
                    errors.append(f"Missing required fields: {missing}")
            else:
                errors.append(
                    f"Expected object (dict), got {type(parsed).__name__}"
                )

    return StructuredResult(
        data=parsed,
        raw_text=raw_text,
        is_valid=len(errors) == 0,
        validation_errors=errors,
    )
