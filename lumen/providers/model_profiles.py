"""
Model Profile Registry — dynamic tool format adaptation for any model.

核心设计思想：
  用户不需要查阅每个模型的技术文档。
  框架自动检测模型家族，注入对应的工具格式适配器。

  Model name → ModelProfile → ToolAdapter
                             → ResponseParser
                             → Capabilities

每个模型家族在预训练阶段用了不同的工具格式：
  - OpenAI GPT: function calling (JSON string arguments)
  - Anthropic Claude: tool_use content blocks
  - DeepSeek: OpenAI-compat + <｜tool▁call▁begin｜> special tokens
  - Qwen 2.5: <tool_call> XML in pretraining, OpenAI API compat
  - Llama 3.1+: <|python_tag|> + JSON tool calls
  - Mistral: [TOOL_CALLS] token, OpenAI API compat
  - GLM-4: custom protocol
  - Gemini: Google's function_call format

框架负责：
  1. 从 model name 自动识别家族
  2. 选择该家族最优的 tool schema 格式
  3. 正确解析该家族的 tool call 响应
  4. 用户可以动态注册新模型 profile
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Tool Schema Format — 模型预训练时用的工具描述格式
# ═════════════════════════════════════════════════════════════════════════════

class ToolSchemaFormat(str, Enum):
    """How to format tool definitions for the API."""
    OPENAI_FUNCTION = "openai_function"          # {"type":"function","function":{...}}
    ANTHROPIC_TOOL_USE = "anthropic_tool_use"    # {"name","description","input_schema"}
    OPENAI_STRICT = "openai_strict"              # OpenAI strict mode (JSON schema)
    GEMINI_FUNCTION = "gemini_function"          # Google function declarations


class ToolCallFormat(str, Enum):
    """How the model returns tool calls in responses."""
    OPENAI_TOOL_CALLS = "openai_tool_calls"      # message.tool_calls[]
    ANTHROPIC_CONTENT_BLOCKS = "anthropic_content_blocks"  # content[].type=="tool_use"
    GEMINI_FUNCTION_CALL = "gemini_function_call"  # candidates[].content.parts[].functionCall


class MessageFormat(str, Enum):
    """Wire format for messages."""
    OPENAI = "openai"       # role + content string
    ANTHROPIC = "anthropic" # role + content blocks (alternating required)


# ═════════════════════════════════════════════════════════════════════════════
# Model Capabilities
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelCapabilities:
    """What a model can do — inferred from profile."""
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_parallel_tool_calls: bool = True
    supports_system_message: bool = True
    supports_tool_choice: bool = True
    max_tools: int = 128                # Max number of tools in one request
    requires_alternating_roles: bool = False  # Anthropic requires this
    system_role_name: str = "system"    # "system" or "developer" (o1)
    # 预训练特性
    native_tool_tokens: bool = False    # Model was pretrained with tool tokens
    tool_token_prefix: str | None = None  # e.g. "<|python_tag|>" for Llama


# ═════════════════════════════════════════════════════════════════════════════
# Model Profile — the complete adapter for one model family
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelProfile:
    """
    Complete specification for how to talk to a model family.

    Encapsulates:
      - Which wire format to use for messages
      - Which schema format to use for tools
      - Which format to expect for tool call responses
      - Model capabilities and limits
      - Optional schema/response transformers for fine-grained control
    """
    family: str                          # "openai", "anthropic", "deepseek", etc.
    display_name: str                    # Human-readable name
    message_format: MessageFormat = MessageFormat.OPENAI
    tool_schema_format: ToolSchemaFormat = ToolSchemaFormat.OPENAI_FUNCTION
    tool_call_format: ToolCallFormat = ToolCallFormat.OPENAI_TOOL_CALLS
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    # ── Optional transformers ────────────────────────────────────────────
    # These allow per-model tweaks without subclassing

    # Transform tool schema before sending to API
    # fn(base_schema: dict) -> dict
    schema_transformer: Callable[[dict], dict] | None = None

    # Transform API parameters (e.g., add model-specific fields)
    # fn(payload: dict) -> dict
    payload_transformer: Callable[[dict], dict] | None = None

    # Extra metadata
    notes: str = ""


# ═════════════════════════════════════════════════════════════════════════════
# Schema Converters — from Pydantic tool definition to model-specific format
# ═════════════════════════════════════════════════════════════════════════════

def tool_to_openai_function(tool_def: dict) -> dict:
    """Convert universal tool def → OpenAI function calling format."""
    schema = dict(tool_def.get("parameters", {}))
    schema.pop("$defs", None)
    return {
        "type": "function",
        "function": {
            "name": tool_def["name"],
            "description": tool_def["description"],
            "parameters": schema,
        },
    }


def tool_to_openai_strict(tool_def: dict) -> dict:
    """Convert → OpenAI strict function calling (additionalProperties enforcement)."""
    schema = dict(tool_def.get("parameters", {}))
    schema.pop("$defs", None)
    # Strict mode: all properties required, no additionalProperties
    if "properties" in schema:
        schema.setdefault("required", list(schema["properties"].keys()))
        schema["additionalProperties"] = False
    return {
        "type": "function",
        "function": {
            "name": tool_def["name"],
            "description": tool_def["description"],
            "parameters": schema,
            "strict": True,
        },
    }


def tool_to_anthropic(tool_def: dict) -> dict:
    """Convert → Anthropic tool use format."""
    schema = dict(tool_def.get("parameters", {}))
    schema.pop("$defs", None)
    return {
        "name": tool_def["name"],
        "description": tool_def["description"],
        "input_schema": schema,
    }


def tool_to_gemini(tool_def: dict) -> dict:
    """Convert → Gemini function declaration format."""
    schema = dict(tool_def.get("parameters", {}))
    schema.pop("$defs", None)
    # Gemini uses a slightly different schema format
    return {
        "name": tool_def["name"],
        "description": tool_def["description"],
        "parameters": schema,
    }


_SCHEMA_CONVERTERS: dict[ToolSchemaFormat, Callable] = {
    ToolSchemaFormat.OPENAI_FUNCTION: tool_to_openai_function,
    ToolSchemaFormat.OPENAI_STRICT: tool_to_openai_strict,
    ToolSchemaFormat.ANTHROPIC_TOOL_USE: tool_to_anthropic,
    ToolSchemaFormat.GEMINI_FUNCTION: tool_to_gemini,
}


def convert_tool_schema(
    tool_def: dict,
    profile: ModelProfile,
) -> dict:
    """
    Convert a universal tool definition to model-specific format.

    This is the core adapter function:
      Universal → _SCHEMA_CONVERTERS[format] → optional schema_transformer → final
    """
    converter = _SCHEMA_CONVERTERS.get(profile.tool_schema_format)
    if converter is None:
        # Fallback to OpenAI format
        converter = tool_to_openai_function

    result = converter(tool_def)

    # Apply model-specific transformer if present
    if profile.schema_transformer:
        result = profile.schema_transformer(result)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Built-in Model Profiles
# ═════════════════════════════════════════════════════════════════════════════

def _openai_gpt_profile() -> ModelProfile:
    """GPT-4o, GPT-4.1, GPT-4 Turbo, etc."""
    return ModelProfile(
        family="openai",
        display_name="OpenAI GPT",
        message_format=MessageFormat.OPENAI,
        tool_schema_format=ToolSchemaFormat.OPENAI_FUNCTION,
        tool_call_format=ToolCallFormat.OPENAI_TOOL_CALLS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            supports_vision=True,
            max_tools=128,
        ),
    )


def _openai_o_series_profile() -> ModelProfile:
    """o1, o3, o3-mini — reasoning models with different API surface."""
    return ModelProfile(
        family="openai_reasoning",
        display_name="OpenAI Reasoning (o-series)",
        message_format=MessageFormat.OPENAI,
        tool_schema_format=ToolSchemaFormat.OPENAI_FUNCTION,
        tool_call_format=ToolCallFormat.OPENAI_TOOL_CALLS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_parallel_tool_calls=True,
            system_role_name="developer",  # o1 uses "developer" role
        ),
    )


def _anthropic_claude_profile() -> ModelProfile:
    """Claude 3.5, Claude 4.5/4.6 (Sonnet, Opus, Haiku)."""
    return ModelProfile(
        family="anthropic",
        display_name="Anthropic Claude",
        message_format=MessageFormat.ANTHROPIC,
        tool_schema_format=ToolSchemaFormat.ANTHROPIC_TOOL_USE,
        tool_call_format=ToolCallFormat.ANTHROPIC_CONTENT_BLOCKS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            supports_vision=True,
            requires_alternating_roles=True,
            max_tools=128,
            native_tool_tokens=True,
        ),
    )


def _deepseek_profile() -> ModelProfile:
    """DeepSeek Chat / DeepSeek Coder / DeepSeek Reasoner."""

    def _deepseek_schema_tweak(schema: dict) -> dict:
        """DeepSeek works best with concise descriptions."""
        # DeepSeek's function calling is OpenAI-compatible but
        # benefits from shorter, more structured descriptions
        return schema

    return ModelProfile(
        family="deepseek",
        display_name="DeepSeek",
        message_format=MessageFormat.OPENAI,
        tool_schema_format=ToolSchemaFormat.OPENAI_FUNCTION,
        tool_call_format=ToolCallFormat.OPENAI_TOOL_CALLS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            native_tool_tokens=True,
            tool_token_prefix="<｜tool▁call▁begin｜>",
        ),
        schema_transformer=_deepseek_schema_tweak,
        notes="Pretrained with special tool tokens. Uses OpenAI-compat API.",
    )


def _qwen_profile() -> ModelProfile:
    """Qwen 2.5 / Qwen Max / Qwen Plus."""
    return ModelProfile(
        family="qwen",
        display_name="Qwen",
        message_format=MessageFormat.OPENAI,
        tool_schema_format=ToolSchemaFormat.OPENAI_FUNCTION,
        tool_call_format=ToolCallFormat.OPENAI_TOOL_CALLS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            native_tool_tokens=True,
            tool_token_prefix="<tool_call>",
        ),
        notes="Pretrained with <tool_call> XML format. API uses OpenAI compat.",
    )


def _llama_profile() -> ModelProfile:
    """Llama 3.1+, Llama 3.2, Llama 4."""
    return ModelProfile(
        family="llama",
        display_name="Meta Llama",
        message_format=MessageFormat.OPENAI,
        tool_schema_format=ToolSchemaFormat.OPENAI_FUNCTION,
        tool_call_format=ToolCallFormat.OPENAI_TOOL_CALLS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            native_tool_tokens=True,
            tool_token_prefix="<|python_tag|>",
        ),
        notes="Pretrained with <|python_tag|> and JSON tool calls since 3.1.",
    )


def _mistral_profile() -> ModelProfile:
    """Mistral, Mixtral, Mistral Large."""
    return ModelProfile(
        family="mistral",
        display_name="Mistral",
        message_format=MessageFormat.OPENAI,
        tool_schema_format=ToolSchemaFormat.OPENAI_FUNCTION,
        tool_call_format=ToolCallFormat.OPENAI_TOOL_CALLS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            native_tool_tokens=True,
            tool_token_prefix="[TOOL_CALLS]",
        ),
        notes="Pretrained with [TOOL_CALLS] token. API is OpenAI-compat.",
    )


def _gemini_profile() -> ModelProfile:
    """Google Gemini 1.5, Gemini 2.0."""
    return ModelProfile(
        family="gemini",
        display_name="Google Gemini",
        message_format=MessageFormat.OPENAI,  # When used via OpenAI-compat
        tool_schema_format=ToolSchemaFormat.OPENAI_FUNCTION,  # Via compat layer
        tool_call_format=ToolCallFormat.OPENAI_TOOL_CALLS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_parallel_tool_calls=True,
            supports_vision=True,
        ),
        notes="Native API uses functionDeclarations. OpenAI-compat layer available.",
    )


def _generic_openai_compat_profile() -> ModelProfile:
    """Fallback for any unknown model using OpenAI-compatible API."""
    return ModelProfile(
        family="generic",
        display_name="Generic OpenAI-Compatible",
        message_format=MessageFormat.OPENAI,
        tool_schema_format=ToolSchemaFormat.OPENAI_FUNCTION,
        tool_call_format=ToolCallFormat.OPENAI_TOOL_CALLS,
        capabilities=ModelCapabilities(
            supports_tools=True,
            supports_parallel_tool_calls=False,  # Conservative default
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Profile Registry — auto-detect model family from name
# ═════════════════════════════════════════════════════════════════════════════

# Pattern → profile factory
# Order matters: first match wins
_MODEL_PATTERNS: list[tuple[str, Callable[[], ModelProfile]]] = [
    # OpenAI reasoning models (before generic GPT)
    (r"^o[13]", _openai_o_series_profile),
    # OpenAI GPT models
    (r"^gpt-", _openai_gpt_profile),
    # Anthropic Claude
    (r"^claude", _anthropic_claude_profile),
    # DeepSeek
    (r"deepseek", _deepseek_profile),
    # Qwen
    (r"qwen", _qwen_profile),
    # Llama
    (r"llama", _llama_profile),
    # Mistral / Mixtral
    (r"mistral|mixtral", _mistral_profile),
    # Gemini
    (r"gemini", _gemini_profile),
    # Phi
    (r"phi", _generic_openai_compat_profile),
]

# Custom user-registered profiles (checked first)
_CUSTOM_PROFILES: dict[str, ModelProfile] = {}
_CUSTOM_PATTERNS: list[tuple[str, Callable[[], ModelProfile]]] = []


def detect_profile(model: str, base_url: str | None = None) -> ModelProfile:
    """
    Auto-detect the model profile from model name and/or base URL.

    Detection order:
      1. Exact match in custom profiles
      2. Pattern match in custom patterns
      3. Base URL hints (anthropic.com → Claude)
      4. Pattern match in built-in patterns
      5. Fallback to generic OpenAI-compat

    This is the main entry point. Users should call this, not hardcode profiles.
    """
    model_lower = model.lower()

    # 1. Custom exact matches
    if model_lower in _CUSTOM_PROFILES:
        logger.debug("Profile: custom exact match for %s", model)
        return _CUSTOM_PROFILES[model_lower]

    # 2. Custom patterns
    for pattern, factory in _CUSTOM_PATTERNS:
        if re.search(pattern, model_lower):
            logger.debug("Profile: custom pattern %s matched %s", pattern, model)
            return factory()

    # 3. Base URL hints
    if base_url:
        if "anthropic.com" in base_url:
            return _anthropic_claude_profile()

    # 4. Built-in patterns
    for pattern, factory in _MODEL_PATTERNS:
        if re.search(pattern, model_lower):
            logger.debug("Profile: built-in pattern %s matched %s", pattern, model)
            return factory()

    # 5. Fallback
    logger.debug("Profile: no match for %s, using generic", model)
    return _generic_openai_compat_profile()


# ═════════════════════════════════════════════════════════════════════════════
# Dynamic Registration API — users can add their own model profiles
# ═════════════════════════════════════════════════════════════════════════════

def register_profile(model_name: str, profile: ModelProfile) -> None:
    """
    Register a profile for an exact model name.

    Example:
        register_profile("my-custom-model", ModelProfile(
            family="custom",
            display_name="My Custom Model",
            tool_schema_format=ToolSchemaFormat.OPENAI_STRICT,
            capabilities=ModelCapabilities(supports_parallel_tool_calls=False),
        ))
    """
    _CUSTOM_PROFILES[model_name.lower()] = profile
    logger.info("Registered custom profile for %s: %s", model_name, profile.display_name)


def register_pattern(pattern: str, profile_factory: Callable[[], ModelProfile]) -> None:
    """
    Register a regex pattern → profile factory.

    Example:
        register_pattern(r"my-org/.*", lambda: ModelProfile(
            family="my_org",
            display_name="My Org Models",
        ))
    """
    _CUSTOM_PATTERNS.insert(0, (pattern, profile_factory))
    logger.info("Registered custom pattern: %s", pattern)


def list_supported_families() -> list[str]:
    """List all supported model families."""
    families = set()
    for _, factory in _MODEL_PATTERNS:
        p = factory()
        families.add(f"{p.family} ({p.display_name})")
    for name, profile in _CUSTOM_PROFILES.items():
        families.add(f"{profile.family} ({profile.display_name}) [custom: {name}]")
    return sorted(families)
