"""
Skills — reusable task templates and plugin system.

A Skill is a pre-defined task template with:
  - A name and description
  - A prompt template (with placeholders)
  - Required tools
  - Optional pre/post processing
  - Composability (skills can invoke other skills)

Examples:
  - "commit": stages changes, generates commit message, commits
  - "review-pr": fetches PR diff, analyzes, posts review
  - "refactor": identifies code smells, proposes changes, applies
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    """A reusable task template."""
    name: str
    description: str
    prompt_template: str  # with {placeholders}
    required_tools: list[str] = field(default_factory=list)
    pre_process: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    post_process: Callable[[str], str] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SkillResult:
    """Result from executing a skill."""
    output: str
    success: bool
    skill_name: str
    execution_time_ms: float


# ── Built-in skills ─────────────────────────────────────────────────────────

BUILTIN_SKILLS: list[Skill] = [
    Skill(
        name="commit",
        description="Review staged changes and create a commit with a descriptive message.",
        prompt_template=(
            "Review the staged changes and create a commit with a descriptive "
            "message. {instructions}"
        ),
        required_tools=["bash"],
        tags=["git", "commit", "vcs"],
    ),
    Skill(
        name="explain",
        description="Explain the code in a file in detail.",
        prompt_template="Explain the code in {file_path} in detail. Focus on: {focus}",
        required_tools=["read_file"],
        tags=["explain", "read", "understand"],
    ),
    Skill(
        name="test",
        description="Write tests for a given file.",
        prompt_template="Write tests for {file_path}. Framework: {framework}",
        required_tools=["read_file", "write_file"],
        tags=["test", "testing", "quality"],
    ),
]


# ── Registry ─────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Registry for discovering and managing skills.

    Supports programmatic registration and loading from YAML/JSON files.
    """

    def __init__(self, *, load_builtins: bool = True) -> None:
        self._skills: dict[str, Skill] = {}
        if load_builtins:
            for skill in BUILTIN_SKILLS:
                self._skills[skill.name] = skill

    def register(self, skill: Skill) -> None:
        """Register a skill. Overwrites if name already exists."""
        self._skills[skill.name] = skill
        logger.debug("Registered skill: %s", skill.name)

    def unregister(self, name: str) -> None:
        """Remove a skill by name. No-op if not found."""
        if name in self._skills:
            del self._skills[name]
            logger.debug("Unregistered skill: %s", name)

    def get(self, name: str) -> Skill | None:
        """Get a skill by exact name."""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """Return all registered skills, sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def search(self, query: str) -> list[Skill]:
        """
        Fuzzy search skills by name, description, and tags.

        Returns skills sorted by relevance (best match first).
        """
        query_lower = query.lower()
        scored: list[tuple[float, Skill]] = []

        for skill in self._skills.values():
            score = 0.0

            # Exact name match
            if skill.name == query_lower:
                score += 10.0
            # Name contains query
            elif query_lower in skill.name.lower():
                score += 5.0
            # Query contains name
            elif skill.name.lower() in query_lower:
                score += 3.0

            # Tag match
            for tag in skill.tags:
                if query_lower in tag.lower():
                    score += 2.0
                elif tag.lower() in query_lower:
                    score += 1.0

            # Description match
            if query_lower in skill.description.lower():
                score += 1.0

            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored]

    def load_from_directory(self, path: Path) -> int:
        """
        Load skill definitions from .yaml and .json files in a directory.

        Each file should define a single skill with keys:
          name, description, prompt_template, required_tools, tags

        Returns the number of skills loaded.
        """
        path = Path(path)
        if not path.is_dir():
            logger.warning("Skill directory does not exist: %s", path)
            return 0

        loaded = 0
        for file_path in sorted(path.iterdir()):
            if file_path.suffix == ".json":
                loaded += self._load_json_skill(file_path)
            elif file_path.suffix in (".yaml", ".yml"):
                loaded += self._load_yaml_skill(file_path)

        logger.info("Loaded %d skills from %s", loaded, path)
        return loaded

    def _load_json_skill(self, file_path: Path) -> int:
        """Load a skill from a JSON file."""
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            skill = self._dict_to_skill(data)
            self.register(skill)
            return 1
        except Exception as e:
            logger.error("Failed to load skill from %s: %s", file_path, e)
            return 0

    def _load_yaml_skill(self, file_path: Path) -> int:
        """Load a skill from a YAML file. Requires PyYAML."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "PyYAML not installed — cannot load %s. "
                "Install with: pip install pyyaml",
                file_path,
            )
            return 0

        try:
            data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            skill = self._dict_to_skill(data)
            self.register(skill)
            return 1
        except Exception as e:
            logger.error("Failed to load skill from %s: %s", file_path, e)
            return 0

    @staticmethod
    def _dict_to_skill(data: dict[str, Any]) -> Skill:
        """Convert a dictionary to a Skill dataclass."""
        return Skill(
            name=data["name"],
            description=data.get("description", ""),
            prompt_template=data["prompt_template"],
            required_tools=data.get("required_tools", []),
            tags=data.get("tags", []),
            # pre_process and post_process cannot be defined in files
        )

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


# ── Executor ─────────────────────────────────────────────────────────────────

class SkillExecutor:
    """
    Executes skills by filling prompt templates and running them through an Agent.
    """

    async def execute(
        self,
        skill: Skill,
        agent: Agent,
        params: dict[str, Any] | None = None,
    ) -> SkillResult:
        """
        Execute a skill.

        1. Validate required tools are available on the agent
        2. Run pre_process on params (if defined)
        3. Fill the prompt template with params
        4. Call agent.chat() with the filled prompt
        5. Run post_process on the output (if defined)

        Parameters
        ----------
        skill : Skill
            The skill to execute.
        agent : Agent
            The agent to run the skill on.
        params : dict
            Template parameters (keys matching {placeholders}).

        Returns
        -------
        SkillResult with the output and execution metadata.
        """
        params = params or {}
        start_time = time.monotonic()

        try:
            # 1. Check required tools
            missing = self._check_required_tools(skill, agent)
            if missing:
                return SkillResult(
                    output=f"Missing required tools: {', '.join(missing)}",
                    success=False,
                    skill_name=skill.name,
                    execution_time_ms=self._elapsed_ms(start_time),
                )

            # 2. Pre-process
            if skill.pre_process:
                params = skill.pre_process(params)

            # 3. Fill template
            prompt = self._fill_template(skill.prompt_template, params)

            # 4. Execute via agent
            response = await agent.chat(prompt)

            # 5. Post-process
            output = response.content
            if skill.post_process:
                output = skill.post_process(output)

            return SkillResult(
                output=output,
                success=True,
                skill_name=skill.name,
                execution_time_ms=self._elapsed_ms(start_time),
            )

        except Exception as e:
            logger.error("Skill '%s' execution failed: %s", skill.name, e)
            return SkillResult(
                output=f"Execution error: {e}",
                success=False,
                skill_name=skill.name,
                execution_time_ms=self._elapsed_ms(start_time),
            )

    @staticmethod
    def _check_required_tools(skill: Skill, agent: Agent) -> list[str]:
        """Check if the agent has all required tools. Returns missing tool names."""
        missing: list[str] = []
        for tool_name in skill.required_tools:
            if agent._tool_registry.get(tool_name) is None:
                missing.append(tool_name)
        return missing

    @staticmethod
    def _fill_template(template: str, params: dict[str, Any]) -> str:
        """
        Fill a prompt template with parameters.

        Missing keys are replaced with empty strings to avoid KeyError.
        """
        # Build a defaultdict-like approach: replace known keys, leave unknowns empty
        result = template
        for key, value in params.items():
            result = result.replace(f"{{{key}}}", str(value))

        # Replace any remaining {placeholders} with empty string
        import re
        result = re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", "", result)
        return result.strip()

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.monotonic() - start) * 1000
