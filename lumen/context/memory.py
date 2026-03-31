"""
Memory file discovery and loading.

Memory file discovery and loading — four priority layers,
directory traversal,
@include directives, and content limits.

Priority order (lowest → highest, later files win):
  1. SYSTEM  — /etc/engram/ENGRAM.md       (org-wide policy)
  2. USER    — ~/.engram/ENGRAM.md         (personal, all projects)
  3. PROJECT — ./ENGRAM.md, ./.engram/rules/*.md  (team, checked-in)
  4. LOCAL   — ./ENGRAM.local.md           (private, gitignored)

Files discovered closer to cwd take priority over parent directories.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .._types import MemoryFile, MemoryLayer

logger = logging.getLogger(__name__)

# ── limits (MAX_MEMORY_CHARS = 40_000) ───────
MAX_MEMORY_CHARS = 40_000
MAX_INCLUDE_DEPTH = 10  # prevent infinite @include loops

# ── instruction header injected before memory content ────────────────────────
MEMORY_HEADER = (
    "Codebase and user instructions are shown below. "
    "Be sure to adhere to these instructions. "
    "IMPORTANT: These instructions OVERRIDE any default behaviour "
    "and you MUST follow them exactly as written."
)

# Allowed text file extensions for @include
_TEXT_EXTENSIONS = {
    ".md", ".txt", ".text", ".json", ".yaml", ".yml", ".toml",
    ".env", ".ini", ".cfg", ".conf", ".config", ".properties",
    ".py", ".pyi", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
    ".java", ".kt", ".cs", ".swift", ".rb", ".sh", ".bash",
    ".sql", ".graphql", ".html", ".css", ".xml", ".csv",
}


# ── @include resolution ───────────────────────────────────────────────────────

def _resolve_include(ref: str, relative_to: Path) -> Optional[Path]:
    """Resolve an @include path to an absolute Path, or None if invalid."""
    ref = ref.strip()
    if ref.startswith("~/"):
        path = Path.home() / ref[2:]
    elif ref.startswith("/"):
        path = Path(ref)
    else:
        # ./relative or bare relative
        clean = ref.lstrip("./") if ref.startswith("./") else ref
        path = relative_to / (ref[2:] if ref.startswith("./") else ref)

    if path.suffix.lower() not in _TEXT_EXTENSIONS:
        logger.debug("@include ignored (unsupported extension): %s", path)
        return None
    return path.resolve()


def _expand_includes(
    content: str,
    source_path: Path,
    visited: set[Path],
    depth: int = 0,
) -> str:
    """
    Recursively expand @include directives in memory file content.

    Lines matching /^@\S+/ outside code fences are treated as includes.
    Circular references are silently skipped. Missing files are ignored.
    """
    if depth >= MAX_INCLUDE_DEPTH:
        return content

    lines = content.splitlines(keepends=True)
    result: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        # Track code fence state
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if not in_code_block and stripped.startswith("@") and len(stripped) > 1:
            ref = stripped[1:]
            resolved = _resolve_include(ref, source_path.parent)
            if resolved is None or resolved in visited or not resolved.exists():
                result.append(line)
                continue
            visited.add(resolved)
            try:
                included = resolved.read_text(encoding="utf-8", errors="replace")
                expanded = _expand_includes(included, resolved, visited, depth + 1)
                result.append(f"\n<!-- @include: {resolved} -->\n")
                result.append(expanded)
                result.append("\n")
            except OSError:
                result.append(line)
        else:
            result.append(line)

    return "".join(result)


# ── file loading ──────────────────────────────────────────────────────────────

def _load_file(path: Path, layer: MemoryLayer) -> Optional[MemoryFile]:
    """Load a single memory file, expanding @includes. Returns None if unreadable."""
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Cannot read memory file %s: %s", path, e)
        return None

    # Strip frontmatter (--- ... ---)
    content = re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL)
    # Expand @include directives
    content = _expand_includes(content, path, visited={path.resolve()})
    return MemoryFile(path=str(path), layer=layer, content=content.strip())


def _load_dir_rules(rules_dir: Path, layer: MemoryLayer) -> list[MemoryFile]:
    """Load all .md files from a .engram/rules/ directory."""
    files: list[MemoryFile] = []
    if rules_dir.is_dir():
        for p in sorted(rules_dir.glob("*.md")):
            f = _load_file(p, layer)
            if f:
                files.append(f)
    return files


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_memory_files(cwd: Optional[Path] = None) -> list[MemoryFile]:
    """
    Discover all ENGRAM.md memory files applicable to *cwd*.

    Returns files ordered lowest-to-highest priority (highest priority last).
    Files loaded lowest-to-highest priority; later entries override earlier.
    """
    cwd = (cwd or Path.cwd()).resolve()
    files: list[MemoryFile] = []

    # ── Layer 1: System ────────────────────────────────────────────────────
    system_path = Path("/etc/engram/ENGRAM.md")
    f = _load_file(system_path, MemoryLayer.SYSTEM)
    if f:
        files.append(f)

    # ── Layer 2: User ──────────────────────────────────────────────────────
    user_path = Path.home() / ".engram" / "ENGRAM.md"
    f = _load_file(user_path, MemoryLayer.USER)
    if f:
        files.append(f)

    # ── Layers 3 & 4: Walk up from cwd to root ─────────────────────────────
    # Collect directories root→cwd so that cwd files load last (highest priority)
    ancestors: list[Path] = []
    current = cwd
    while True:
        ancestors.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    ancestors.reverse()  # root first, cwd last

    for directory in ancestors:
        # Layer 3 — project files
        for candidate in [
            directory / "ENGRAM.md",
            directory / ".engram" / "ENGRAM.md",
        ]:
            f = _load_file(candidate, MemoryLayer.PROJECT)
            if f:
                files.append(f)

        rules_files = _load_dir_rules(directory / ".engram" / "rules", MemoryLayer.PROJECT)
        files.extend(rules_files)

        # Layer 4 — local private file
        local = directory / "ENGRAM.local.md"
        f = _load_file(local, MemoryLayer.LOCAL)
        if f:
            files.append(f)

    logger.debug(
        "Discovered %d memory file(s): %s",
        len(files),
        [f.path for f in files],
    )
    return files


def build_memory_prompt(files: list[MemoryFile]) -> str:
    """
    Concatenate memory files into a single prompt string.

    Applies the MAX_MEMORY_CHARS limit (40,000 chars).
    """
    if not files:
        return ""

    parts: list[str] = [MEMORY_HEADER, ""]
    total_chars = len(MEMORY_HEADER) + 1

    for mf in files:
        label = f"[{mf.layer.value.upper()}: {mf.path}]"
        block = f"{label}\n{mf.content}\n"
        if total_chars + len(block) > MAX_MEMORY_CHARS:
            remaining = MAX_MEMORY_CHARS - total_chars
            if remaining > 100:
                block = block[:remaining] + "\n... (truncated)"
                parts.append(block)
            break
        parts.append(block)
        total_chars += len(block)

    return "\n".join(parts)
