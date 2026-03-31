"""
Project Scanner — auto-discover project structure at session start.

Runs once when the Agent is created, producing a "project overview" block
that is injected into the system prompt. This gives the model instant
awareness of:
  - Project type and tech stack
  - Directory structure (smart tree)
  - README summary
  - Entry points and key config files
  - Total codebase size estimate

Mirrors the startup project context prefetch (
startDeferredPrefetches in main.tsx), but focused on code-reading use cases.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

# ── Project type detection ────────────────────────────────────────────────────

_MARKERS: list[tuple[list[str], str]] = [
    (["pyproject.toml", "setup.py", "setup.cfg"],          "Python"),
    (["package.json"],                                      "Node.js / JavaScript"),
    (["tsconfig.json"],                                     "TypeScript"),
    (["Cargo.toml"],                                        "Rust"),
    (["go.mod"],                                            "Go"),
    (["pom.xml", "build.gradle", "build.gradle.kts"],       "Java / JVM"),
    (["CMakeLists.txt", "Makefile"],                        "C / C++"),
    (["mix.exs"],                                           "Elixir"),
    (["pubspec.yaml"],                                      "Dart / Flutter"),
    (["Gemfile", "*.gemspec"],                              "Ruby"),
    (["composer.json"],                                     "PHP"),
    (["*.sln", "*.csproj"],                                 "C# / .NET"),
]

_NOISE_DIRS = {
    ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "vendor", ".idea", ".vscode",
}

_NOISE_EXTS = {
    ".pyc", ".pyo", ".lock", ".so", ".dylib", ".dll", ".exe",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".zip", ".gz", ".tar",
}

_MAX_TREE_LINES = 80
_MAX_README_CHARS = 1500
_MAX_TREE_DEPTH = 3


def _detect_project_type(root: Path) -> str:
    for markers, label in _MARKERS:
        for marker in markers:
            if "*" in marker:
                # glob check
                if any(root.glob(marker)):
                    return label
            else:
                if (root / marker).exists():
                    return label
    return "Unknown"


def _smart_tree(root: Path, prefix: str = "", depth: int = 1) -> list[str]:
    if depth > _MAX_TREE_DEPTH:
        return []
    lines: list[str] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return []

    dirs = [e for e in entries if e.is_dir() and e.name not in _NOISE_DIRS
            and not e.name.endswith(".egg-info")]
    files = [e for e in entries if e.is_file()
             and e.suffix.lower() not in _NOISE_EXTS]

    children = dirs + files
    for i, entry in enumerate(children[:_MAX_TREE_LINES]):
        connector = "└── " if i == len(children) - 1 else "├── "
        ext = "    " if i == len(children) - 1 else "│   "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            lines.extend(_smart_tree(entry, prefix + ext, depth + 1))
        else:
            try:
                sz = entry.stat().st_size
                sz_str = f"  ({sz/1024:.1f} KB)" if sz > 1024 else ""
            except OSError:
                sz_str = ""
            lines.append(f"{prefix}{connector}{entry.name}{sz_str}")
    if len(children) > _MAX_TREE_LINES:
        lines.append(f"{prefix}... ({len(children) - _MAX_TREE_LINES} more)")
    return lines


def _read_readme(root: Path) -> Optional[str]:
    for name in ["README.md", "README.rst", "README.txt", "README", "readme.md"]:
        p = root / name
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                if len(text) > _MAX_README_CHARS:
                    text = text[:_MAX_README_CHARS] + "\n… (truncated)"
                return text
            except OSError:
                pass
    return None


def _find_entry_points(root: Path, project_type: str) -> list[str]:
    candidates = []
    patterns = {
        "Python":               ["main.py", "app.py", "run.py", "cli.py",
                                  "src/main.py", "src/app.py", "__main__.py"],
        "Node.js / JavaScript": ["index.js", "src/index.js", "app.js", "server.js"],
        "TypeScript":           ["src/index.ts", "index.ts", "src/main.ts"],
        "Go":                   ["main.go", "cmd/main.go"],
        "Rust":                 ["src/main.rs", "src/lib.rs"],
    }
    for pattern in patterns.get(project_type, []):
        p = root / pattern
        if p.exists():
            candidates.append(str(p.relative_to(root)))
    return candidates


def _count_source_files(root: Path) -> dict[str, int]:
    """Count source files by extension for a quick codebase size estimate."""
    ext_count: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune noise dirs in-place
        dirnames[:] = [d for d in dirnames if d not in _NOISE_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext and ext not in _NOISE_EXTS:
                ext_count[ext] = ext_count.get(ext, 0) + 1
    return ext_count


async def scan_project(cwd: Path) -> str:
    """
    Build a project overview string for injection into the system prompt.
    Runs in a thread to avoid blocking the event loop on large repos.
    """
    def _scan() -> str:
        parts: list[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        project_type = _detect_project_type(cwd)
        parts.append(
            f"## Project Overview: `{cwd.name}`\n"
            f"**Root:** {cwd}\n"
            f"**Tech stack:** {project_type}"
        )

        # ── File count ────────────────────────────────────────────────────────
        try:
            ext_counts = _count_source_files(cwd)
            if ext_counts:
                top = sorted(ext_counts.items(), key=lambda x: -x[1])[:6]
                summary = "  |  ".join(f"{ext}: {n}" for ext, n in top)
                parts.append(f"**Source files:** {sum(ext_counts.values())} total  ({summary})")
        except Exception:
            pass

        # ── Directory tree ─────────────────────────────────────────────────────
        tree_lines = [f"{cwd.name}/"] + _smart_tree(cwd)
        parts.append("**Directory structure:**\n```\n" + "\n".join(tree_lines) + "\n```")

        # ── Entry points ──────────────────────────────────────────────────────
        entry_points = _find_entry_points(cwd, project_type)
        if entry_points:
            parts.append(
                "**Entry points:** " + "  |  ".join(entry_points)
            )

        # ── README ────────────────────────────────────────────────────────────
        readme = _read_readme(cwd)
        if readme:
            parts.append(f"**README:**\n```\n{readme}\n```")

        return "\n\n".join(parts)

    # Run synchronous I/O in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _scan)
