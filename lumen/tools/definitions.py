"""
Definitions tool — extract all symbols (classes, functions, methods) from a file.

This is the "quick map" tool: get the structure of a file without reading all its
lines. Use it to understand what a file contains before diving into specific parts.

Supports:
  Python     — uses AST for 100% accuracy
  TypeScript / JavaScript — regex-based
  Go, Rust, Java, C/C++  — regex-based
  Generic fallback        — finds common definition keywords
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult


@dataclass
class Symbol:
    kind: str        # class / function / method / interface / type / const / etc.
    name: str
    line: int
    parent: str | None = None   # for methods: the enclosing class name


# ── Python (AST) ──────────────────────────────────────────────────────────────

def _extract_python(source: str) -> list[Symbol]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    symbols: list[Symbol] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self._class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            symbols.append(Symbol("class", node.name, node.lineno))
            self._class_stack.append(node.name)
            self.generic_visit(node)
            self._class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            parent = self._class_stack[-1] if self._class_stack else None
            kind = "method" if parent else "function"
            symbols.append(Symbol(kind, node.name, node.lineno, parent))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            parent = self._class_stack[-1] if self._class_stack else None
            kind = "async method" if parent else "async function"
            symbols.append(Symbol(kind, node.name, node.lineno, parent))
            self.generic_visit(node)

    Visitor().visit(tree)
    return symbols


# ── TypeScript / JavaScript ───────────────────────────────────────────────────

_TS_PATTERNS: list[tuple[str, str]] = [
    (r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", "class"),
    (r"^(?:export\s+)?interface\s+(\w+)", "interface"),
    (r"^(?:export\s+)?type\s+(\w+)\s*=", "type"),
    (r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
    (r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
    (r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?function", "function"),
    (r"^\s+(?:(?:public|private|protected|static|async|readonly)\s+)*(\w+)\s*\(", "method"),
    (r"^(?:export\s+)?enum\s+(\w+)", "enum"),
]


def _extract_ts(source: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    current_class: str | None = None
    brace_depth = 0

    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()

        # Track brace depth to detect class exits
        brace_depth += stripped.count("{") - stripped.count("}")

        for pattern, kind in _TS_PATTERNS:
            m = re.match(pattern, stripped)
            if m:
                name = m.group(1)
                if kind == "class":
                    current_class = name
                    symbols.append(Symbol(kind, name, lineno))
                elif kind == "method" and current_class:
                    # Skip constructor noise and getters/setters if too noisy
                    if name not in ("if", "for", "while", "switch", "return", "const",
                                    "let", "var", "new", "throw", "catch"):
                        symbols.append(Symbol(kind, name, lineno, current_class))
                else:
                    symbols.append(Symbol(kind, name, lineno))
                break   # only one pattern per line

    return symbols


# ── Go ────────────────────────────────────────────────────────────────────────

_GO_PATTERNS: list[tuple[str, str]] = [
    (r"^func\s+\((\w+)\s+\*?(\w+)\)\s+(\w+)", "method"),   # method on type
    (r"^func\s+(\w+)\s*\(", "function"),
    (r"^type\s+(\w+)\s+struct", "struct"),
    (r"^type\s+(\w+)\s+interface", "interface"),
    (r"^type\s+(\w+)\s+", "type"),
    (r"^var\s+(\w+)\s+", "var"),
    (r"^const\s+(\w+)\s+", "const"),
]


def _extract_go(source: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        for pattern, kind in _GO_PATTERNS:
            m = re.match(pattern, line)
            if m:
                if kind == "method":
                    receiver_type = m.group(2)
                    method_name = m.group(3)
                    symbols.append(Symbol(kind, method_name, lineno, receiver_type))
                else:
                    symbols.append(Symbol(kind, m.group(1), lineno))
                break
    return symbols


# ── Rust ──────────────────────────────────────────────────────────────────────

_RUST_PATTERNS: list[tuple[str, str]] = [
    (r"^(?:pub(?:\(.*?\))?\s+)?struct\s+(\w+)", "struct"),
    (r"^(?:pub(?:\(.*?\))?\s+)?enum\s+(\w+)", "enum"),
    (r"^(?:pub(?:\(.*?\))?\s+)?trait\s+(\w+)", "trait"),
    (r"^impl(?:<[^>]*>)?\s+(\w+)", "impl"),
    (r"^(?:pub(?:\(.*?\))?\s+)?(?:async\s+)?fn\s+(\w+)", "function"),
    (r"^(?:pub(?:\(.*?\))?\s+)?type\s+(\w+)\s*=", "type"),
]


def _extract_rust(source: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        for pattern, kind in _RUST_PATTERNS:
            m = re.match(pattern, stripped)
            if m:
                symbols.append(Symbol(kind, m.group(1), lineno))
                break
    return symbols


# ── Java / Kotlin ─────────────────────────────────────────────────────────────

_JAVA_PATTERNS: list[tuple[str, str]] = [
    (r"(?:public|private|protected|abstract|final|static|\s)+class\s+(\w+)", "class"),
    (r"(?:public|private|protected|\s)+interface\s+(\w+)", "interface"),
    (r"(?:public|private|protected|static|final|void|\w+(?:<[^>]*>)?)\s+(\w+)\s*\(", "method"),
]


def _extract_java(source: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        for pattern, kind in _JAVA_PATTERNS:
            m = re.search(pattern, stripped)
            if m:
                name = m.group(1)
                # Skip keywords that look like method names
                if name not in ("if", "for", "while", "switch", "new", "return", "catch"):
                    symbols.append(Symbol(kind, name, lineno))
                break
    return symbols


# ── Generic fallback ──────────────────────────────────────────────────────────

_GENERIC_PATTERNS: list[tuple[str, str]] = [
    (r"^\s*(?:class|struct|interface|trait|enum)\s+(\w+)", "class"),
    (r"^\s*(?:def|function|func|fn|sub|procedure)\s+(\w+)", "function"),
    (r"^\s*(?:const|let|var|val)\s+(\w+)\s*=", "constant"),
]


def _extract_generic(source: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        for pattern, kind in _GENERIC_PATTERNS:
            m = re.match(pattern, line)
            if m:
                symbols.append(Symbol(kind, m.group(1), lineno))
                break
    return symbols


# ── Dispatcher ────────────────────────────────────────────────────────────────

def extract_symbols(path: Path, source: str) -> list[Symbol]:
    ext = path.suffix.lower()
    if ext == ".py":
        return _extract_python(source)
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        return _extract_ts(source)
    if ext == ".go":
        return _extract_go(source)
    if ext == ".rs":
        return _extract_rust(source)
    if ext in (".java", ".kt", ".kts"):
        return _extract_java(source)
    return _extract_generic(source)


# ── Tool ──────────────────────────────────────────────────────────────────────

class DefinitionsInput(BaseModel):
    file_path: str = Field(
        description="Absolute path to the file to extract symbols from"
    )


class DefinitionsTool(Tool[DefinitionsInput]):
    """
    Extract all classes, functions, and methods from a file with their line numbers.

    Use this BEFORE read_file to get a map of what's in a file.
    Then use read_file with offset+limit to read specific symbols of interest.

    Supports: Python (AST), TypeScript/JavaScript, Go, Rust, Java/Kotlin.
    """

    @property
    def name(self) -> str:
        return "definitions"

    @property
    def description(self) -> str:
        return (
            "Extract all classes, functions, and methods from a source file "
            "with their line numbers.\n\n"
            "Use this to get a symbol map of a file BEFORE reading it in detail.\n"
            "Then use read_file with offset+limit to read specific symbols.\n\n"
            "Supports: Python (AST), TypeScript/JS, Go, Rust, Java/Kotlin.\n\n"
            "Example workflow:\n"
            "  1. definitions('/path/agent.py')  → see all classes/functions + line numbers\n"
            "  2. read_file('/path/agent.py', offset=38, limit=60)  → read Agent class\n\n"
            'Example: {"file_path": "/abs/path/to/agent.py"}'
        )

    @property
    def input_schema(self) -> type[DefinitionsInput]:
        return DefinitionsInput

    async def execute(self, input_data: DefinitionsInput) -> ToolResult:
        try:
            path = Path(input_data.file_path).expanduser().resolve()

            if not path.exists():
                return ToolResult(
                    success=False, output="",
                    error=f"File not found: {input_data.file_path}",
                )
            if not path.is_file():
                return ToolResult(
                    success=False, output="",
                    error=f"Not a file: {input_data.file_path}",
                )

            source = path.read_text(encoding="utf-8", errors="replace")
            symbols = extract_symbols(path, source)

            if not symbols:
                total_lines = source.count("\n") + 1
                return ToolResult(
                    success=True,
                    output=(
                        f"[definitions] No symbols found in {path.name} "
                        f"({total_lines} lines).\n"
                        "Use read_file to read the full file."
                    ),
                )

            # Format output
            lines_out: list[str] = [
                f"[definitions] {path.name}  ({source.count(chr(10)) + 1} lines total)\n"
            ]

            current_class: str | None = None
            for sym in symbols:
                if sym.kind == "class":
                    current_class = sym.name
                    lines_out.append(f"  line {sym.line:4d}  class {sym.name}")
                elif sym.parent:
                    # Method — indent under its class
                    lines_out.append(f"  line {sym.line:4d}    ├─ {sym.kind}: {sym.name}")
                else:
                    if current_class and sym.kind not in ("class", "struct", "interface", "impl"):
                        current_class = None
                    lines_out.append(f"  line {sym.line:4d}  {sym.kind}: {sym.name}")

            lines_out.append(
                f"\n[definitions] {len(symbols)} symbol(s) found. "
                "Use read_file with offset+limit to read specific sections."
            )

            return ToolResult(success=True, output="\n".join(lines_out))

        except Exception as e:
            return ToolResult(success=False, output="", error=f"definitions error: {e}")
