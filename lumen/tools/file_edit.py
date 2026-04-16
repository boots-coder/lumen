"""
FileEdit tool — surgical find-and-replace in existing files.

Design decisions:
- Exact string matching (not regex) for safety and predictability
- old_string must be unique in the file (prevents ambiguous edits)
- replace_all flag for renaming across the file
- Shows diff-style preview of changes
- Preserves file encoding (UTF-8)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .base import Tool, ToolResult


class FileEditInput(BaseModel):
    """Input schema for FileEdit tool."""

    file_path: str = Field(
        description="Absolute path to the file to edit."
    )
    old_string: str = Field(
        description=(
            "The exact text to find and replace. "
            "Must be unique in the file (unless replace_all=true). "
            "Include enough surrounding context to make it unique."
        )
    )
    new_string: str = Field(
        description=(
            "The replacement text. Must be different from old_string. "
            "Use empty string to delete the matched text."
        )
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "If true, replace ALL occurrences of old_string. "
            "Useful for renaming variables or identifiers across a file."
        ),
    )


class FileEditTool(Tool[FileEditInput]):
    """
    Edit a file by replacing exact string matches.

    This is the preferred tool for modifying existing files — it's surgical,
    safe, and produces clear diffs. Use write_file only when the entire file
    needs to be replaced.
    """

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by finding and replacing an exact string.\n\n"
            "IMPORTANT:\n"
            "  - old_string must match EXACTLY (whitespace, indentation matter)\n"
            "  - old_string must be UNIQUE in the file (or use replace_all=true)\n"
            "  - new_string must be DIFFERENT from old_string\n"
            "  - Include enough context in old_string to make it unique\n"
            "  - Always read the file first before editing!\n\n"
            "Parameters:\n"
            "  file_path    — absolute path to edit (required)\n"
            "  old_string   — exact text to find (required)\n"
            "  new_string   — replacement text (required, use '' to delete)\n"
            "  replace_all  — replace all occurrences (default false)\n\n"
            "Examples:\n"
            '  Single edit:  {"file_path": "/abs/path/f.py", "old_string": "def old_name(", "new_string": "def new_name("}\n'
            '  Rename all:   {"file_path": "/abs/path/f.py", "old_string": "old_var", "new_string": "new_var", "replace_all": true}\n'
            '  Delete code:  {"file_path": "/abs/path/f.py", "old_string": "# TODO: remove this\\nold_code()", "new_string": ""}'
        )

    @property
    def input_schema(self) -> type[FileEditInput]:
        return FileEditInput

    async def execute(self, input_data: FileEditInput) -> ToolResult:
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

            # Read current content
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ToolResult(
                    success=False, output="",
                    error="File contains non-UTF-8 content. Cannot edit binary files.",
                )

            old = input_data.old_string
            new = input_data.new_string

            # Validate
            if old == new:
                return ToolResult(
                    success=False, output="",
                    error="old_string and new_string are identical. Nothing to change.",
                )

            if old not in content:
                # Try to help: show similar lines
                lines = content.splitlines()
                old_first_line = old.splitlines()[0] if old.splitlines() else old
                similar = [
                    f"  line {i+1}: {line.rstrip()}"
                    for i, line in enumerate(lines)
                    if old_first_line.strip() in line
                ]
                hint = ""
                if similar:
                    hint = "\n\nSimilar lines found:\n" + "\n".join(similar[:5])
                    hint += "\n\nCheck indentation and whitespace — they must match exactly."

                return ToolResult(
                    success=False, output="",
                    error=f"old_string not found in {path.name}.{hint}",
                )

            # Check uniqueness
            count = content.count(old)
            if count > 1 and not input_data.replace_all:
                return ToolResult(
                    success=False, output="",
                    error=(
                        f"old_string appears {count} times in {path.name}. "
                        "Include more surrounding context to make it unique, "
                        "or set replace_all=true to replace all occurrences."
                    ),
                )

            # Perform replacement
            if input_data.replace_all:
                new_content = content.replace(old, new)
                replaced_count = count
            else:
                new_content = content.replace(old, new, 1)
                replaced_count = 1

            # Write back
            path.write_text(new_content, encoding="utf-8")

            # Build diff preview
            old_lines = old.splitlines()
            new_lines = new.splitlines()

            diff_parts = []
            for line in old_lines:
                diff_parts.append(f"- {line}")
            for line in new_lines:
                diff_parts.append(f"+ {line}")

            diff_preview = "\n".join(diff_parts)

            # Find the line number where the edit occurred
            before_edit = content[:content.index(old)]
            edit_line = before_edit.count("\n") + 1

            return ToolResult(
                success=True,
                output=(
                    f"[edit_file] {path.name}  (line {edit_line}, "
                    f"{replaced_count} replacement{'s' if replaced_count > 1 else ''})\n\n"
                    f"{diff_preview}\n\n"
                    f"[edit_file] Edit applied successfully."
                ),
            )

        except PermissionError:
            return ToolResult(
                success=False, output="",
                error=f"Permission denied: {input_data.file_path}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=f"edit_file error: {e}")
