"""
Compaction prompt templates.

Compaction prompt templates.

The compaction prompt instructs the model to produce a structured summary
of the conversation so far. The summary is then injected as a replacement
for the full history, drastically reducing token usage while preserving
essential context.

Two modes:
  BASE    — summarise the entire conversation
  PARTIAL — summarise only the recent portion; earlier context is kept intact
"""

from __future__ import annotations

# ── Preamble / trailer (tool-call suppression) ────────────────────────────────
# Added to prevent the model from calling tools during
# compaction (it only has one shot and tool calls would waste it).

NO_TOOLS_PREAMBLE = """\
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- You already have all the context you need in the conversation above.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block."
)

# ── Analysis instructions ─────────────────────────────────────────────────────

_ANALYSIS_INSTRUCTION_BASE = """\
Before providing your final summary, wrap your analysis in <analysis> tags to organise \
your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyse each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if \
the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.\
"""

_ANALYSIS_INSTRUCTION_PARTIAL = """\
Before providing your final summary, wrap your analysis in <analysis> tags to organise \
your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Analyse the recent messages chronologically. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if \
the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.\
"""

# ── Summary structure (9 sections) ──────────────────────────

_SUMMARY_SECTIONS = """\
Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. \
Pay special attention to the most recent messages and include full code snippets where applicable \
and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention \
to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for \
understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this \
summary request, paying special attention to the most recent messages from both user and assistant. \
Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work \
you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent \
explicit requests, and the task you were working on immediately before this summary request. \
If your last task was concluded, then only list next steps if they are explicitly in line with \
the users request. Do not start on tangential requests or really old requests that were already \
completed without confirming with the user first.
   If there is a next step, include direct quotes from the most recent conversation showing exactly \
what task you were working on and where you left off. This should be verbatim to ensure there's no \
drift in task interpretation.\
"""

_EXAMPLE_STRUCTURE = """\
Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Important Code Snippet]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]

7. Pending Tasks:
   - [Task 1]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]
</summary>
</example>\
"""

# ── Full prompt templates ─────────────────────────────────────────────────────

BASE_COMPACT_PROMPT = (
    NO_TOOLS_PREAMBLE
    + "Your task is to create a detailed summary of the conversation so far, "
    "paying close attention to the user's explicit requests and your previous actions.\n"
    "This summary should be thorough in capturing technical details, code patterns, "
    "and architectural decisions that would be essential for continuing development "
    "work without losing context.\n\n"
    + _ANALYSIS_INSTRUCTION_BASE
    + "\n\n"
    + _SUMMARY_SECTIONS
    + "\n\n"
    + _EXAMPLE_STRUCTURE
    + "\n\nPlease provide your summary based on the conversation so far, following "
    "this structure and ensuring precision and thoroughness in your response."
    + NO_TOOLS_TRAILER
)

PARTIAL_COMPACT_PROMPT = (
    NO_TOOLS_PREAMBLE
    + "Your task is to create a detailed summary of the RECENT portion of the "
    "conversation — the messages that follow earlier retained context. "
    "The earlier messages are being kept intact and do NOT need to be summarised. "
    "Focus your summary on what was discussed, learned, and accomplished in the recent messages only.\n\n"
    + _ANALYSIS_INSTRUCTION_PARTIAL
    + "\n\n"
    + _SUMMARY_SECTIONS
    + "\n\n"
    + _EXAMPLE_STRUCTURE
    + "\n\nPlease provide your summary based on the RECENT messages only "
    "(after the retained earlier context), following this structure and ensuring "
    "precision and thoroughness in your response."
    + NO_TOOLS_TRAILER
)


# ── Post-processing ───────────────────────────────────────────────────────────

def extract_summary(raw_response: str) -> str:
    """
    Extract the <summary> block from the model's compaction response.

    Strips the <analysis> scratchpad (which was only for the model's reasoning)
    and returns the formatted summary text.

    Extracts the clean summary from the raw model response.
    """
    import re

    # Remove <analysis>...</analysis> block entirely
    cleaned = re.sub(r"<analysis>.*?</analysis>", "", raw_response, flags=re.DOTALL).strip()

    # Extract content inside <summary>...</summary>
    match = re.search(r"<summary>(.*?)</summary>", cleaned, flags=re.DOTALL)
    if match:
        return "Summary:\n" + match.group(1).strip()

    # Fallback: return the cleaned text as-is if tags are missing
    return "Summary:\n" + cleaned


def build_compact_user_message(summary: str) -> str:
    """
    Format the summary as a user message that replaces the full history.

    The message starts with a clear marker so both the model and future
    compaction passes recognise it as a compaction boundary.
    """
    return (
        "[CONTEXT COMPACTED]\n\n"
        "The conversation history above this point has been summarised to manage "
        "context window size. The summary below captures all essential information:\n\n"
        + summary
    )
