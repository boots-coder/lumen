"""
Session Memory — dynamic key-fact extraction across conversations.

Automatically extracts and persists:
  - User preferences and corrections
  - Project context (tech stack, architecture decisions)
  - Recurring patterns and solutions
  - Important file paths and their purposes

Design: runs extraction after each conversation turn, stores in a lightweight
JSON file, loads on session start. Uses the LLM itself for extraction.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .._types import Message, Role

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Types
# ═════════════════════════════════════════════════════════════════════════════

class MemoryCategory(str, Enum):
    """Categories of extracted session memories."""
    PREFERENCE = "preference"    # User preferences and corrections
    PROJECT = "project"          # Tech stack, architecture decisions
    PATTERN = "pattern"          # Recurring patterns and solutions
    REFERENCE = "reference"      # Important file paths and their purposes
    CORRECTION = "correction"    # Explicit corrections from the user


@dataclass
class MemoryEntry:
    """A single extracted memory fact."""
    content: str
    category: MemoryCategory
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_turn: int = 0
    relevance_score: float = 0.5  # 0.0–1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "source_turn": self.source_turn,
            "relevance_score": self.relevance_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            content=data["content"],
            category=MemoryCategory(data["category"]),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            source_turn=data.get("source_turn", 0),
            relevance_score=data.get("relevance_score", 0.5),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Extraction prompt — sent to the LLM to extract key facts
# ═════════════════════════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """\
You are a memory extraction system. Analyse the following conversation messages \
and extract key facts that would be useful to remember for future conversations.

Extract facts in these categories:
- preference: User preferences, style choices, conventions they follow
- project: Tech stack, architecture decisions, project structure
- pattern: Recurring patterns, solutions, approaches that work
- reference: Important file paths and their purposes
- correction: Explicit corrections the user made ("no, do it this way")

Return a JSON array of objects with these fields:
- "content": the fact (concise, one sentence)
- "category": one of the categories above
- "relevance_score": 0.0 to 1.0 (how important/reusable this fact is)

Return ONLY the JSON array, no other text. If there are no facts worth \
extracting, return an empty array [].

Messages:
"""


# ═════════════════════════════════════════════════════════════════════════════
# TF-IDF keyword relevance (lightweight, no external deps)
# ═════════════════════════════════════════════════════════════════════════════

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "not", "with", "this", "that", "from", "by", "as", "be",
    "was", "were", "are", "been", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can",
    "i", "you", "we", "they", "he", "she", "my", "your", "our",
})


def _tokenise(text: str) -> list[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return [
        w for w in re.findall(r"[a-z0-9_./]+", text.lower())
        if w not in _STOP_WORDS and len(w) > 1
    ]


def _tf_idf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    """Compute a simple TF-IDF relevance score between query and document."""
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_counts = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for token in query_tokens:
        tf = doc_counts.get(token, 0) / doc_len if doc_len > 0 else 0.0
        score += tf * idf.get(token, 0.0)
    return score


# ═════════════════════════════════════════════════════════════════════════════
# SessionMemoryManager
# ═════════════════════════════════════════════════════════════════════════════

class SessionMemoryManager:
    """
    Manages dynamic session memory — extraction, storage, and retrieval.

    Memory is stored as a lightweight JSON file. Extraction uses the LLM
    itself to identify key facts from conversation turns. Retrieval uses
    simple TF-IDF keyword matching (no vector DB needed).
    """

    def __init__(self, storage_path: Path, max_entries: int = 200) -> None:
        self._storage_path = Path(storage_path)
        self._max_entries = max_entries
        self._entries: list[MemoryEntry] = []
        self._turn_count: int = 0

    # ── Extraction ───────────────────────────────────────────────────────────

    async def extract_from_turn(
        self,
        messages: list[Message],
        provider: Any,
        model: str,
    ) -> list[MemoryEntry]:
        """
        Extract key facts from recent messages using the LLM.

        Sends the last few messages to the LLM with an extraction prompt,
        parses the structured response, deduplicates against existing entries,
        and appends new entries.
        """
        self._turn_count += 1

        # Build message text for extraction (last 6 messages max)
        recent = messages[-6:] if len(messages) > 6 else messages
        conversation_text = "\n".join(
            f"[{m.role.value}]: {m.content[:500]}"
            for m in recent
            if m.content and m.role in (Role.USER, Role.ASSISTANT)
        )

        if not conversation_text.strip():
            return []

        extraction_input = _EXTRACTION_PROMPT + conversation_text

        try:
            response = await provider.chat(
                messages=[{"role": "user", "content": extraction_input}],
                system="You extract structured facts from conversations. Return only valid JSON.",
                max_tokens=1000,
                temperature=0.0,
            )

            raw_content = response.content or ""
            new_entries = self._parse_extraction_response(raw_content)
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return []

        # Deduplicate against existing entries
        deduplicated = self._deduplicate(new_entries)

        # Set source turn
        for entry in deduplicated:
            entry.source_turn = self._turn_count

        self._entries.extend(deduplicated)

        # Prune if over limit
        if len(self._entries) > self._max_entries:
            self.prune(self._max_entries)

        logger.debug(
            "Extracted %d new memories (%d total)",
            len(deduplicated),
            len(self._entries),
        )
        return deduplicated

    def _parse_extraction_response(self, content: str) -> list[MemoryEntry]:
        """Parse the LLM's JSON response into MemoryEntry objects."""
        # Try to find JSON array in the response
        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON array from surrounding text
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.debug("Could not parse extraction response as JSON")
                    return []
            else:
                return []

        if not isinstance(data, list):
            return []

        entries: list[MemoryEntry] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                category = MemoryCategory(item.get("category", "reference"))
            except ValueError:
                category = MemoryCategory.REFERENCE

            score = item.get("relevance_score", 0.5)
            if not isinstance(score, (int, float)):
                score = 0.5
            score = max(0.0, min(1.0, float(score)))

            content_text = item.get("content", "")
            if content_text and isinstance(content_text, str):
                entries.append(MemoryEntry(
                    content=content_text,
                    category=category,
                    relevance_score=score,
                ))
        return entries

    def _deduplicate(self, new_entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Remove entries that are too similar to existing ones."""
        deduplicated: list[MemoryEntry] = []
        existing_tokens = [_tokenise(e.content) for e in self._entries]

        for entry in new_entries:
            entry_tokens = _tokenise(entry.content)
            is_duplicate = False

            for existing in existing_tokens:
                if not entry_tokens or not existing:
                    continue
                # Jaccard similarity
                set_a = set(entry_tokens)
                set_b = set(existing)
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                similarity = intersection / union if union > 0 else 0.0
                if similarity > 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(entry)
                existing_tokens.append(entry_tokens)

        return deduplicated

    # ── Retrieval ────────────────────────────────────────────────────────────

    def get_relevant_context(self, query: str, top_k: int = 10) -> str:
        """
        Retrieve the most relevant memory entries for a query.

        Uses simple TF-IDF keyword matching — no vector DB needed.
        Returns a formatted context string suitable for system prompt injection.
        """
        if not self._entries:
            return ""

        query_tokens = _tokenise(query)
        if not query_tokens:
            # No meaningful query tokens — return top entries by relevance score
            sorted_entries = sorted(
                self._entries, key=lambda e: e.relevance_score, reverse=True
            )
            top = sorted_entries[:top_k]
        else:
            # Build IDF from all entries
            num_docs = len(self._entries)
            doc_freq: Counter[str] = Counter()
            entry_tokens_list: list[list[str]] = []
            for entry in self._entries:
                tokens = _tokenise(entry.content)
                entry_tokens_list.append(tokens)
                for token in set(tokens):
                    doc_freq[token] += 1

            idf: dict[str, float] = {}
            for token, freq in doc_freq.items():
                idf[token] = math.log(num_docs / (1 + freq)) + 1

            # Score each entry
            scored: list[tuple[float, MemoryEntry]] = []
            for entry, tokens in zip(self._entries, entry_tokens_list):
                text_score = _tf_idf_score(query_tokens, tokens, idf)
                # Blend text relevance with stored relevance score
                combined = 0.7 * text_score + 0.3 * entry.relevance_score
                scored.append((combined, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = [entry for _, entry in scored[:top_k]]

        if not top:
            return ""

        # Format as context string
        lines = ["# Session Memory (auto-extracted)", ""]
        for entry in top:
            tag = entry.category.value.upper()
            lines.append(f"- [{tag}] {entry.content}")

        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist memory entries to JSON file."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "turn_count": self._turn_count,
            "entries": [e.to_dict() for e in self._entries],
        }
        self._storage_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Session memory saved to %s (%d entries)", self._storage_path, len(self._entries))

    def load(self) -> None:
        """Load memory entries from JSON file."""
        if not self._storage_path.exists():
            logger.debug("No session memory file at %s", self._storage_path)
            return

        try:
            raw = self._storage_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load session memory from %s: %s", self._storage_path, e)
            return

        self._turn_count = data.get("turn_count", 0)
        self._entries = []
        for entry_data in data.get("entries", []):
            try:
                self._entries.append(MemoryEntry.from_dict(entry_data))
            except (KeyError, ValueError) as e:
                logger.debug("Skipping malformed memory entry: %s", e)

        logger.debug(
            "Loaded %d session memories from %s",
            len(self._entries),
            self._storage_path,
        )

    # ── Pruning ──────────────────────────────────────────────────────────────

    def prune(self, max_entries: int | None = None) -> int:
        """
        Remove lowest-relevance entries when over the limit.

        Returns the number of entries removed.
        """
        limit = max_entries or self._max_entries
        if len(self._entries) <= limit:
            return 0

        # Sort by relevance score (ascending) — remove lowest first
        self._entries.sort(key=lambda e: e.relevance_score)
        to_remove = len(self._entries) - limit
        self._entries = self._entries[to_remove:]
        logger.debug("Pruned %d low-relevance memory entries", to_remove)
        return to_remove

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def entries(self) -> list[MemoryEntry]:
        """Read-only access to memory entries."""
        return list(self._entries)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"SessionMemoryManager(entries={len(self._entries)}, "
            f"turns={self._turn_count}, "
            f"path={self._storage_path})"
        )
