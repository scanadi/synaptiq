"""Token counting and budget-aware response truncation.

Provides lightweight token estimation (chars / 4) and structural
truncation so MCP tool responses stay within agent context budgets.
"""

from __future__ import annotations

import re


def count_tokens(text: str) -> int:
    """Estimate token count using chars / 4 heuristic."""
    return max(1, len(text) // 4)


def wrap_with_metadata(text: str) -> str:
    """Append token-count metadata to a tool response."""
    tokens = count_tokens(text)
    return f"{text}\n\n--- tokens: {tokens} ---"


_METADATA_FOOTER = re.compile(r"\n*--- tokens: \d+ ---\s*$")


def strip_metadata(text: str) -> str:
    """Remove the footer added by :func:`wrap_with_metadata`.

    Lives here so the footer format has exactly one owner — consumers
    (e.g. the CLI) must not hardcode the pattern.
    """
    return _METADATA_FOOTER.sub("", text)


def truncate_response(text: str, max_tokens: int) -> str:
    """Truncate *text* to fit within *max_tokens*.

    Uses a structural approach:

    - If the text contains numbered list items (``1. ...``, ``2. ...``),
      removes items from the end until the response fits.
    - If the text contains section headers (``=== ... ===`` or lines
      followed by ``===``), removes trailing sections.
    - Otherwise falls back to character-level truncation.
    """
    if max_tokens <= 0 or count_tokens(text) <= max_tokens:
        return text

    max_chars = max_tokens * 4

    # Strategy 1: numbered-list truncation.
    lines = text.split("\n")
    list_indices = [i for i, line in enumerate(lines) if _is_numbered_item(line)]
    if len(list_indices) >= 3:
        return _truncate_list_items(lines, list_indices, max_chars)

    # Strategy 2: section-based truncation.
    sections = _split_sections(text)
    if len(sections) >= 2:
        return _truncate_sections(sections, max_chars)

    # Strategy 3: hard character truncation.
    truncated = text[:max_chars]
    return truncated + "\n\n[... truncated to fit token budget]"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _is_numbered_item(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    # Match "1. ", "  2. ", "10. " etc.
    i = 0
    while i < len(stripped) and stripped[i].isdigit():
        i += 1
    return i > 0 and stripped[i : i + 2] == ". "


def _truncate_list_items(
    lines: list[str], list_indices: list[int], max_chars: int
) -> str:
    """Remove list items from the end until the result fits."""
    kept_up_to = len(lines)
    for idx in reversed(list_indices):
        kept_up_to = idx
        candidate = "\n".join(lines[:kept_up_to])
        if len(candidate) <= max_chars:
            total_items = len(list_indices)
            shown = sum(1 for i in list_indices if i < kept_up_to)
            suffix = f"\n\n[... showing {shown}/{total_items} items to fit token budget]"
            return candidate + suffix
    # Even the first item doesn't fit — hard truncate.
    return "\n".join(lines[:kept_up_to])[:max_chars] + "\n\n[... truncated]"


def _split_sections(text: str) -> list[str]:
    """Split text on ``=== ... ===`` section headers."""
    parts = re.split(r"(?=^===\s)", text, flags=re.MULTILINE)
    return [p for p in parts if p.strip()]


def _truncate_sections(sections: list[str], max_chars: int) -> str:
    """Drop trailing sections until the response fits."""
    result = ""
    kept = 0
    for section in sections:
        candidate = result + section
        if len(candidate) > max_chars and kept > 0:
            break
        result = candidate
        kept += 1
    total = len(sections)
    if kept < total:
        result += f"\n\n[... showing {kept}/{total} sections to fit token budget]"
    return result
