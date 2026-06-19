"""Pre-tool routing hints for AI agents.

Given a natural language question, returns a suggested sequence of Synaptiq
tool calls so agents avoid trial-and-error discovery.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from synaptiq.core.storage.base import StorageBackend


@dataclass
class ToolSuggestion:
    """A single suggested tool call."""

    tool_name: str
    arguments: dict
    reason: str


# ------------------------------------------------------------------
# Question-type classification rules (order matters — first match wins).
# ------------------------------------------------------------------

_RULES: list[tuple[re.Pattern[str], str]] = [
    # Dead code
    (re.compile(r"\bdead\s*code\b", re.I), "dead_code"),
    # Impact / blast radius
    (re.compile(r"\b(?:impact|blast\s*radius|what\s+breaks|affect)\b", re.I), "impact"),
    # Call path
    (re.compile(r"\b(?:path|chain|route)\s+(?:from|between)\b", re.I), "call_path"),
    # Callers / who uses
    (re.compile(r"\b(?:who\s+(?:calls|uses)|callers?\s+of|what\s+calls)\b", re.I), "context"),
    # Coupling
    (re.compile(r"\b(?:coupl\w*|co-?change|change\s+together)\b", re.I), "coupling"),
    # Cycles
    (re.compile(r"\b(?:circular|cycle|cyclic)\b", re.I), "cycles"),
    # Test impact
    (
        re.compile(
            r"\b(?:tests?\s+(?:\w+\s+)*(?:impact|affected|cover)|test\s+impact)\b",
            re.I,
        ),
        "test_impact",
    ),
    # Communities
    (re.compile(r"\b(?:communit\w*|cluster\w*|module\s+group)\b", re.I), "communities"),
    # File context
    (re.compile(r"\b(?:file|what.s\s+in)\s+\S+\.(?:py|ts|js|tsx|jsx)\b", re.I), "file_context"),
    # Explain
    (re.compile(r"\b(?:explain|describe|what\s+(?:is|does))\b", re.I), "explain"),
    # Review risk
    (re.compile(r"\b(?:review|risk|pr\s+risk)\b", re.I), "review_risk"),
]

# Patterns for extracting symbol names from questions, paired with an
# optional per-pattern filter.  PascalCase requires two uppercase humps so
# capitalized English words don't match; digits and acronym runs are allowed
# (KPIData, Base64Encoder) but pure acronyms (USA) are filtered — unless the
# user quoted them explicitly.
_SYMBOL_PATTERNS: list[tuple[re.Pattern[str], Callable[[str], bool] | None]] = [
    (re.compile(r"[`'\"]([A-Za-z_]\w*(?:\.\w+)*)[`'\"]"), None),  # Quoted
    (re.compile(r"\b([A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)\b"), str.isupper),  # PascalCase
    (re.compile(r"\b([a-z]+(?:[A-Z][a-z0-9]*)+)\b"), None),  # lowerCamelCase
    (re.compile(r"\b([a-z_][a-z0-9]*(?:_[a-z0-9]+)+)\b"), None),  # snake_case
]

# Pattern for extracting file paths.
_FILE_PATH_PATTERN = re.compile(r"\b(\S+\.(?:py|ts|js|tsx|jsx))\b")


def suggest_tools(question: str, storage: StorageBackend | None = None) -> list[ToolSuggestion]:
    """Classify a question and return suggested tool calls.

    Parameters
    ----------
    question:
        Natural language question from the agent.
    storage:
        Optional storage backend for validating extracted symbols.
    """
    question = question.strip()
    if not question:
        return [ToolSuggestion("synaptiq_query", {"query": ""}, "No question provided.")]

    symbols = _extract_symbols(question, storage)
    file_paths = _FILE_PATH_PATTERN.findall(question)
    category = _classify(question)

    return _build_suggestions(category, question, symbols, file_paths)


def _classify(question: str) -> str:
    """Return the question category based on regex rules."""
    for pattern, category in _RULES:
        if pattern.search(question):
            return category
    return "query"


def _extract_symbols(question: str, storage: StorageBackend | None) -> list[str]:
    """Extract potential symbol names from the question text."""
    candidates: list[str] = []
    for pattern, reject in _SYMBOL_PATTERNS:
        candidates.extend(
            m for m in pattern.findall(question) if reject is None or not reject(m)
        )

    if not candidates or storage is None:
        return candidates[:3]

    # Validate against the graph.
    validated: list[str] = []
    for sym in candidates[:5]:
        try:
            if hasattr(storage, "exact_name_search"):
                results = storage.exact_name_search(sym, limit=1)
            else:
                results = storage.fts_search(sym, limit=1)
            if results:
                validated.append(sym)
        except Exception:
            pass

    return validated if validated else candidates[:3]


def _build_suggestions(
    category: str,
    question: str,
    symbols: list[str],
    file_paths: list[str],
) -> list[ToolSuggestion]:
    """Build tool suggestions based on classification and extracted entities."""
    sym = symbols[0] if symbols else None
    fp = file_paths[0] if file_paths else None

    if category == "dead_code":
        return [ToolSuggestion("synaptiq_dead_code", {}, "List all dead code symbols.")]

    if category == "impact":
        if sym:
            return [
                ToolSuggestion("synaptiq_impact", {"symbol": sym}, f"Blast radius for '{sym}'."),
            ]
        return [
            ToolSuggestion(
                "synaptiq_query",
                {"query": question[:100]},
                "Find the symbol in question.",
            ),
            ToolSuggestion(
                "synaptiq_impact",
                {"symbol": "<symbol from query results>"},
                "Then get its blast radius.",
            ),
        ]

    if category == "call_path" and len(symbols) >= 2:
        return [
            ToolSuggestion(
                "synaptiq_call_path",
                {"from_symbol": symbols[0], "to_symbol": symbols[1]},
                f"Call chain from '{symbols[0]}' to '{symbols[1]}'.",
            ),
        ]

    if category == "context" and sym:
        return [
            ToolSuggestion("synaptiq_context", {"symbol": sym}, f"360-degree view of '{sym}'."),
        ]

    if category == "coupling" and fp:
        return [
            ToolSuggestion(
                "synaptiq_coupling",
                {"file_path": fp},
                f"Temporal coupling for '{fp}'.",
            ),
        ]

    if category == "cycles":
        return [ToolSuggestion("synaptiq_cycles", {}, "Detect circular dependencies.")]

    if category == "test_impact" and sym:
        return [
            ToolSuggestion(
                "synaptiq_test_impact",
                {"symbols": [sym]},
                f"Tests affected by '{sym}'.",
            ),
        ]

    if category == "communities":
        return [ToolSuggestion("synaptiq_communities", {}, "List detected communities.")]

    if category == "file_context" and fp:
        return [
            ToolSuggestion("synaptiq_file_context", {"file_path": fp}, f"Context for '{fp}'."),
        ]

    if category == "explain" and sym:
        return [
            ToolSuggestion("synaptiq_query", {"query": sym}, f"Search for '{sym}'."),
            ToolSuggestion("synaptiq_explain", {"symbol": sym}, f"Explain '{sym}'."),
        ]

    if category == "review_risk":
        return [
            ToolSuggestion(
                "synaptiq_review_risk",
                {"diff": "<provide git diff>"},
                "Assess PR risk from a diff.",
            ),
        ]

    # Fallback: general search.
    query = sym or question[:100]
    suggestions = [
        ToolSuggestion("synaptiq_query", {"query": query}, f"Search for '{query}'."),
    ]
    if sym:
        suggestions.append(
            ToolSuggestion("synaptiq_context", {"symbol": sym}, f"Then get context for '{sym}'."),
        )
    return suggestions
