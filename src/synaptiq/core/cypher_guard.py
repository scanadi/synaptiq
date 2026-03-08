"""Shared Cypher query safety utilities.

Provides a compiled regex for detecting write keywords in Cypher queries.
Used by the MCP tools layer to enforce read-only query execution.
"""

from __future__ import annotations

import re

_COMMENT_PATTERN = re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL)

WRITE_KEYWORDS = re.compile(
    r"\b(DELETE|DROP|CREATE|SET|REMOVE|MERGE|DETACH|INSTALL|LOAD|COPY)\b",
    re.IGNORECASE,
)


def sanitize_cypher(query: str) -> str:
    """Strip comments from a Cypher query before safety checking."""
    return _COMMENT_PATTERN.sub('', query)
