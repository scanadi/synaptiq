"""Shared Cypher query safety utilities.

Enforces read-only Cypher for the MCP ``synaptiq_cypher`` tool using two
layers:

1. An **allow-list** on the first clause — the query must start with a
   read clause (``MATCH``, ``OPTIONAL MATCH``, ``RETURN``, ``WITH``,
   ``UNWIND``) or an allow-listed read-only ``CALL`` procedure.
2. A **deny-list** of write/DDL/transaction keywords anywhere in the
   query, catching embedded write clauses (``MATCH ... CREATE ...``).

The deny-list alone is insufficient: LadybugDB also mutates state via
``ALTER``, ``EXPORT DATABASE`` (writes files to disk), ``IMPORT
DATABASE``, and ``ATTACH`` — none of which contain a classic write
keyword.
"""

from __future__ import annotations

import re

_COMMENT_PATTERN = re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL)

# Quoted string literals (single or double, with backslash escapes).
# Stripped before keyword scanning so a read query like
# ``WHERE n.content CONTAINS 'import os'`` is not rejected for the
# IMPORT inside its literal.
_STRING_PATTERN = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")

WRITE_KEYWORDS = re.compile(
    r"\b(DELETE|DROP|CREATE|SET|REMOVE|MERGE|DETACH|INSTALL|LOAD|COPY"
    r"|ALTER|EXPORT|IMPORT|ATTACH|BEGIN|COMMIT|ROLLBACK|CHECKPOINT|USE)\b",
    re.IGNORECASE,
)

# A query must begin with one of these read clauses.
_ALLOWED_FIRST_CLAUSE = re.compile(
    r"^\s*(MATCH|OPTIONAL\s+MATCH|RETURN|WITH|UNWIND|CALL)\b",
    re.IGNORECASE,
)

# CALL is only allowed for known read-only procedures — checked for every
# CALL occurrence in the query, not just a leading one.
_READONLY_PROCEDURES = frozenset({
    "query_fts_index", "show_tables", "table_info", "show_connection",
    "current_setting", "db_version",
})

_ANY_CALL = re.compile(r"\bCALL\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def sanitize_cypher(query: str) -> str:
    """Strip comments and string literals from a Cypher query for safety
    checking.

    Comments are removed so keywords cannot hide in ``/* ... */`` blocks;
    string literals are replaced with empty quotes so keywords *inside*
    them (common when searching code content for ``import``/``use``/...)
    do not trigger false rejections.
    """
    without_comments = _COMMENT_PATTERN.sub('', query)
    return _STRING_PATTERN.sub("''", without_comments)


def check_read_only(query: str) -> str | None:
    """Validate that *query* is a read-only Cypher query.

    Returns ``None`` when the query is allowed, or a human-readable
    rejection reason otherwise.  Keyword scanning runs on a sanitized
    copy with comments and string literals removed.
    """
    cleaned = sanitize_cypher(query)

    if not cleaned.strip():
        return "Query rejected: empty query."

    first = _ALLOWED_FIRST_CLAUSE.match(cleaned)
    if first is None:
        return (
            "Query rejected: must start with a read clause "
            "(MATCH, OPTIONAL MATCH, RETURN, WITH, UNWIND, or an "
            "allow-listed CALL procedure)."
        )

    for call_match in _ANY_CALL.finditer(cleaned):
        if call_match.group(1).lower() not in _READONLY_PROCEDURES:
            return (
                "Query rejected: only read-only CALL procedures are allowed "
                "(QUERY_FTS_INDEX, SHOW_TABLES, TABLE_INFO, SHOW_CONNECTION, "
                "CURRENT_SETTING, DB_VERSION)."
            )

    keyword = WRITE_KEYWORDS.search(cleaned)
    if keyword is not None:
        return (
            "Query rejected: only read-only queries (MATCH/RETURN) are allowed. "
            "Write operations (DELETE, DROP, CREATE, SET, MERGE) are not permitted."
        )

    return None
