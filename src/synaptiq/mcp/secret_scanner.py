"""Secret scanner for MCP tool responses.

Scans text for potential secrets (API keys, tokens, passwords, connection
strings) and redacts them before they are returned to MCP clients.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SecretMatch:
    """A detected secret in text."""

    secret_type: str
    start: int
    end: int


# Compiled patterns for common secret formats.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS_KEY", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GITHUB_TOKEN", re.compile(r"gh[ps]_[A-Za-z0-9]{36}")),
    ("GITHUB_FINE_GRAINED", re.compile(r"github_pat_[A-Za-z0-9_]{82}")),
    ("SLACK_TOKEN", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("JWT_TOKEN", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")),
    (
        "CONNECTION_STRING",
        re.compile(r"(?:mongodb|postgresql|mysql|redis|amqp):\/\/[^\s'\"]{10,}"),
    ),
    (
        "API_KEY_ASSIGNMENT",
        re.compile(
            r"(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token)"
            r"\s*[:=]\s*['\"][A-Za-z0-9/+=]{20,}['\"]",
            re.IGNORECASE,
        ),
    ),
    (
        "PASSWORD_ASSIGNMENT",
        re.compile(
            r"(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]",
            re.IGNORECASE,
        ),
    ),
    ("ANTHROPIC_KEY", re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}")),
    ("OPENAI_KEY", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("STRIPE_KEY", re.compile(r"[sr]k_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("SENDGRID_KEY", re.compile(r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}")),
]


def scan(text: str) -> list[SecretMatch]:
    """Scan *text* for potential secrets.

    Returns a list of :class:`SecretMatch` instances sorted by position.
    """
    matches: list[SecretMatch] = []
    for secret_type, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            matches.append(SecretMatch(secret_type=secret_type, start=m.start(), end=m.end()))
    matches.sort(key=lambda m: m.start)
    return matches


def redact(text: str) -> tuple[str, int]:
    """Redact detected secrets from *text*.

    Returns a ``(redacted_text, count)`` tuple.  The redacted text replaces
    each secret with ``[REDACTED: <type>]``.
    """
    matches = scan(text)
    if not matches:
        return text, 0

    # Build result from non-overlapping matches (process right-to-left).
    result = text
    seen_ranges: list[tuple[int, int]] = []
    count = 0

    for m in reversed(matches):
        # Skip overlapping matches.
        if any(m.start < end and m.end > start for start, end in seen_ranges):
            continue
        replacement = f"[REDACTED: {m.secret_type}]"
        result = result[: m.start] + replacement + result[m.end :]
        seen_ranges.append((m.start, m.end))
        count += 1

    return result, count
