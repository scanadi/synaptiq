"""Ruby language parser using tree-sitter.

Minimal stub introduced in Task 1 (plumbing).  Full symbol/import/call/heritage
extraction is implemented in later tasks (Task 3+).  Until then ``parse``
returns an empty :class:`ParseResult` so the pipeline can route ``.rb`` files
without error.
"""

from __future__ import annotations

from synaptiq.core.parsers.base import LanguageParser, ParseResult


class RubyParser(LanguageParser):
    """Tree-sitter based parser for Ruby source code."""

    def parse(self, content: str, file_path: str) -> ParseResult:
        return ParseResult()
