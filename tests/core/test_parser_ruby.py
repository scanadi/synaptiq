"""Tests for the Ruby parser and its dispatch wiring."""

from __future__ import annotations

from synaptiq.core.ingestion.parser_phase import get_parser
from synaptiq.core.parsers.base import LanguageParser, ParseResult


class TestRubyParserDispatch:
    """get_parser must return a RubyParser for the 'ruby' language."""

    def test_get_parser_returns_language_parser(self) -> None:
        parser = get_parser("ruby")
        assert isinstance(parser, LanguageParser)

    def test_parse_returns_parse_result(self) -> None:
        parser = get_parser("ruby")
        result = parser.parse("puts 'hello'", "hello.rb")
        assert isinstance(result, ParseResult)
