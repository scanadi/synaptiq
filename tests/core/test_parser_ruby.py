"""Tests for the Ruby parser and its dispatch wiring."""

from __future__ import annotations

import pytest

from synaptiq.core.ingestion.parser_phase import get_parser
from synaptiq.core.parsers.base import LanguageParser, ParseResult
from synaptiq.core.parsers.ruby_lang import RubyParser


@pytest.fixture
def parser() -> RubyParser:
    return RubyParser()


class TestRubyParserDispatch:
    """get_parser must return a RubyParser for the 'ruby' language."""

    def test_get_parser_returns_language_parser(self) -> None:
        parser = get_parser("ruby")
        assert isinstance(parser, LanguageParser)

    def test_parse_returns_parse_result(self) -> None:
        parser = get_parser("ruby")
        result = parser.parse("puts 'hello'", "hello.rb")
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Top-level functions
# ---------------------------------------------------------------------------


class TestParseTopLevelFunction:
    """A top-level ``def`` is a function (no owning class)."""

    CODE = "def greet(name)\n  puts name\nend\n"

    def test_symbol_count(self, parser: RubyParser) -> None:
        result = parser.parse(self.CODE, "test.rb")
        assert len(result.symbols) == 1

    def test_name_and_kind(self, parser: RubyParser) -> None:
        func = parser.parse(self.CODE, "test.rb").symbols[0]
        assert func.name == "greet"
        assert func.kind == "function"

    def test_no_class_name(self, parser: RubyParser) -> None:
        func = parser.parse(self.CODE, "test.rb").symbols[0]
        assert func.class_name == ""

    def test_lines(self, parser: RubyParser) -> None:
        func = parser.parse(self.CODE, "test.rb").symbols[0]
        assert func.start_line == 1
        assert func.end_line == 3

    def test_signature(self, parser: RubyParser) -> None:
        func = parser.parse(self.CODE, "test.rb").symbols[0]
        assert func.signature == "def greet(name)"

    def test_content_includes_body(self, parser: RubyParser) -> None:
        func = parser.parse(self.CODE, "test.rb").symbols[0]
        assert "puts name" in func.content

    def test_predicate_and_bang_names(self, parser: RubyParser) -> None:
        code = "def valid?\nend\ndef save!\nend\ndef name=(v)\nend\n"
        names = {s.name for s in parser.parse(code, "test.rb").symbols}
        assert names == {"valid?", "save!", "name="}


# ---------------------------------------------------------------------------
# Classes with methods
# ---------------------------------------------------------------------------


class TestParseClassWithMethods:
    """A class and its instance methods carry ``class_name``."""

    CODE = (
        "class User\n"
        "  def initialize(name)\n"
        "    @name = name\n"
        "  end\n"
        "\n"
        "  def save\n"
        "    true\n"
        "  end\n"
        "end\n"
    )

    def test_symbol_count(self, parser: RubyParser) -> None:
        result = parser.parse(self.CODE, "test.rb")
        # 1 class + 2 methods
        assert len(result.symbols) == 3

    def test_class_symbol(self, parser: RubyParser) -> None:
        cls = [s for s in parser.parse(self.CODE, "test.rb").symbols if s.kind == "class"]
        assert len(cls) == 1
        assert cls[0].name == "User"
        assert cls[0].class_name == ""

    def test_methods_kind_and_owner(self, parser: RubyParser) -> None:
        methods = [s for s in parser.parse(self.CODE, "test.rb").symbols if s.kind == "method"]
        assert {m.name for m in methods} == {"initialize", "save"}
        assert all(m.class_name == "User" for m in methods)

    def test_no_function_kind_inside_class(self, parser: RubyParser) -> None:
        kinds = {s.kind for s in parser.parse(self.CODE, "test.rb").symbols}
        assert "function" not in kinds


# ---------------------------------------------------------------------------
# Modules with methods
# ---------------------------------------------------------------------------


class TestParseModuleWithMethods:
    """A module is a ``module`` symbol; its defs are methods owned by it."""

    CODE = "module Greeter\n  def hello\n    puts 'hi'\n  end\nend\n"

    def test_module_symbol(self, parser: RubyParser) -> None:
        mods = [s for s in parser.parse(self.CODE, "test.rb").symbols if s.kind == "module"]
        assert len(mods) == 1
        assert mods[0].name == "Greeter"

    def test_method_owned_by_module(self, parser: RubyParser) -> None:
        methods = [s for s in parser.parse(self.CODE, "test.rb").symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "hello"
        assert methods[0].class_name == "Greeter"


# ---------------------------------------------------------------------------
# Singleton / self. class methods
# ---------------------------------------------------------------------------


class TestParseSingletonMethods:
    """``def self.foo`` is a method whose signature shows the receiver."""

    CODE = "class Repo\n  def self.find(id)\n    nil\n  end\nend\n"

    def test_singleton_method_owner(self, parser: RubyParser) -> None:
        methods = [s for s in parser.parse(self.CODE, "test.rb").symbols if s.kind == "method"]
        assert len(methods) == 1
        assert methods[0].name == "find"
        assert methods[0].class_name == "Repo"

    def test_singleton_signature(self, parser: RubyParser) -> None:
        method = [s for s in parser.parse(self.CODE, "test.rb").symbols if s.kind == "method"][0]
        assert method.signature == "def self.find(id)"

    def test_top_level_singleton_is_function(self, parser: RubyParser) -> None:
        # A singleton def outside any class/module is treated as a function.
        code = "def Foo.bar\nend\n"
        syms = parser.parse(code, "test.rb").symbols
        assert len(syms) == 1
        assert syms[0].kind == "function"
        assert syms[0].name == "bar"
        assert syms[0].signature == "def Foo.bar"


# ---------------------------------------------------------------------------
# Nested classes / deeply nested modules
# ---------------------------------------------------------------------------


class TestParseNesting:
    """Nested classes attribute their own methods; modules nest arbitrarily."""

    def test_nested_class_methods(self, parser: RubyParser) -> None:
        code = (
            "class Outer\n"
            "  def outer_m\n"
            "  end\n"
            "  class Inner\n"
            "    def inner_m\n"
            "    end\n"
            "  end\n"
            "end\n"
        )
        syms = parser.parse(code, "test.rb").symbols
        by_name = {s.name: s for s in syms}
        assert by_name["Outer"].kind == "class"
        assert by_name["Inner"].kind == "class"
        assert by_name["outer_m"].class_name == "Outer"
        assert by_name["inner_m"].class_name == "Inner"

    def test_deeply_nested_modules(self, parser: RubyParser) -> None:
        code = (
            "module A\n  module B\n    module C\n      def deep\n      end\n    end\n  end\nend\n"
        )
        syms = parser.parse(code, "test.rb").symbols
        modules = {s.name for s in syms if s.kind == "module"}
        assert modules == {"A", "B", "C"}
        deep = [s for s in syms if s.kind == "method"][0]
        assert deep.name == "deep"
        assert deep.class_name == "C"

    def test_namespaced_definition_uses_last_segment(self, parser: RubyParser) -> None:
        code = "class Foo::Bar\n  def m\n  end\nend\n"
        syms = parser.parse(code, "test.rb").symbols
        cls = [s for s in syms if s.kind == "class"][0]
        assert cls.name == "Bar"
        method = [s for s in syms if s.kind == "method"][0]
        assert method.class_name == "Bar"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestParseConstants:
    """Constant assignments are emitted as ``constant`` symbols."""

    def test_top_level_constant(self, parser: RubyParser) -> None:
        code = "MAX_RETRIES = 5\n"
        syms = parser.parse(code, "test.rb").symbols
        consts = [s for s in syms if s.kind == "constant"]
        assert len(consts) == 1
        assert consts[0].name == "MAX_RETRIES"
        assert consts[0].class_name == ""

    def test_constant_inside_class(self, parser: RubyParser) -> None:
        code = "class C\n  DEFAULT = 1\nend\n"
        consts = [s for s in parser.parse(code, "test.rb").symbols if s.kind == "constant"]
        assert len(consts) == 1
        assert consts[0].name == "DEFAULT"
        assert consts[0].class_name == "C"

    def test_local_and_instance_assignments_are_not_constants(self, parser: RubyParser) -> None:
        code = "x = 1\n@y = 2\n@@z = 3\n"
        syms = parser.parse(code, "test.rb").symbols
        assert [s for s in syms if s.kind == "constant"] == []


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


class TestParseErrorHandling:
    """Empty and malformed sources degrade gracefully."""

    def test_empty_file(self, parser: RubyParser) -> None:
        result = parser.parse("", "empty.rb")
        assert result.symbols == []

    def test_whitespace_only(self, parser: RubyParser) -> None:
        result = parser.parse("\n\n   \n", "ws.rb")
        assert result.symbols == []

    def test_syntax_error_does_not_raise(self, parser: RubyParser) -> None:
        # Unterminated class — tree-sitter is error-tolerant but must not raise.
        code = "class Broken\n  def half(\n"
        result = parser.parse(code, "broken.rb")
        assert isinstance(result, ParseResult)

    def test_partial_error_still_extracts_valid_defs(self, parser: RubyParser) -> None:
        # A valid class followed by a garbled line: the valid def is still found.
        code = "class Good\n  def ok\n  end\nend\n@@@ bad tokens @@@\n"
        result = parser.parse(code, "partial.rb")
        names = {s.name for s in result.symbols}
        assert "Good" in names
        assert "ok" in names

    def test_comments_only(self, parser: RubyParser) -> None:
        result = parser.parse("# just a comment\n# another\n", "c.rb")
        assert result.symbols == []
