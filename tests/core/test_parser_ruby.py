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
# Imports (require / require_relative / autoload / load)
# ---------------------------------------------------------------------------


class TestParseImports:
    """``require``-family calls become :class:`ImportInfo` entries."""

    def test_require_is_absolute(self, parser: RubyParser) -> None:
        imports = parser.parse('require "json"\n', "test.rb").imports
        assert len(imports) == 1
        assert imports[0].module == "json"
        assert imports[0].is_relative is False

    def test_require_relative_is_relative(self, parser: RubyParser) -> None:
        imports = parser.parse('require_relative "../lib/foo"\n', "test.rb").imports
        assert len(imports) == 1
        assert imports[0].module == "../lib/foo"
        assert imports[0].is_relative is True

    def test_autoload_records_constant_name_and_path(self, parser: RubyParser) -> None:
        imports = parser.parse('autoload :Bar, "bar"\n', "test.rb").imports
        assert len(imports) == 1
        assert imports[0].module == "bar"
        assert imports[0].names == ["Bar"]
        assert imports[0].is_relative is False

    def test_load_is_an_import(self, parser: RubyParser) -> None:
        imports = parser.parse('load "x.rb"\n', "test.rb").imports
        assert len(imports) == 1
        assert imports[0].module == "x.rb"

    def test_multiple_requires(self, parser: RubyParser) -> None:
        code = 'require "json"\nrequire "set"\nrequire_relative "./util"\n'
        imports = parser.parse(code, "test.rb").imports
        assert [(i.module, i.is_relative) for i in imports] == [
            ("json", False),
            ("set", False),
            ("./util", True),
        ]

    def test_require_inside_class_is_captured(self, parser: RubyParser) -> None:
        code = 'class C\n  require "json"\n  def m\n  end\nend\n'
        imports = parser.parse(code, "test.rb").imports
        assert [i.module for i in imports] == ["json"]

    def test_single_quoted_require(self, parser: RubyParser) -> None:
        imports = parser.parse("require 'yaml'\n", "test.rb").imports
        assert [i.module for i in imports] == ["yaml"]


class TestParseImportsEdgeCases:
    """Non-literal / dynamic requires are ignored without error."""

    def test_dynamic_require_identifier_ignored(self, parser: RubyParser) -> None:
        imports = parser.parse("require name\n", "test.rb").imports
        assert imports == []

    def test_dynamic_require_method_call_ignored(self, parser: RubyParser) -> None:
        imports = parser.parse('require File.expand_path("y")\n', "test.rb").imports
        assert imports == []

    def test_interpolated_require_ignored(self, parser: RubyParser) -> None:
        imports = parser.parse('require "a/#{x}"\n', "test.rb").imports
        assert imports == []

    def test_non_import_call_is_not_an_import(self, parser: RubyParser) -> None:
        imports = parser.parse('puts "hi"\n', "test.rb").imports
        assert imports == []

    def test_autoload_without_path_ignored(self, parser: RubyParser) -> None:
        imports = parser.parse("autoload :Bar\n", "test.rb").imports
        assert imports == []


# ---------------------------------------------------------------------------
# Calls (receivers, self, blocks, paren-less)
# ---------------------------------------------------------------------------


def _calls_by_name(result: ParseResult) -> dict[str, list]:
    by_name: dict[str, list] = {}
    for call in result.calls:
        by_name.setdefault(call.name, []).append(call)
    return by_name


class TestParseCalls:
    """``call`` nodes become :class:`CallInfo` entries."""

    def test_function_call_with_parens(self, parser: RubyParser) -> None:
        calls = parser.parse("foo()\n", "test.rb").calls
        assert len(calls) == 1
        assert calls[0].name == "foo"
        assert calls[0].receiver == ""

    def test_method_call_with_receiver(self, parser: RubyParser) -> None:
        by_name = _calls_by_name(parser.parse("obj.bar(x)\n", "test.rb"))
        assert "bar" in by_name
        call = by_name["bar"][0]
        assert call.receiver == "obj"
        assert call.arguments == ["x"]

    def test_self_receiver(self, parser: RubyParser) -> None:
        by_name = _calls_by_name(parser.parse("self.baz\n", "test.rb"))
        assert "baz" in by_name
        assert by_name["baz"][0].receiver == "self"

    def test_constant_receiver(self, parser: RubyParser) -> None:
        by_name = _calls_by_name(parser.parse("Foo.create(attrs)\n", "test.rb"))
        assert "create" in by_name
        assert by_name["create"][0].receiver == "Foo"

    def test_bare_call_in_method_body(self, parser: RubyParser) -> None:
        code = "def perform\n  validate\n  save\nend\n"
        by_name = _calls_by_name(parser.parse(code, "test.rb"))
        assert "validate" in by_name
        assert "save" in by_name
        assert by_name["validate"][0].receiver == ""

    def test_parenless_call_with_argument(self, parser: RubyParser) -> None:
        by_name = _calls_by_name(parser.parse("render arg\n", "test.rb"))
        assert "render" in by_name
        assert by_name["render"][0].arguments == ["arg"]

    def test_chained_calls(self, parser: RubyParser) -> None:
        by_name = _calls_by_name(parser.parse("a.b.c(x)\n", "test.rb"))
        # both the inner ``.b`` and the outer ``.c`` are method calls.
        assert "b" in by_name
        assert "c" in by_name
        assert by_name["c"][0].receiver == "a"
        assert by_name["b"][0].receiver == "a"

    def test_block_callback_calls(self, parser: RubyParser) -> None:
        code = "items.map { |i| transform(i) }\n"
        by_name = _calls_by_name(parser.parse(code, "test.rb"))
        assert "map" in by_name
        assert by_name["map"][0].receiver == "items"
        # the call inside the block body is also extracted.
        assert "transform" in by_name
        assert by_name["transform"][0].receiver == ""

    def test_do_block_body_calls(self, parser: RubyParser) -> None:
        code = "items.each do |i|\n  process i\nend\n"
        by_name = _calls_by_name(parser.parse(code, "test.rb"))
        assert "each" in by_name
        assert "process" in by_name

    def test_call_lines(self, parser: RubyParser) -> None:
        code = "foo\n\nbar\n"
        by_name = _calls_by_name(parser.parse(code, "test.rb"))
        assert by_name["foo"][0].line == 1
        assert by_name["bar"][0].line == 3


class TestParseCallsEdgeCases:
    """Operators, safe-navigation, and local-variable references."""

    def test_operator_is_not_a_call(self, parser: RubyParser) -> None:
        by_name = _calls_by_name(parser.parse("a + b\n", "test.rb"))
        assert "+" not in by_name
        assert by_name == {}

    def test_safe_navigation_call(self, parser: RubyParser) -> None:
        by_name = _calls_by_name(parser.parse("user&.name\n", "test.rb"))
        assert "name" in by_name
        assert by_name["name"][0].receiver == "user"

    def test_local_variable_reference_is_not_a_call(self, parser: RubyParser) -> None:
        code = "def m\n  count = compute\n  count\nend\n"
        by_name = _calls_by_name(parser.parse(code, "test.rb"))
        # ``count`` is a local variable, not a method call.
        assert "count" not in by_name

    def test_method_parameter_is_not_a_call(self, parser: RubyParser) -> None:
        code = "def m(value)\n  value\nend\n"
        by_name = _calls_by_name(parser.parse(code, "test.rb"))
        assert "value" not in by_name

    def test_block_parameter_is_not_a_call(self, parser: RubyParser) -> None:
        code = "list.each do |row|\n  row\nend\n"
        by_name = _calls_by_name(parser.parse(code, "test.rb"))
        assert "row" not in by_name

    def test_assignment_rhs_call(self, parser: RubyParser) -> None:
        code = "def m\n  result = fetch_data\n  result\nend\n"
        by_name = _calls_by_name(parser.parse(code, "test.rb"))
        assert "fetch_data" in by_name
        assert "result" not in by_name

    def test_no_calls_for_definitions_only(self, parser: RubyParser) -> None:
        code = "class C\n  def m\n  end\nend\n"
        assert parser.parse(code, "test.rb").calls == []


# ---------------------------------------------------------------------------
# Heritage & mixins (<, include/extend/prepend, attr_*)
# ---------------------------------------------------------------------------


class TestParseHeritage:
    """Superclass and mixin relationships become heritage tuples."""

    def test_superclass_extends(self, parser: RubyParser) -> None:
        heritage = parser.parse("class A < B\nend\n", "test.rb").heritage
        assert ("A", "extends", "B") in heritage

    def test_no_superclass_no_extends(self, parser: RubyParser) -> None:
        heritage = parser.parse("class A\nend\n", "test.rb").heritage
        assert all(kind != "extends" for _, kind, _ in heritage)

    def test_namespaced_superclass_uses_last_segment(self, parser: RubyParser) -> None:
        heritage = parser.parse("class A < Foo::Bar\nend\n", "test.rb").heritage
        assert ("A", "extends", "Bar") in heritage

    def test_include_is_mixin(self, parser: RubyParser) -> None:
        code = "class A\n  include M\nend\n"
        heritage = parser.parse(code, "test.rb").heritage
        assert ("A", "mixin", "M") in heritage

    def test_extend_is_mixin(self, parser: RubyParser) -> None:
        code = "class A\n  extend M\nend\n"
        heritage = parser.parse(code, "test.rb").heritage
        assert ("A", "mixin", "M") in heritage

    def test_prepend_is_mixin(self, parser: RubyParser) -> None:
        code = "class A\n  prepend M\nend\n"
        heritage = parser.parse(code, "test.rb").heritage
        assert ("A", "mixin", "M") in heritage

    def test_mixin_inside_module(self, parser: RubyParser) -> None:
        code = "module A\n  include K\nend\n"
        heritage = parser.parse(code, "test.rb").heritage
        assert ("A", "mixin", "K") in heritage

    def test_multiple_includes(self, parser: RubyParser) -> None:
        code = "class A\n  include M1, M2\nend\n"
        heritage = parser.parse(code, "test.rb").heritage
        assert ("A", "mixin", "M1") in heritage
        assert ("A", "mixin", "M2") in heritage

    def test_namespaced_mixin_uses_last_segment(self, parser: RubyParser) -> None:
        code = "class A\n  include Foo::Bar\nend\n"
        heritage = parser.parse(code, "test.rb").heritage
        assert ("A", "mixin", "Bar") in heritage

    def test_extends_and_mixin_combined(self, parser: RubyParser) -> None:
        code = "class A < Base\n  include M\nend\n"
        heritage = parser.parse(code, "test.rb").heritage
        assert ("A", "extends", "Base") in heritage
        assert ("A", "mixin", "M") in heritage

    def test_top_level_include_is_not_a_mixin(self, parser: RubyParser) -> None:
        # ``include M`` outside any class/module has no owning type.
        heritage = parser.parse("include M\n", "test.rb").heritage
        assert heritage == []


class TestParseAttrAccessors:
    """attr_accessor/reader/writer names are recorded on the owning type."""

    def _class_symbol(self, parser: RubyParser, code: str) -> object:
        syms = parser.parse(code, "test.rb").symbols
        return next(s for s in syms if s.kind in ("class", "module"))

    def test_attr_accessor_recorded(self, parser: RubyParser) -> None:
        cls = self._class_symbol(parser, "class A\n  attr_accessor :x, :y\nend\n")
        assert "attr_accessor:x" in cls.decorators
        assert "attr_accessor:y" in cls.decorators

    def test_attr_reader_recorded(self, parser: RubyParser) -> None:
        cls = self._class_symbol(parser, "class A\n  attr_reader :z\nend\n")
        assert "attr_reader:z" in cls.decorators

    def test_attr_writer_recorded(self, parser: RubyParser) -> None:
        cls = self._class_symbol(parser, "class A\n  attr_writer :w\nend\n")
        assert "attr_writer:w" in cls.decorators

    def test_attr_on_module(self, parser: RubyParser) -> None:
        cls = self._class_symbol(parser, "module M\n  attr_reader :a\nend\n")
        assert "attr_reader:a" in cls.decorators

    def test_no_attrs_means_empty_decorators(self, parser: RubyParser) -> None:
        cls = self._class_symbol(parser, "class A\n  def m\n  end\nend\n")
        assert cls.decorators == []


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
