"""Tests for the Go parser and its dispatch wiring."""

from __future__ import annotations

import pytest

from synaptiq.core.ingestion.parser_phase import get_parser
from synaptiq.core.parsers.base import LanguageParser, ParseResult
from synaptiq.core.parsers.go_lang import GoParser


@pytest.fixture
def parser() -> GoParser:
    return GoParser()


def _by_kind(result: ParseResult, kind: str) -> list:
    return [s for s in result.symbols if s.kind == kind]


def _calls_by_name(result: ParseResult) -> dict[str, list]:
    by_name: dict[str, list] = {}
    for call in result.calls:
        by_name.setdefault(call.name, []).append(call)
    return by_name


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestGoParserDispatch:
    """get_parser must return a GoParser for the 'go' language."""

    def test_get_parser_returns_language_parser(self) -> None:
        assert isinstance(get_parser("go"), LanguageParser)

    def test_parse_returns_parse_result(self) -> None:
        result = get_parser("go").parse("package main\n", "main.go")
        assert isinstance(result, ParseResult)


# ---------------------------------------------------------------------------
# Package → module
# ---------------------------------------------------------------------------


class TestParsePackage:
    """The ``package`` clause becomes a ``module`` symbol (one per file)."""

    def test_package_is_module(self, parser: GoParser) -> None:
        mods = _by_kind(parser.parse("package main\n", "main.go"), "module")
        assert len(mods) == 1
        assert mods[0].name == "main"

    def test_package_content_and_line(self, parser: GoParser) -> None:
        mod = _by_kind(parser.parse("package widgets\n", "w.go"), "module")[0]
        assert mod.content == "package widgets"
        assert mod.start_line == 1


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class TestParseFunctions:
    """Top-level ``func`` declarations are ``function`` symbols."""

    CODE = "package p\nfunc Greet(name string) string {\n\treturn name\n}\n"

    def test_kind_and_name(self, parser: GoParser) -> None:
        funcs = _by_kind(parser.parse(self.CODE, "p.go"), "function")
        assert len(funcs) == 1
        assert funcs[0].name == "Greet"
        assert funcs[0].class_name == ""

    def test_lines(self, parser: GoParser) -> None:
        fn = _by_kind(parser.parse(self.CODE, "p.go"), "function")[0]
        assert fn.start_line == 2
        assert fn.end_line == 4

    def test_signature_excludes_body(self, parser: GoParser) -> None:
        fn = _by_kind(parser.parse(self.CODE, "p.go"), "function")[0]
        assert fn.signature == "func Greet(name string) string"

    def test_content_includes_body(self, parser: GoParser) -> None:
        fn = _by_kind(parser.parse(self.CODE, "p.go"), "function")[0]
        assert "return name" in fn.content

    def test_generic_function_signature(self, parser: GoParser) -> None:
        code = "package p\nfunc Map[T any, U any](s []T, f func(T) U) []U {\n\treturn nil\n}\n"
        fn = _by_kind(parser.parse(code, "p.go"), "function")[0]
        assert fn.name == "Map"
        assert fn.signature == "func Map[T any, U any](s []T, f func(T) U) []U"


# ---------------------------------------------------------------------------
# Methods & receivers
# ---------------------------------------------------------------------------


class TestParseMethods:
    """Receiver functions are ``method`` symbols owned by the receiver type."""

    def test_pointer_receiver_class_name(self, parser: GoParser) -> None:
        code = "package p\nfunc (s *Server) Start() {}\n"
        m = _by_kind(parser.parse(code, "p.go"), "method")[0]
        assert m.name == "Start"
        assert m.class_name == "Server"  # pointer stripped

    def test_value_receiver_class_name(self, parser: GoParser) -> None:
        code = "package p\nfunc (s Server) Name() string {\n\treturn s.name\n}\n"
        m = _by_kind(parser.parse(code, "p.go"), "method")[0]
        assert m.class_name == "Server"

    def test_generic_receiver_base_type(self, parser: GoParser) -> None:
        code = "package p\nfunc (s *Stack[T]) Push(v T) {}\n"
        m = _by_kind(parser.parse(code, "p.go"), "method")[0]
        assert m.name == "Push"
        assert m.class_name == "Stack"

    def test_method_signature(self, parser: GoParser) -> None:
        code = "package p\nfunc (s *Server) Start(addr string) error {\n\treturn nil\n}\n"
        m = _by_kind(parser.parse(code, "p.go"), "method")[0]
        assert m.signature == "func (s *Server) Start(addr string) error"

    def test_receiver_recorded_as_variable_type(self, parser: GoParser) -> None:
        # The receiver variable maps to its type so intra-type calls resolve.
        code = "package p\nfunc (s *Server) Start() {}\n"
        vts = parser.parse(code, "p.go").variable_types
        assert ("s", "Server") in [(v.var_name, v.type_name) for v in vts]


# ---------------------------------------------------------------------------
# Types: struct / interface / alias
# ---------------------------------------------------------------------------


class TestParseTypes:
    """``type`` declarations map to class / interface / type_alias."""

    def test_struct_is_class(self, parser: GoParser) -> None:
        code = "package p\ntype Server struct {\n\tName string\n}\n"
        cls = _by_kind(parser.parse(code, "p.go"), "class")
        assert len(cls) == 1
        assert cls[0].name == "Server"

    def test_interface_is_interface(self, parser: GoParser) -> None:
        code = "package p\ntype Reader interface {\n\tRead() int\n}\n"
        ifaces = _by_kind(parser.parse(code, "p.go"), "interface")
        assert len(ifaces) == 1
        assert ifaces[0].name == "Reader"

    def test_explicit_alias_is_type_alias(self, parser: GoParser) -> None:
        code = "package p\ntype MyString = string\n"
        aliases = _by_kind(parser.parse(code, "p.go"), "type_alias")
        assert [a.name for a in aliases] == ["MyString"]

    def test_defined_type_is_type_alias(self, parser: GoParser) -> None:
        code = "package p\ntype Celsius float64\n"
        aliases = _by_kind(parser.parse(code, "p.go"), "type_alias")
        assert [a.name for a in aliases] == ["Celsius"]

    def test_grouped_type_block(self, parser: GoParser) -> None:
        code = (
            "package p\n"
            "type (\n"
            "\tFoo struct{ x int }\n"
            "\tBar interface{ M() }\n"
            "\tBaz = string\n"
            "\tQux int\n"
            ")\n"
        )
        result = parser.parse(code, "p.go")
        kinds = {s.name: s.kind for s in result.symbols}
        assert kinds["Foo"] == "class"
        assert kinds["Bar"] == "interface"
        assert kinds["Baz"] == "type_alias"
        assert kinds["Qux"] == "type_alias"

    def test_interface_methods_are_not_symbols(self, parser: GoParser) -> None:
        # Interface method specs are part of the interface node, not separate
        # method symbols (interface method sets are not materialised).
        code = "package p\ntype Reader interface {\n\tRead() int\n\tClose() error\n}\n"
        assert _by_kind(parser.parse(code, "p.go"), "method") == []


# ---------------------------------------------------------------------------
# Heritage: struct & interface embedding
# ---------------------------------------------------------------------------


class TestParseHeritage:
    """Embedding (struct anonymous fields, interface elements) → extends."""

    def test_struct_embedding_extends(self, parser: GoParser) -> None:
        code = "package p\ntype Server struct {\n\tName string\n\tLogger\n}\n"
        heritage = parser.parse(code, "p.go").heritage
        assert ("Server", "extends", "Logger") in heritage

    def test_struct_embedding_pointer(self, parser: GoParser) -> None:
        code = "package p\ntype Server struct {\n\t*Logger\n}\n"
        heritage = parser.parse(code, "p.go").heritage
        assert ("Server", "extends", "Logger") in heritage

    def test_struct_embedding_qualified_last_segment(self, parser: GoParser) -> None:
        code = "package p\ntype Server struct {\n\thttp.Handler\n}\n"
        heritage = parser.parse(code, "p.go").heritage
        assert ("Server", "extends", "Handler") in heritage

    def test_named_field_is_not_embedding(self, parser: GoParser) -> None:
        code = "package p\ntype Server struct {\n\tlog Logger\n}\n"
        heritage = parser.parse(code, "p.go").heritage
        assert all(kind != "extends" for _, kind, _ in heritage)

    def test_interface_embedding_extends(self, parser: GoParser) -> None:
        code = (
            "package p\n"
            "type ReadWriter interface {\n"
            "\tReader\n"
            "\tWriter\n"
            "\tExtra() int\n"
            "}\n"
        )
        heritage = parser.parse(code, "p.go").heritage
        assert ("ReadWriter", "extends", "Reader") in heritage
        assert ("ReadWriter", "extends", "Writer") in heritage

    def test_no_implements_ever_emitted(self, parser: GoParser) -> None:
        # Interface satisfaction is not statically derivable — never guessed.
        code = (
            "package p\n"
            "type Speaker interface{ Speak() string }\n"
            "type Dog struct{}\n"
            "func (d Dog) Speak() string { return \"woof\" }\n"
        )
        heritage = parser.parse(code, "p.go").heritage
        assert all(kind != "implements" for _, kind, _ in heritage)


# ---------------------------------------------------------------------------
# const / var → constant
# ---------------------------------------------------------------------------


class TestParseConstVar:
    """Top-level const/var declarations emit ``constant`` symbols."""

    def test_top_level_const(self, parser: GoParser) -> None:
        consts = _by_kind(parser.parse("package p\nconst MaxSize = 100\n", "p.go"), "constant")
        assert [c.name for c in consts] == ["MaxSize"]

    def test_top_level_var(self, parser: GoParser) -> None:
        consts = _by_kind(parser.parse('package p\nvar name string = "x"\n', "p.go"), "constant")
        assert [c.name for c in consts] == ["name"]

    def test_grouped_const_and_var_blocks(self, parser: GoParser) -> None:
        code = "package p\nconst (\n\tA = 1\n\tB = 2\n)\nvar (\n\tx int\n\ty string\n)\n"
        names = {c.name for c in _by_kind(parser.parse(code, "p.go"), "constant")}
        assert names == {"A", "B", "x", "y"}

    def test_local_var_is_not_a_constant_symbol(self, parser: GoParser) -> None:
        # A ``var`` inside a function body is a local, not a package constant.
        code = "package p\nfunc f() {\n\tvar local Thing\n\t_ = local\n}\n"
        assert _by_kind(parser.parse(code, "p.go"), "constant") == []


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestParseImports:
    """Import specs become :class:`ImportInfo` entries (never relative)."""

    def test_single_import(self, parser: GoParser) -> None:
        imports = parser.parse('package p\nimport "net/http"\n', "p.go").imports
        assert len(imports) == 1
        assert imports[0].module == "net/http"
        assert imports[0].alias == ""
        assert imports[0].is_relative is False

    def test_grouped_imports(self, parser: GoParser) -> None:
        code = 'package p\nimport (\n\t"fmt"\n\t"os"\n)\n'
        modules = [i.module for i in parser.parse(code, "p.go").imports]
        assert modules == ["fmt", "os"]

    def test_aliased_import(self, parser: GoParser) -> None:
        imp = parser.parse('package p\nimport m "math/rand"\n', "p.go").imports[0]
        assert imp.module == "math/rand"
        assert imp.alias == "m"

    def test_blank_import(self, parser: GoParser) -> None:
        imp = parser.parse('package p\nimport _ "database/sql"\n', "p.go").imports[0]
        assert imp.module == "database/sql"
        assert imp.alias == "_"

    def test_dot_import(self, parser: GoParser) -> None:
        imp = parser.parse('package p\nimport . "strings"\n', "p.go").imports[0]
        assert imp.module == "strings"
        assert imp.alias == "."


# ---------------------------------------------------------------------------
# Calls (functions, selectors, goroutines, defer, composite literals)
# ---------------------------------------------------------------------------


class TestParseCalls:
    """Call expressions become :class:`CallInfo` entries."""

    def test_plain_function_call(self, parser: GoParser) -> None:
        by_name = _calls_by_name(parser.parse("package p\nfunc f() {\n\thelper()\n}\n", "p.go"))
        assert "helper" in by_name
        assert by_name["helper"][0].receiver == ""

    def test_selector_call_has_receiver(self, parser: GoParser) -> None:
        by_name = _calls_by_name(
            parser.parse("package p\nfunc f() {\n\tfmt.Println(x)\n}\n", "p.go")
        )
        assert "Println" in by_name
        assert by_name["Println"][0].receiver == "fmt"

    def test_identifier_arguments_captured(self, parser: GoParser) -> None:
        by_name = _calls_by_name(
            parser.parse("package p\nfunc f() {\n\tregister(handler)\n}\n", "p.go")
        )
        assert by_name["register"][0].arguments == ["handler"]

    def test_goroutine_unwraps_to_inner_call(self, parser: GoParser) -> None:
        by_name = _calls_by_name(
            parser.parse("package p\nfunc f() {\n\tgo background()\n}\n", "p.go")
        )
        assert "background" in by_name
        assert len(by_name["background"]) == 1

    def test_defer_unwraps_to_inner_call(self, parser: GoParser) -> None:
        by_name = _calls_by_name(
            parser.parse("package p\nfunc f() {\n\tdefer cleanup()\n}\n", "p.go")
        )
        assert "cleanup" in by_name
        assert len(by_name["cleanup"]) == 1

    def test_composite_literal_is_a_call(self, parser: GoParser) -> None:
        # Struct instantiation links to the type (Go analogue of ``new``).
        by_name = _calls_by_name(
            parser.parse("package p\nfunc f() {\n\ts := Server{}\n\t_ = s\n}\n", "p.go")
        )
        assert "Server" in by_name

    def test_pointer_composite_literal_is_a_call(self, parser: GoParser) -> None:
        by_name = _calls_by_name(
            parser.parse("package p\nfunc f() {\n\ts := &Server{}\n\t_ = s\n}\n", "p.go")
        )
        assert "Server" in by_name

    def test_call_line_numbers(self, parser: GoParser) -> None:
        by_name = _calls_by_name(
            parser.parse("package p\nfunc f() {\n\ta()\n\n\tb()\n}\n", "p.go")
        )
        assert by_name["a"][0].line == 3
        assert by_name["b"][0].line == 5

    def test_no_calls_for_definitions_only(self, parser: GoParser) -> None:
        code = "package p\ntype S struct{ x int }\nfunc (s S) M() {}\n"
        assert parser.parse(code, "p.go").calls == []


# ---------------------------------------------------------------------------
# Types (param / return / var references)
# ---------------------------------------------------------------------------


class TestParseTypeRefs:
    """Parameter, return, and variable type references become TypeRefs."""

    def test_param_and_return_types(self, parser: GoParser) -> None:
        code = "package p\nfunc F(cfg Config, s *Server) *Result {\n\treturn nil\n}\n"
        refs = parser.parse(code, "p.go").type_refs
        params = {(r.param_name, r.name) for r in refs if r.kind == "param"}
        returns = {r.name for r in refs if r.kind == "return"}
        assert ("cfg", "Config") in params
        assert ("s", "Server") in params  # pointer stripped
        assert "Result" in returns

    def test_slice_and_map_stripped_to_base(self, parser: GoParser) -> None:
        code = "package p\nfunc F(items []Config, m map[string]*Server) {}\n"
        names = {r.name for r in parser.parse(code, "p.go").type_refs}
        assert "Config" in names  # []Config -> Config
        assert "Server" in names  # map[string]*Server -> Server

    def test_builtins_are_filtered(self, parser: GoParser) -> None:
        code = "package p\nfunc F(a int, b string) (bool, error) {\n\treturn true, nil\n}\n"
        assert parser.parse(code, "p.go").type_refs == []

    def test_multi_return_types(self, parser: GoParser) -> None:
        code = "package p\nfunc F() (*Thing, Other) {\n\treturn nil, Other{}\n}\n"
        returns = {r.name for r in parser.parse(code, "p.go").type_refs if r.kind == "return"}
        assert returns == {"Thing", "Other"}

    def test_var_declaration_type(self, parser: GoParser) -> None:
        code = "package p\nfunc f() {\n\tvar c Config\n\t_ = c\n}\n"
        result = parser.parse(code, "p.go")
        assert any(r.name == "Config" and r.kind == "variable" for r in result.type_refs)
        assert ("c", "Config") in [(v.var_name, v.type_name) for v in result.variable_types]

    def test_short_var_composite_literal_infers_type(self, parser: GoParser) -> None:
        code = "package p\nfunc f() {\n\tc := Config{}\n\t_ = c\n}\n"
        vts = parser.parse(code, "p.go").variable_types
        assert ("c", "Config") in [(v.var_name, v.type_name) for v in vts]


# ---------------------------------------------------------------------------
# Exports (Go's upper-case convention)
# ---------------------------------------------------------------------------


class TestExports:
    """Upper-cased identifiers are recorded as exported."""

    def test_exported_and_unexported_functions(self, parser: GoParser) -> None:
        code = "package p\nfunc Public() {}\nfunc private() {}\n"
        exports = set(parser.parse(code, "p.go").exports)
        assert "Public" in exports
        assert "private" not in exports

    def test_export_by_method_name_not_receiver(self, parser: GoParser) -> None:
        code = "package p\nfunc (s *server) Exported() {}\nfunc (s *Server) hidden() {}\n"
        exports = set(parser.parse(code, "p.go").exports)
        assert "Exported" in exports
        assert "hidden" not in exports

    def test_exported_types(self, parser: GoParser) -> None:
        code = "package p\ntype Public struct{}\ntype private struct{}\n"
        exports = set(parser.parse(code, "p.go").exports)
        assert "Public" in exports
        assert "private" not in exports


# ---------------------------------------------------------------------------
# Non-ASCII content (byte-offset vs str-index regression — B1)
# ---------------------------------------------------------------------------


class TestParseNonAsciiContent:
    """Multi-byte UTF-8 before a symbol must not shift its content window.

    tree-sitter reports byte offsets into the UTF-8 source, not str indices;
    every extractor slices ``node.text`` (the node's exact bytes), so a
    non-ASCII character earlier in the file must not desync a later symbol.
    """

    CODE = (
        "package main\n"
        "// émojis 🎉🚀 and åäö accented\n"
        'const Greeting = "héllo wörld"\n'
        "\n"
        "type Grüßer struct {\n"
        "\t// ünïcödé field comment\n"
        "\tName string\n"
        "}\n"
        "\n"
        "func (g *Grüßer) Hello(name string) string {\n"
        '\treturn "héllo, " + name\n'
        "}\n"
    )

    def _symbol(self, parser: GoParser, name: str):
        return next(s for s in parser.parse(self.CODE, "grüßer.go").symbols if s.name == name)

    def test_const_content_exact(self, parser: GoParser) -> None:
        const = self._symbol(parser, "Greeting")
        assert const.content == 'Greeting = "héllo wörld"'
        assert const.start_line == 3

    def test_struct_content_exact(self, parser: GoParser) -> None:
        cls = self._symbol(parser, "Grüßer")
        assert cls.content == (
            "Grüßer struct {\n"
            "\t// ünïcödé field comment\n"
            "\tName string\n"
            "}"
        )
        assert cls.start_line == 5
        assert cls.end_line == 8

    def test_method_content_and_signature_exact(self, parser: GoParser) -> None:
        method = self._symbol(parser, "Hello")
        assert method.content == (
            'func (g *Grüßer) Hello(name string) string {\n'
            '\treturn "héllo, " + name\n'
            "}"
        )
        assert method.signature == "func (g *Grüßer) Hello(name string) string"
        assert method.class_name == "Grüßer"
        assert method.start_line == 10
        assert method.end_line == 12


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


class TestParseErrorHandling:
    """Empty and malformed sources degrade gracefully."""

    def test_empty_file(self, parser: GoParser) -> None:
        assert parser.parse("", "empty.go").symbols == []

    def test_whitespace_only(self, parser: GoParser) -> None:
        assert parser.parse("\n\n   \n", "ws.go").symbols == []

    def test_syntax_error_does_not_raise(self, parser: GoParser) -> None:
        result = parser.parse("package p\nfunc broken(\n", "broken.go")
        assert isinstance(result, ParseResult)

    def test_partial_error_still_extracts_valid_defs(self, parser: GoParser) -> None:
        code = "package p\nfunc Good() {}\n@@@ bad tokens @@@\n"
        names = {s.name for s in parser.parse(code, "partial.go").symbols}
        assert "Good" in names

    def test_comments_only(self, parser: GoParser) -> None:
        # Only the package's module symbol — no functions/types from comments.
        result = parser.parse("package p\n// just a comment\n// another\n", "c.go")
        assert [s.kind for s in result.symbols] == ["module"]
