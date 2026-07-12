"""Go language parser using tree-sitter.

Extracts the language-agnostic :class:`ParseResult` IR from Go source so the
downstream ingestion phases can treat Go like Python/TypeScript/Ruby.

Symbol mapping (see W4.2 scope):

* ``func`` (no receiver)            → ``function``
* ``func (r T) m()`` (receiver)     → ``method`` with ``class_name`` = the
  receiver *type* name (pointer ``*T`` stripped to ``T``)
* ``type X struct { ... }``         → ``class`` (+ ``extends`` heritage for
  anonymous/embedded fields)
* ``type X interface { ... }``      → ``interface`` (+ ``extends`` heritage for
  embedded interfaces)
* ``type X = Y`` / ``type X Y``     → ``type_alias``
* ``const`` / ``var`` (top-level)   → ``constant`` (an unmapped kind — parsed
  but not materialised as a graph node, matching the existing Ruby behaviour)
* ``package foo``                   → ``module`` (one per file, like a Ruby
  module marker)

Go's *exported* convention (an identifier is public iff its first letter is
upper-case) is surfaced by adding every exported symbol name to
:attr:`ParseResult.exports`; the parser phase then sets ``is_exported`` on the
node, which the dead-code phase already treats as an exemption.

The walk is single-pass (post-W2.2 standard): definitions, calls, imports, and
type references are all collected in one recursive descent. Byte-offset-correct
slicing is used throughout (``node.text`` holds the node's exact source bytes),
so multi-byte UTF-8 earlier in a file never shifts a later symbol's window.
"""

from __future__ import annotations

import tree_sitter_go as tsgo
from tree_sitter import Language, Node, Parser

from synaptiq.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
    VarTypeInfo,
)

GO_LANGUAGE = Language(tsgo.language())

# Go predeclared (builtin) type names.  Type references to these never resolve
# to a user-defined node, so they are filtered out before emitting a TypeRef —
# mirroring the ``_BUILTIN_TYPES`` blocklists in the Python/TS parsers.
_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "bool",
        "string",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uintptr",
        "byte",
        "rune",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "error",
        "any",
        "comparable",
    }
)


class GoParser(LanguageParser):
    """Parses Go source code using tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(GO_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse Go source and return structured information.

        Degrades to an empty :class:`ParseResult` if the source cannot be
        parsed (tree-sitter is error-tolerant, but byte conversion or a
        malformed tree could still raise) — matching the other parsers.
        """
        result = ParseResult()
        try:
            tree = self._parser.parse(bytes(content, "utf8"))
        except Exception:
            return result

        self._walk(tree.root_node, result)
        return result

    # ------------------------------------------------------------------
    # Single-pass walk
    # ------------------------------------------------------------------

    def _walk(self, node: Node, result: ParseResult) -> None:
        """Walk the tree once, dispatching on node type.

        Each node is visited exactly once: the extractors never recurse back
        into ``_walk`` — only the single child loop at the bottom does.
        Goroutine (``go f()``) and deferred (``defer f()``) calls need no
        special handling: their inner ``call_expression`` is reached naturally
        by this descent.
        """
        ntype = node.type

        if ntype == "package_clause":
            self._extract_package(node, result)
        elif ntype == "import_declaration":
            self._extract_imports(node, result)
        elif ntype == "function_declaration":
            self._extract_function(node, result)
        elif ntype == "method_declaration":
            self._extract_method(node, result)
        elif ntype == "type_declaration":
            self._extract_type_declaration(node, result)
        elif ntype in ("const_declaration", "var_declaration"):
            self._extract_const_var(node, result)
        elif ntype == "short_var_declaration":
            self._extract_short_var(node, result)
        elif ntype == "call_expression":
            self._extract_call(node, result)
        elif ntype == "composite_literal":
            self._extract_composite_literal(node, result)

        for child in node.children:
            self._walk(child, result)

    # ------------------------------------------------------------------
    # Package
    # ------------------------------------------------------------------

    def _extract_package(self, node: Node, result: ParseResult) -> None:
        """Emit a ``module`` symbol for the file's ``package`` clause."""
        for child in node.children:
            if child.type == "package_identifier":
                name = child.text.decode("utf8")
                result.symbols.append(
                    SymbolInfo(
                        name=name,
                        kind="module",
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        content=node.text.decode("utf8"),
                    )
                )
                self._add_export(name, result)
                return

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def _extract_imports(self, node: Node, result: ParseResult) -> None:
        """Extract ``import`` specs (single, grouped, aliased, blank, dot).

        ``import "net/http"``          → module=``net/http``
        ``import m "math/rand"``       → module=``math/rand``, alias=``m``
        ``import _ "database/sql"``    → module=``database/sql``, alias=``_``
        ``import . "strings"``         → module=``strings``, alias=``.``
        """
        for spec in self._iter_import_specs(node):
            path_node = spec.child_by_field_name("path")
            if path_node is None:
                continue
            module = self._string_value(path_node)
            if not module:
                continue
            alias = ""
            name_node = spec.child_by_field_name("name")
            if name_node is not None:
                # package_identifier ("m"), blank_identifier ("_"), or dot (".")
                alias = name_node.text.decode("utf8")
            result.imports.append(ImportInfo(module=module, alias=alias))

    @staticmethod
    def _iter_import_specs(node: Node):
        """Yield every ``import_spec`` under an ``import_declaration``.

        Handles both the single form (``import "x"`` — an ``import_spec`` child)
        and the grouped form (``import ( ... )`` — an ``import_spec_list``).
        """
        for child in node.children:
            if child.type == "import_spec":
                yield child
            elif child.type == "import_spec_list":
                for sub in child.children:
                    if sub.type == "import_spec":
                        yield sub

    # ------------------------------------------------------------------
    # Functions & methods
    # ------------------------------------------------------------------

    def _extract_function(self, node: Node, result: ParseResult) -> None:
        """Extract a top-level ``func`` (no receiver) as a ``function``."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = name_node.text.decode("utf8")
        body = node.child_by_field_name("body")

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="function",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=node.text.decode("utf8"),
                signature=self._signature(node, body),
            )
        )
        self._add_export(name, result)
        self._extract_signature_types(node, result)

    def _extract_method(self, node: Node, result: ParseResult) -> None:
        """Extract a ``func (r T) m()`` as a ``method`` owned by type ``T``.

        The receiver variable is recorded as a variable→type mapping so that
        intra-type calls (``r.other()`` inside a method) resolve against the
        receiver's type.
        """
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = name_node.text.decode("utf8")
        body = node.child_by_field_name("body")

        recv_name, recv_type = self._receiver_info(node)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=node.text.decode("utf8"),
                signature=self._signature(node, body),
                class_name=recv_type,
            )
        )
        self._add_export(name, result)
        self._extract_signature_types(node, result)

        if recv_name and recv_type and recv_type not in _BUILTIN_TYPES:
            result.variable_types.append(
                VarTypeInfo(
                    var_name=recv_name,
                    type_name=recv_type,
                    line=node.start_point[0] + 1,
                )
            )

    def _receiver_info(self, method_node: Node) -> tuple[str, str]:
        """Return ``(receiver_var_name, receiver_type_name)`` for a method.

        ``func (s *Server) Start()`` → ``("s", "Server")``; the pointer,
        qualifier, and any generic type arguments are stripped to the base
        type name.  Returns ``("", "")`` for a malformed receiver.
        """
        receiver = method_node.child_by_field_name("receiver")
        if receiver is None:
            return "", ""
        for child in receiver.children:
            if child.type != "parameter_declaration":
                continue
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            recv_name = name_node.text.decode("utf8") if name_node is not None else ""
            recv_type = self._base_type_name(type_node) if type_node is not None else ""
            return recv_name, recv_type or ""
        return "", ""

    def _extract_signature_types(self, node: Node, result: ParseResult) -> None:
        """Emit param/return :class:`TypeRef` entries for a func/method node."""
        params = node.child_by_field_name("parameters")
        if params is not None:
            for param in params.children:
                if param.type != "parameter_declaration":
                    continue
                name_node = param.child_by_field_name("name")
                param_name = name_node.text.decode("utf8") if name_node is not None else ""
                type_node = param.child_by_field_name("type")
                if type_node is None:
                    continue
                for type_name in self._collect_type_names(type_node):
                    result.type_refs.append(
                        TypeRef(
                            name=type_name,
                            kind="param",
                            line=param.start_point[0] + 1,
                            param_name=param_name,
                        )
                    )

        result_node = node.child_by_field_name("result")
        if result_node is None:
            return
        if result_node.type == "parameter_list":
            # Named or multiple returns: (a int, b error) / (*Thing, error).
            for param in result_node.children:
                if param.type != "parameter_declaration":
                    continue
                type_node = param.child_by_field_name("type")
                if type_node is None:
                    continue
                self._emit_return_types(type_node, result)
        else:
            # Single unnamed return type.
            self._emit_return_types(result_node, result)

    def _emit_return_types(self, type_node: Node, result: ParseResult) -> None:
        for type_name in self._collect_type_names(type_node):
            result.type_refs.append(
                TypeRef(name=type_name, kind="return", line=type_node.start_point[0] + 1)
            )

    # ------------------------------------------------------------------
    # Types (struct / interface / alias) + heritage
    # ------------------------------------------------------------------

    def _extract_type_declaration(self, node: Node, result: ParseResult) -> None:
        """Extract every ``type_spec`` / ``type_alias`` in a type declaration.

        Handles both the single form and the grouped ``type ( ... )`` block.
        """
        for spec in node.children:
            if spec.type == "type_alias":
                self._extract_type_alias(spec, result)
            elif spec.type == "type_spec":
                self._extract_type_spec(spec, result)

    def _extract_type_alias(self, spec: Node, result: ParseResult) -> None:
        """Handle ``type X = Y`` (an explicit alias)."""
        name_node = spec.child_by_field_name("name")
        if name_node is None:
            return
        name = name_node.text.decode("utf8")
        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="type_alias",
                start_line=spec.start_point[0] + 1,
                end_line=spec.end_point[0] + 1,
                content=spec.text.decode("utf8"),
            )
        )
        self._add_export(name, result)

    def _extract_type_spec(self, spec: Node, result: ParseResult) -> None:
        """Handle ``type X <underlying>`` — struct, interface, or defined type."""
        name_node = spec.child_by_field_name("name")
        if name_node is None:
            return
        name = name_node.text.decode("utf8")
        type_node = spec.child_by_field_name("type")

        if type_node is not None and type_node.type == "struct_type":
            kind = "class"
        elif type_node is not None and type_node.type == "interface_type":
            kind = "interface"
        else:
            # ``type Celsius float64`` / ``type Qux int`` — a defined type.
            kind = "type_alias"

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind=kind,
                start_line=spec.start_point[0] + 1,
                end_line=spec.end_point[0] + 1,
                content=spec.text.decode("utf8"),
            )
        )
        self._add_export(name, result)

        if kind == "class":
            self._extract_struct_embedding(name, type_node, result)
        elif kind == "interface":
            self._extract_interface_embedding(name, type_node, result)

    def _extract_struct_embedding(
        self, struct_name: str, struct_node: Node, result: ParseResult
    ) -> None:
        """Emit ``extends`` heritage for anonymous (embedded) struct fields.

        An embedded field is a ``field_declaration`` with no ``name`` field
        (e.g. ``http.Handler`` or ``*Logger`` inside a struct body).  Embedding
        is the closest Go analogue to inheritance, so it maps to ``extends``.
        """
        for child in struct_node.children:
            if child.type != "field_declaration_list":
                continue
            for field in child.children:
                if field.type != "field_declaration":
                    continue
                if field.child_by_field_name("name") is not None:
                    continue  # named field — not embedded
                type_node = field.child_by_field_name("type")
                base = self._base_type_name(type_node) if type_node is not None else None
                if base:
                    result.heritage.append((struct_name, "extends", base))

    def _extract_interface_embedding(
        self, iface_name: str, iface_node: Node, result: ParseResult
    ) -> None:
        """Emit ``extends`` heritage for embedded interfaces.

        Inside an interface body, a bare type element (``type_elem``, e.g.
        ``Reader``) embeds another interface — a static, syntactic form of
        interface extension, mapped to ``extends`` (mirrors TS
        ``interface A extends B``).  Method elements (``method_elem``) are not
        emitted as symbols (interface method sets are not materialised).
        """
        for child in iface_node.children:
            if child.type != "type_elem":
                continue
            for sub in child.named_children:
                base = self._base_type_name(sub)
                if base:
                    result.heritage.append((iface_name, "extends", base))

    # ------------------------------------------------------------------
    # const / var
    # ------------------------------------------------------------------

    def _extract_const_var(self, node: Node, result: ParseResult) -> None:
        """Extract ``const``/``var`` specs.

        Top-level (package-scope) specs emit a ``constant`` symbol (an unmapped
        kind — parsed but not materialised, matching Ruby's constants).  Every
        ``var`` spec that carries an explicit type also contributes a
        variable→type mapping and a variable :class:`TypeRef`, regardless of
        scope, so local declarations still feed receiver resolution and
        ``USES_TYPE``.
        """
        is_top_level = node.parent is not None and node.parent.type == "source_file"
        is_var = node.type == "var_declaration"

        for spec in self._iter_value_specs(node):
            names = [c for c in spec.children if c.type == "identifier"]
            type_node = spec.child_by_field_name("type")

            for name_node in names:
                name = name_node.text.decode("utf8")
                if is_top_level:
                    result.symbols.append(
                        SymbolInfo(
                            name=name,
                            kind="constant",
                            start_line=spec.start_point[0] + 1,
                            end_line=spec.end_point[0] + 1,
                            content=spec.text.decode("utf8"),
                        )
                    )
                    self._add_export(name, result)

                if is_var and type_node is not None:
                    base = self._base_type_name(type_node)
                    if base and base not in _BUILTIN_TYPES:
                        result.variable_types.append(
                            VarTypeInfo(
                                var_name=name,
                                type_name=base,
                                line=spec.start_point[0] + 1,
                            )
                        )

            if is_var and type_node is not None:
                for type_name in self._collect_type_names(type_node):
                    result.type_refs.append(
                        TypeRef(
                            name=type_name,
                            kind="variable",
                            line=spec.start_point[0] + 1,
                        )
                    )

    @staticmethod
    def _iter_value_specs(node: Node):
        """Yield every ``const_spec`` / ``var_spec`` in a declaration.

        Handles the single form and the grouped ``( ... )`` block (whose specs
        may sit directly under the declaration or inside a ``*_spec_list``).
        """
        for child in node.children:
            if child.type in ("const_spec", "var_spec"):
                yield child
            elif child.type in ("const_spec_list", "var_spec_list"):
                for sub in child.children:
                    if sub.type in ("const_spec", "var_spec"):
                        yield sub

    def _extract_short_var(self, node: Node, result: ParseResult) -> None:
        """Infer a variable→type mapping from ``x := T{}`` / ``x := &T{}``.

        Only composite-literal right-hand sides yield a type (the Go analogue
        of TS's ``const pool = new Pool()`` inference); call results and
        literals are left unresolved.  Only single-name assignments are handled.
        """
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return
        left_ids = [c for c in left.children if c.type == "identifier"]
        right_vals = [c for c in right.named_children]
        if len(left_ids) != 1 or len(right_vals) != 1:
            return

        var_name = left_ids[0].text.decode("utf8")
        value = right_vals[0]
        if value.type == "unary_expression":
            # &T{}
            operand = value.child_by_field_name("operand")
            if operand is not None:
                value = operand
        if value.type != "composite_literal":
            return
        type_node = value.child_by_field_name("type")
        base = self._base_type_name(type_node) if type_node is not None else None
        if base and base not in _BUILTIN_TYPES:
            result.variable_types.append(
                VarTypeInfo(
                    var_name=var_name,
                    type_name=base,
                    line=node.start_point[0] + 1,
                )
            )

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def _extract_call(self, node: Node, result: ParseResult) -> None:
        """Extract a ``call_expression`` into a :class:`CallInfo`.

        ``f(x)``          → name=``f``
        ``pkg.F(x)``      → name=``F``, receiver=``pkg``
        ``r.method(x)``   → name=``method``, receiver=``r``
        """
        func_node = node.child_by_field_name("function")
        if func_node is None:
            return
        if func_node.type == "parenthesized_expression":
            inner = func_node.named_child(0)
            if inner is not None:
                func_node = inner

        line = node.start_point[0] + 1
        arguments = self._identifier_arguments(node)

        if func_node.type == "identifier":
            result.calls.append(
                CallInfo(name=func_node.text.decode("utf8"), line=line, arguments=arguments)
            )
        elif func_node.type == "selector_expression":
            field = func_node.child_by_field_name("field")
            operand = func_node.child_by_field_name("operand")
            if field is not None:
                receiver = operand.text.decode("utf8") if operand is not None else ""
                result.calls.append(
                    CallInfo(
                        name=field.text.decode("utf8"),
                        line=line,
                        receiver=receiver,
                        arguments=arguments,
                    )
                )

    def _extract_composite_literal(self, node: Node, result: ParseResult) -> None:
        """Emit a :class:`CallInfo` for a struct literal ``T{...}`` / ``&T{...}``.

        This is the Go analogue of TS's ``new Expr`` handling: instantiating a
        struct is a use of that type, so it links the containing symbol to the
        struct (class) node and keeps instantiated-but-uncalled structs off the
        dead-code list.  Restricted to plain and generic named types
        (``type_identifier`` / ``generic_type``); qualified (cross-package),
        slice, and map literals are skipped to avoid false last-segment matches.
        """
        type_node = node.child_by_field_name("type")
        if type_node is None or type_node.type not in ("type_identifier", "generic_type"):
            return
        base = self._base_type_name(type_node)
        if base and base not in _BUILTIN_TYPES:
            result.calls.append(CallInfo(name=base, line=node.start_point[0] + 1))

    @staticmethod
    def _identifier_arguments(call_node: Node) -> list[str]:
        """Return bare-identifier arguments of a call (callback references)."""
        args_node = call_node.child_by_field_name("arguments")
        if args_node is None:
            return []
        return [c.text.decode("utf8") for c in args_node.children if c.type == "identifier"]

    # ------------------------------------------------------------------
    # Type-name helpers
    # ------------------------------------------------------------------

    def _base_type_name(self, node: Node | None) -> str | None:
        """Reduce a type node to its single base named type.

        Strips pointer/slice/array wrappers and generic type arguments, and
        returns the trailing segment of a qualified type (``pkg.T`` → ``T``).
        Returns ``None`` for composite types with no single base
        (maps, functions, channels, ...).
        """
        if node is None:
            return None
        t = node.type
        if t == "type_identifier":
            return node.text.decode("utf8")
        if t == "qualified_type":
            name = node.child_by_field_name("name")
            return name.text.decode("utf8") if name is not None else None
        if t == "generic_type":
            return self._base_type_name(node.child_by_field_name("type"))
        if t in ("slice_type", "array_type"):
            return self._base_type_name(node.child_by_field_name("element"))
        if t == "pointer_type":
            for child in node.named_children:
                base = self._base_type_name(child)
                if base:
                    return base
            return None
        return None

    def _collect_type_names(self, node: Node) -> list[str]:
        """Collect every non-builtin base type name reachable in a type node.

        Descends through pointers, slices, arrays, maps (key + value), generics
        (base + arguments), etc., gathering ``type_identifier`` /
        ``qualified_type`` leaves.  Builtins and duplicates are dropped.  Used
        for param/return/variable :class:`TypeRef` emission.
        """
        names: list[str] = []
        self._gather_type_names(node, names)
        seen: set[str] = set()
        out: list[str] = []
        for name in names:
            if name in _BUILTIN_TYPES or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    def _gather_type_names(self, node: Node, out: list[str]) -> None:
        if node.type == "type_identifier":
            out.append(node.text.decode("utf8"))
            return
        if node.type == "qualified_type":
            name = node.child_by_field_name("name")
            if name is not None:
                out.append(name.text.decode("utf8"))
            return
        for child in node.named_children:
            self._gather_type_names(child, out)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_export(name: str, result: ParseResult) -> None:
        """Record *name* as exported when it follows Go's upper-case convention."""
        if name and name[0].isupper():
            result.exports.append(name)

    @staticmethod
    def _signature(node: Node, body: Node | None) -> str:
        """Return the declaration signature — everything up to the body block.

        Sliced off ``node.text`` (the node's exact bytes) by the body's byte
        offset, so it stays correct with non-ASCII content earlier in the file.
        """
        if body is not None:
            return node.text[: body.start_byte - node.start_byte].decode("utf8").strip()
        return node.text.decode("utf8").strip()

    @staticmethod
    def _string_value(node: Node) -> str:
        """Extract the value of an ``interpreted_string_literal`` import path."""
        for child in node.children:
            if child.type == "interpreted_string_literal_content":
                return child.text.decode("utf8")
        text = node.text.decode("utf8")
        if len(text) >= 2 and text[0] in ("'", '"', "`") and text[-1] in ("'", '"', "`"):
            return text[1:-1]
        return text
