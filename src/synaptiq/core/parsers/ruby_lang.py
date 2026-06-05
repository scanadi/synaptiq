"""Ruby language parser using tree-sitter.

Extracts the language-agnostic :class:`ParseResult` IR from Ruby source so the
downstream ingestion phases can treat Ruby like Python/TypeScript.

This module is built incrementally across the Ruby-parity tasks:

* Task 3 — symbol extraction (methods, classes, modules, constants).
* Task 4 — imports (``require`` / ``require_relative`` / ``autoload``).
* Task 5 — calls (receivers, ``self``, blocks, paren-less, bare calls).
* Task 6 — heritage & mixins.

Each ``parse`` failure degrades gracefully to an empty :class:`ParseResult`,
matching the behaviour of the existing parsers.
"""

from __future__ import annotations

import tree_sitter_ruby as tsruby
from tree_sitter import Language, Node, Parser

from synaptiq.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
)

RUBY_LANGUAGE = Language(tsruby.language())

# ``require``-family methods that bring another file/feature into scope.
_IMPORT_METHODS = frozenset({"require", "require_relative", "autoload", "load"})

# Node types whose direct ``identifier`` children sit in statement/value
# position, where a bare identifier (not a known local) is a method call.
_STMT_CONTAINERS = frozenset({"program", "body_statement", "block_body", "then", "else", "ensure"})


class RubyParser(LanguageParser):
    """Parses Ruby source code using tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(RUBY_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse Ruby source and return structured information.

        Returns an empty :class:`ParseResult` if the source cannot be parsed
        (tree-sitter is error-tolerant, but byte conversion or a malformed
        tree could still raise).
        """
        result = ParseResult()
        try:
            tree = self._parser.parse(bytes(content, "utf8"))
        except Exception:
            return result

        self._walk(tree.root_node, content, result, class_name="")
        self._extract_calls(tree.root_node, result, locals_=set())
        return result

    # ------------------------------------------------------------------
    # Definition walking
    # ------------------------------------------------------------------

    def _walk(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Recursively walk the AST collecting definitions.

        ``class_name`` is the name of the lexically enclosing class/module, used
        to attribute methods and constants.  Definition nodes are dispatched to
        their dedicated extractors; everything else is descended into so that a
        definition nested inside e.g. an ``if`` block is still discovered.
        """
        for child in node.children:
            match child.type:
                case "method":
                    self._extract_method(child, content, result, class_name)
                case "singleton_method":
                    self._extract_singleton_method(child, content, result, class_name)
                case "class":
                    self._extract_class(child, content, result, class_name)
                case "module":
                    self._extract_module(child, content, result, class_name)
                case "assignment":
                    self._extract_constant(child, content, result, class_name)
                case "call":
                    # A ``call`` may be a ``require``-family import; either way we
                    # still descend so definitions nested in a block are found.
                    self._extract_import(child, result)
                    self._walk(child, content, result, class_name)
                case _:
                    self._walk(child, content, result, class_name)

    def _extract_method(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract an instance method (``def foo``) or top-level function."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        kind = "method" if class_name else "function"

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind=kind,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
                signature=self._build_signature(node, name),
                class_name=class_name,
            )
        )

        # Descend into the body for nested definitions (rare in Ruby, but a
        # ``def`` may itself contain a class/module).  Nested defs are treated
        # as standalone functions, matching the Python parser.
        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, content, result, class_name="")

    def _extract_singleton_method(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a singleton/class method (``def self.foo`` / ``def Foo.bar``)."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        kind = "method" if class_name else "function"

        obj_node = node.child_by_field_name("object")
        receiver = obj_node.text.decode("utf8") if obj_node is not None else "self"

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind=kind,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
                signature=self._build_signature(node, name, receiver=receiver),
                class_name=class_name,
            )
        )

        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, content, result, class_name="")

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a class definition and walk its body for members."""
        name = self._definition_name(node)
        if not name:
            return

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="class",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
            )
        )

        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, content, result, class_name=name)

    def _extract_module(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a module definition and walk its body for members."""
        name = self._definition_name(node)
        if not name:
            return

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="module",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
            )
        )

        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, content, result, class_name=name)

    def _extract_constant(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a constant assignment (``CONFIG = ...``).

        Only assignments whose left-hand side is a bare ``constant`` are
        captured; instance/local-variable assignments are ignored.
        """
        left = node.child_by_field_name("left")
        if left is None or left.type != "constant":
            return

        name = left.text.decode("utf8")
        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="constant",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content[node.start_byte : node.end_byte],
                class_name=class_name,
            )
        )

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, result: ParseResult) -> None:
        """Extract a ``require``/``require_relative``/``autoload``/``load`` call.

        Only string-literal arguments are honoured; dynamic requires (an
        identifier, a method call, or an interpolated string) are ignored so we
        never record a bogus module path.
        """
        method_node = node.child_by_field_name("method")
        if method_node is None or method_node.type != "identifier":
            return
        method = method_node.text.decode("utf8")
        if method not in _IMPORT_METHODS:
            return

        args_node = node.child_by_field_name("arguments")
        if args_node is None:
            return
        args = [c for c in args_node.children if c.is_named]
        if not args:
            return

        if method == "autoload":
            # ``autoload :Const, "path"`` — first arg is the constant symbol,
            # second is the feature path.
            if len(args) < 2:
                return
            module = self._string_literal(args[1])
            if module is None:
                return
            const = args[0]
            names = [const.text.decode("utf8").lstrip(":")] if const.type == "simple_symbol" else []
            result.imports.append(ImportInfo(module=module, names=names))
            return

        module = self._string_literal(args[0])
        if module is None:
            return
        result.imports.append(ImportInfo(module=module, is_relative=method == "require_relative"))

    @staticmethod
    def _string_literal(node: Node) -> str | None:
        """Return the text of a plain string literal, or ``None``.

        ``None`` is returned for any non-string node or for an interpolated
        string (``"a/#{x}"``), both of which represent dynamic values.
        """
        if node.type != "string":
            return None
        parts: list[str] = []
        for child in node.children:
            if child.type == "string_content":
                parts.append(child.text.decode("utf8"))
            elif child.type == "interpolation":
                return None
        return "".join(parts)

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

    def _extract_calls(self, node: Node, result: ParseResult, locals_: set[str]) -> None:
        """Recursively collect method calls into ``result.calls``.

        ``locals_`` holds the names bound as local variables / parameters in the
        current scope.  Ruby resolves a bare identifier as a method call only
        when no local of that name exists, so this set distinguishes a real
        zero-arg call (``validate``) from a local-variable read (``count``).

        Scope rules mirror Ruby's: a ``def`` opens a *fresh* scope (methods do
        not capture surrounding locals) seeded with its parameters, while a
        block inherits the enclosing locals plus its own block parameters.
        """
        for child in node.children:
            ctype = child.type
            if ctype == "call":
                self._extract_call_node(child, result)
                # Descend to find nested calls (arguments, receiver chains,
                # block bodies).  A ``call`` is not a statement container, so a
                # receiver identifier here is never mistaken for a bare call.
                self._extract_calls(child, result, locals_)
            elif ctype in ("method", "singleton_method"):
                body = child.child_by_field_name("body")
                if body is not None:
                    self._extract_calls(body, result, self._parameter_names(child))
            elif ctype in ("block", "do_block"):
                self._extract_calls(child, result, locals_ | self._block_parameter_names(child))
            elif ctype == "assignment":
                right = child.child_by_field_name("right")
                if right is not None and right.type == "identifier":
                    name = right.text.decode("utf8")
                    if name not in locals_:
                        result.calls.append(CallInfo(name=name, line=right.start_point[0] + 1))
                else:
                    self._extract_calls(child, result, locals_)
                left = child.child_by_field_name("left")
                if left is not None and left.type == "identifier":
                    locals_.add(left.text.decode("utf8"))
            elif ctype == "identifier" and node.type in _STMT_CONTAINERS:
                name = child.text.decode("utf8")
                if name not in locals_:
                    result.calls.append(CallInfo(name=name, line=child.start_point[0] + 1))
            else:
                self._extract_calls(child, result, locals_)

    def _extract_call_node(self, node: Node, result: ParseResult) -> None:
        """Extract a single ``call`` node into a :class:`CallInfo`."""
        method_node = node.child_by_field_name("method")
        if method_node is None:
            return
        name = method_node.text.decode("utf8")

        receiver = ""
        recv_node = node.child_by_field_name("receiver")
        if recv_node is not None:
            receiver = self._receiver_name(recv_node)

        result.calls.append(
            CallInfo(
                name=name,
                line=node.start_point[0] + 1,
                receiver=receiver,
                arguments=self._identifier_arguments(node),
            )
        )

    def _receiver_name(self, node: Node) -> str:
        """Return a textual receiver for a call (``self``, identifier, constant)."""
        if node.type == "self":
            return "self"
        if node.type in ("identifier", "constant", "scope_resolution"):
            return node.text.decode("utf8")
        # Chained call or other expression — fall back to the leftmost name.
        return self._root_identifier(node)

    @staticmethod
    def _root_identifier(node: Node) -> str:
        """Walk down the leftmost children to the first name-like node."""
        current: Node | None = node
        while current is not None:
            if current.type in ("identifier", "constant", "self"):
                return current.text.decode("utf8")
            current = current.children[0] if current.children else None
        return ""

    @staticmethod
    def _identifier_arguments(node: Node) -> list[str]:
        """Return bare-identifier arguments of a call (callback references)."""
        args_node = node.child_by_field_name("arguments")
        if args_node is None:
            return []
        return [c.text.decode("utf8") for c in args_node.children if c.type == "identifier"]

    @staticmethod
    def _parameter_names(node: Node) -> set[str]:
        """Collect local names bound by a method's parameter list."""
        params_node = node.child_by_field_name("parameters")
        if params_node is None:
            return set()
        return RubyParser._collect_parameter_identifiers(params_node)

    @staticmethod
    def _block_parameter_names(node: Node) -> set[str]:
        """Collect local names bound by a block's ``|params|`` list."""
        for child in node.children:
            if child.type == "block_parameters":
                return RubyParser._collect_parameter_identifiers(child)
        return set()

    @staticmethod
    def _collect_parameter_identifiers(params_node: Node) -> set[str]:
        """Pull every plain identifier name out of a parameter list.

        Handles simple, optional (``a=1``), keyword (``a:``), splat (``*a``),
        and block (``&blk``) parameters — each wraps or contains an identifier.
        """
        names: set[str] = set()
        for child in params_node.children:
            if child.type == "identifier":
                names.add(child.text.decode("utf8"))
            else:
                inner = child.child_by_field_name("name")
                if inner is not None and inner.type == "identifier":
                    names.add(inner.text.decode("utf8"))
                else:
                    for grand in child.children:
                        if grand.type == "identifier":
                            names.add(grand.text.decode("utf8"))
                            break
        return names

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _definition_name(node: Node) -> str:
        """Return the name of a ``class``/``module`` node.

        Handles both bare (``class Foo``) and namespaced (``class Foo::Bar``)
        definitions by using the textual name; the trailing segment is the
        most useful for method attribution.
        """
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return ""
        text = name_node.text.decode("utf8")
        # For ``class Foo::Bar`` keep the final segment as the symbol name.
        return text.rsplit("::", 1)[-1]

    @staticmethod
    def _build_signature(node: Node, name: str, *, receiver: str = "") -> str:
        """Build a ``def name(params)`` signature string for a method.

        ``receiver`` is set for singleton methods (``def self.foo`` →
        ``receiver="self"``) and prefixed before the name.
        """
        params_node = node.child_by_field_name("parameters")
        params = params_node.text.decode("utf8") if params_node is not None else ""
        qualified = f"{receiver}.{name}" if receiver else name
        return f"def {qualified}{params}"
