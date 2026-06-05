"""Ruby language parser using tree-sitter.

Extracts the language-agnostic :class:`ParseResult` IR from Ruby source so the
downstream ingestion phases can treat Ruby like Python/TypeScript.

This module is built incrementally across the Ruby-parity tasks:

* Task 3 — symbol extraction (methods, classes, modules, constants).
* Task 4 — imports (``require`` / ``require_relative`` / ``autoload``).
* Task 5 — calls.
* Task 6 — heritage & mixins.

Each ``parse`` failure degrades gracefully to an empty :class:`ParseResult`,
matching the behaviour of the existing parsers.
"""

from __future__ import annotations

import tree_sitter_ruby as tsruby
from tree_sitter import Language, Node, Parser

from synaptiq.core.parsers.base import (
    LanguageParser,
    ParseResult,
    SymbolInfo,
)

RUBY_LANGUAGE = Language(tsruby.language())


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
