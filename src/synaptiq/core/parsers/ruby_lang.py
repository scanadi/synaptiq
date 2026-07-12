"""Ruby language parser using tree-sitter.

Extracts the language-agnostic :class:`ParseResult` IR from Ruby source so the
downstream ingestion phases can treat Ruby like Python/TypeScript.

This module is built incrementally across the Ruby-parity tasks:

* Task 3 — symbol extraction (methods, classes, modules, constants).
* Task 4 — imports (``require`` / ``require_relative`` / ``autoload``).
* Task 5 — calls (receivers, ``self``, blocks, paren-less, bare calls).
* Task 6 — heritage (``<``), mixins (include/extend/prepend), ``attr_*`` capture.

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

# Mixin macros: ``include``/``extend``/``prepend`` pull a module into the
# enclosing class/module, modelled as a ``"mixin"`` heritage tuple.
_MIXIN_METHODS = frozenset({"include", "extend", "prepend"})

# Accessor-generating macros.  Their symbol arguments name methods that Ruby
# synthesises at load time; we record them on the owning type so dead-code
# detection (Task 10) can treat the generated accessors as used.
_ATTR_METHODS = frozenset({"attr_accessor", "attr_reader", "attr_writer"})

# Rails callback macros.  Their symbol arguments name methods the framework
# invokes indirectly (never via a direct call edge), so we record them on the
# owning type for the same dead-code exemption path as ``attr_*``.
_CALLBACK_METHODS = frozenset({
    "before_action", "after_action", "around_action",
    "before_filter", "after_filter", "around_filter",
    "before_save", "after_save", "around_save",
    "before_create", "after_create", "around_create",
    "before_update", "after_update", "around_update",
    "before_destroy", "after_destroy", "around_destroy",
    "before_validation", "after_validation",
    "after_commit", "after_rollback", "after_initialize",
    "after_find", "after_touch",
})

# Macros whose symbol arguments name framework-invoked methods recorded on the
# owning type's ``decorators`` (``attr_*`` accessors + Rails callbacks).
_SYMBOL_RECORDING_METHODS = _ATTR_METHODS | _CALLBACK_METHODS

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

        self._walk(
            tree.root_node,
            content,
            result,
            class_name="",
            owner=None,
            locals_=set(),
            collect_symbols=True,
        )
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
        owner: SymbolInfo | None,
        locals_: set[str],
        collect_symbols: bool,
    ) -> None:
        """Walk the AST once, collecting definitions **and** calls together.

        This folds the former second ``_extract_calls`` pass into the single
        definition walk. Two independent kinds of state are threaded down:

        * *Symbol state* — ``class_name`` (the lexically enclosing class/module,
          for method/constant attribution) and ``owner`` (that type's
          :class:`SymbolInfo`, for ``attr_*``/callback macros). Symbol
          extraction only runs when ``collect_symbols`` is set.
        * *Call state* — ``locals_``, the names bound as locals/parameters in
          the current scope. Ruby resolves a bare identifier as a method call
          only when no local of that name exists, so this set separates a
          zero-arg call (``validate``) from a local read (``count``). Call
          extraction always runs.

        ``collect_symbols`` is cleared for the sub-regions the old symbol walk
        never descended into but the old call walk did — an assignment RHS and a
        class/module's non-body children — so those yield calls only, exactly as
        before. Scope rules mirror Ruby's: a ``def`` opens a *fresh* scope
        seeded with its parameters; a block inherits a *copy* of the enclosing
        locals plus its block parameters; a class/module body shares the
        enclosing locals.
        """
        for child in node.children:
            ctype = child.type

            if ctype == "call":
                # Call site is always recorded; a bare ``call`` may also be a
                # ``require`` import or a class macro (mixin / attr_*), which are
                # symbol-side concerns.
                self._extract_call_node(child, result)
                if collect_symbols:
                    self._extract_import(child, result)
                    self._extract_class_macro(child, result, class_name, owner)
                self._walk(
                    child, content, result, class_name, owner, locals_, collect_symbols
                )

            elif ctype in ("method", "singleton_method"):
                if collect_symbols:
                    if ctype == "method":
                        self._extract_method(child, content, result, class_name)
                    else:
                        self._extract_singleton_method(child, content, result, class_name)
                # A ``def`` opens a fresh scope seeded with its parameters and
                # only its body is descended (a paren-less identifier in the
                # parameter list is never a method call).
                body = child.child_by_field_name("body")
                if body is not None:
                    self._walk(
                        body,
                        content,
                        result,
                        class_name="",
                        owner=None,
                        locals_=self._parameter_names(child),
                        collect_symbols=collect_symbols,
                    )

            elif ctype in ("class", "module"):
                symbol: SymbolInfo | None = None
                if collect_symbols:
                    if ctype == "class":
                        symbol = self._extract_class(child, content, result, class_name)
                    else:
                        symbol = self._extract_module(child, content, result, class_name)
                if symbol is None:
                    # Unnamed/malformed, or already in a calls-only region: the
                    # old symbol walk skipped the whole node while the call walk
                    # descended it — so recurse every child for calls only.
                    self._walk(
                        child, content, result, class_name, None, locals_, False
                    )
                else:
                    # Calls in the superclass expression (``class A < Base.for(x)``)
                    # precede the body in source order; the ``name`` field is a
                    # constant and never holds a call, so it is skipped. The body
                    # shares the enclosing locals (only defs reset the scope).
                    superclass = child.child_by_field_name("superclass")
                    if superclass is not None:
                        self._walk(
                            superclass, content, result, class_name, None, locals_, False
                        )
                    body = child.child_by_field_name("body")
                    if body is not None:
                        self._walk(
                            body,
                            content,
                            result,
                            symbol.name,
                            symbol,
                            locals_,
                            collect_symbols,
                        )

            elif ctype in ("block", "do_block"):
                # A block inherits a COPY of the enclosing locals plus its own
                # block parameters, so its bindings do not leak outward.
                self._walk(
                    child,
                    content,
                    result,
                    class_name,
                    owner,
                    locals_ | self._block_parameter_names(child),
                    collect_symbols,
                )

            elif ctype in ("assignment", "operator_assignment"):
                # ``x = foo`` / ``x ||= foo`` — a bare identifier RHS is a
                # paren-less call; the LHS name then becomes a local.
                if collect_symbols and ctype == "assignment":
                    self._extract_constant(child, content, result, class_name)
                right = child.child_by_field_name("right")
                if right is not None and right.type == "identifier":
                    name = right.text.decode("utf8")
                    if name not in locals_:
                        result.calls.append(
                            CallInfo(name=name, line=right.start_point[0] + 1)
                        )
                else:
                    # The old symbol walk descended into an ``operator_assignment``
                    # but not a plain ``assignment`` (only its constant LHS
                    # mattered); the call walk descended into both. Preserve the
                    # split so a ``require`` hidden in an assignment RHS is not
                    # newly promoted to an import.
                    recurse_symbols = collect_symbols and ctype == "operator_assignment"
                    self._walk(
                        child, content, result, class_name, owner, locals_, recurse_symbols
                    )
                left = child.child_by_field_name("left")
                if left is not None and left.type == "identifier":
                    locals_.add(left.text.decode("utf8"))

            elif ctype == "identifier" and node.type in _STMT_CONTAINERS:
                name = child.text.decode("utf8")
                if name not in locals_:
                    result.calls.append(
                        CallInfo(name=name, line=child.start_point[0] + 1)
                    )

            else:
                self._walk(
                    child, content, result, class_name, owner, locals_, collect_symbols
                )

    def _extract_method(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract an instance method (``def foo``) or top-level function.

        The body (nested definitions and calls) is descended by the unified
        ``_walk``; nested defs are treated as standalone functions.
        """
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
                # node.text is the node's exact source bytes — slicing the str
                # with byte offsets would drift on any non-ASCII content.
                content=node.text.decode("utf8"),
                signature=self._build_signature(node, name),
                class_name=class_name,
            )
        )

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
                # node.text is the node's exact source bytes — slicing the str
                # with byte offsets would drift on any non-ASCII content.
                content=node.text.decode("utf8"),
                signature=self._build_signature(node, name, receiver=receiver),
                class_name=class_name,
            )
        )
        # The body is descended by the unified _walk.

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> SymbolInfo | None:
        """Extract a class definition; return its symbol for scope threading.

        The body (members and calls) is descended by the unified ``_walk``;
        this records only the class symbol and its ``extends`` heritage.
        Returns ``None`` for an unnamed (malformed) class.
        """
        name = self._definition_name(node)
        if not name:
            return None

        symbol = SymbolInfo(
            name=name,
            kind="class",
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            # node.text is the node's exact source bytes — slicing the str
            # with byte offsets would drift on any non-ASCII content.
            content=node.text.decode("utf8"),
        )
        result.symbols.append(symbol)

        # ``class A < B`` — the superclass field holds ``< Parent``.
        superclass = node.child_by_field_name("superclass")
        if superclass is not None:
            parent = self._superclass_name(superclass)
            if parent:
                result.heritage.append((name, "extends", parent))
        return symbol

    def _extract_module(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> SymbolInfo | None:
        """Extract a module definition; return its symbol for scope threading.

        The body (members and calls) is descended by the unified ``_walk``.
        Returns ``None`` for an unnamed (malformed) module.
        """
        name = self._definition_name(node)
        if not name:
            return None

        symbol = SymbolInfo(
            name=name,
            kind="module",
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            # node.text is the node's exact source bytes — slicing the str
            # with byte offsets would drift on any non-ASCII content.
            content=node.text.decode("utf8"),
        )
        result.symbols.append(symbol)
        return symbol

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
                # node.text is the node's exact source bytes — slicing the str
                # with byte offsets would drift on any non-ASCII content.
                content=node.text.decode("utf8"),
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
    # Class macros (mixins & accessors)
    # ------------------------------------------------------------------

    def _extract_class_macro(
        self,
        node: Node,
        result: ParseResult,
        class_name: str,
        owner: SymbolInfo | None,
    ) -> None:
        """Extract mixin and accessor macros from a bare ``call`` node.

        ``include``/``extend``/``prepend M`` inside a class/module emit a
        ``(class_name, "mixin", "M")`` heritage tuple; ``attr_accessor``/
        ``attr_reader``/``attr_writer`` accessor symbols and Rails callback
        symbols (``before_action``/``after_save``/...) are recorded on the
        owning type's ``decorators`` (as ``"{macro}:{name}"``) so Task 10 can
        exempt the framework-invoked methods from dead-code analysis.  All only
        apply within a type, and only to bare calls (no receiver).
        """
        if not class_name or owner is None:
            return
        if node.child_by_field_name("receiver") is not None:
            return
        method_node = node.child_by_field_name("method")
        if method_node is None or method_node.type != "identifier":
            return
        method = method_node.text.decode("utf8")

        args_node = node.child_by_field_name("arguments")
        if args_node is None:
            return

        if method in _MIXIN_METHODS:
            for arg in args_node.children:
                if arg.type in ("constant", "scope_resolution"):
                    result.heritage.append((class_name, "mixin", self._constant_last_segment(arg)))
        elif method in _SYMBOL_RECORDING_METHODS:
            for arg in args_node.children:
                if arg.type == "simple_symbol":
                    sym = arg.text.decode("utf8").lstrip(":")
                    owner.decorators.append(f"{method}:{sym}")

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------

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
    def _superclass_name(node: Node) -> str:
        """Return the parent name from a ``superclass`` node (``< Parent``).

        The trailing constant segment is used so namespaced parents
        (``< Foo::Bar``) match the way classes/modules are named.
        """
        for child in node.children:
            if child.type in ("constant", "scope_resolution"):
                return RubyParser._constant_last_segment(child)
        return ""

    @staticmethod
    def _constant_last_segment(node: Node) -> str:
        """Return the final segment of a ``constant``/``scope_resolution`` node.

        ``Foo::Bar`` → ``Bar``; a bare ``Foo`` is returned unchanged.
        """
        return node.text.decode("utf8").rsplit("::", 1)[-1]

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
