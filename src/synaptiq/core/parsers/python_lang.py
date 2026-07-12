"""Python language parser using tree-sitter.

Extracts functions, classes, methods, imports, calls, type annotations,
and inheritance relationships from Python source code.
"""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from synaptiq.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)

PY_LANGUAGE = Language(tspython.language())

_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "None",
        "list",
        "dict",
        "set",
        "tuple",
        "Any",
        "Optional",
        "bytes",
        "complex",
        "object",
        "type",
    }
)

class PythonParser(LanguageParser):
    """Parses Python source code using tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse Python source and return structured information."""
        tree = self._parser.parse(bytes(content, "utf8"))
        result = ParseResult()
        self._walk(tree.root_node, content, result, class_name="")
        return result

    def _walk(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Walk the tree once, extracting definitions, annotations, and calls.

        Every node is visited exactly once: the extractors never recurse back
        into ``_walk`` — only the single child loop at the bottom does (the
        TypeScript parser's structure).  Call and exception-class references
        are pulled at *every* node, so ``result.calls`` is byte-identical to
        the previous dedicated full-tree pass and every call is attributed
        exactly once.

        ``class_name`` is the enclosing class used for method attribution: it
        resets to ``""`` inside a function body (nested defs are standalone
        functions) and becomes the class name inside a class body; otherwise it
        propagates down unchanged.
        """
        ntype = node.type

        # Calls and exception-class references — collected at every node.
        if ntype == "call":
            self._extract_call(node, result)
        elif ntype == "except_clause":
            self._extract_except_clause(node, result)
        elif ntype == "raise_statement":
            self._extract_raise_statement(node, result)

        # Definitions, imports, and annotations.
        child_class_name = class_name
        if ntype == "function_definition":
            self._extract_function(node, content, result, class_name)
            # Nested functions/classes inside a function body are standalone
            # symbols, so their scope carries no class name.
            child_class_name = ""
        elif ntype == "class_definition":
            new_name = self._extract_class(node, content, result)
            if new_name:
                child_class_name = new_name
        elif ntype == "import_statement":
            self._extract_import(node, result)
        elif ntype == "import_from_statement":
            self._extract_import_from(node, result)
        elif ntype == "expression_statement":
            self._extract_annotations_from_expression(node, result)

        for child in node.children:
            self._walk(child, content, result, child_class_name)

    def _extract_function(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a function or method definition."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        # node.text is the node's exact source bytes — slicing the str with
        # byte offsets would drift on any non-ASCII content.
        node_content = node.text.decode("utf8")

        kind = "method" if class_name else "function"
        signature = self._build_signature(node, content)

        symbol = SymbolInfo(
            name=name,
            kind=kind,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            signature=signature,
            class_name=class_name,
        )
        result.symbols.append(symbol)
        decorators = self._decorators_from_parent(node)
        if decorators:
            symbol.decorators = decorators

        self._extract_param_types(node, result)

        return_type = node.child_by_field_name("return_type")
        if return_type is not None:
            type_name = self._extract_type_name(return_type)
            if type_name and type_name not in _BUILTIN_TYPES:
                result.type_refs.append(
                    TypeRef(
                        name=type_name,
                        kind="return",
                        line=return_type.start_point[0] + 1,
                    )
                )
        # The body (nested defs and calls) is visited by the unified _walk.

    def _build_signature(self, func_node: Node, content: str) -> str:
        """Build a human-readable signature string for a function."""
        name_node = func_node.child_by_field_name("name")
        params_node = func_node.child_by_field_name("parameters")
        return_type = func_node.child_by_field_name("return_type")

        if name_node is None or params_node is None:
            return ""

        name = name_node.text.decode("utf8")
        params = params_node.text.decode("utf8")
        sig = f"def {name}{params}"

        if return_type is not None:
            sig += f" -> {return_type.text.decode('utf8')}"

        return sig

    def _decorators_from_parent(self, node: Node) -> list[str]:
        """Return decorator names if *node* is the inner def of a decoration.

        Tree-sitter wraps decorated definitions in a ``decorated_definition``
        node whose children are one or more ``decorator`` nodes followed by the
        actual ``function_definition`` / ``class_definition``.  The unified
        walk reaches that inner def directly, so decorators are recovered by
        looking one level up rather than dispatching the wrapper.
        """
        parent = node.parent
        if parent is None or parent.type != "decorated_definition":
            return []
        decorators: list[str] = []
        for child in parent.children:
            if child.type == "decorator":
                dec_name = self._extract_decorator_name(child)
                if dec_name:
                    decorators.append(dec_name)
        return decorators

    def _extract_decorator_name(self, decorator_node: Node) -> str:
        """Extract the dotted name from a decorator node.

        Handles three forms::

            @staticmethod          -> "staticmethod"
            @app.route             -> "app.route"
            @server.list_tools()   -> "server.list_tools"
        """
        for child in decorator_node.children:
            if child.type == "identifier":
                return child.text.decode("utf8")
            if child.type == "attribute":
                return child.text.decode("utf8")
            if child.type == "call":
                func = child.child_by_field_name("function")
                if func is not None:
                    return func.text.decode("utf8")
        return ""

    def _extract_param_types(self, func_node: Node, result: ParseResult) -> None:
        """Extract type annotations from function parameters."""
        params_node = func_node.child_by_field_name("parameters")
        if params_node is None:
            return

        for param in params_node.children:
            if param.type == "typed_parameter":
                self._extract_typed_param(param, result)
            elif param.type == "typed_default_parameter":
                self._extract_typed_param(param, result)

    def _extract_typed_param(self, param_node: Node, result: ParseResult) -> None:
        """Extract a single typed parameter's type reference."""
        param_name = ""
        for child in param_node.children:
            if child.type == "identifier":
                param_name = child.text.decode("utf8")
                break

        type_node = param_node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._extract_type_name(type_node)
        if type_name and type_name not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="param",
                    line=type_node.start_point[0] + 1,
                    param_name=param_name,
                )
            )

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> str:
        """Extract a class definition; return its name for scope threading.

        The class body (its members and calls) is visited by the unified
        ``_walk``; this records only the class symbol and its heritage.
        Returns ``""`` for an unnamed (malformed) class.
        """
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return ""

        class_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = node.text.decode("utf8")

        symbol = SymbolInfo(
            name=class_name,
            kind="class",
            start_line=start_line,
            end_line=end_line,
            content=node_content,
        )
        result.symbols.append(symbol)
        decorators = self._decorators_from_parent(node)
        if decorators:
            symbol.decorators = decorators

        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            for child in superclasses.children:
                if not child.is_named:
                    continue
                if child.type == "identifier":
                    parent_name = child.text.decode("utf8")
                    result.heritage.append((class_name, "extends", parent_name))
                elif child.type == "attribute":
                    # e.g. class Foo(module.Base): — capture "module.Base"
                    parent_name = child.text.decode("utf8")
                    result.heritage.append((class_name, "extends", parent_name))
                elif child.type == "subscript":
                    # e.g. class Foo(Generic[T]): — capture "Generic"
                    base = child.child_by_field_name("value")
                    if base is not None:
                        parent_name = base.text.decode("utf8")
                        result.heritage.append((class_name, "extends", parent_name))
        # The class body is visited by the unified _walk.
        return class_name

    def _extract_import(self, node: Node, result: ParseResult) -> None:
        """Extract a plain ``import X`` statement."""
        # ``import_statement`` children: "import", dotted_name [, ",", dotted_name ...]
        for child in node.children:
            if child.type == "dotted_name":
                module = child.text.decode("utf8")
                # For ``import os.path`` the imported name available locally is "path"
                # (the last segment), but the module is the full dotted path.
                parts = module.split(".")
                result.imports.append(
                    ImportInfo(
                        module=module,
                        names=[parts[-1]],
                    )
                )
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node is not None:
                    module = name_node.text.decode("utf8")
                    parts = module.split(".")
                    alias = alias_node.text.decode("utf8") if alias_node else ""
                    result.imports.append(
                        ImportInfo(
                            module=module,
                            names=[parts[-1]],
                            alias=alias,
                        )
                    )

    def _extract_import_from(self, node: Node, result: ParseResult) -> None:
        """Extract a ``from X import Y`` statement."""
        module_name_node = node.child_by_field_name("module_name")
        if module_name_node is None:
            return

        is_relative = module_name_node.type == "relative_import"
        module = module_name_node.text.decode("utf8")

        names: list[str] = []
        past_import = False
        for child in node.children:
            if child.type == "import":
                past_import = True
                continue
            if not past_import:
                continue
            # Handle both bare names and parenthesized import lists.
            if child.type in ("dotted_name", "identifier"):
                names.append(child.text.decode("utf8"))
            elif child.type == "import_as_names":
                for sub in child.children:
                    if sub.type in ("dotted_name", "identifier"):
                        names.append(sub.text.decode("utf8"))
                    elif sub.type == "aliased_import":
                        name_node = sub.child_by_field_name("name")
                        if name_node:
                            names.append(name_node.text.decode("utf8"))
            elif child.type == "wildcard_import":
                names.append("*")

        result.imports.append(
            ImportInfo(
                module=module,
                names=names,
                is_relative=is_relative,
            )
        )

    def _extract_annotations_from_expression(
        self,
        node: Node,
        result: ParseResult,
    ) -> None:
        """Extract variable annotations and __all__ from an expression_statement."""
        for child in node.children:
            if child.type == "assignment":
                self._try_extract_variable_annotation(child, result)
                self._try_extract_all_exports(child, result)

    def _try_extract_variable_annotation(self, assignment_node: Node, result: ParseResult) -> None:
        """Extract a type reference from a variable annotation if present."""
        type_node = assignment_node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._extract_type_name(type_node)
        if type_name and type_name not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="variable",
                    line=type_node.start_point[0] + 1,
                )
            )

    @staticmethod
    def _try_extract_all_exports(assignment_node: Node, result: ParseResult) -> None:
        """Extract names from ``__all__ = [...]`` or ``__all__ = (...)`` assignments."""
        left = assignment_node.child_by_field_name("left")
        right = assignment_node.child_by_field_name("right")
        if left is None or right is None:
            return
        if left.type != "identifier" or left.text.decode("utf8") != "__all__":
            return
        if right.type not in ("list", "tuple"):
            return

        for child in right.children:
            if child.type == "string":
                text = child.text.decode("utf8")
                # Strip surrounding quotes (single, double, or triple).
                for quote in ('"""', "'''", '"', "'"):
                    if text.startswith(quote) and text.endswith(quote):
                        text = text[len(quote):-len(quote)]
                        break
                if text:
                    result.exports.append(text)

    def _extract_except_clause(self, node: Node, result: ParseResult) -> None:
        """Extract exception-class references from an ``except`` clause.

        ``except SomeError:`` references the exception class; the same applies
        to tuple forms and ``as`` bindings.  Descent into the clause body (for
        the calls it contains) is handled by the unified ``_walk``.
        """
        for child in node.children:
            if child.type == "identifier":
                result.calls.append(
                    CallInfo(
                        name=child.text.decode("utf8"),
                        line=child.start_point[0] + 1,
                    )
                )
            elif child.type == "tuple":
                # except (ErrorA, ErrorB): — extract each exception type.
                for elem in child.children:
                    if elem.type == "identifier":
                        result.calls.append(
                            CallInfo(
                                name=elem.text.decode("utf8"),
                                line=elem.start_point[0] + 1,
                            )
                        )
            elif child.type == "as_pattern":
                # except ErrorA as e  OR  except (ErrorA, ErrorB) as e
                for sub in child.children:
                    if sub.type == "identifier":
                        result.calls.append(
                            CallInfo(
                                name=sub.text.decode("utf8"),
                                line=sub.start_point[0] + 1,
                            )
                        )
                        break
                    if sub.type == "tuple":
                        for elem in sub.children:
                            if elem.type == "identifier":
                                result.calls.append(
                                    CallInfo(
                                        name=elem.text.decode("utf8"),
                                        line=elem.start_point[0] + 1,
                                    )
                                )
                        break

    def _extract_raise_statement(self, node: Node, result: ParseResult) -> None:
        """Extract a bare ``raise SomeError`` exception-class reference.

        ``raise SomeError(...)`` is handled as a normal call node during the
        walk; only the paren-less form needs explicit handling here.
        """
        for child in node.children:
            if child.type == "identifier":
                result.calls.append(
                    CallInfo(
                        name=child.text.decode("utf8"),
                        line=child.start_point[0] + 1,
                    )
                )

    def _extract_call(self, call_node: Node, result: ParseResult) -> None:
        """Extract a single call node into a CallInfo."""
        func_node = call_node.child_by_field_name("function")
        if func_node is None:
            # Fallback: first named child is the function.
            for child in call_node.children:
                if child.is_named:
                    func_node = child
                    break
        if func_node is None:
            return

        line = call_node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(call_node)

        if func_node.type == "identifier":
            result.calls.append(
                CallInfo(
                    name=func_node.text.decode("utf8"),
                    line=line,
                    arguments=arguments,
                )
            )
        elif func_node.type == "attribute":
            name, receiver = self._extract_attribute_call(func_node)
            result.calls.append(
                CallInfo(
                    name=name,
                    line=line,
                    receiver=receiver,
                    arguments=arguments,
                )
            )

    def _extract_attribute_call(self, attr_node: Node) -> tuple[str, str]:
        """Extract (method_name, receiver) from an attribute node.

        For chained calls like ``obj.method1().method2()``, the outer call's
        function is ``attribute(call(...), "method2")``.  We extract
        ``method2`` as the name and the first identifier in the chain as
        the receiver.
        """
        method_name = ""
        for child in reversed(attr_node.children):
            if child.type == "identifier":
                method_name = child.text.decode("utf8")
                break

        receiver = ""
        obj_node = attr_node.children[0] if attr_node.children else None
        if obj_node is not None:
            if obj_node.type == "identifier":
                receiver = obj_node.text.decode("utf8")
            elif obj_node.type == "attribute":
                # Nested attribute access like ``self.logger.info()`` — use the root.
                receiver = self._root_identifier(obj_node)
            elif obj_node.type == "call":
                # Chained call like ``get_user().save()`` — try the innermost identifier.
                receiver = self._root_identifier(obj_node)

        return method_name, receiver

    @staticmethod
    def _extract_identifier_arguments(call_node: Node) -> list[str]:
        """Extract bare identifier arguments from a call node.

        Returns names of arguments that are plain identifiers (not literals,
        calls, or attribute accesses) — these are likely callback references
        like ``map(transform, items)`` or ``Depends(get_db)``.
        """
        args_node = call_node.child_by_field_name("arguments")
        if args_node is None:
            return []

        identifiers: list[str] = []
        for child in args_node.children:
            if child.type == "identifier":
                identifiers.append(child.text.decode("utf8"))
            elif child.type == "keyword_argument":
                value_node = child.child_by_field_name("value")
                if value_node is not None and value_node.type == "identifier":
                    identifiers.append(value_node.text.decode("utf8"))
        return identifiers

    def _root_identifier(self, node: Node) -> str:
        """Walk down into the leftmost identifier of an expression."""
        current = node
        while current is not None:
            if current.type == "identifier":
                return current.text.decode("utf8")
            if current.children:
                current = current.children[0]
            else:
                break
        return ""

    @staticmethod
    def _extract_type_name(type_node: Node) -> str:
        """Extract the primary type name from a type annotation node.

        For simple types like ``User``, returns ``"User"``.
        For generic types like ``list[User]``, returns ``"list"``.
        For complex types, returns the text of the first identifier found.
        """
        if type_node.type == "type" and type_node.children:
            inner = type_node.children[0]
            if inner.type == "identifier":
                return inner.text.decode("utf8")
            if inner.type == "generic_type":
                # e.g., ``Optional[User]`` — return "Optional"
                for child in inner.children:
                    if child.type == "identifier":
                        return child.text.decode("utf8")
            # Fallback: return text of first identifier found anywhere.
            return PythonParser._find_first_identifier(inner)
        if type_node.type == "identifier":
            return type_node.text.decode("utf8")
        return PythonParser._find_first_identifier(type_node)

    @staticmethod
    def _find_first_identifier(node: Node) -> str:
        """DFS for the first identifier node."""
        if node.type == "identifier":
            return node.text.decode("utf8")
        for child in node.children:
            found = PythonParser._find_first_identifier(child)
            if found:
                return found
        return ""
