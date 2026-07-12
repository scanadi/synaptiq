"""Tests for the dead code detection phase (Phase 10)."""

from __future__ import annotations

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from synaptiq.core.ingestion.dead_code import process_dead_code

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_file_node(graph: KnowledgeGraph, path: str) -> str:
    """Add a File node and return its ID."""
    node_id = generate_id(NodeLabel.FILE, path)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=NodeLabel.FILE,
            name=path.rsplit("/", 1)[-1],
            file_path=path,
        )
    )
    return node_id


def _add_symbol_node(
    graph: KnowledgeGraph,
    label: NodeLabel,
    file_path: str,
    name: str,
    *,
    is_entry_point: bool = False,
    is_exported: bool = False,
    class_name: str = "",
) -> str:
    """Add a symbol node and return its ID."""
    symbol_name = (
        f"{class_name}.{name}" if label == NodeLabel.METHOD and class_name else name
    )
    node_id = generate_id(label, file_path, symbol_name)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=label,
            name=name,
            file_path=file_path,
            class_name=class_name,
            is_entry_point=is_entry_point,
            is_exported=is_exported,
        )
    )
    return node_id


def _add_calls_relationship(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
) -> None:
    """Add a CALLS relationship from *source_id* to *target_id*."""
    rel_id = f"calls:{source_id}->{target_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.CALLS,
            source=source_id,
            target=target_id,
        )
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the test fixture specification.

    - Function:src/main.py:main         (entry point, no incoming calls)
    - Function:src/auth.py:validate     (has incoming calls from main)
    - Function:src/auth.py:unused_helper (no calls, not entry point) -> DEAD
    - Method:src/models.py:User.__init__ (no calls, constructor)    -> NOT dead
    - Function:src/tests/test_auth.py:test_validate (test function) -> NOT dead
    - Function:src/utils.py:orphan_function (no calls, not entry)   -> DEAD
    """
    g = KnowledgeGraph()

    # Files
    _add_file_node(g, "src/main.py")
    _add_file_node(g, "src/auth.py")
    _add_file_node(g, "src/models.py")
    _add_file_node(g, "src/tests/test_auth.py")
    _add_file_node(g, "src/utils.py")

    # Symbols
    main_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/main.py", "main", is_entry_point=True
    )
    validate_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/auth.py", "validate"
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/auth.py", "unused_helper"
    )
    _add_symbol_node(
        g,
        NodeLabel.METHOD,
        "src/models.py",
        "__init__",
        class_name="User",
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/tests/test_auth.py", "test_validate"
    )
    _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/utils.py", "orphan_function"
    )

    # CALLS: main -> validate
    _add_calls_relationship(g, main_id, validate_id)

    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectsUnusedFunction:
    """Unused helper functions with no incoming calls are flagged as dead."""

    def test_detects_unused_function(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        unused_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "unused_helper"
        )
        node = graph.get_node(unused_id)
        assert node is not None
        assert node.is_dead is True


class TestSkipsEntryPoints:
    """Entry points are never flagged as dead, even without incoming calls."""

    def test_skips_entry_points(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        main_id = generate_id(NodeLabel.FUNCTION, "src/main.py", "main")
        node = graph.get_node(main_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsCalledFunctions:
    """Functions with incoming CALLS relationships are not flagged."""

    def test_skips_called_functions(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        validate_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        node = graph.get_node(validate_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsConstructors:
    """__init__ and __new__ methods are never flagged as dead."""

    def test_skips_constructors(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        init_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__init__"
        )
        node = graph.get_node(init_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsTestFunctions:
    """Test functions (test_*) are never flagged as dead."""

    def test_skips_test_functions(self, graph: KnowledgeGraph) -> None:
        process_dead_code(graph)

        test_id = generate_id(
            NodeLabel.FUNCTION, "src/tests/test_auth.py", "test_validate"
        )
        node = graph.get_node(test_id)
        assert node is not None
        assert node.is_dead is False


class TestSkipsDunderMethods:
    """Dunder methods (__str__, __repr__, etc.) are never flagged as dead."""

    def test_skips_dunder_methods(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "__str__",
            class_name="User",
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.py",
            "__repr__",
            class_name="User",
        )

        process_dead_code(g)

        str_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__str__"
        )
        repr_id = generate_id(
            NodeLabel.METHOD, "src/models.py", "User.__repr__"
        )

        str_node = g.get_node(str_id)
        repr_node = g.get_node(repr_id)

        assert str_node is not None
        assert str_node.is_dead is False

        assert repr_node is not None
        assert repr_node.is_dead is False


class TestReturnsCount:
    """process_dead_code returns the correct count of dead symbols."""

    def test_returns_count(self, graph: KnowledgeGraph) -> None:
        count = process_dead_code(graph)

        # unused_helper and orphan_function are the two dead symbols.
        assert count == 2


class TestEmptyGraph:
    """An empty graph produces zero dead symbols."""

    def test_empty_graph(self) -> None:
        g = KnowledgeGraph()
        count = process_dead_code(g)
        assert count == 0


# ---------------------------------------------------------------------------
# Helpers for USES_TYPE and EXTENDS relationships
# ---------------------------------------------------------------------------


def _add_uses_type_relationship(
    graph: KnowledgeGraph,
    source_id: str,
    target_id: str,
) -> None:
    """Add a USES_TYPE relationship from *source_id* to *target_id*."""
    rel_id = f"uses_type:{source_id}->{target_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.USES_TYPE,
            source=source_id,
            target=target_id,
        )
    )


# ---------------------------------------------------------------------------
# USES_TYPE tests
# ---------------------------------------------------------------------------


class TestSkipsTypeReferencedClasses:
    """Classes with incoming USES_TYPE edges are not flagged as dead."""

    def test_class_with_uses_type_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_file_node(g, "src/handler.py")

        class_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/models.py", "Status"
        )
        func_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/handler.py", "handle",
            is_entry_point=True,
        )
        _add_uses_type_relationship(g, func_id, class_id)

        process_dead_code(g)

        node = g.get_node(class_id)
        assert node is not None
        assert node.is_dead is False

    def test_function_with_only_uses_type_still_dead(self) -> None:
        """Functions referenced only as types ARE dead (not classes)."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.py")
        _add_file_node(g, "src/handler.py")

        func_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.py", "unused_func"
        )
        other_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/handler.py", "handle",
            is_entry_point=True,
        )
        _add_uses_type_relationship(g, other_id, func_id)

        process_dead_code(g)

        node = g.get_node(func_id)
        assert node is not None
        assert node.is_dead is True


# ---------------------------------------------------------------------------
# Framework decorator tests
# ---------------------------------------------------------------------------


class TestSkipsFrameworkDecoratedFunctions:
    """Functions with framework-registration decorators are not flagged dead."""

    def test_framework_decorated_function_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/server.py")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/server.py", "list_tools"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["decorators"] = ["server.list_tools"]

        process_dead_code(g)

        assert node.is_dead is False

    def test_simple_decorator_still_dead(self) -> None:
        """Decorators without dots (e.g., @staticmethod) do not exempt."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.py")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.py", "unused"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["decorators"] = ["staticmethod"]

        process_dead_code(g)

        assert node.is_dead is True

    def test_typing_overload_decorator_exempts(self) -> None:
        """@typing.overload stubs are not dead — they define type signatures."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.py")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.py", "overloaded"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["decorators"] = ["typing.overload"]

        process_dead_code(g)

        assert node.is_dead is False

    def test_functools_wraps_still_dead(self) -> None:
        """Known non-framework dotted decorators (functools.wraps) don't exempt."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.py")
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.py", "wrapper"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["decorators"] = ["functools.wraps"]

        process_dead_code(g)

        assert node.is_dead is True


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Methods on classes structurally conforming to a Protocol are not dead."""

    def test_conforming_class_methods_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/base.py")
        _add_file_node(g, "src/impl.py")
        _add_file_node(g, "src/main.py")

        # Protocol class with is_protocol annotation
        proto_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/base.py", "StorageBackend"
        )
        proto_node = g.get_node(proto_id)
        assert proto_node is not None
        proto_node.properties["is_protocol"] = True

        # Protocol methods
        proto_init_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/base.py", "initialize",
            class_name="StorageBackend",
        )
        proto_close_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/base.py", "close",
            class_name="StorageBackend",
        )

        # Concrete class structurally conforming (has both methods)
        _add_symbol_node(
            g, NodeLabel.CLASS, "src/impl.py", "LadybugBackend"
        )
        impl_init_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/impl.py", "initialize",
            class_name="LadybugBackend",
        )
        impl_close_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/impl.py", "close",
            class_name="LadybugBackend",
        )

        # A caller calls StorageBackend.initialize (not LadybugBackend)
        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/main.py", "main",
            is_entry_point=True,
        )
        _add_calls_relationship(g, caller_id, proto_init_id)
        _add_calls_relationship(g, caller_id, proto_close_id)

        process_dead_code(g)

        # Protocol methods are alive (have incoming CALLS)
        assert g.get_node(proto_init_id).is_dead is False
        assert g.get_node(proto_close_id).is_dead is False

        # Concrete methods should be un-flagged by Protocol conformance
        assert g.get_node(impl_init_id).is_dead is False
        assert g.get_node(impl_close_id).is_dead is False

    def test_non_conforming_class_still_dead(self) -> None:
        """A class with only some protocol methods is still flagged dead."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/base.py")
        _add_file_node(g, "src/partial.py")

        proto_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/base.py", "Backend"
        )
        proto_node = g.get_node(proto_id)
        assert proto_node is not None
        proto_node.properties["is_protocol"] = True

        _add_symbol_node(
            g, NodeLabel.METHOD, "src/base.py", "initialize",
            class_name="Backend",
        )
        _add_symbol_node(
            g, NodeLabel.METHOD, "src/base.py", "close",
            class_name="Backend",
        )

        # Partial class only has "initialize", not "close"
        _add_symbol_node(
            g, NodeLabel.CLASS, "src/partial.py", "Partial"
        )
        partial_method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/partial.py", "initialize",
            class_name="Partial",
        )

        process_dead_code(g)

        partial_method = g.get_node(partial_method_id)
        assert partial_method is not None
        assert partial_method.is_dead is True


# ---------------------------------------------------------------------------
# TypeScript constructor tests
# ---------------------------------------------------------------------------


class TestSkipsTypeScriptConstructors:
    """TypeScript ``constructor`` methods are never flagged as dead."""

    def test_ts_constructor_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.ts")
        node_id = _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/models.ts",
            "constructor",
            class_name="MyClass",
        )

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is False


# ---------------------------------------------------------------------------
# Test file suffix tests (Jest/Vitest/Storybook conventions)
# ---------------------------------------------------------------------------


class TestSkipsTestFileSuffixes:
    """Symbols in files with .test.ts, .spec.tsx, .stories.tsx suffixes are NOT flagged."""

    @pytest.mark.parametrize(
        "file_path",
        [
            "src/components/Button.test.ts",
            "src/components/Button.test.tsx",
            "src/components/Button.spec.ts",
            "src/components/Button.spec.tsx",
            "src/components/Button.stories.tsx",
            "src/utils/helpers.test.js",
            "src/utils/helpers.spec.jsx",
        ],
    )
    def test_test_suffix_files_not_dead(self, file_path: str) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, file_path)
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, file_path, "someHelper"
        )

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is False


# ---------------------------------------------------------------------------
# __tests__, __mocks__, __fixtures__ directory tests
# ---------------------------------------------------------------------------


class TestSkipsMockAndFixtureDirectories:
    """Symbols in __tests__/, __mocks__/, __fixtures__/ paths are NOT flagged."""

    @pytest.mark.parametrize(
        "file_path",
        [
            "src/__tests__/Button.tsx",
            "src/components/__tests__/Form.test.ts",
            "src/__mocks__/api.ts",
            "src/__fixtures__/data.ts",
        ],
    )
    def test_test_directories_not_dead(self, file_path: str) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, file_path)
        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, file_path, "mockHelper"
        )

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is False


# ---------------------------------------------------------------------------
# Alive class method tests
# ---------------------------------------------------------------------------


class TestSkipsAliveClassMethods:
    """Non-private methods on alive classes are not flagged dead."""

    def test_method_on_alive_class_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/pool.ts")
        _add_file_node(g, "src/main.ts")

        class_id = _add_symbol_node(g, NodeLabel.CLASS, "src/pool.ts", "Pool")
        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/main.ts", "main", is_entry_point=True
        )
        _add_calls_relationship(g, caller_id, class_id)

        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/pool.ts", "acquire", class_name="Pool"
        )

        process_dead_code(g)

        node = g.get_node(method_id)
        assert node is not None
        assert node.is_dead is False

    def test_method_on_dead_class_still_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/unused.ts")

        _add_symbol_node(g, NodeLabel.CLASS, "src/unused.ts", "Unused")
        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/unused.ts", "doStuff", class_name="Unused"
        )

        process_dead_code(g)

        node = g.get_node(method_id)
        assert node is not None
        assert node.is_dead is True

    def test_private_method_on_alive_class_still_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/pool.ts")
        _add_file_node(g, "src/main.ts")

        class_id = _add_symbol_node(g, NodeLabel.CLASS, "src/pool.ts", "Pool")
        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/main.ts", "main", is_entry_point=True
        )
        _add_calls_relationship(g, caller_id, class_id)

        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/pool.ts", "_internalHelper", class_name="Pool"
        )

        process_dead_code(g)

        node = g.get_node(method_id)
        assert node is not None
        assert node.is_dead is True


# ---------------------------------------------------------------------------
# Inner function tests
# ---------------------------------------------------------------------------


class TestSkipsInnerFunctions:
    """Inner functions inside alive parent functions are not flagged dead."""

    def test_inner_function_in_alive_parent_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/component.tsx")
        _add_file_node(g, "src/app.tsx")

        parent_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/component.tsx", "MyComponent"
        )
        parent_node = g.get_node(parent_id)
        assert parent_node is not None
        parent_node.start_line = 1
        parent_node.end_line = 50

        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.tsx", "App", is_entry_point=True
        )
        _add_calls_relationship(g, caller_id, parent_id)

        inner_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/component.tsx", "handleClick"
        )
        inner_node = g.get_node(inner_id)
        assert inner_node is not None
        inner_node.start_line = 10
        inner_node.end_line = 15

        process_dead_code(g)

        node = g.get_node(inner_id)
        assert node is not None
        assert node.is_dead is False

    def test_top_level_function_still_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/utils.ts")

        node_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/utils.ts", "unusedUtil"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.start_line = 1
        node.end_line = 10

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is True


# ---------------------------------------------------------------------------
# Config file hook tests
# ---------------------------------------------------------------------------


class TestSkipsConfigFileHooks:
    """Vite/config plugin hooks are not flagged dead."""

    @pytest.mark.parametrize(
        "name,file_path",
        [
            ("resolveId", "apps/web/vite.config.ts"),
            ("closeBundle", "apps/server/vite.config.ts"),
            ("manualChunks", "apps/web/vite.config.ts"),
            ("configure", "apps/web/server/index.ts"),
            ("onShellError", "apps/web/pages/entry.server.tsx"),
        ],
    )
    def test_config_hooks_not_dead(self, name: str, file_path: str) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, file_path)
        node_id = _add_symbol_node(g, NodeLabel.FUNCTION, file_path, name)

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is False


# ---------------------------------------------------------------------------
# Framework entry file tests
# ---------------------------------------------------------------------------


class TestSkipsFrameworkEntryFiles:
    """All symbols in entry.server.tsx / entry.client.tsx are exempt."""

    def test_entry_server_symbols_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "apps/web/pages/entry.server.tsx")
        node_id = _add_symbol_node(
            g, NodeLabel.METHOD, "apps/web/pages/entry.server.tsx", "[readyOption]"
        )

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is False


# ---------------------------------------------------------------------------
# Framework model base class tests
# ---------------------------------------------------------------------------


class TestSkipsFrameworkModelClasses:
    """Classes extending framework model bases (BaseModel, Model, etc.) are exempt."""

    @pytest.mark.parametrize(
        "base_name",
        ["BaseModel", "BaseSettings", "Model", "Manager", "Base", "DeclarativeBase",
         "BaseEntity"],
    )
    def test_framework_model_class_not_dead(self, base_name: str) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        node_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/models.py", "MyModel"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["bases"] = [base_name]

        process_dead_code(g)

        assert node.is_dead is False

    def test_class_without_framework_base_still_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        node_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/models.py", "OrphanClass"
        )

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is True

    @pytest.mark.parametrize(
        "base_name",
        ["TestCase"],
    )
    def test_additional_framework_bases_not_dead(self, base_name: str) -> None:
        """Classes extending unittest.TestCase and similar bases are not flagged dead."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/tests.py")
        node_id = _add_symbol_node(
            g, NodeLabel.CLASS, "src/tests.py", "MyTestCase"
        )
        node = g.get_node(node_id)
        assert node is not None
        node.properties["bases"] = [base_name]

        process_dead_code(g)

        assert node.is_dead is False


# ---------------------------------------------------------------------------
# Object-literal method alive pass tests
# ---------------------------------------------------------------------------


class TestSkipsAliveObjectLiteralMethods:
    """Methods on object literals referenced by alive code are not flagged dead."""

    def test_object_literal_method_referenced_by_alive_function(self) -> None:
        """const visitors = { enter() {} } — if alive code references 'visitors', un-flag."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/plugin.ts")
        _add_file_node(g, "src/main.ts")

        # An alive function that references "visitors" in its content
        alive_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/plugin.ts", "createPlugin",
            is_exported=True,
        )
        alive_node = g.get_node(alive_id)
        assert alive_node is not None
        alive_node.start_line = 1
        alive_node.end_line = 20
        alive_node.content = (
            "function createPlugin() {\n"
            "  const visitors = { enter() {}, exit() {} };\n"
            "  return visitors;\n"
            "}"
        )

        # Methods on the object literal "visitors"
        enter_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/plugin.ts", "enter",
            class_name="visitors",
        )
        exit_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/plugin.ts", "exit",
            class_name="visitors",
        )

        process_dead_code(g)

        enter_node = g.get_node(enter_id)
        exit_node = g.get_node(exit_id)
        assert enter_node is not None
        assert exit_node is not None
        assert enter_node.is_dead is False
        assert exit_node.is_dead is False

    def test_object_literal_method_no_reference_still_dead(self) -> None:
        """Methods on unreferenced object literals remain dead."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/unused.ts")

        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/unused.ts", "doStuff",
            class_name="orphanObj",
        )

        process_dead_code(g)

        method_node = g.get_node(method_id)
        assert method_node is not None
        assert method_node.is_dead is True


# ---------------------------------------------------------------------------
# Ruby exemption tests (Task 10)
# ---------------------------------------------------------------------------


class TestRubyConstructorExempt:
    """Ruby ``initialize`` is treated as a constructor and never flagged dead."""

    def test_initialize_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/models/user.rb")
        node_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/models/user.rb", "initialize",
            class_name="User",
        )

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is False


class TestRubyMetaprogrammingHooksExempt:
    """method_missing / respond_to_missing? and friends are never flagged dead."""

    @pytest.mark.parametrize(
        "name",
        ["method_missing", "respond_to_missing?", "const_missing", "inherited"],
    )
    def test_metaprogramming_hook_not_dead(self, name: str) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/models/proxy.rb")
        node_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/models/proxy.rb", name, class_name="Proxy"
        )

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is False


class TestRubyTestFilesExempt:
    """Symbols in *_spec.rb / *_test.rb files and spec/ dirs are not flagged."""

    @pytest.mark.parametrize(
        "file_path",
        [
            "spec/models/user_spec.rb",
            "test/models/user_test.rb",
            "app/services/foo_spec.rb",
            "spec/support/helper.rb",
        ],
    )
    def test_ruby_test_file_symbol_not_dead(self, file_path: str) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, file_path)
        node_id = _add_symbol_node(g, NodeLabel.FUNCTION, file_path, "build_widget")

        process_dead_code(g)

        node = g.get_node(node_id)
        assert node is not None
        assert node.is_dead is False


class TestRubyAttrAccessorExempt:
    """Methods generated/named by attr_* macros (recorded on the class) are exempt."""

    def test_attr_accessor_method_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/models/user.rb")
        class_id = _add_symbol_node(g, NodeLabel.CLASS, "app/models/user.rb", "User")
        cls = g.get_node(class_id)
        assert cls is not None
        cls.properties["decorators"] = ["attr_reader:name", "attr_accessor:email"]

        name_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/models/user.rb", "name", class_name="User"
        )
        email_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/models/user.rb", "email", class_name="User"
        )

        process_dead_code(g)

        assert g.get_node(name_id).is_dead is False
        assert g.get_node(email_id).is_dead is False

    def test_attr_macro_on_module_method_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/models/concerns/named.rb")
        mod_id = _add_symbol_node(
            g, NodeLabel.MODULE, "app/models/concerns/named.rb", "Named"
        )
        mod = g.get_node(mod_id)
        assert mod is not None
        mod.properties["decorators"] = ["attr_writer:label"]

        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/models/concerns/named.rb", "label",
            class_name="Named",
        )

        process_dead_code(g)

        assert g.get_node(method_id).is_dead is False


class TestRubyCallbackTargetsExempt:
    """Methods registered as Rails callbacks (before_action, etc.) are exempt."""

    def test_callback_target_method_not_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/controllers/users_controller.rb")
        class_id = _add_symbol_node(
            g, NodeLabel.CLASS, "app/controllers/users_controller.rb",
            "UsersController",
        )
        cls = g.get_node(class_id)
        assert cls is not None
        cls.properties["decorators"] = [
            "before_action:authenticate_user",
            "after_save:notify",
        ]

        auth_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/controllers/users_controller.rb",
            "authenticate_user", class_name="UsersController",
        )
        notify_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/controllers/users_controller.rb",
            "notify", class_name="UsersController",
        )

        process_dead_code(g)

        assert g.get_node(auth_id).is_dead is False
        assert g.get_node(notify_id).is_dead is False


class TestRubyRailsModelBasesExempt:
    """Classes extending Rails base classes are not flagged dead."""

    @pytest.mark.parametrize(
        "base_name",
        ["ApplicationRecord", "ApplicationController", "ApplicationJob",
         "ApplicationMailer", "Base"],
    )
    def test_rails_base_class_not_dead(self, base_name: str) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/models/widget.rb")
        node_id = _add_symbol_node(g, NodeLabel.CLASS, "app/models/widget.rb", "Widget")
        node = g.get_node(node_id)
        assert node is not None
        node.properties["bases"] = [base_name]

        process_dead_code(g)

        assert node.is_dead is False


class TestRubyUnusedPrivateMethodStillDead:
    """A genuinely unused private Ruby method (no macro, no caller) is flagged dead."""

    def test_unused_private_method_is_dead(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/services/report.rb")
        # An alive class so the alive-class pass could otherwise un-flag publics,
        # but a leading-underscore method must remain dead.
        class_id = _add_symbol_node(
            g, NodeLabel.CLASS, "app/services/report.rb", "Report"
        )
        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "app/main.rb", "run", is_entry_point=True
        )
        _add_file_node(g, "app/main.rb")
        _add_calls_relationship(g, caller_id, class_id)

        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/services/report.rb", "_compute_totals",
            class_name="Report",
        )

        process_dead_code(g)

        node = g.get_node(method_id)
        assert node is not None
        assert node.is_dead is True


def test_ruby_macro_prefixes_stay_in_sync_with_parser():
    """The dead-code exemption list must match the parser's recorded macros.

    ``dead_code._RUBY_MACRO_PREFIXES`` is intentionally duplicated from
    ``ruby_lang._SYMBOL_RECORDING_METHODS`` (to avoid importing the tree-sitter
    parser into every dead-code run).  If the parser starts recording a new
    macro and this list is not updated, those methods would be wrongly flagged
    dead.  This test fails loudly on that drift.
    """
    from synaptiq.core.ingestion.dead_code import _RUBY_MACRO_PREFIXES
    from synaptiq.core.parsers.ruby_lang import _SYMBOL_RECORDING_METHODS

    assert set(_RUBY_MACRO_PREFIXES) == set(_SYMBOL_RECORDING_METHODS)
