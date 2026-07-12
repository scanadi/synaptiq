"""Regression tests for bugs found in the 2026-06 code review.

Each test pins a verified bug fix:

- Python parser sliced str content with byte offsets (non-ASCII drift).
- JS/TS parent-relative imports (``../``) never resolved.
- Same-named symbols in one file silently replaced each other.
- Containment lookup used a fixed window that missed wide enclosing symbols.
- USES_TYPE edges with different roles collapsed during bulk load.
- Vector search returned ghost results for deleted nodes; embeddings
  were never removed with their file.
- The watcher's global phase wiped embeddings without re-storing them.
- Concurrent MemoryStore writers could truncate each other's temp file.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from synaptiq.core.ingestion.imports import resolve_import_path
from synaptiq.core.ingestion.parser_phase import process_parsing
from synaptiq.core.ingestion.pipeline import commit_full_index
from synaptiq.core.ingestion.symbol_lookup import (
    build_file_symbol_index,
    find_containing_symbol,
)
from synaptiq.core.ingestion.walker import FileEntry
from synaptiq.core.memory import MemoryStore
from synaptiq.core.parsers.base import ImportInfo
from synaptiq.core.parsers.python_lang import PythonParser
from synaptiq.core.storage.base import NodeEmbedding
from synaptiq.core.storage.ladybug_backend import LadybugBackend


class TestPythonParserNonAscii:
    def test_content_not_shifted_by_multibyte_chars(self) -> None:
        src = '# comment with émojis 🎉🎉\ndef target_function():\n    return 42\n'
        result = PythonParser().parse(src, "t.py")
        sym = result.symbols[0]
        assert sym.name == "target_function"
        assert sym.content == "def target_function():\n    return 42"

    def test_class_content_not_shifted(self) -> None:
        src = '"""Üñïçödé docstring."""\n\nclass Config:\n    pass\n'
        result = PythonParser().parse(src, "t.py")
        cls = next(s for s in result.symbols if s.kind == "class")
        assert cls.content == "class Config:\n    pass"


class TestParentRelativeImports:
    FILE_INDEX = {
        "src/models/user.ts": "file:src/models/user.ts:",
        "src/components/Button.tsx": "file:src/components/Button.tsx:",
        "src/utils/index.mjs": "file:src/utils/index.mjs:",
    }

    def test_dotdot_import_resolves(self) -> None:
        imp = ImportInfo(module="../models/user", names=["User"], is_relative=True)
        result = resolve_import_path("src/components/Button.tsx", imp, self.FILE_INDEX)
        assert result == "file:src/models/user.ts:"

    def test_double_dotdot_import_resolves(self) -> None:
        index = {"lib/core.ts": "file:lib/core.ts:"}
        imp = ImportInfo(module="../../lib/core", names=["x"], is_relative=True)
        result = resolve_import_path("src/components/Button.tsx", imp, index)
        assert result == "file:lib/core.ts:"

    def test_mjs_directory_index_resolves(self) -> None:
        imp = ImportInfo(module="../utils", names=["helper"], is_relative=True)
        result = resolve_import_path("src/components/Button.tsx", imp, self.FILE_INDEX)
        assert result == "file:src/utils/index.mjs:"


class TestSymbolIdCollisions:
    def test_same_named_nested_functions_both_kept(self) -> None:
        src = (
            "def outer_a():\n"
            "    def helper():\n"
            "        pass\n"
            "\n"
            "def outer_b():\n"
            "    def helper():\n"
            "        pass\n"
        )
        graph = KnowledgeGraph()
        files = [FileEntry(path="m.py", content=src, language="python")]
        graph.add_node(GraphNode(
            id="file:m.py:", label=NodeLabel.FILE, name="m.py", file_path="m.py",
        ))
        process_parsing(files, graph)
        helpers = [
            n for n in graph.get_nodes_by_label(NodeLabel.FUNCTION)
            if n.name == "helper"
        ]
        assert len(helpers) == 2
        assert len({n.id for n in helpers}) == 2


class TestContainmentLookup:
    def test_wide_enclosing_symbol_found_despite_many_siblings(self) -> None:
        """A line between methods deep inside a class must resolve to the
        class even when dozens of narrow siblings sit between them in the
        index (the old fixed ±10 window missed this)."""
        graph = KnowledgeGraph()
        graph.add_node(GraphNode(
            id="class:big.py:Big", label=NodeLabel.CLASS, name="Big",
            file_path="big.py", start_line=1, end_line=500,
        ))
        for i in range(40):
            start = 2 + i * 3
            graph.add_node(GraphNode(
                id=f"method:big.py:Big.m{i}", label=NodeLabel.METHOD, name=f"m{i}",
                file_path="big.py", start_line=start, end_line=start + 1,
                class_name="Big",
            ))
        index = build_file_symbol_index(
            graph, (NodeLabel.CLASS, NodeLabel.METHOD)
        )
        # Line 130 is inside Big but past every method (last ends at 120).
        found = find_containing_symbol(130, "big.py", index)
        assert found == "class:big.py:Big"


@pytest.fixture(scope="module")
def kuzu(tmp_path_factory: pytest.TempPathFactory) -> LadybugBackend:
    # Module-scoped: every test below starts with bulk_load, which resets
    # the database anyway — per-test schema creation would only add ~3s each.
    backend = LadybugBackend()
    backend.initialize(tmp_path_factory.mktemp("regressions") / "kuzu")
    yield backend
    backend.close()


def _two_role_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node(GraphNode(
        id="function:a.py:f", label=NodeLabel.FUNCTION, name="f",
        file_path="a.py", start_line=1, end_line=2,
    ))
    g.add_node(GraphNode(
        id="class:a.py:T", label=NodeLabel.CLASS, name="T",
        file_path="a.py", start_line=4, end_line=6,
    ))
    g.add_relationship(GraphRelationship(
        id="uses_type:function:a.py:f->class:a.py:T:param",
        type=RelType.USES_TYPE, source="function:a.py:f", target="class:a.py:T",
        properties={"role": "param"},
    ))
    g.add_relationship(GraphRelationship(
        id="uses_type:function:a.py:f->class:a.py:T:return",
        type=RelType.USES_TYPE, source="function:a.py:f", target="class:a.py:T",
        properties={"role": "return"},
    ))
    return g


class TestKuzuRegressionFixes:
    def test_uses_type_roles_both_persisted(self, kuzu: LadybugBackend) -> None:
        kuzu.bulk_load(_two_role_graph())
        rows = kuzu.execute_raw(
            "MATCH (a)-[r:CodeRelation]->(b) "
            "WHERE r.rel_type = 'uses_type' RETURN r.role ORDER BY r.role"
        )
        assert [r[0] for r in rows] == ["param", "return"]

    def test_node_properties_round_trip(self, kuzu: LadybugBackend) -> None:
        g = KnowledgeGraph()
        g.add_node(GraphNode(
            id="community:community_0:", label=NodeLabel.COMMUNITY, name="Core",
            properties={"cohesion": 0.5, "symbol_count": 3},
        ))
        kuzu.bulk_load(g)
        node = kuzu.get_node("community:community_0:")
        assert node is not None
        assert node.properties.get("cohesion") == 0.5
        assert node.properties.get("symbol_count") == 3

    def test_vector_search_skips_orphaned_embeddings(self, kuzu: LadybugBackend) -> None:
        g = KnowledgeGraph()
        g.add_node(GraphNode(
            id="function:a.py:f", label=NodeLabel.FUNCTION, name="f",
            file_path="a.py", start_line=1, end_line=2,
        ))
        kuzu.bulk_load(g)
        kuzu.store_embeddings([
            NodeEmbedding(node_id="function:a.py:f", embedding=[1.0, 0.0]),
            NodeEmbedding(node_id="function:gone.py:ghost", embedding=[0.9, 0.1]),
        ])
        results = kuzu.vector_search([1.0, 0.0], limit=10)
        ids = [r.node_id for r in results]
        assert "function:a.py:f" in ids
        assert "function:gone.py:ghost" not in ids

    def test_remove_nodes_by_file_removes_embeddings(self, kuzu: LadybugBackend) -> None:
        g = KnowledgeGraph()
        g.add_node(GraphNode(
            id="function:a.py:f", label=NodeLabel.FUNCTION, name="f",
            file_path="a.py", start_line=1, end_line=2,
        ))
        kuzu.bulk_load(g)
        kuzu.store_embeddings([
            NodeEmbedding(node_id="function:a.py:f", embedding=[1.0, 0.0]),
        ])
        kuzu.remove_nodes_by_file("a.py")
        rows = kuzu.execute_raw("MATCH (e:Embedding) RETURN count(e)")
        assert rows[0][0] == 0

    def test_global_phase_commit_preserves_embeddings(self, kuzu: LadybugBackend) -> None:
        """bulk_load resets the whole DB — the shared commit step must
        re-store embeddings or vector search silently dies in watch mode."""
        g = _two_role_graph()
        embeddings = [NodeEmbedding(node_id="function:a.py:f", embedding=[0.5, 0.5])]
        commit_full_index(kuzu, g, embeddings)
        results = kuzu.vector_search([0.5, 0.5], limit=5)
        assert [r.node_id for r in results] == ["function:a.py:f"]


class TestCypherGuardLiterals:
    def test_keywords_inside_string_literals_allowed(self) -> None:
        from synaptiq.core.cypher_guard import check_read_only

        for query in (
            "MATCH (f:File) WHERE f.content CONTAINS 'import os' RETURN f.file_path",
            'MATCH (n) WHERE n.content CONTAINS "use strict" RETURN n.name',
            "MATCH (n) WHERE n.name = 'begin' RETURN n",
            "MATCH (n) WHERE n.content CONTAINS 'export default' RETURN n.name",
        ):
            assert check_read_only(query) is None, query

    def test_keywords_outside_literals_still_rejected(self) -> None:
        from synaptiq.core.cypher_guard import check_read_only

        for query in (
            "MATCH (n) SET n.name = 'x' RETURN n",
            "MATCH (n) WITH n CALL something() RETURN n",
            "EXPORT DATABASE '/tmp/x'",
        ):
            assert check_read_only(query) is not None, query


class TestSchemaMigration:
    def test_old_database_gains_properties_json_column(self, tmp_path: Path) -> None:
        """A database created before the properties_json column must be
        migrated on open — without it every node SELECT binder-errors and
        is swallowed, so all lookups silently return nothing."""
        import ladybug as engine

        db_path = tmp_path / "kuzu"
        # Simulate a pre-upgrade database: old 12-column schema, one row.
        old_props = (
            "id STRING, name STRING, file_path STRING, start_line INT64, "
            "end_line INT64, content STRING, signature STRING, language STRING, "
            "class_name STRING, is_dead BOOL, is_entry_point BOOL, "
            "is_exported BOOL, PRIMARY KEY (id)"
        )
        db = engine.Database(str(db_path))
        conn = engine.Connection(db)
        conn.execute(f"CREATE NODE TABLE Function({old_props})")
        conn.execute(
            "CREATE (:Function {id: 'function:a.py:f', name: 'f', "
            "file_path: 'a.py', start_line: 1, end_line: 2})"
        )
        conn.close()
        db.close()

        backend = LadybugBackend()
        backend.initialize(db_path)
        try:
            node = backend.get_node("function:a.py:f")
            assert node is not None, "pre-upgrade row must remain readable"
            assert node.name == "f"
            assert node.properties == {}
        finally:
            backend.close()


class TestDecoratorEdgeAttribution:
    def test_decorator_edge_attaches_to_suffixed_duplicate(self) -> None:
        """Decorator CALLS edges must use the same #L collision suffix as
        the parser phase, or they attach to the wrong same-named symbol."""
        from synaptiq.core.ingestion.calls import process_calls
        from synaptiq.core.ingestion.parser_phase import process_parsing

        src = (
            "def my_decorator(fn):\n"
            "    return fn\n"
            "\n"
            "def outer_a():\n"
            "    def helper():\n"
            "        pass\n"
            "\n"
            "def outer_b():\n"
            "    @my_decorator\n"
            "    def helper():\n"
            "        pass\n"
        )
        graph = KnowledgeGraph()
        graph.add_node(GraphNode(
            id="file:m.py:", label=NodeLabel.FILE, name="m.py", file_path="m.py",
        ))
        files = [FileEntry(path="m.py", content=src, language="python")]
        parse_data = process_parsing(files, graph)
        process_calls(parse_data, graph)

        decorator_edges = [
            r for r in graph.get_relationships_by_type(RelType.CALLS)
            if r.target == "function:m.py:my_decorator" and "helper" in r.source
        ]
        assert len(decorator_edges) == 1
        # The decorated helper is the second (suffixed) one.
        assert "#L" in decorator_edges[0].source


class TestWeakRefEdges:
    def test_shorthand_fuzzy_resolution_keeps_low_confidence_edge(self) -> None:
        """Weak refs resolved by global fuzzy matching keep a down-weighted
        edge — dropping it entirely made dead-code flag live symbols."""
        from synaptiq.core.ingestion.calls import process_calls
        from synaptiq.core.ingestion.parser_phase import process_parsing

        handlers_src = "export function onSave() {\n  return 1;\n}\n"
        usage_src = "function register() {\n  return { onSave };\n}\nregister();\n"

        graph = KnowledgeGraph()
        for path in ("src/handlers.ts", "src/app.ts"):
            graph.add_node(GraphNode(
                id=f"file:{path}:", label=NodeLabel.FILE, name=path, file_path=path,
            ))
        files = [
            FileEntry(path="src/handlers.ts", content=handlers_src, language="typescript"),
            FileEntry(path="src/app.ts", content=usage_src, language="typescript"),
        ]
        parse_data = process_parsing(files, graph)
        process_calls(parse_data, graph)

        edges = [
            r for r in graph.get_relationships_by_type(RelType.CALLS)
            if r.target == "function:src/handlers.ts:onSave"
        ]
        assert edges, "weak shorthand reference must still produce a CALLS edge"
        assert all(r.properties.get("confidence", 1.0) <= 0.5 for r in edges)


class TestWatchFilter:
    def test_git_and_synaptiq_paths_excluded(self) -> None:
        """The watch filter must KEEP watchfiles' default exclusions (.git,
        __pycache__, ...) — replacing them made every git command trigger
        reindex work — and add .synaptiq on top."""
        from watchfiles import Change

        from synaptiq.core.ingestion.watcher import _SynaptiqFilter

        f = _SynaptiqFilter()
        assert not f(Change.deleted, "/repo/.git/index.lock")
        assert not f(Change.modified, "/repo/__pycache__/m.cpython-311.pyc")
        assert not f(Change.modified, "/repo/.synaptiq/kuzu/data.kz")
        assert f(Change.modified, "/repo/src/app.py")


class TestMemoryStoreConcurrency:
    def test_concurrent_remember_loses_no_facts(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        errors: list[Exception] = []

        def write(start: int) -> None:
            try:
                for i in range(start, start + 10):
                    MemoryStore(tmp_path).remember(f"key_{i}", f"value_{i}")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(n * 10,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        facts = store.list_all()
        assert len(facts) == 40
