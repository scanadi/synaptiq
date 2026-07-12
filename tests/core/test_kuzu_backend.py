"""Tests for the KuzuDB storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from synaptiq.core.storage.kuzu_backend import KuzuBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend(tmp_path: Path) -> KuzuBackend:
    """Return a KuzuBackend initialised in a temporary directory."""
    db_path = tmp_path / "test_db"
    b = KuzuBackend()
    b.initialize(db_path)
    yield b
    b.close()


def _make_node(
    label: NodeLabel = NodeLabel.FUNCTION,
    file_path: str = "src/app.py",
    name: str = "my_func",
    content: str = "",
) -> GraphNode:
    """Helper to build a GraphNode with a deterministic id."""
    return GraphNode(
        id=generate_id(label, file_path, name),
        label=label,
        name=name,
        file_path=file_path,
        content=content,
    )


def _make_rel(
    source: str,
    target: str,
    rel_type: RelType = RelType.CALLS,
    rel_id: str | None = None,
) -> GraphRelationship:
    """Helper to build a GraphRelationship."""
    return GraphRelationship(
        id=rel_id or f"{rel_type.value}:{source}->{target}",
        type=rel_type,
        source=source,
        target=target,
    )


def _build_small_graph() -> KnowledgeGraph:
    """Build a small KnowledgeGraph with 2 functions and 1 CALLS relationship."""
    graph = KnowledgeGraph()

    caller = _make_node(name="caller", file_path="src/a.py")
    callee = _make_node(name="callee", file_path="src/a.py")
    graph.add_node(caller)
    graph.add_node(callee)

    rel = _make_rel(caller.id, callee.id)
    graph.add_relationship(rel)

    return graph


# ---------------------------------------------------------------------------
# Initialize and close
# ---------------------------------------------------------------------------


class TestInitializeAndClose:
    def test_initialize_creates_db(self, backend: KuzuBackend) -> None:
        """After initialize, internal handles should be set."""
        assert backend._db is not None
        assert backend._conn is not None

    def test_close_releases_handles(self, tmp_path: Path) -> None:
        b = KuzuBackend()
        b.initialize(tmp_path / "close_test")
        b.close()
        assert b._db is None
        assert b._conn is None


# ---------------------------------------------------------------------------
# bulk_load
# ---------------------------------------------------------------------------


class TestBulkLoad:
    def test_bulk_load_inserts_nodes_and_relationships(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        # Both function nodes should be retrievable.
        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        callee_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "callee")

        caller = backend.get_node(caller_id)
        callee = backend.get_node(callee_id)

        assert caller is not None
        assert caller.name == "caller"
        assert callee is not None
        assert callee.name == "callee"

    def test_bulk_load_replaces_existing(self, backend: KuzuBackend) -> None:
        """Calling bulk_load twice should not duplicate data."""
        graph = _build_small_graph()
        backend.bulk_load(graph)
        backend.bulk_load(graph)

        rows = backend.execute_raw("MATCH (n:Function) RETURN n.id")
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


class TestGetNode:
    def test_returns_correct_node(self, backend: KuzuBackend) -> None:
        node = _make_node(name="target_func", file_path="src/x.py")
        backend.add_nodes([node])

        result = backend.get_node(node.id)
        assert result is not None
        assert result.id == node.id
        assert result.name == "target_func"
        assert result.file_path == "src/x.py"
        assert result.label == NodeLabel.FUNCTION

    def test_returns_none_for_missing(self, backend: KuzuBackend) -> None:
        result = backend.get_node("function:nonexistent.py:ghost")
        assert result is None

    def test_returns_none_for_unknown_label(self, backend: KuzuBackend) -> None:
        result = backend.get_node("unknown_label:foo:bar")
        assert result is None

    def test_preserves_boolean_fields(self, backend: KuzuBackend) -> None:
        node = GraphNode(
            id=generate_id(NodeLabel.FUNCTION, "src/b.py", "entry"),
            label=NodeLabel.FUNCTION,
            name="entry",
            file_path="src/b.py",
            is_entry_point=True,
            is_exported=True,
        )
        backend.add_nodes([node])

        result = backend.get_node(node.id)
        assert result is not None
        assert result.is_entry_point is True
        assert result.is_exported is True
        assert result.is_dead is False


# ---------------------------------------------------------------------------
# Module nodes and MIXES_IN relationships (Ruby module/mixin support)
# ---------------------------------------------------------------------------


class TestModuleAndMixesIn:
    def test_module_node_round_trips(self, backend: KuzuBackend) -> None:
        """A MODULE node persists to the auto-derived Module table."""
        node = _make_node(
            label=NodeLabel.MODULE,
            file_path="lib/greetable.rb",
            name="Greetable",
        )
        backend.add_nodes([node])

        result = backend.get_node(node.id)
        assert result is not None
        assert result.label == NodeLabel.MODULE
        assert result.name == "Greetable"

        rows = backend.execute_raw("MATCH (n:Module) RETURN n.name")
        assert rows == [["Greetable"]]

    def test_mixes_in_relationship_round_trips(self, backend: KuzuBackend) -> None:
        """A MIXES_IN edge from a class to a module persists in the REL TABLE GROUP."""
        klass = _make_node(
            label=NodeLabel.CLASS,
            file_path="lib/user.rb",
            name="User",
        )
        module = _make_node(
            label=NodeLabel.MODULE,
            file_path="lib/greetable.rb",
            name="Greetable",
        )
        backend.add_nodes([klass, module])
        backend.add_relationships([_make_rel(klass.id, module.id, rel_type=RelType.MIXES_IN)])

        rows = backend.execute_raw(
            "MATCH (c:Class)-[r:CodeRelation]->(m:Module) "
            "WHERE r.rel_type = 'mixes_in' RETURN c.name, m.name"
        )
        assert rows == [["User", "Greetable"]]


# ---------------------------------------------------------------------------
# get_callers / get_callees
# ---------------------------------------------------------------------------


class TestCallersAndCallees:
    def test_get_callers(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        callee_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "callee")
        callers = backend.get_callers(callee_id)

        assert len(callers) == 1
        assert callers[0].name == "caller"

    def test_get_callees(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        callees = backend.get_callees(caller_id)

        assert len(callees) == 1
        assert callees[0].name == "callee"

    def test_get_callers_empty(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        # The caller has no one calling it.
        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        callers = backend.get_callers(caller_id)
        assert callers == []

    def test_get_callees_empty(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        # The callee does not call anyone.
        callee_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "callee")
        callees = backend.get_callees(callee_id)
        assert callees == []


# ---------------------------------------------------------------------------
# execute_raw
# ---------------------------------------------------------------------------


class TestExecuteRaw:
    def test_simple_cypher(self, backend: KuzuBackend) -> None:
        backend.add_nodes([_make_node(name="raw_test")])

        rows = backend.execute_raw("MATCH (n:Function) RETURN n.name")
        assert len(rows) == 1
        assert rows[0][0] == "raw_test"

    def test_return_expression(self, backend: KuzuBackend) -> None:
        rows = backend.execute_raw("RETURN 1 + 2 AS result")
        assert rows == [[3]]


# ---------------------------------------------------------------------------
# get_indexed_files
# ---------------------------------------------------------------------------


class TestGetIndexedFiles:
    def test_returns_empty_initially(self, backend: KuzuBackend) -> None:
        result = backend.get_indexed_files()
        assert result == {}

    def test_returns_files_after_insert(self, backend: KuzuBackend) -> None:
        file_node = _make_node(
            label=NodeLabel.FILE,
            file_path="src/main.py",
            name="main.py",
            content="print('hello')",
        )
        backend.add_nodes([file_node])

        result = backend.get_indexed_files()
        assert "src/main.py" in result
        # The hash should be the sha256 of the content.
        import hashlib

        expected_hash = hashlib.sha256(b"print('hello')").hexdigest()
        assert result["src/main.py"] == expected_hash


# ---------------------------------------------------------------------------
# remove_nodes_by_file
# ---------------------------------------------------------------------------


class TestRemoveNodesByFile:
    def test_removes_matching_nodes(self, backend: KuzuBackend) -> None:
        n1 = _make_node(name="f1", file_path="src/a.py")
        n2 = _make_node(name="f2", file_path="src/a.py")
        n3 = _make_node(name="f3", file_path="src/b.py")
        backend.add_nodes([n1, n2, n3])

        backend.remove_nodes_by_file("src/a.py")

        assert backend.get_node(n1.id) is None
        assert backend.get_node(n2.id) is None
        assert backend.get_node(n3.id) is not None

    def test_returns_zero_for_no_match(self, backend: KuzuBackend) -> None:
        result = backend.remove_nodes_by_file("nonexistent.py")
        assert result == 0


# ---------------------------------------------------------------------------
# traverse
# ---------------------------------------------------------------------------


class TestTraverse:
    def test_traverse_one_hop(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        nodes = backend.traverse(caller_id, depth=1, direction="callees")

        assert len(nodes) == 1
        assert nodes[0].name == "callee"

    def test_traverse_zero_depth(self, backend: KuzuBackend) -> None:
        graph = _build_small_graph()
        backend.bulk_load(graph)

        caller_id = generate_id(NodeLabel.FUNCTION, "src/a.py", "caller")
        nodes = backend.traverse(caller_id, depth=0, direction="callees")
        assert nodes == []


# ---------------------------------------------------------------------------
# add_nodes with different labels
# ---------------------------------------------------------------------------


class TestMultipleLabels:
    def test_class_and_function(self, backend: KuzuBackend) -> None:
        fn = _make_node(label=NodeLabel.FUNCTION, name="my_fn", file_path="src/c.py")
        cls = _make_node(label=NodeLabel.CLASS, name="MyClass", file_path="src/c.py")
        backend.add_nodes([fn, cls])

        assert backend.get_node(fn.id) is not None
        assert backend.get_node(cls.id) is not None
        assert backend.get_node(fn.id).label == NodeLabel.FUNCTION
        assert backend.get_node(cls.id).label == NodeLabel.CLASS


class TestRecoveryAndRebuildSafety:
    """Regression tests for the stale-WAL incident (v1.4.0 → v1.4.1)."""

    def test_remove_db_files_takes_wal_and_shadow_siblings(self, tmp_path: Path) -> None:
        """Sibling WAL/shadow files must die with the database file.

        A stale WAL left beside a recreated database gets replayed into
        the fresh file by kuzu and corrupts it (unordered_map::at).
        """
        db = tmp_path / "kuzu.rebuild"
        for p in (db, Path(str(db) + ".wal"), Path(str(db) + ".shadow")):
            p.write_bytes(b"x")

        KuzuBackend._remove_db_files(db)

        assert not db.exists()
        assert not Path(str(db) + ".wal").exists()
        assert not Path(str(db) + ".shadow").exists()

    def test_open_with_recovery_handles_native_map_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IndexError('unordered_map::at...') triggers recovery, not a crash."""
        from synaptiq.core.storage.kuzu_backend import open_with_recovery

        db = tmp_path / "kuzu"
        stale_wal = tmp_path / "kuzu.wal"
        stale_wal.write_bytes(b"stale")

        calls = {"n": 0}
        real_init = KuzuBackend.initialize

        def flaky_init(self, path, *, read_only=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise IndexError("unordered_map::at: key not found")
            return real_init(self, path, read_only=read_only)

        monkeypatch.setattr(KuzuBackend, "initialize", flaky_init)

        storage = open_with_recovery(db, tmp_path / "meta.json")
        try:
            assert storage._db is not None
            assert not stale_wal.exists()
        finally:
            storage.close()

    def test_failed_rebuild_leaves_live_index_intact(self, tmp_path: Path) -> None:
        """bulk_load builds aside and swaps — a failed build keeps old data."""
        from synaptiq.core.graph.model import GraphNode, NodeLabel

        db = tmp_path / "kuzu"
        backend = KuzuBackend()
        backend.initialize(db)
        node = GraphNode(
            id="function:src/a.py:keep_me",
            label=NodeLabel.FUNCTION,
            name="keep_me",
            file_path="src/a.py",
        )
        g1 = KnowledgeGraph()
        g1.add_node(node)
        backend.bulk_load(g1)
        assert backend.get_node(node.id) is not None

        # Sabotage the builder so the rebuild fails mid-flight.
        class BoomError(Exception):
            pass

        original = KuzuBackend._bulk_load_nodes_csv

        def exploding(self, graph):
            raise BoomError("simulated mid-rebuild failure")

        KuzuBackend._bulk_load_nodes_csv = exploding
        try:
            with pytest.raises(BoomError):
                backend.bulk_load(KnowledgeGraph())
        finally:
            KuzuBackend._bulk_load_nodes_csv = original

        # Old data still served; no .rebuild leftovers.
        assert backend.get_node(node.id) is not None
        assert not (tmp_path / "kuzu.rebuild").exists()
        backend.close()


class TestBulkLoadCopyPath:
    """Regression tests: CSV COPY must handle source code with embedded newlines.

    Kuzu's parallel CSV reader rejects quoted newlines, so COPY of real code
    used to fail silently and fall back to ~50x slower row-by-row inserts.
    PARALLEL=false in ``_csv_copy`` keeps bulk_load on the fast path.
    """

    def _gnarly_node(self) -> GraphNode:
        # newlines, double-quotes, and commas — all CSV hazards.
        return GraphNode(
            id="function:src/x.ts:f",
            label=NodeLabel.FUNCTION,
            name="f",
            file_path="src/x.ts",
            content='export const f = (a: string, b: number) => {\n  return `hi, "${a}"`;\n};\n',
            signature='(a: string, b: number) => `hi, "x"`',
            language="typescript",
        )

    def test_node_copy_used_for_content_with_newlines(self, backend: KuzuBackend) -> None:
        """_bulk_load_nodes_csv stays on the fast COPY path (returns True)."""
        g = KnowledgeGraph()
        g.add_node(self._gnarly_node())
        assert backend._bulk_load_nodes_csv(g) is True

    def test_rel_copy_used_for_content_with_newlines(self, backend: KuzuBackend) -> None:
        """Relationship COPY succeeds once nodes are present (no fallback)."""
        g = KnowledgeGraph()
        a = self._gnarly_node()
        b = GraphNode(
            id="function:src/x.ts:g",
            label=NodeLabel.FUNCTION,
            name="g",
            file_path="src/x.ts",
        )
        g.add_node(a)
        g.add_node(b)
        g.add_relationship(
            GraphRelationship(
                id="r0",
                type=RelType.CALLS,
                source=a.id,
                target=b.id,
                properties={"rel_type": "calls", "confidence": 1.0},
            )
        )
        backend.add_nodes(list(g.iter_nodes()))
        assert backend._bulk_load_rels_csv(g) is True

    def test_bulk_load_roundtrips_content_with_newlines(self, backend: KuzuBackend) -> None:
        """End-to-end: gnarly multi-line content survives bulk_load intact."""
        node = self._gnarly_node()
        g = KnowledgeGraph()
        g.add_node(node)
        backend.bulk_load(g)
        stored = backend.get_node(node.id)
        assert stored is not None
        assert stored.content == node.content
