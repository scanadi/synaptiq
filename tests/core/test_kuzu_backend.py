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

        # Patch the loader dispatcher so the failure is injected regardless of
        # which COPY path (Arrow or CSV) bulk_load selects.
        original = KuzuBackend._bulk_load_nodes

        def exploding(self, graph):
            raise BoomError("simulated mid-rebuild failure")

        KuzuBackend._bulk_load_nodes = exploding
        try:
            with pytest.raises(BoomError):
                backend.bulk_load(KnowledgeGraph())
        finally:
            KuzuBackend._bulk_load_nodes = original

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


# ---------------------------------------------------------------------------
# Transactional batched inserts + prepared statements (W1.6)
# ---------------------------------------------------------------------------


class _ConnSpy:
    """Transparent proxy over a kuzu Connection that logs string queries.

    Lets tests observe transaction control statements (BEGIN/COMMIT/ROLLBACK)
    while forwarding everything — including prepared-statement execution — to
    the real connection.
    """

    def __init__(self, real: object) -> None:
        self._real = real
        self.log: list[str] = []

    def execute(self, query: object, parameters: object = None) -> object:
        if isinstance(query, str):
            self.log.append(query)
        return self._real.execute(query, parameters=parameters)

    def prepare(self, query: str) -> object:
        return self._real.prepare(query)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


class TestTransactionalBatchInserts:
    """add_nodes/add_relationships batch in one transaction, reuse prepared
    statements, and roll back atomically on failure."""

    def test_batch_inserts_nodes_and_rels_with_properties_intact(
        self, backend: KuzuBackend
    ) -> None:
        fn = _make_node(label=NodeLabel.FUNCTION, name="caller", file_path="src/m.py")
        cls = _make_node(label=NodeLabel.CLASS, name="Widget", file_path="src/m.py")
        meth = GraphNode(
            id=generate_id(NodeLabel.METHOD, "src/m.py", "Widget.render"),
            label=NodeLabel.METHOD,
            name="Widget.render",
            file_path="src/m.py",
            start_line=10,
            end_line=20,
            signature="render(self) -> str",
            class_name="Widget",
            is_entry_point=True,
        )
        backend.add_nodes([fn, cls, meth])

        # Two distinct source/target table pairs.
        r_call = _make_rel(fn.id, meth.id, rel_type=RelType.CALLS)
        r_contains = _make_rel(cls.id, meth.id, rel_type=RelType.CONTAINS)
        backend.add_relationships([r_call, r_contains])

        stored = backend.get_node(meth.id)
        assert stored is not None
        assert stored.name == "Widget.render"
        assert stored.start_line == 10
        assert stored.end_line == 20
        assert stored.signature == "render(self) -> str"
        assert stored.class_name == "Widget"
        assert stored.is_entry_point is True

        assert {n.name for n in backend.get_callees(fn.id)} == {"Widget.render"}
        rows = backend.execute_raw(
            "MATCH (a)-[r:CodeRelation]->(b) RETURN r.rel_type ORDER BY r.rel_type"
        )
        assert [row[0] for row in rows] == ["calls", "contains"]

    def test_prepared_statements_cached_per_label_and_pair(self, backend: KuzuBackend) -> None:
        fn = _make_node(label=NodeLabel.FUNCTION, name="f", file_path="src/p.py")
        cls = _make_node(label=NodeLabel.CLASS, name="C", file_path="src/p.py")
        meth = GraphNode(
            id=generate_id(NodeLabel.METHOD, "src/p.py", "C.m"),
            label=NodeLabel.METHOD,
            name="C.m",
            file_path="src/p.py",
        )
        backend.add_nodes([fn, cls, meth])
        backend.add_relationships(
            [
                _make_rel(fn.id, meth.id, rel_type=RelType.CALLS),
                _make_rel(cls.id, meth.id, rel_type=RelType.CONTAINS),
            ]
        )

        # Two node labels and two rel-table pairs were prepared and cached.
        assert "node:Function" in backend._prepared
        assert "node:Class" in backend._prepared
        assert "node:Method" in backend._prepared
        assert "rel:Function:Method" in backend._prepared
        assert "rel:Class:Method" in backend._prepared

        # Adding more Function nodes reuses the SAME prepared statement object.
        cached = backend._prepared["node:Function"]
        backend.add_nodes([_make_node(label=NodeLabel.FUNCTION, name="g", file_path="src/p.py")])
        assert backend._prepared["node:Function"] is cached

    def test_node_batch_runs_in_a_single_transaction(self, backend: KuzuBackend) -> None:
        spy = _ConnSpy(backend._conn)
        backend._conn = spy  # type: ignore[assignment]

        backend.add_nodes(
            [
                _make_node(name="a", file_path="src/t.py"),
                _make_node(name="b", file_path="src/t.py"),
                _make_node(name="c", file_path="src/t.py"),
            ]
        )

        assert spy.log.count("BEGIN TRANSACTION") == 1
        assert spy.log.count("COMMIT") == 1
        assert "ROLLBACK" not in spy.log

    def test_failed_node_batch_rolls_back_atomically(
        self, backend: KuzuBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed = _make_node(name="seed", file_path="src/a.py")
        backend.add_nodes([seed])
        before = backend.execute_raw("MATCH (n:Function) RETURN count(n)")[0][0]

        real_insert = backend._insert_node
        state = {"n": 0}

        def flaky(node: GraphNode) -> None:
            state["n"] += 1
            if state["n"] == 2:
                raise RuntimeError("induced node failure")
            real_insert(node)

        monkeypatch.setattr(backend, "_insert_node", flaky)

        n1 = _make_node(name="first", file_path="src/a.py")
        n2 = _make_node(name="second", file_path="src/a.py")
        # n1 is really inserted inside the txn; n2 raises -> whole batch rolls back.
        with pytest.raises(RuntimeError, match="induced node failure"):
            backend.add_nodes([n1, n2])

        # Atomic: n1 (inserted inside the failed transaction) was rolled back.
        after = backend.execute_raw("MATCH (n:Function) RETURN count(n)")[0][0]
        assert after == before
        assert backend.get_node(n1.id) is None
        assert backend.get_node(seed.id) is not None

        # Connection recovers for subsequent writes.
        monkeypatch.undo()
        backend.add_nodes([_make_node(name="later", file_path="src/a.py")])
        assert backend.execute_raw("MATCH (n:Function) RETURN count(n)")[0][0] == before + 1

    def test_reinserting_existing_node_upserts_idempotently(self, backend: KuzuBackend) -> None:
        # Mirrors the incremental re-index path: a persistent structural node
        # (e.g. an ancestor Folder) is re-inserted on every rebuild and must
        # upsert via MERGE, not raise a duplicate-primary-key error.
        first = _make_node(name="dir", file_path="src", content="v1")
        second = _make_node(name="dir", file_path="src", content="v2")  # same id
        assert second.id == first.id

        backend.add_nodes([first])
        backend.add_nodes([second])  # must not raise; must not duplicate

        stored = backend.get_node(first.id)
        assert stored is not None
        assert stored.content == "v2"  # properties refreshed
        count = backend.execute_raw(
            "MATCH (n:Function) WHERE n.id = $id RETURN count(n)",
            {"id": first.id},
        )[0][0]
        assert count == 1

    def test_failed_relationship_batch_rolls_back_atomically(
        self, backend: KuzuBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = _make_node(name="A", file_path="src/r.py")
        b = _make_node(name="B", file_path="src/r.py")
        c = _make_node(name="C", file_path="src/r.py")
        backend.add_nodes([a, b, c])
        # Pre-existing edge in its own committed transaction.
        backend.add_relationships([_make_rel(a.id, b.id, rel_type=RelType.CALLS)])

        real_insert = backend._insert_relationship
        state = {"n": 0}

        def flaky(rel: GraphRelationship) -> None:
            state["n"] += 1
            if state["n"] == 2:
                raise RuntimeError("induced relationship failure")
            real_insert(rel)

        monkeypatch.setattr(backend, "_insert_relationship", flaky)

        # First rel (A->C) inserts inside the txn, second raises -> whole batch rolls back.
        with pytest.raises(RuntimeError, match="induced relationship failure"):
            backend.add_relationships(
                [
                    _make_rel(a.id, c.id, rel_type=RelType.CALLS),
                    _make_rel(b.id, c.id, rel_type=RelType.CALLS),
                ]
            )

        # A->C was rolled back; the pre-existing A->B survived.
        assert {n.name for n in backend.get_callees(a.id)} == {"B"}
        total = backend.execute_raw("MATCH ()-[r:CodeRelation]->() RETURN count(r)")[0][0]
        assert total == 1

    def test_empty_batches_open_no_transaction(self, backend: KuzuBackend) -> None:
        spy = _ConnSpy(backend._conn)
        backend._conn = spy  # type: ignore[assignment]

        backend.add_nodes([])
        backend.add_relationships([])
        assert "BEGIN TRANSACTION" not in spy.log

        # No dangling transaction was left open: a real batch still succeeds
        # (a leaked BEGIN would make the next BEGIN raise).
        backend.add_nodes([_make_node(name="ok", file_path="src/e.py")])
        assert spy.log.count("BEGIN TRANSACTION") == 1
        assert spy.log.count("COMMIT") == 1

    def test_prepared_cache_cleared_on_close(self, tmp_path: Path) -> None:
        b = KuzuBackend()
        b.initialize(tmp_path / "cache_db")
        b.add_nodes([_make_node(name="x", file_path="src/x.py")])
        assert b._prepared  # populated
        b.close()
        assert b._prepared == {}  # cleared with the connection


# ---------------------------------------------------------------------------
# Arrow / Parquet bulk COPY with CSV fallback (W2.3)
# ---------------------------------------------------------------------------


def _rich_graph() -> KnowledgeGraph:
    """A graph exercising every bulk-load edge case: multiple labels, empty and
    non-empty strings, unicode + embedded newlines, node properties, every rel
    property, and duplicate node/rel ids (to check dedup parity)."""
    from synaptiq.core.graph.model import generate_id as gid

    g = KnowledgeGraph()

    folder = GraphNode(id=gid(NodeLabel.FOLDER, "src", ""), label=NodeLabel.FOLDER, name="src")
    file_a = GraphNode(
        id=gid(NodeLabel.FILE, "src/a.py", ""),
        label=NodeLabel.FILE,
        name="a.py",
        file_path="src/a.py",
        language="python",
    )
    # Rich content: unicode, newlines, quotes, commas — all CSV hazards.
    rich = GraphNode(
        id=gid(NodeLabel.FUNCTION, "src/a.py", "café"),
        label=NodeLabel.FUNCTION,
        name="café",
        file_path="src/a.py",
        start_line=1,
        end_line=9,
        content='def café(x):\n    """doc, with "quotes"\n    and, commas"""\n    return x  # ☕',
        signature="def café(x)",
        language="python",
        is_dead=True,
        is_entry_point=True,
        is_exported=True,
        properties={"decorators": ["staticmethod"], "count": 3},
    )
    # Empty content/signature/class_name -> must land as NULL, like CSV.
    empty = GraphNode(
        id=gid(NodeLabel.FUNCTION, "src/a.py", "bare"),
        label=NodeLabel.FUNCTION,
        name="bare",
        file_path="src/a.py",
        content="",
        signature="",
        class_name="",
    )
    klass = GraphNode(
        id=gid(NodeLabel.CLASS, "src/a.py", "Widget"),
        label=NodeLabel.CLASS,
        name="Widget",
        file_path="src/a.py",
        content="class Widget:\n    pass",
    )
    method = GraphNode(
        id=gid(NodeLabel.METHOD, "src/a.py", "Widget.save"),
        label=NodeLabel.METHOD,
        name="save",
        file_path="src/a.py",
        class_name="Widget",
    )
    module = GraphNode(
        id=gid(NodeLabel.MODULE, "src/a.rb", "Helpers"),
        label=NodeLabel.MODULE,
        name="Helpers",
        file_path="src/a.rb",
        language="ruby",
    )
    for n in (folder, file_a, rich, empty, klass, method, module):
        g.add_node(n)
    # Duplicate id with different content — dedup must keep the last occurrence
    # identically in both paths.
    g.add_node(
        GraphNode(
            id=rich.id,
            label=NodeLabel.FUNCTION,
            name="café",
            file_path="src/a.py",
            content="LAST WINS\nsecond line",
            signature="def café(x)",
        )
    )

    # Relationships covering every property and an empty-property edge.
    g.add_relationship(
        GraphRelationship(
            id="r-calls",
            type=RelType.CALLS,
            source=rich.id,
            target=method.id,
            properties={"confidence": 0.8, "role": "receiver"},
        )
    )
    g.add_relationship(
        GraphRelationship(
            id="r-contains", type=RelType.CONTAINS, source=folder.id, target=file_a.id
        )
    )
    g.add_relationship(
        GraphRelationship(
            id="r-step",
            type=RelType.STEP_IN_PROCESS,
            source=rich.id,
            target=empty.id,
            properties={"step_number": 2, "role": "entry"},
        )
    )
    g.add_relationship(
        GraphRelationship(
            id="r-coupled",
            type=RelType.COUPLED_WITH,
            source=file_a.id,
            target=module.id,
            properties={"strength": 0.5, "co_changes": 4, "symbols": "a,b,c"},
        )
    )
    g.add_relationship(
        GraphRelationship(id="r-defines", type=RelType.DEFINES, source=klass.id, target=method.id)
    )
    # Duplicate edge identity — dedup must collapse identically in both paths.
    g.add_relationship(
        GraphRelationship(
            id="r-calls-dup",
            type=RelType.CALLS,
            source=rich.id,
            target=method.id,
            properties={"confidence": 0.8, "role": "receiver"},
        )
    )
    return g


class TestArrowCsvEquivalence:
    """The Arrow and CSV bulk paths must produce byte-identical databases."""

    def _load(self, backend: KuzuBackend, graph: KnowledgeGraph, *, arrow: bool) -> None:
        if arrow:
            assert backend._bulk_load_nodes_arrow(graph) is True
            assert backend._bulk_load_rels_arrow(graph) is True
        else:
            assert backend._bulk_load_nodes_csv(graph) is True
            assert backend._bulk_load_rels_csv(graph) is True

    def _dump_nodes(self, backend: KuzuBackend, table: str) -> list[list[object]]:
        from synaptiq.core.storage.kuzu_backend import _node_columns

        return backend.execute_raw(f"MATCH (n:{table}) RETURN {_node_columns('n')} ORDER BY n.id")

    def _dump_rels(self, backend: KuzuBackend) -> list[list[object]]:
        return backend.execute_raw(
            "MATCH (a)-[r:CodeRelation]->(b) "
            "RETURN a.id, b.id, r.rel_type, r.confidence, r.role, r.step_number, "
            "r.strength, r.co_changes, r.symbols "
            "ORDER BY a.id, b.id, r.rel_type, r.role, r.step_number"
        )

    def test_nodes_and_rels_identical(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        from synaptiq.core.storage.kuzu_backend import _NODE_TABLE_NAMES

        graph = _rich_graph()

        arrow_be = KuzuBackend()
        arrow_be.initialize(tmp_path / "arrow_db")
        csv_be = KuzuBackend()
        csv_be.initialize(tmp_path / "csv_db")
        try:
            self._load(arrow_be, graph, arrow=True)
            self._load(csv_be, graph, arrow=False)

            for table in _NODE_TABLE_NAMES:
                arrow_rows = self._dump_nodes(arrow_be, table)
                csv_rows = self._dump_nodes(csv_be, table)
                assert arrow_rows == csv_rows, f"node table {table} differs"

            # Sanity: the dedup'd rich function kept the last content in both.
            fn_rows = self._dump_nodes(arrow_be, "Function")
            contents = [r[5] for r in fn_rows]
            assert "LAST WINS\nsecond line" in contents
            # Empty strings landed as NULL (None), matching CSV's reader.
            assert any(r[6] is None for r in fn_rows)  # bare's empty signature

            assert self._dump_rels(arrow_be) == self._dump_rels(csv_be)
        finally:
            arrow_be.close()
            csv_be.close()

    def test_embeddings_identical(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        from synaptiq.core.storage.base import NodeEmbedding

        dim = 8
        embs = [
            NodeEmbedding(
                node_id=f"function:src/a.py:f{i}",
                embedding=[float(i) - 3.5 + j * 0.1 for j in range(dim)],
                text_sha=f"sha{i}",
            )
            for i in range(5)
        ]

        arrow_be = KuzuBackend()
        arrow_be.initialize(tmp_path / "arrow_emb")
        csv_be = KuzuBackend()
        csv_be.initialize(tmp_path / "csv_emb")
        try:
            assert arrow_be._bulk_store_embeddings_arrow(embs) is True
            assert csv_be._bulk_store_embeddings_csv(embs) is True

            q = "MATCH (e:Embedding) RETURN e.node_id, e.vec, e.text_sha ORDER BY e.node_id"
            arrow_rows = arrow_be.execute_raw(q)
            csv_rows = csv_be.execute_raw(q)
            assert len(arrow_rows) == len(embs)
            assert arrow_rows == csv_rows
        finally:
            arrow_be.close()
            csv_be.close()


class TestArrowBulkPath:
    """End-to-end bulk_load on the Arrow path (pyarrow installed)."""

    def test_dispatch_prefers_arrow_when_available(self) -> None:
        pytest.importorskip("pyarrow")
        import synaptiq.core.storage.kuzu_backend as kb

        assert kb._HAS_PYARROW is True

    def test_bulk_load_roundtrips_gnarly_content(self, backend: KuzuBackend) -> None:
        graph = _rich_graph()
        backend.bulk_load(graph)
        stored = backend.get_node("function:src/a.py:café")
        assert stored is not None
        assert stored.content == "LAST WINS\nsecond line"
        # Relationship survived with its properties.
        rows = backend.execute_raw(
            "MATCH (a)-[r:CodeRelation]->(b) WHERE r.rel_type = 'coupled_with' "
            "RETURN r.strength, r.co_changes, r.symbols"
        )
        assert rows == [[0.5, 4, "a,b,c"]]


class TestCsvFallbackWithoutPyarrow:
    """With pyarrow force-disabled, bulk_load must transparently use CSV."""

    def test_bulk_load_uses_csv_when_pyarrow_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import synaptiq.core.storage.kuzu_backend as kb

        monkeypatch.setattr(kb, "_HAS_PYARROW", False)

        called: dict[str, bool] = {}
        orig_csv = KuzuBackend._bulk_load_nodes_csv

        def spy(self, graph):
            called["csv"] = True
            return orig_csv(self, graph)

        monkeypatch.setattr(KuzuBackend, "_bulk_load_nodes_csv", spy)

        backend = KuzuBackend()
        backend.initialize(tmp_path / "csv_only_db")
        try:
            backend.bulk_load(_rich_graph())
            assert called.get("csv") is True
            stored = backend.get_node("function:src/a.py:café")
            assert stored is not None
            assert stored.content == "LAST WINS\nsecond line"
        finally:
            backend.close()


class TestBulkLoadSkipsEmptyFtsBuild:
    """W2.3b: the .rebuild schema skips the (empty) FTS build; rebuild_fts_indexes
    builds them over the populated tables, and the query path stays safe."""

    def _searchable(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id="function:src/a.py:findmez9",
                label=NodeLabel.FUNCTION,
                name="findmez9",
                file_path="src/a.py",
                content="def findmez9():\n    return 1",
            )
        )
        return g

    def test_builder_skips_fts_yet_search_works_after_bulk_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_flags: list[bool] = []
        orig = KuzuBackend._create_schema

        def spy(self, *, build_fts: bool = True):
            seen_flags.append(build_fts)
            return orig(self, build_fts=build_fts)

        monkeypatch.setattr(KuzuBackend, "_create_schema", spy)

        backend = KuzuBackend()
        backend.initialize(tmp_path / "fts_db")
        try:
            backend.bulk_load(self._searchable())
            # The bulk_load builder created its schema without FTS indexes.
            assert False in seen_flags
            # ...but the populated FTS index built by rebuild_fts_indexes works.
            results = backend.fts_search("findmez9", limit=5)
            assert any(r.node_id == "function:src/a.py:findmez9" for r in results)
        finally:
            backend.close()

    def test_fts_search_tolerates_missing_indexes(self, tmp_path: Path) -> None:
        # A DB whose FTS indexes were never built must not crash on search.
        backend = KuzuBackend()
        backend.initialize(tmp_path / "nofts_db", _build_fts_indexes=False)
        try:
            backend.add_nodes(
                [
                    GraphNode(
                        id="function:src/a.py:x",
                        label=NodeLabel.FUNCTION,
                        name="x",
                        file_path="src/a.py",
                    )
                ]
            )
            assert backend.fts_search("x", limit=5) == []
        finally:
            backend.close()
