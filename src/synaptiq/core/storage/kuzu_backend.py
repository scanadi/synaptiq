"""KuzuDB storage backend for Synaptiq.

Implements the :class:`StorageBackend` protocol using KuzuDB, an embedded
graph database that speaks Cypher. Each :class:`NodeLabel` maps to a
separate node table, and a single ``CodeRelation`` relationship table group
covers all source-to-target combinations.

Concurrency
-----------
Read methods use a thread-safe connection pool (``_read_conn`` context
manager) so multiple threads can query in parallel. Write methods use a
dedicated ``self._conn`` handle — callers must ensure exclusivity via an
external ``AsyncRWLock``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import kuzu

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, GraphRelationship, NodeLabel
from synaptiq.core.storage.base import NodeEmbedding, SearchResult

logger = logging.getLogger(__name__)

_NODE_TABLE_NAMES: list[str] = [label.name.title().replace("_", "") for label in NodeLabel]

_LABEL_TO_TABLE: dict[str, str] = {
    label.value: label.name.title().replace("_", "") for label in NodeLabel
}

_LABEL_MAP: dict[str, NodeLabel] = {label.value: label for label in NodeLabel}

_SEARCHABLE_TABLES: list[str] = [
    t for t in _NODE_TABLE_NAMES
    if t not in ("Folder", "Community", "Process")
]

_NODE_PROPERTIES = (
    "id STRING, "
    "name STRING, "
    "file_path STRING, "
    "start_line INT64, "
    "end_line INT64, "
    "content STRING, "
    "signature STRING, "
    "language STRING, "
    "class_name STRING, "
    "is_dead BOOL, "
    "is_entry_point BOOL, "
    "is_exported BOOL, "
    "properties_json STRING, "
    "PRIMARY KEY (id)"
)

# Column order used by every node SELECT — must match _NODE_PROPERTIES.
# Explicit column lists (instead of ``RETURN n.*``) keep row decoding
# independent of Kuzu's internal property ordering.
_NODE_COLUMN_NAMES: tuple[str, ...] = (
    "id", "name", "file_path", "start_line", "end_line", "content",
    "signature", "language", "class_name", "is_dead", "is_entry_point",
    "is_exported", "properties_json",
)


def _node_columns(alias: str) -> str:
    """Return the explicit node column list for *alias* (e.g. ``n.id, n.name, ...``)."""
    return ", ".join(f"{alias}.{c}" for c in _NODE_COLUMN_NAMES)

_REL_PROPERTIES = (
    "rel_type STRING, "
    "confidence DOUBLE, "
    "role STRING, "
    "step_number INT64, "
    "strength DOUBLE, "
    "co_changes INT64, "
    "symbols STRING"
)

def _escape(value: str) -> str:
    """Escape a string for inclusion in a Cypher literal.

    Internal only — used for FTS/fuzzy search literals that Kuzu cannot
    parameterize. Everything else must use parameterized queries.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _table_for_id(node_id: str) -> str | None:
    """Extract the table name from a node ID by mapping its label prefix."""
    prefix = node_id.split(":", 1)[0]
    return _LABEL_TO_TABLE.get(prefix)


def _serialize_properties(properties: dict[str, Any] | None) -> str:
    """Serialize a node's properties dict to JSON for storage ('' when empty)."""
    if not properties:
        return ""
    try:
        return json.dumps(properties, default=str)
    except (TypeError, ValueError):
        return ""


def deserialize_properties(raw: Any) -> dict[str, Any]:
    """Parse a stored ``properties_json`` value back into a dict.

    Shared by row hydration here and by tool handlers that read the
    column via raw Cypher — one parser for one format.
    """
    if not raw or not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}

# Embedding vectors use a fixed-dimension FLOAT column so Kuzu's HNSW
# vector index can be built on them; 384 matches BAAI/bge-small-en-v1.5.
# The bulk store path recreates the table from the actual embedding width,
# so this constant only shapes the empty table created at schema time.
EMBEDDING_DIM = 384

_VECTOR_INDEX_NAME = "embedding_vec_idx"


def _embedding_ddl(dim: int) -> str:
    """Embedding table column DDL for *dim*-wide vectors."""
    return f"node_id STRING, vec FLOAT[{dim}], PRIMARY KEY(node_id)"

# Maximum number of read connections to keep in the pool.
_MAX_POOL_SIZE = 8


def open_with_recovery(
    db_path: Path,
    meta_path: Path | None = None,
    *,
    read_only: bool = False,
) -> KuzuBackend:
    """Open a KuzuBackend at *db_path*, rebuilding on corruption.

    A corrupted database (e.g. duplicate primary key from a mid-write kill)
    is deleted along with *meta_path* so the next index run rebuilds
    cleanly.  In read-write mode the empty database is re-initialised; in
    read-only mode a bare (uninitialised) backend is returned since there
    is nothing left to open.

    Non-corruption errors (e.g. the database is locked by another
    process) propagate unchanged.
    """
    storage = KuzuBackend()
    try:
        storage.initialize(db_path, read_only=read_only)
        return storage
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "primary key" not in msg and "corrupt" not in msg:
            raise

    logger.warning("Corrupted index detected, removing %s", db_path)
    storage.close()
    if db_path.is_dir():
        shutil.rmtree(db_path, ignore_errors=True)
    else:
        db_path.unlink(missing_ok=True)
        # Clean up WAL/shadow files left alongside a file-based DB.
        for suffix in (".wal", ".shadow"):
            db_path.with_suffix(suffix).unlink(missing_ok=True)
    if meta_path is not None:
        meta_path.unlink(missing_ok=True)

    storage = KuzuBackend()
    if not read_only:
        storage.initialize(db_path)
    return storage


class KuzuBackend:
    """StorageBackend implementation backed by KuzuDB.

    Usage::

        backend = KuzuBackend()
        backend.initialize(Path("/tmp/synaptiq_db"))
        backend.bulk_load(graph)
        node = backend.get_node("function:src/app.py:main")
        backend.close()
    """

    def __init__(self) -> None:
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None
        self._db_path: Path | None = None
        # Thread-safe pool of read connections, tagged with the database
        # generation they were created against.  The generation increments
        # on every (re)initialize so connections bound to a closed/deleted
        # database are never handed out again.
        self._read_pool: list[tuple[int, kuzu.Connection]] = []
        self._pool_lock = threading.Lock()
        self._generation = 0
        # In-flight read tracking so destructive operations (_reset_database)
        # can drain readers that outlived their dispatch timeout.
        self._reads_cv = threading.Condition()
        self._active_reads = 0

    @property
    def generation(self) -> int:
        """Monotonic counter bumped on every (re)initialize.

        Cache key for data derived from the graph (e.g. the PageRank
        projection): a bump means the underlying data may have changed.
        """
        return self._generation

    def initialize(self, path: Path, *, read_only: bool = False) -> None:
        """Open or create the KuzuDB database at *path* and set up the schema.

        Args:
            path: Filesystem path to the KuzuDB database directory.
            read_only: If ``True``, open the database in read-only mode.
                This allows multiple concurrent readers (e.g. MCP server
                instances) without lock conflicts.  Schema creation is
                skipped since the database must already exist — but the
                schema is verified so a database created by an older
                synaptiq fails loudly instead of silently returning
                empty results for every query.
        """
        from synaptiq.core.resources import current_limits

        limits = current_limits()
        self._db_path = path
        # 0 for either cap means Kuzu's library default (all cores /
        # default buffer pool) — the interactive profile resolves to that.
        self._db = kuzu.Database(
            str(path),
            read_only=read_only,
            max_num_threads=limits.kuzu_threads,
            buffer_pool_size=limits.kuzu_buffer_bytes,
        )
        self._conn = kuzu.Connection(self._db)
        with self._pool_lock:
            self._generation += 1
        if not read_only:
            self._create_schema()
        else:
            self._verify_schema()

    def _table_columns(self, table: str) -> set[str] | None:
        """Return the column names of *table*, or ``None`` if it doesn't exist."""
        assert self._conn is not None
        try:
            rows = self._drain(self._conn.execute(f"CALL TABLE_INFO('{table}') RETURN *"))
        except Exception:
            return None
        return {row[1] for row in rows}

    def _migrate_schema(self) -> None:
        """Add columns introduced after an existing database was created.

        ``CREATE NODE TABLE IF NOT EXISTS`` is a no-op for pre-existing
        tables, so without this step every node SELECT on an upgraded
        index would binder-error on the missing column — and those errors
        are swallowed at debug level, surfacing as 'symbol not found'.
        """
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            cols = self._table_columns(table)
            if cols is None:
                continue
            for column, col_type in (("properties_json", "STRING"),):
                if column not in cols:
                    logger.info("Migrating table %s: adding column %s", table, column)
                    try:
                        self._conn.execute(f"ALTER TABLE {table} ADD {column} {col_type}")
                    except Exception:
                        logger.warning(
                            "Schema migration failed for %s.%s", table, column, exc_info=True
                        )

    def _verify_schema(self) -> None:
        """Raise a clear error when a read-only database has an old schema."""
        cols = self._table_columns(_NODE_TABLE_NAMES[0])
        if cols is not None and "properties_json" not in cols:
            raise RuntimeError(
                "This index was created by an older synaptiq version and "
                "cannot be migrated in read-only mode. Run `synaptiq analyze` "
                "to rebuild it."
            )

    @staticmethod
    def _close_quietly(obj: object) -> None:
        """Call ``close()`` on a Kuzu object, ignoring errors."""
        try:
            obj.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    def close(self) -> None:
        """Release all connections and the database handle.

        Uses the explicit ``close()`` methods on Kuzu connections and the
        database so file locks are released and data is flushed
        deterministically (not at GC time).
        """
        with self._pool_lock:
            pool = self._read_pool
            self._read_pool = []
            self._generation += 1
        for _gen, conn in pool:
            self._close_quietly(conn)

        if self._conn is not None:
            self._close_quietly(self._conn)
            self._conn = None
        if self._db is not None:
            self._close_quietly(self._db)
            self._db = None

    # ------------------------------------------------------------------
    # Connection pool for concurrent reads
    # ------------------------------------------------------------------

    def _acquire_read_conn(self) -> tuple[int, kuzu.Connection]:
        """Get a pooled connection or create a new one (thread-safe).

        Returns ``(generation, connection)``; the generation is passed back
        on release so stale connections are closed instead of re-pooled.
        """
        with self._pool_lock:
            while self._read_pool:
                gen, conn = self._read_pool.pop()
                if gen == self._generation:
                    return gen, conn
                self._close_quietly(conn)
            gen = self._generation
            db = self._db
        if db is None:
            raise RuntimeError(
                "No database open — the index is missing or was removed. "
                "Run `synaptiq analyze` first."
            )
        return gen, kuzu.Connection(db)

    def _release_read_conn(self, gen: int, conn: kuzu.Connection) -> None:
        """Return a connection to the pool, or close it if stale."""
        with self._pool_lock:
            if gen == self._generation and len(self._read_pool) < _MAX_POOL_SIZE:
                self._read_pool.append((gen, conn))
                return
        self._close_quietly(conn)

    @contextmanager
    def _read_conn(self) -> Iterator[kuzu.Connection]:
        """Context manager for read connections from the pool.

        Tracks in-flight reads so :meth:`_reset_database` can wait for
        stragglers before deleting database files out from under them.
        The counter is incremented BEFORE acquiring the connection — a
        reader inside connection creation must already be visible to
        the drain, or the reset could delete files under it.
        """
        with self._reads_cv:
            self._active_reads += 1
        try:
            gen, conn = self._acquire_read_conn()
            try:
                yield conn
            finally:
                self._release_read_conn(gen, conn)
        finally:
            with self._reads_cv:
                self._active_reads -= 1
                self._reads_cv.notify_all()

    def _wait_for_readers(self, timeout: float = 30.0) -> bool:
        """Block until no reads are in flight, or *timeout* elapses.

        Returns ``True`` when the read count reached zero.  Used before
        destructive operations: a read whose ``asyncio.wait_for`` dispatch
        timed out keeps running in its thread even after the caller released
        the RW lock, so the lock alone does not guarantee quiescence.
        """
        with self._reads_cv:
            ok = self._reads_cv.wait_for(lambda: self._active_reads == 0, timeout=timeout)
        if not ok:
            logger.warning(
                "Proceeding with database reset while %d read(s) still in flight",
                self._active_reads,
            )
        return ok

    @staticmethod
    def _drain(result: Any) -> list[list[Any]]:
        """Materialize all rows from a QueryResult and close it.

        QueryResults hold native resources tied to their connection; closing
        them eagerly keeps pooled connections reusable and lets ``close()``
        release the database deterministically.
        """
        rows: list[list[Any]] = []
        try:
            while result.has_next():
                rows.append(result.get_next())
        finally:
            try:
                result.close()
            except Exception:
                pass
        return rows

    # ------------------------------------------------------------------
    # Write operations (use self._conn — must be externally serialized)
    # ------------------------------------------------------------------

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Insert nodes into their respective label tables."""
        for node in nodes:
            self._insert_node(node)

    def add_relationships(self, rels: list[GraphRelationship]) -> None:
        """Insert relationships by matching source and target nodes."""
        for rel in rels:
            self._insert_relationship(rel)

    def remove_nodes_by_file(self, file_path: str) -> int:
        """Delete all nodes whose ``file_path`` matches across every table.

        Also removes embedding rows for the file's symbols so vector search
        does not return ghost results pointing at deleted nodes.

        Returns:
            Always 0 — exact count is not tracked for performance.
        """
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            try:
                self._conn.execute(
                    f"MATCH (n:{table}) WHERE n.file_path = $fp DETACH DELETE n",
                    parameters={"fp": file_path},
                )
            except Exception:
                logger.debug("Failed to remove nodes from table %s", table, exc_info=True)

        # Embedding node_ids have the form ``label:file_path:symbol``; the
        # path segment is delimited by colons and paths contain no colons.
        try:
            self._conn.execute(
                "MATCH (e:Embedding) WHERE e.node_id CONTAINS $pat DELETE e",
                parameters={"pat": f":{file_path}:"},
            )
        except Exception:
            logger.debug("Failed to remove embeddings for %s", file_path, exc_info=True)
        return 0

    # ------------------------------------------------------------------
    # Read operations (use pooled connections — safe for concurrent use)
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a single node by ID, or ``None`` if not found."""
        table = _table_for_id(node_id)
        if table is None:
            return None

        query = f"MATCH (n:{table}) WHERE n.id = $nid RETURN {_node_columns('n')}"
        with self._read_conn() as conn:
            try:
                rows = self._drain(conn.execute(query, parameters={"nid": node_id}))
                if rows:
                    return self._row_to_node(rows[0], node_id)
            except Exception:
                logger.debug("get_node failed for %s", node_id, exc_info=True)
        return None

    def get_callers(self, node_id: str) -> list[GraphNode]:
        """Return nodes that CALL the node identified by *node_id*."""
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (caller)-[r:CodeRelation]->(callee:{table}) "
            f"WHERE callee.id = $nid AND r.rel_type = 'calls' "
            f"RETURN {_node_columns('caller')}"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_callees(self, node_id: str) -> list[GraphNode]:
        """Return nodes called by the node identified by *node_id*."""
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (caller:{table})-[r:CodeRelation]->(callee) "
            f"WHERE caller.id = $nid AND r.rel_type = 'calls' "
            f"RETURN {_node_columns('callee')}"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def get_type_refs(self, node_id: str) -> list[GraphNode]:
        """Return nodes referenced via USES_TYPE from *node_id*."""
        table = _table_for_id(node_id)
        if table is None:
            return []

        query = (
            f"MATCH (src:{table})-[r:CodeRelation]->(tgt) "
            f"WHERE src.id = $nid AND r.rel_type = 'uses_type' "
            f"RETURN {_node_columns('tgt')}"
        )
        return self._query_nodes(query, parameters={"nid": node_id})

    def traverse(self, start_id: str, depth: int, direction: str = "callers") -> list[GraphNode]:
        """Batched BFS traversal through CALLS edges up to *depth* hops.

        Instead of querying one node at a time (N+1 pattern), queries all
        nodes at the current BFS level in a single Cypher call per level.
        This reduces round-trips from O(nodes) to O(depth).

        Args:
            direction: ``"callers"`` follows incoming CALLS (blast radius),
                       ``"callees"`` follows outgoing CALLS (dependencies).
        """
        start_table = _table_for_id(start_id)
        if start_table is None:
            return []

        visited: set[str] = {start_id}
        result_list: list[GraphNode] = []
        current_ids: list[str] = [start_id]

        with self._read_conn() as conn:
            for _level in range(depth):
                if not current_ids:
                    break

                # Batch query: get all neighbors of current level at once.
                neighbors = self._get_neighbors_batch(
                    conn, current_ids, direction
                )

                next_ids: list[str] = []
                for node in neighbors:
                    if node.id not in visited:
                        visited.add(node.id)
                        result_list.append(node)
                        next_ids.append(node.id)

                current_ids = next_ids

        return result_list

    def _get_neighbors_batch(
        self,
        conn: kuzu.Connection,
        node_ids: list[str],
        direction: str,
    ) -> list[GraphNode]:
        """Get all CALLS neighbors of *node_ids* in a single query per table."""
        nodes: list[GraphNode] = []
        ids_by_table: dict[str, list[str]] = {}
        for nid in node_ids:
            table = _table_for_id(nid)
            if table:
                ids_by_table.setdefault(table, []).append(nid)

        for table, ids in ids_by_table.items():
            if direction == "callers":
                query = (
                    f"MATCH (caller)-[r:CodeRelation]->(callee:{table}) "
                    f"WHERE callee.id IN $ids AND r.rel_type = 'calls' "
                    f"RETURN {_node_columns('caller')}"
                )
            else:
                query = (
                    f"MATCH (caller:{table})-[r:CodeRelation]->(callee) "
                    f"WHERE caller.id IN $ids AND r.rel_type = 'calls' "
                    f"RETURN {_node_columns('callee')}"
                )
            try:
                for row in self._drain(conn.execute(query, parameters={"ids": ids})):
                    node = self._row_to_node(row)
                    if node is not None:
                        nodes.append(node)
            except Exception:
                logger.debug(
                    "_get_neighbors_batch failed for table %s", table, exc_info=True
                )

        return nodes

    def traverse_with_depth(
        self, start_id: str, depth: int, direction: str = "callers"
    ) -> list[tuple[GraphNode, int]]:
        """Like :meth:`traverse` but returns ``(node, hop_distance)`` pairs."""
        start_table = _table_for_id(start_id)
        if start_table is None:
            return []

        visited: set[str] = {start_id}
        result_list: list[tuple[GraphNode, int]] = []
        current_ids: list[str] = [start_id]

        with self._read_conn() as conn:
            for level in range(1, depth + 1):
                if not current_ids:
                    break
                neighbors = self._get_neighbors_batch(conn, current_ids, direction)
                next_ids: list[str] = []
                for node in neighbors:
                    if node.id not in visited:
                        visited.add(node.id)
                        result_list.append((node, level))
                        next_ids.append(node.id)
                current_ids = next_ids

        return result_list

    def get_callers_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return callers paired with the CALLS edge confidence score."""
        table = _table_for_id(node_id)
        if table is None:
            return []

        results: list[tuple[GraphNode, float]] = []
        with self._read_conn() as conn:
            for src_table in _SEARCHABLE_TABLES:
                query = (
                    f"MATCH (caller:{src_table})-[r:CodeRelation]->(callee:{table}) "
                    f"WHERE callee.id = $nid AND r.rel_type = 'calls' "
                    f"RETURN {_node_columns('caller')}, r.confidence"
                )
                try:
                    for row in self._drain(conn.execute(query, parameters={"nid": node_id})):
                        conf = float(row[-1]) if row[-1] is not None else 1.0
                        node = self._row_to_node(row[:-1])
                        if node is not None:
                            results.append((node, conf))
                except Exception:
                    logger.debug(
                        "get_callers_with_confidence failed on table %s",
                        src_table,
                        exc_info=True,
                    )
        return results

    def get_callees_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return callees paired with the CALLS edge confidence score."""
        table = _table_for_id(node_id)
        if table is None:
            return []

        results: list[tuple[GraphNode, float]] = []
        with self._read_conn() as conn:
            for tgt_table in _SEARCHABLE_TABLES:
                query = (
                    f"MATCH (caller:{table})-[r:CodeRelation]->(callee:{tgt_table}) "
                    f"WHERE caller.id = $nid AND r.rel_type = 'calls' "
                    f"RETURN {_node_columns('callee')}, r.confidence"
                )
                try:
                    for row in self._drain(conn.execute(query, parameters={"nid": node_id})):
                        conf = float(row[-1]) if row[-1] is not None else 1.0
                        node = self._row_to_node(row[:-1])
                        if node is not None:
                            results.append((node, conf))
                except Exception:
                    logger.debug(
                        "get_callees_with_confidence failed on table %s",
                        tgt_table,
                        exc_info=True,
                    )
        return results

    def get_process_memberships(self, node_ids: list[str]) -> dict[str, str]:
        """Return ``{node_id: process_name}`` for nodes that belong to a process."""
        mapping: dict[str, str] = {}
        ids_by_table: dict[str, list[str]] = {}
        for nid in node_ids:
            table = _table_for_id(nid)
            if table:
                ids_by_table.setdefault(table, []).append(nid)

        with self._read_conn() as conn:
            for table, ids in ids_by_table.items():
                query = (
                    f"MATCH (n:{table})-[r:CodeRelation]->(p:Process) "
                    f"WHERE n.id IN $ids AND r.rel_type = 'step_in_process' "
                    f"RETURN n.id, p.name"
                )
                try:
                    for row in self._drain(conn.execute(query, parameters={"ids": ids})):
                        if row[0] and row[1]:
                            mapping[row[0]] = row[1]
                except Exception:
                    logger.debug(
                        "get_process_memberships failed on table %s", table, exc_info=True
                    )
        return mapping

    def load_graph(self) -> KnowledgeGraph:
        """Load the full graph into an in-memory KnowledgeGraph."""
        from synaptiq.core.graph.model import RelType as RelTypeEnum

        graph = KnowledgeGraph()
        with self._read_conn() as conn:
            for table in _NODE_TABLE_NAMES:
                try:
                    rows = self._drain(
                        conn.execute(f"MATCH (n:{table}) RETURN {_node_columns('n')}")
                    )
                    for row in rows:
                        node = self._row_to_node(row)
                        if node is not None:
                            graph.add_node(node)
                except Exception:
                    logger.debug("load_graph: failed to load %s nodes", table, exc_info=True)

            # One label-less query covers every table pair in the rel group.
            try:
                rel_rows = self._drain(conn.execute(
                    "MATCH (a)-[r:CodeRelation]->(b) "
                    "RETURN a.id, b.id, r.rel_type, r.confidence, r.symbols, "
                    "r.strength, r.co_changes, r.step_number, r.role"
                ))
            except Exception:
                logger.debug("load_graph: relationship query failed", exc_info=True)
                rel_rows = []

            for row in rel_rows:
                rel_type_str = row[2] or "calls"
                try:
                    rel_type = RelTypeEnum(rel_type_str)
                except ValueError:
                    continue
                rel_id = f"{rel_type_str}:{row[0]}->{row[1]}"
                props: dict[str, Any] = {}
                if row[3] is not None:
                    props["confidence"] = float(row[3])
                if row[4]:
                    props["symbols"] = str(row[4])
                if row[5] is not None:
                    props["strength"] = float(row[5])
                if row[6] is not None:
                    props["co_changes"] = int(row[6])
                if row[7] is not None:
                    props["step_number"] = int(row[7])
                if row[8]:
                    props["role"] = str(row[8])
                graph.add_relationship(GraphRelationship(
                    id=rel_id,
                    type=rel_type,
                    source=row[0],
                    target=row[1],
                    properties=props,
                ))
        return graph

    def execute_raw(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[list[Any]]:
        """Execute a raw Cypher query and return all result rows.

        Args:
            query: Cypher query string. Use ``$param`` placeholders for
                user-supplied values.
            parameters: Optional parameter dict for parameterized queries.
        """
        with self._read_conn() as conn:
            return self._drain(conn.execute(query, parameters=parameters or {}))

    def exact_name_search(self, name: str, limit: int = 5) -> list[SearchResult]:
        """Search for nodes with an exact name match across all searchable tables.

        Returns results sorted by label priority (functions/methods first),
        preferring source files over test files.
        """
        limit = max(1, int(limit))
        candidates: list[SearchResult] = []

        with self._read_conn() as conn:
            for table in _SEARCHABLE_TABLES:
                cypher = (
                    f"MATCH (n:{table}) WHERE n.name = $name "
                    f"RETURN n.id, n.name, n.file_path, n.content, n.signature "
                    f"LIMIT {limit}"
                )
                try:
                    for row in self._drain(conn.execute(cypher, parameters={"name": name})):
                        node_id = row[0] or ""
                        node_name = row[1] or ""
                        file_path = row[2] or ""
                        content = row[3] or ""
                        signature = row[4] or ""
                        label_prefix = node_id.split(":", 1)[0] if node_id else ""
                        snippet = content[:200] if content else signature[:200]
                        score = 2.0 if "/tests/" not in file_path else 1.0
                        candidates.append(
                            SearchResult(
                                node_id=node_id,
                                score=score,
                                node_name=node_name,
                                file_path=file_path,
                                label=label_prefix,
                                snippet=snippet,
                            )
                        )
                except Exception:
                    logger.debug("exact_name_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def fts_search(self, query: str, limit: int) -> list[SearchResult]:
        """BM25 full-text search using KuzuDB's native FTS extension.

        Searches across all node tables using pre-built FTS indexes on
        ``name``, ``content``, and ``signature`` fields.  Results are
        ranked by BM25 relevance score.

        Returns the top *limit* results sorted by score descending.
        """
        limit = max(1, int(limit))
        escaped_q = _escape(query)
        candidates: list[SearchResult] = []

        with self._read_conn() as conn:
            for table in _SEARCHABLE_TABLES:
                idx_name = f"{table.lower()}_fts"
                cypher = (
                    f"CALL QUERY_FTS_INDEX('{table}', '{idx_name}', '{escaped_q}') "
                    f"RETURN node.id, node.name, node.file_path, node.content, "
                    f"node.signature, score "
                    f"ORDER BY score DESC LIMIT {limit}"
                )
                try:
                    for row in self._drain(conn.execute(cypher)):
                        node_id = row[0] or ""
                        name = row[1] or ""
                        file_path = row[2] or ""
                        content = row[3] or ""
                        signature = row[4] or ""
                        bm25_score = float(row[5]) if row[5] is not None else 0.0

                        # Demote test file results — mirrors exact_name_search penalty.
                        if "/tests/" in file_path or "/test_" in file_path:
                            bm25_score *= 0.5

                        label_prefix = node_id.split(":", 1)[0] if node_id else ""

                        # Boost top-level definitions in source files.
                        if label_prefix in ("function", "class") and "/tests/" not in file_path:
                            bm25_score *= 1.2

                        snippet = content[:200] if content else signature[:200]

                        candidates.append(
                            SearchResult(
                                node_id=node_id,
                                score=bm25_score,
                                node_name=name,
                                file_path=file_path,
                                label=label_prefix,
                                snippet=snippet,
                            )
                        )
                except Exception:
                    logger.debug("fts_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def fuzzy_search(
        self, query: str, limit: int, max_distance: int = 2
    ) -> list[SearchResult]:
        """Fuzzy name search using Levenshtein edit distance.

        Scans all node tables for symbols whose name is within
        *max_distance* edits of *query*.  Converts edit distance to a
        score (0 edits = 1.0, *max_distance* edits = 0.3).
        """
        limit = max(1, int(limit))
        max_distance = int(max_distance)
        escaped_q = _escape(query.lower())
        candidates: list[SearchResult] = []

        with self._read_conn() as conn:
            for table in _SEARCHABLE_TABLES:
                cypher = (
                    f"MATCH (n:{table}) "
                    f"WHERE levenshtein(lower(n.name), '{escaped_q}') <= {max_distance} "
                    f"RETURN n.id, n.name, n.file_path, n.content, "
                    f"levenshtein(lower(n.name), '{escaped_q}') AS dist "
                    f"ORDER BY dist LIMIT {limit}"
                )
                try:
                    for row in self._drain(conn.execute(cypher)):
                        node_id = row[0] or ""
                        name = row[1] or ""
                        file_path = row[2] or ""
                        content = row[3] or ""
                        dist = int(row[4]) if row[4] is not None else max_distance

                        score = max(0.3, 1.0 - (dist * 0.3))
                        label_prefix = node_id.split(":", 1)[0] if node_id else ""

                        candidates.append(
                            SearchResult(
                                node_id=node_id,
                                score=score,
                                node_name=name,
                                file_path=file_path,
                                label=label_prefix,
                                snippet=content[:200] if content else "",
                            )
                        )
                except Exception:
                    logger.debug("fuzzy_search failed on table %s", table, exc_info=True)

        candidates.sort(key=lambda r: (-r.score, r.node_id))
        return candidates[:limit]

    def store_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """Persist embedding vectors and build the HNSW vector index.

        Attempts batch CSV COPY FROM first, falls back to individual MERGE.
        """
        assert self._conn is not None
        if not embeddings:
            return

        # The HNSW index pins the table: DROP TABLE (bulk path) and SET on
        # the indexed column (fallback path) both fail while it exists.
        self._drop_vector_index()

        if not self._bulk_store_embeddings_csv(embeddings):
            dim = len(embeddings[0].embedding)
            for emb in embeddings:
                try:
                    self._conn.execute(
                        "MERGE (e:Embedding {node_id: $nid}) "
                        f"SET e.vec = CAST($vec AS FLOAT[{dim}])",
                        parameters={"nid": emb.node_id, "vec": emb.embedding},
                    )
                except Exception:
                    logger.debug(
                        "store_embeddings failed for node %s", emb.node_id, exc_info=True
                    )

        self._create_vector_index()

    def _create_vector_index(self) -> None:
        """(Re)build the HNSW index over Embedding.vec.

        Failure is non-fatal — :meth:`vector_search` falls back to a full
        cosine-similarity scan when the index is missing.
        """
        assert self._conn is not None
        try:
            self._conn.execute("LOAD EXTENSION vector")
        except Exception:
            pass  # statically linked or already loaded
        self._drop_vector_index()
        try:
            self._conn.execute(
                f"CALL CREATE_VECTOR_INDEX('Embedding', '{_VECTOR_INDEX_NAME}', "
                f"'vec', metric := 'cosine')"
            )
        except Exception:
            logger.warning(
                "Vector index creation failed — semantic search will use a full scan",
                exc_info=True,
            )

    def _drop_vector_index(self) -> None:
        """Drop the HNSW index if present (no-op when absent)."""
        assert self._conn is not None
        try:
            self._conn.execute(
                f"CALL DROP_VECTOR_INDEX('Embedding', '{_VECTOR_INDEX_NAME}')"
            )
        except Exception:
            pass

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        """Find the closest nodes to *vector* via the HNSW vector index.

        Falls back to a full ``array_cosine_similarity`` scan when the index
        is unavailable (pre-index database or failed index build).  Joins
        with node tables to fetch metadata in a single query.
        """
        limit = max(1, int(limit))
        # Vector literals must be inlined — KuzuDB cannot bind a parameter
        # in the index-function argument position, nor distinguish DOUBLE[]
        # from LIST for array_cosine_similarity.
        vec_literal = "[" + ", ".join(str(float(v)) for v in vector) + "]"
        dim = len(vector)

        with self._read_conn() as conn:
            emb_rows = self._vector_index_query(conn, vec_literal, dim, limit)
            if emb_rows is None:
                emb_rows = self._vector_scan_query(conn, vec_literal, dim, limit)
            if not emb_rows:
                return []

            node_cache: dict[str, GraphNode] = {}
            node_ids = [r[0] for r in emb_rows]
            ids_by_table: dict[str, list[str]] = {}
            for nid in node_ids:
                table = _table_for_id(nid)
                if table:
                    ids_by_table.setdefault(table, []).append(nid)

            for table, ids in ids_by_table.items():
                try:
                    q = f"MATCH (n:{table}) WHERE n.id IN $ids RETURN {_node_columns('n')}"
                    for row in self._drain(conn.execute(q, parameters={"ids": ids})):
                        node = self._row_to_node(row)
                        if node:
                            node_cache[node.id] = node
                except Exception:
                    logger.debug("Batch node fetch failed for table %s", table, exc_info=True)

        results: list[SearchResult] = []
        for node_id, sim in emb_rows:
            node = node_cache.get(node_id)
            if node is None:
                # Orphaned embedding — its node was deleted. Skip rather than
                # returning a ghost result with empty metadata.
                continue
            label_prefix = node_id.split(":", 1)[0] if node_id else ""
            results.append(
                SearchResult(
                    node_id=node_id,
                    score=sim,
                    node_name=node.name,
                    file_path=node.file_path,
                    label=label_prefix,
                    snippet=(node.content[:200] if node.content else ""),
                )
            )
        return results

    @classmethod
    def _vector_index_query(
        cls, conn: kuzu.Connection, vec_literal: str, dim: int, limit: int
    ) -> list[tuple[str, float]] | None:
        """K-nearest via the HNSW index; ``None`` when the index is unavailable.

        Cosine *distance* from the index is converted to similarity
        (``1 - distance``) so scores match the full-scan fallback.
        """
        query = (
            f"CALL QUERY_VECTOR_INDEX('Embedding', '{_VECTOR_INDEX_NAME}', "
            f"CAST({vec_literal} AS FLOAT[{dim}]), {limit}) "
            f"RETURN node.node_id, distance ORDER BY distance"
        )
        try:
            rows = cls._drain(conn.execute(query))
        except Exception:
            logger.debug("Vector index unavailable, falling back to scan", exc_info=True)
            return None
        return [
            (row[0] or "", 1.0 - float(row[1]) if row[1] is not None else 0.0)
            for row in rows
        ]

    @classmethod
    def _vector_scan_query(
        cls, conn: kuzu.Connection, vec_literal: str, dim: int, limit: int
    ) -> list[tuple[str, float]]:
        """Full-scan cosine similarity over all embeddings.

        Tries the plain literal first (matches the legacy ``DOUBLE[]``
        column), then a ``FLOAT[dim]`` cast (new column type without an
        index, e.g. after a failed index build).
        """
        for vec_expr in (vec_literal, f"CAST({vec_literal} AS FLOAT[{dim}])"):
            query = (
                f"MATCH (e:Embedding) "
                f"RETURN e.node_id, "
                f"array_cosine_similarity(e.vec, {vec_expr}) AS sim "
                f"ORDER BY sim DESC LIMIT {limit}"
            )
            try:
                rows = cls._drain(conn.execute(query))
            except Exception:
                logger.debug("vector scan failed for %s", vec_expr[:40], exc_info=True)
                continue
            return [
                (row[0] or "", float(row[1]) if row[1] is not None else 0.0)
                for row in rows
            ]
        return []

    def get_indexed_files(self) -> dict[str, str]:
        """Return ``{file_path: sha256(content)}`` for all File nodes."""
        mapping: dict[str, str] = {}
        with self._read_conn() as conn:
            try:
                rows = self._drain(conn.execute(
                    "MATCH (n:File) RETURN n.file_path, n.content"
                ))
                for row in rows:
                    fp = row[0] or ""
                    content = row[1] or ""
                    mapping[fp] = hashlib.sha256(content.encode()).hexdigest()
            except Exception:
                logger.debug("get_indexed_files failed", exc_info=True)
        return mapping

    def _reset_database(self) -> None:
        """Close, delete, and reinitialize the database from scratch.

        This avoids ``MATCH (n) DETACH DELETE n`` which triggers a segfault
        in Kuzu's native layer on large datasets (2000+ files).

        Waits for in-flight reads to drain first: a read whose dispatch
        timed out keeps running in its thread after the RW lock is released,
        and deleting the database files under a live native query risks a
        crash in Kuzu's native layer.
        """
        assert self._db_path is not None
        db_path = self._db_path
        self._wait_for_readers()
        self.close()
        if db_path.exists():
            if db_path.is_dir():
                shutil.rmtree(db_path)
            else:
                db_path.unlink()
        self.initialize(db_path)

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """Replace the entire store with the contents of *graph*.

        Uses CSV-based COPY FROM for bulk loading nodes and relationships,
        falling back to individual inserts if COPY FROM fails.

        Recreates the database from scratch to avoid a Kuzu native segfault
        on large ``DETACH DELETE`` operations.
        """
        self._reset_database()
        assert self._conn is not None

        if not self._bulk_load_nodes_csv(graph):
            self.add_nodes(list(graph.iter_nodes()))

        if not self._bulk_load_rels_csv(graph):
            self.add_relationships(list(graph.iter_relationships()))

        self.rebuild_fts_indexes()

    def rebuild_fts_indexes(self) -> None:
        """Drop and recreate all FTS indexes.

        Must be called after any bulk data change so the BM25 indexes
        reflect the current node contents.
        """
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            try:
                self._conn.execute(f"CALL DROP_FTS_INDEX('{table}', '{idx_name}')")
            except Exception:
                pass
            try:
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])"
                )
            except Exception:
                logger.debug("FTS index rebuild failed for %s", table, exc_info=True)

    def _csv_copy(self, table: str, rows: list[list[Any]]) -> None:
        """Write *rows* to a temporary CSV and COPY FROM into *table*.

        Always cleans up the temp file, even on failure.
        """
        assert self._conn is not None
        csv_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                csv_path = f.name
            self._conn.execute(f'COPY {table} FROM "{csv_path}" (HEADER=false)')
        finally:
            if csv_path:
                Path(csv_path).unlink(missing_ok=True)

    def _bulk_load_nodes_csv(self, graph: KnowledgeGraph) -> bool:
        """Load all nodes via temporary CSV files + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        by_table: dict[str, list[GraphNode]] = {}
        for node in graph.iter_nodes():
            table = _LABEL_TO_TABLE.get(node.label.value)
            if table:
                by_table.setdefault(table, []).append(node)

        # Deduplicate by node.id within each table, keeping the last occurrence.
        for table, nodes in by_table.items():
            seen: dict[str, int] = {}
            for i, node in enumerate(nodes):
                seen[node.id] = i
            if len(seen) < len(nodes):
                logger.debug(
                    "Deduplicated %d duplicate node(s) in table %s",
                    len(nodes) - len(seen),
                    table,
                )
                by_table[table] = [nodes[i] for i in sorted(seen.values())]

        try:
            for table, nodes in by_table.items():
                self._csv_copy(table, [
                    [node.id, node.name, node.file_path, node.start_line,
                     node.end_line, node.content, node.signature, node.language,
                     node.class_name, node.is_dead, node.is_entry_point,
                     node.is_exported, _serialize_properties(node.properties)]
                    for node in nodes
                ])
            return True
        except Exception:
            logger.debug("CSV bulk_load_nodes failed, falling back", exc_info=True)
            return False

    def _bulk_load_rels_csv(self, graph: KnowledgeGraph) -> bool:
        """Load all relationships via temporary CSV files + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        by_pair: dict[tuple[str, str], list[GraphRelationship]] = {}
        for rel in graph.iter_relationships():
            src_table = _table_for_id(rel.source)
            dst_table = _table_for_id(rel.target)
            if src_table and dst_table:
                by_pair.setdefault((src_table, dst_table), []).append(rel)

        # Deduplicate by full edge identity — including role and step_number,
        # so e.g. USES_TYPE edges with different roles between the same pair
        # survive (mirrors the in-memory relationship ID semantics).
        for pair_key, rels in by_pair.items():
            seen: dict[tuple[str, str, str, str, int], int] = {}
            for i, rel in enumerate(rels):
                props = rel.properties or {}
                key = (
                    rel.source,
                    rel.target,
                    rel.type.value,
                    str(props.get("role", "")),
                    int(props.get("step_number", 0)),
                )
                seen[key] = i
            if len(seen) < len(rels):
                logger.debug(
                    "Deduplicated %d duplicate rel(s) in %s->%s",
                    len(rels) - len(seen),
                    pair_key[0],
                    pair_key[1],
                )
                by_pair[pair_key] = [rels[i] for i in sorted(seen.values())]

        try:
            for (src_table, dst_table), rels in by_pair.items():
                self._csv_copy(f"CodeRelation_{src_table}_{dst_table}", [
                    [rel.source, rel.target, rel.type.value,
                     float((rel.properties or {}).get("confidence", 1.0)),
                     str((rel.properties or {}).get("role", "")),
                     int((rel.properties or {}).get("step_number", 0)),
                     float((rel.properties or {}).get("strength", 0.0)),
                     int((rel.properties or {}).get("co_changes", 0)),
                     str((rel.properties or {}).get("symbols", ""))]
                    for rel in rels
                ])
            return True
        except Exception:
            logger.debug("CSV bulk_load_rels failed, falling back", exc_info=True)
            return False

    def _bulk_store_embeddings_csv(self, embeddings: list[NodeEmbedding]) -> bool:
        """Store embeddings via temporary CSV + COPY FROM.

        Returns True on success, False if COPY FROM is not available.
        """
        assert self._conn is not None
        try:
            try:
                self._conn.execute("DROP TABLE Embedding")
            except Exception:
                pass
            # Recreate at the actual vector width so the FLOAT[dim] column
            # (required by the HNSW index) matches whatever model produced
            # these embeddings.
            self._conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS "
                f"Embedding({_embedding_ddl(len(embeddings[0].embedding))})"
            )

            self._csv_copy("Embedding", [
                [emb.node_id,
                 "[" + ",".join(str(v) for v in emb.embedding) + "]"]
                for emb in embeddings
            ])
            return True
        except Exception:
            logger.debug("CSV bulk_store_embeddings failed, falling back", exc_info=True)
            return False

    def _create_schema(self) -> None:
        """Create node/rel/embedding tables and the FTS extension."""
        assert self._conn is not None

        try:
            self._conn.execute("INSTALL fts")
            self._conn.execute("LOAD EXTENSION fts")
        except Exception:
            logger.debug("FTS extension load skipped (may already be loaded)", exc_info=True)

        try:
            self._conn.execute("INSTALL vector")
            self._conn.execute("LOAD EXTENSION vector")
        except Exception:
            logger.debug("Vector extension load skipped (may already be loaded)", exc_info=True)

        for table in _NODE_TABLE_NAMES:
            stmt = f"CREATE NODE TABLE IF NOT EXISTS {table}({_NODE_PROPERTIES})"
            self._conn.execute(stmt)

        # Bring pre-existing tables (no-ops for CREATE IF NOT EXISTS) up to
        # the current schema.
        self._migrate_schema()

        self._conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS Embedding({_embedding_ddl(EMBEDDING_DIM)})"
        )

        # Build the REL TABLE GROUP covering all table-to-table combinations.
        from_to_pairs: list[str] = []
        for src in _NODE_TABLE_NAMES:
            for dst in _NODE_TABLE_NAMES:
                from_to_pairs.append(f"FROM {src} TO {dst}")

        pairs_clause = ", ".join(from_to_pairs)
        rel_stmt = (
            f"CREATE REL TABLE GROUP IF NOT EXISTS CodeRelation("
            f"{pairs_clause}, {_REL_PROPERTIES})"
        )
        try:
            self._conn.execute(rel_stmt)
        except Exception:
            logger.debug("REL TABLE GROUP creation skipped", exc_info=True)

        self._create_fts_indexes()

    def _create_fts_indexes(self) -> None:
        """Create FTS indexes for every node table (idempotent)."""
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            idx_name = f"{table.lower()}_fts"
            try:
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])"
                )
            except Exception:
                # Index may already exist — that's fine.
                pass

    def _insert_node(self, node: GraphNode) -> None:
        """INSERT a single node into the appropriate label table using parameterized query."""
        assert self._conn is not None
        table = _LABEL_TO_TABLE.get(node.label.value)
        if table is None:
            logger.warning("Unknown label %s for node %s", node.label, node.id)
            return

        query = (
            f"CREATE (:{table} {{"
            f"id: $id, name: $name, file_path: $file_path, "
            f"start_line: $start_line, end_line: $end_line, "
            f"content: $content, signature: $signature, "
            f"language: $language, class_name: $class_name, "
            f"is_dead: $is_dead, is_entry_point: $is_entry_point, "
            f"is_exported: $is_exported, properties_json: $properties_json"
            f"}})"
        )
        params = {
            "id": node.id,
            "name": node.name,
            "file_path": node.file_path,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "content": node.content,
            "signature": node.signature,
            "language": node.language,
            "class_name": node.class_name,
            "is_dead": node.is_dead,
            "is_entry_point": node.is_entry_point,
            "is_exported": node.is_exported,
            "properties_json": _serialize_properties(node.properties),
        }
        try:
            self._conn.execute(query, parameters=params)
        except Exception:
            logger.debug("Insert node failed for %s", node.id, exc_info=True)

    def _insert_relationship(self, rel: GraphRelationship) -> None:
        """MATCH source and target, then CREATE the relationship using parameterized query."""
        assert self._conn is not None
        src_table = _table_for_id(rel.source)
        tgt_table = _table_for_id(rel.target)
        if src_table is None or tgt_table is None:
            logger.warning(
                "Cannot resolve tables for relationship %s -> %s",
                rel.source,
                rel.target,
            )
            return

        props = rel.properties or {}

        query = (
            f"MATCH (a:{src_table}), (b:{tgt_table}) "
            f"WHERE a.id = $src AND b.id = $tgt "
            f"CREATE (a)-[:CodeRelation {{"
            f"rel_type: $rel_type, "
            f"confidence: $confidence, "
            f"role: $role, "
            f"step_number: $step_number, "
            f"strength: $strength, "
            f"co_changes: $co_changes, "
            f"symbols: $symbols"
            f"}}]->(b)"
        )
        params = {
            "src": rel.source,
            "tgt": rel.target,
            "rel_type": rel.type.value,
            "confidence": float(props.get("confidence", 1.0)),
            "role": str(props.get("role", "")),
            "step_number": int(props.get("step_number", 0)),
            "strength": float(props.get("strength", 0.0)),
            "co_changes": int(props.get("co_changes", 0)),
            "symbols": str(props.get("symbols", "")),
        }
        try:
            self._conn.execute(query, parameters=params)
        except Exception:
            logger.debug(
                "Insert relationship failed: %s -> %s", rel.source, rel.target, exc_info=True
            )

    def _query_nodes(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[GraphNode]:
        """Execute a query returning the node column list and convert to GraphNodes."""
        nodes: list[GraphNode] = []
        with self._read_conn() as conn:
            try:
                for row in self._drain(conn.execute(query, parameters=parameters or {})):
                    node = self._row_to_node(row)
                    if node is not None:
                        nodes.append(node)
            except Exception:
                logger.debug("_query_nodes failed: %s", query, exc_info=True)
        return nodes

    @staticmethod
    def _row_to_node(row: list[Any], node_id: str | None = None) -> GraphNode | None:
        """Convert a result row (in ``_NODE_COLUMN_NAMES`` order) into a GraphNode.

        Column order:
        0=id, 1=name, 2=file_path, 3=start_line, 4=end_line,
        5=content, 6=signature, 7=language, 8=class_name,
        9=is_dead, 10=is_entry_point, 11=is_exported, 12=properties_json
        """
        try:
            nid = node_id or row[0]
            prefix = nid.split(":", 1)[0]
            label = _LABEL_MAP.get(prefix, NodeLabel.FILE)

            return GraphNode(
                id=row[0],
                label=label,
                name=row[1] or "",
                file_path=row[2] or "",
                start_line=row[3] or 0,
                end_line=row[4] or 0,
                content=row[5] or "",
                signature=row[6] or "",
                language=row[7] or "",
                class_name=row[8] or "",
                is_dead=bool(row[9]),
                is_entry_point=bool(row[10]),
                is_exported=bool(row[11]),
                properties=deserialize_properties(row[12]) if len(row) > 12 else {},
            )
        except (IndexError, KeyError):
            logger.debug("Failed to convert row to GraphNode: %s", row, exc_info=True)
            return None
