"""LadybugDB storage backend for Synaptiq.

Implements the :class:`StorageBackend` protocol using LadybugDB, the
actively-maintained embedded successor to the archived KuzuDB (owner
decision W2.7 — see ``docs/plans/2026-07-12-storage-successor-evaluation.md``).
LadybugDB is API drop-in for the former ``kuzu`` bindings and speaks the same
Cypher dialect. Each :class:`NodeLabel` maps to a separate node table, and a
single ``CodeRelation`` relationship table group covers all source-to-target
combinations.

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
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import ladybug

try:
    # LadybugDB's Arrow replacement scan (``COPY <table> FROM <local pa.Table>``)
    # resolves modules via ``importlib.import_module`` (see ladybug/_backend.py),
    # which is always available once ``importlib`` is imported — so the
    # ``import importlib.util`` shim that kuzu 0.11.3's native loader required is
    # unnecessary here and was dropped in W2.7 (verified against the engine).
    import pyarrow as pa

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover - exercised in CSV-only environments
    pa = None  # type: ignore[assignment]
    _HAS_PYARROW = False

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, GraphRelationship, NodeLabel
from synaptiq.core.ingestion.manifest import (
    Manifest,
    build_manifest,
    load_manifest_from_rows,
    serialize_file_manifest,
    serialize_index_manifest,
)
from synaptiq.core.storage.base import EdgeRef, GraphDelta, NodeEmbedding, SearchResult

logger = logging.getLogger(__name__)

_NODE_TABLE_NAMES: list[str] = [label.name.title().replace("_", "") for label in NodeLabel]

_LABEL_TO_TABLE: dict[str, str] = {
    label.value: label.name.title().replace("_", "") for label in NodeLabel
}

_LABEL_MAP: dict[str, NodeLabel] = {label.value: label for label in NodeLabel}

_SEARCHABLE_TABLES: list[str] = [
    t for t in _NODE_TABLE_NAMES if t not in ("Folder", "Community", "Process")
]

# Symbol tables whose nodes carry a meaningful ``is_dead`` flag — mirrors
# ``dead_code._SYMBOL_LABELS``. The scoped dead-code recount on the delta path
# only touches these; File/Folder/Module/… are never flagged.
_SYMBOL_TABLE_NAMES: frozenset[str] = frozenset(
    _LABEL_TO_TABLE[label.value]
    for label in (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)
)

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
# independent of the engine's internal property ordering.
_NODE_COLUMN_NAMES: tuple[str, ...] = (
    "id",
    "name",
    "file_path",
    "start_line",
    "end_line",
    "content",
    "signature",
    "language",
    "class_name",
    "is_dead",
    "is_entry_point",
    "is_exported",
    "properties_json",
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

    Internal only — used for FTS/fuzzy search literals that the engine cannot
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


def _arrow_str(value: str | None) -> str | None:
    """Coerce empty/``None`` strings to ``None`` for the Arrow COPY path.

    LadybugDB's CSV reader stores an empty field as ``NULL``, so the CSV bulk
    path already persists ``""`` as ``NULL``. Mirroring that here keeps the Arrow and
    CSV loaders byte-identical: a non-empty string is stored verbatim, while
    ``""``/``None`` land as ``NULL`` in both paths.
    """
    return value if value else None


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


# Embedding vectors use a fixed-dimension FLOAT column so LadybugDB's HNSW
# vector index can be built on them; 384 matches BAAI/bge-small-en-v1.5.
# The bulk store path recreates the table from the actual embedding width,
# so this constant only shapes the empty table created at schema time.
EMBEDDING_DIM = 384

_VECTOR_INDEX_NAME = "embedding_vec_idx"


def _embedding_ddl(dim: int) -> str:
    """Embedding table column DDL for *dim*-wide vectors.

    ``text_sha`` records the hash of the text each vector was generated
    from, so rebuilds can reuse vectors for unchanged symbols.
    """
    return f"node_id STRING, vec FLOAT[{dim}], text_sha STRING, PRIMARY KEY(node_id)"


# Incremental-indexing manifest tables (W3.2a). Two standalone node tables that
# live INSIDE the index DB (design D3, spike-confirmed) so the manifest rides the
# crash-safe ``.rebuild`` swap and ``open_with_recovery`` for free. Their names
# are deliberately NOT in ``NodeLabel`` (hence not in ``_NODE_TABLE_NAMES``), so
# ``_migrate_schema`` / ``_verify_schema`` / FTS / the ``CodeRelation`` rel group
# never touch them and ``load_graph`` never sees them. Column order is
# contractual with ``core.ingestion.manifest`` (FILE_MANIFEST_COLUMNS + the
# read queries below) — keep them in lockstep.
_MANIFEST_FILE_TABLE = "FileManifest"
_MANIFEST_INDEX_TABLE = "IndexManifest"
_MANIFEST_SINGLETON_ID = "singleton"
_FILE_MANIFEST_DDL = (
    f"CREATE NODE TABLE IF NOT EXISTS {_MANIFEST_FILE_TABLE}("
    "path STRING, content_sha STRING, language STRING, "
    "symbol_ids STRING, symbol_sigs STRING, out_edges STRING, "
    "PRIMARY KEY(path))"
)
_INDEX_MANIFEST_DDL = (
    f"CREATE NODE TABLE IF NOT EXISTS {_MANIFEST_INDEX_TABLE}("
    "id STRING, manifest_version INT64, data STRING, PRIMARY KEY(id))"
)


# Maximum number of read connections to keep in the pool.
_MAX_POOL_SIZE = 8


def is_lock_error(exc: BaseException) -> bool:
    """True for a LadybugDB file-lock conflict — another live process owns the DB.

    A lock conflict is **not** corruption: the index must be left untouched and
    the error propagated so the caller's lock handling and the daemon
    primary/proxy hand-off still work.  Shared by :func:`open_with_recovery`,
    the CLI, and the MCP server so all three classify a lock the same way
    (previously three subtly divergent predicates).

    Matches the verified messages ``"Could not set lock on file ..."`` and
    ``"Lock is held by PID ..."``, with a broad ``"lock"`` fallback so a
    reworded lock message is never misread as corruption (fail-safe: keep the
    index rather than wipe it).
    """
    msg = str(exc).lower()
    return "lock on file" in msg or "lock is held" in msg or "lock" in msg


# Verified LadybugDB messages for a derived index that is genuinely unreadable
# and safe to wipe + rebuild.  Kept as a strict ALLOWLIST (see
# :func:`is_recoverable_corruption`) so an *unrecognized* failure never
# destroys a still-good index.
_RECOVERABLE_CORRUPTION_SIGNATURES = (
    "not a valid lbug database file",  # corrupt/partial file from a mid-write kill
    "database path cannot be a directory",  # stale KuzuDB-format index directory
)


def is_recoverable_corruption(exc: BaseException) -> bool:
    """True when *exc* is a verified "the derived index is garbage" signature.

    :func:`open_with_recovery` may heal these by wiping the ``.synaptiq`` index
    (a rebuildable artifact) and reindexing from source.  This is a strict
    ALLOWLIST — anything not listed (a schema mismatch from an older synaptiq,
    OOM, disk quota, a transient native error) returns ``False`` and must be
    re-raised, so a recoverable-but-unrecognized failure never wipes a good
    index (review F1/F15).

    Recoverable cases (verified against LadybugDB):

    * ``IndexError`` from open — the stale-WAL / partially written read, where
      the native layer's ``unordered_map::at`` surfaces as ``IndexError``.
    * ``"not a valid Lbug database file"`` — corruption from a mid-write kill.
    * ``"Database path cannot be a directory"`` — an index directory written by
      the former KuzuDB backend (LadybugDB uses a single-file format).
    """
    if isinstance(exc, IndexError):
        return True
    msg = str(exc).lower()
    return any(sig in msg for sig in _RECOVERABLE_CORRUPTION_SIGNATURES)


def open_with_recovery(
    db_path: Path,
    meta_path: Path | None = None,
    *,
    read_only: bool = False,
    build_fts_indexes: bool = True,
) -> LadybugBackend:
    """Open a LadybugBackend at *db_path*, rebuilding an unreadable index.

    The ``.synaptiq`` index is a rebuildable derived artifact, so any index
    that cannot be opened is deleted (along with *meta_path*) and the next
    ``synaptiq analyze`` rebuilds it from source.  This covers:

    * corruption from a mid-write kill (duplicate primary key, partial file),
    * a stale-WAL / partially written read (``unordered_map::at`` → IndexError),
    * an index directory left behind by the former **KuzuDB** backend —
      LadybugDB uses a single-file on-disk format and rejects a directory path
      with "Database path cannot be a directory", so an upgraded install
      transparently reindexes instead of crashing.

    Only the verified-corruption signatures above (see
    :func:`is_recoverable_corruption`) trigger a wipe.  **Anything else** — a
    schema mismatch from an older synaptiq (read-only opens raise a clear
    "created by an older version" error), OOM, disk quota, a transient native
    error — is re-raised **unchanged with the index left untouched**, so an
    unrecognized failure never destroys a still-good index (review F1/F15).

    After a legitimate corruption wipe: in read-write mode the empty database
    is re-initialised and returned; in **read-only** mode there is nothing left
    to open, so a clear ``RuntimeError`` is raised (rather than returning a bare
    backend that would silently serve an empty graph) directing the caller to
    ``synaptiq analyze``.

    A **lock conflict** (another process holds the database) is not corruption:
    it propagates unchanged so the caller's lock handling and the daemon
    primary/proxy hand-off still work.

    Args:
        build_fts_indexes: Forwarded to :meth:`LadybugBackend.initialize` for
            the read-write path.  ``analyze`` passes ``False`` because it
            immediately ``bulk_load``s — which builds the FTS indexes over the
            populated tables and swaps the fresh database in — making the
            empty-schema FTS build the live open would otherwise do pure waste
            (~2s on this engine).
    """
    storage = LadybugBackend()
    try:
        storage.initialize(db_path, read_only=read_only, _build_fts_indexes=build_fts_indexes)
        return storage
    except (RuntimeError, IndexError) as exc:
        # Release any handle initialize opened before it failed (a read-only
        # schema check raises after the connection is live). Safe on a
        # fresh/partial backend; never deletes files.
        storage.close()
        # Lock conflict → another live process owns the DB; propagate so the
        # caller's lock handling / daemon primary-proxy hand-off runs.
        if is_lock_error(exc):
            raise
        # Only verified-corruption signatures are healed by wiping the derived
        # index. ANYTHING ELSE (schema mismatch, OOM, disk quota, a transient
        # native error) is re-raised UNCHANGED, leaving the index untouched, so
        # an unrecognized failure never destroys a still-good index (F1/F15).
        if not is_recoverable_corruption(exc):
            raise
        # Fall through: a verified-corruption signature — wipe + rebuild.

    logger.warning("Unreadable index at %s — removing it and scheduling a rebuild", db_path)
    LadybugBackend._remove_db_files(db_path)
    if meta_path is not None:
        meta_path.unlink(missing_ok=True)

    if read_only:
        # Nothing left to open after a corruption wipe. Returning a bare,
        # uninitialised backend would silently serve an empty graph (MCP tools
        # answering as if the repo had no code); fail loudly instead so the
        # operator rebuilds.
        raise RuntimeError(
            f"Index at {db_path} was corrupt and has been removed; "
            "run `synaptiq analyze` to rebuild it"
        )

    storage = LadybugBackend()
    storage.initialize(db_path, _build_fts_indexes=build_fts_indexes)
    return storage


class LadybugBackend:
    """StorageBackend implementation backed by LadybugDB.

    Usage::

        backend = LadybugBackend()
        backend.initialize(Path("/tmp/synaptiq_db"))
        backend.bulk_load(graph)
        node = backend.get_node("function:src/app.py:main")
        backend.close()
    """

    def __init__(self) -> None:
        self._db: ladybug.Database | None = None
        self._conn: ladybug.Connection | None = None
        # Prepared statements for the write connection, keyed by node label
        # or source/target rel-table pair. Reused across the rows of a batch
        # so each INSERT is planned once instead of re-planned per row. Bound
        # to the current ``self._conn`` — cleared whenever it is (re)created
        # or closed (see :meth:`initialize` / :meth:`close`).
        self._prepared: dict[str, Any] = {}
        self._db_path: Path | None = None
        # Thread-safe pool of read connections, tagged with the database
        # generation they were created against.  The generation increments
        # on every (re)initialize so connections bound to a closed/deleted
        # database are never handed out again.
        self._read_pool: list[tuple[int, ladybug.Connection]] = []
        self._pool_lock = threading.Lock()
        self._generation = 0
        # In-flight read tracking so destructive operations (the bulk_load swap)
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

    @property
    def data_dir(self) -> Path | None:
        """The ``.synaptiq/`` directory this backend's database lives in.

        ``None`` before :meth:`initialize` has been called. Duck-typed (not
        part of the :class:`~synaptiq.core.storage.base.StorageBackend`
        Protocol — a URI-addressed backend like Neo4j has no natural
        filesystem directory) rather than declared on the Protocol, so
        callers that want it (e.g. ``mcp.tools._get_query_embedding``,
        resolving which embedding tier ``meta.json`` next to the database
        says this index was built with) read it via ``getattr(storage,
        "data_dir", None)``.
        """
        return self._db_path.parent if self._db_path is not None else None

    def initialize(
        self, path: Path, *, read_only: bool = False, _build_fts_indexes: bool = True
    ) -> None:
        """Open or create the LadybugDB database at *path* and set up the schema.

        Args:
            path: Filesystem path to the LadybugDB database file (LadybugDB
                uses a single-file on-disk format, unlike the former KuzuDB
                directory layout — :func:`open_with_recovery` transparently
                rebuilds an index left behind by the old directory format).
            read_only: If ``True``, open the database in read-only mode.
                This allows multiple concurrent readers (e.g. MCP server
                instances) without lock conflicts.  Schema creation is
                skipped since the database must already exist — but the
                schema is verified so a database created by an older
                synaptiq fails loudly instead of silently returning
                empty results for every query.
            _build_fts_indexes: Internal. When ``False``, skip building the
                (empty) FTS indexes during schema creation — used by
                :meth:`bulk_load` for its ``.rebuild`` database, which calls
                :meth:`rebuild_fts_indexes` right after the COPY to build them
                over the populated tables. Building them on empty tables first
                is pure waste (~25% of the storage-load time). The default
                ``True`` preserves behaviour for every other caller, so a
                database queried before any bulk_load still has its indexes.
        """
        from synaptiq.core.resources import current_limits

        limits = current_limits()
        self._db_path = path
        # 0 for either cap means the engine's library default (all cores /
        # default buffer pool) — the interactive profile resolves to that.
        self._db = ladybug.Database(
            str(path),
            read_only=read_only,
            max_num_threads=limits.db_threads,
            buffer_pool_size=limits.db_buffer_bytes,
        )
        self._conn = ladybug.Connection(self._db)
        # Fresh write connection — any statements prepared against a prior
        # one are invalid, so start with an empty cache.
        self._prepared = {}
        with self._pool_lock:
            self._generation += 1
        if not read_only:
            self._create_schema(build_fts=_build_fts_indexes)
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
        """Call ``close()`` on a LadybugDB object, ignoring errors."""
        try:
            obj.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    def close(self) -> None:
        """Release all connections and the database handle.

        Uses the explicit ``close()`` methods on LadybugDB connections and the
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
        # Prepared statements are bound to the now-closed connection.
        self._prepared = {}
        if self._db is not None:
            self._close_quietly(self._db)
            self._db = None

    # ------------------------------------------------------------------
    # Connection pool for concurrent reads
    # ------------------------------------------------------------------

    def _acquire_read_conn(self) -> tuple[int, ladybug.Connection]:
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
        return gen, ladybug.Connection(db)

    def _release_read_conn(self, gen: int, conn: ladybug.Connection) -> None:
        """Return a connection to the pool, or close it if stale."""
        with self._pool_lock:
            if gen == self._generation and len(self._read_pool) < _MAX_POOL_SIZE:
                self._read_pool.append((gen, conn))
                return
        self._close_quietly(conn)

    @contextmanager
    def _read_conn(self) -> Iterator[ladybug.Connection]:
        """Context manager for read connections from the pool.

        Tracks in-flight reads so the ``bulk_load`` swap can wait for
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
        """Insert nodes into their respective label tables.

        The whole batch runs inside a single explicit transaction and
        reuses a prepared statement per node label, so each row is planned
        once and the batch is committed once — rather than the previous
        per-row auto-commit + re-plan. Inserts are idempotent upserts
        (``MERGE`` keyed on ``id``; see :meth:`_insert_node`), so re-adding an
        existing node refreshes it instead of failing. The batch is atomic: if
        an insert fails unexpectedly, the transaction is rolled back (leaving
        the database exactly as it was before the batch) and the error is
        re-raised.
        """
        self._write_batch(nodes, self._insert_node)

    def add_relationships(self, rels: list[GraphRelationship]) -> None:
        """Insert relationships by matching source and target nodes.

        Batched in one transaction with a prepared statement reused per
        source/target table pair; atomic and error-surfacing exactly like
        :meth:`add_nodes`.
        """
        self._write_batch(rels, self._insert_relationship)

    def _write_batch(self, items: list[Any], insert_one: Callable[[Any], None]) -> None:
        """Run ``insert_one`` over ``items`` inside one explicit transaction.

        On any error the transaction is rolled back and the original
        exception re-raised, so a mid-batch failure leaves the database
        untouched. An empty batch is a no-op (no transaction is opened).

        Writes use the single ``self._conn`` handle and callers serialize
        access through the external ``AsyncRWLock``, so no extra locking is
        needed here.
        """
        if not items:
            return
        conn = self._conn
        assert conn is not None
        conn.execute("BEGIN TRANSACTION")
        try:
            for item in items:
                insert_one(item)
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_quietly()
            raise

    def _rollback_quietly(self) -> None:
        """Roll back the active write transaction, tolerating its absence.

        On LadybugDB a statement error inside a transaction leaves the
        transaction open (it does not auto-abort), so this explicit
        ``ROLLBACK`` succeeds and reverts the batch — verified against the
        engine. The ``except`` still guards the case where there is no active
        transaction (e.g. the failure happened before ``BEGIN``), and it also
        absorbs the former KuzuDB behaviour where a statement error auto-aborted
        the transaction and ``ROLLBACK`` then raised "No active transaction".
        Either way the database ends up consistent.
        """
        if self._conn is None:
            return
        try:
            self._conn.execute("ROLLBACK")
        except Exception:
            pass

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

    def remove_nodes_by_id(self, node_ids: list[str]) -> int:
        """Surgically delete only the named nodes (by exact ``id``), any table.

        The scoped counterpart to :meth:`remove_nodes_by_file`. The incremental
        delta path removes *only the symbols that genuinely disappeared* from a
        changed file rather than DETACH-DELETEing the file's whole node set, so
        symbols that survived an edit keep their inbound edges from unchanged
        files — e.g. a ``C -> A.foo`` CALLS edge survives an edit to ``A`` that
        leaves ``A.foo`` in place. This is the remedy for the scoped-reindex
        inbound-edge-loss latent bug (``remove_nodes_by_file`` DETACH-DELETEs a
        surviving symbol together with its cross-file inbound edges, which the
        outbound-only re-insert never restores). DETACH-DELETEing a *genuinely*
        removed symbol still correctly cascades its own now-dangling inbound
        edges. Embedding rows for the removed ids are cleaned up so vector
        search does not return ghosts.

        Transaction-agnostic: runs on ``self._conn`` directly, so it
        auto-commits when called standalone and participates in the caller's
        transaction when invoked from :meth:`apply_graph_delta`.

        Returns:
            The number of nodes actually removed (ids that existed).
        """
        if not node_ids:
            return 0
        removed = self._count_nodes_by_id(node_ids)
        self._delete_nodes_by_id(node_ids)
        return removed

    def _count_nodes_by_id(self, node_ids: list[str]) -> int:
        """Count how many of *node_ids* currently exist across all node tables."""
        assert self._conn is not None
        total = 0
        for table in _NODE_TABLE_NAMES:
            try:
                rows = self._drain(
                    self._conn.execute(
                        f"MATCH (n:{table}) WHERE n.id IN $ids RETURN count(n)",
                        parameters={"ids": node_ids},
                    )
                )
                if rows:
                    total += int(rows[0][0])
            except Exception:
                logger.debug("count-by-id failed for table %s", table, exc_info=True)
        return total

    def _delete_nodes_by_id(self, node_ids: list[str]) -> None:
        """DETACH DELETE nodes by exact id + embedding cleanup. No own transaction.

        Runs on ``self._conn`` so it can be called standalone (auto-commit) or
        inside :meth:`apply_graph_delta`'s explicit transaction. Each per-table
        statement matches only that table's own ids (ids are unique by label
        prefix), so looping every table is safe.
        """
        if not node_ids:
            return
        assert self._conn is not None
        for table in _NODE_TABLE_NAMES:
            try:
                self._conn.execute(
                    f"MATCH (n:{table}) WHERE n.id IN $ids DETACH DELETE n",
                    parameters={"ids": node_ids},
                )
            except Exception:
                logger.debug("Failed to remove nodes by id from table %s", table, exc_info=True)

        try:
            self._conn.execute(
                "MATCH (e:Embedding) WHERE e.node_id IN $ids DELETE e",
                parameters={"ids": node_ids},
            )
        except Exception:
            logger.debug("Failed to remove embeddings by id", exc_info=True)

    def apply_graph_delta(self, delta: GraphDelta) -> None:
        """Apply a scoped :class:`GraphDelta` atomically (incremental design §6.1).

        Everything runs in ONE explicit transaction — reusing :meth:`_write_batch`'s
        ``BEGIN`` / ``COMMIT`` / :meth:`_rollback_quietly` discipline — so a
        mid-delta failure leaves the database exactly as it was. Applied in order:

          1. ``edges_remove`` — delete exactly the edges the re-resolved files
             previously contributed (idempotency without a global DETACH DELETE).
          2. ``nodes_remove`` — surgical by-id removal of genuinely-removed
             symbols (surviving symbols keep their cross-file inbound edges).
          3. ``nodes_upsert`` — idempotent ``MERGE`` node upsert (a body-only
             edit becomes a property refresh).
          4. ``edges_add`` — idempotent ``MERGE`` edge insert (no duplicates).
          5. ``dead_recount`` — recompute ``is_dead`` locally for affected
             symbols, reusing ``process_dead_code``'s exemption predicate
             (imported, not forked): dead iff zero incoming CALLS and not exempt.

        FTS, the HNSW vector index, communities and processes are intentionally
        left stale here and reconciled at consolidation — the same deferral
        precedent as ``apply_reindex`` (W1.2); no rebuild happens on this path.
        The dead-recount reads run on the write connection so they observe the
        edge changes just applied within this same transaction.
        """
        assert self._conn is not None
        if not (
            delta.edges_remove
            or delta.nodes_remove
            or delta.nodes_upsert
            or delta.edges_add
            or delta.dead_recount
        ):
            return

        conn = self._conn
        conn.execute("BEGIN TRANSACTION")
        try:
            for ref in delta.edges_remove:
                self._delete_edge(ref)
            self._delete_nodes_by_id(delta.nodes_remove)
            for node in delta.nodes_upsert:
                self._insert_node(node)
            for rel in delta.edges_add:
                self._insert_relationship(rel)
            self._recount_dead(delta.dead_recount)
            conn.execute("COMMIT")
        except BaseException:
            self._rollback_quietly()
            raise

    def _delete_edge(self, ref: EdgeRef) -> None:
        """Delete the ``(source)-[rel_type]->(target)`` edge. No own transaction.

        Runs on ``self._conn`` so it participates in :meth:`apply_graph_delta`'s
        transaction. Endpoints are matched untyped and keyed on the primary
        ``id``, so the scan is index-bounded regardless of the rel-table group.
        """
        assert self._conn is not None
        self._conn.execute(
            "MATCH (a)-[r:CodeRelation]->(b) "
            "WHERE a.id = $src AND b.id = $tgt AND r.rel_type = $rt "
            "DELETE r",
            parameters={"src": ref.source, "tgt": ref.target, "rt": ref.rel_type},
        )

    def _recount_dead(self, node_ids: set[str]) -> None:
        """Recompute ``is_dead`` locally for each affected symbol.

        Reuses ``process_dead_code``'s exemption predicate verbatim (imported,
        not forked): a symbol is dead iff it has zero incoming CALLS and is not
        exempt. Runs on the write connection inside :meth:`apply_graph_delta`'s
        transaction, so the incoming-CALLS count reflects the edges applied
        above. Only Function/Method/Class ids carry a meaningful ``is_dead``
        flag (``dead_code._SYMBOL_LABELS``); other ids are skipped, as are ids
        whose node no longer exists (it was removed earlier in this delta).
        """
        if not node_ids:
            return
        assert self._conn is not None
        from synaptiq.core.ingestion.dead_code import is_symbol_exempt_from_dead_code

        for node_id in node_ids:
            table = _table_for_id(node_id)
            if table not in _SYMBOL_TABLE_NAMES:
                continue
            incoming_rows = self._drain(
                self._conn.execute(
                    "MATCH ()-[r:CodeRelation]->(n) "
                    "WHERE n.id = $id AND r.rel_type = 'calls' "
                    "RETURN count(r)",
                    parameters={"id": node_id},
                )
            )
            incoming = int(incoming_rows[0][0]) if incoming_rows else 0
            if incoming > 0:
                is_dead = False
            else:
                meta = self._drain(
                    self._conn.execute(
                        f"MATCH (n:{table}) WHERE n.id = $id "
                        f"RETURN n.name, n.is_entry_point, n.is_exported, n.file_path",
                        parameters={"id": node_id},
                    )
                )
                if not meta:
                    continue  # node was removed earlier in this delta
                name, is_ep, is_exp, fpath = meta[0][0], meta[0][1], meta[0][2], meta[0][3]
                is_dead = not is_symbol_exempt_from_dead_code(
                    name or "", bool(is_ep), bool(is_exp), fpath or ""
                )
            self._conn.execute(
                f"MATCH (n:{table}) WHERE n.id = $id SET n.is_dead = $d",
                parameters={"id": node_id, "d": is_dead},
            )

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
                neighbors = self._get_neighbors_batch(conn, current_ids, direction)

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
        conn: ladybug.Connection,
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
                logger.debug("_get_neighbors_batch failed for table %s", table, exc_info=True)

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
                    logger.debug("get_process_memberships failed on table %s", table, exc_info=True)
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
                rel_rows = self._drain(
                    conn.execute(
                        "MATCH (a)-[r:CodeRelation]->(b) "
                        "RETURN a.id, b.id, r.rel_type, r.confidence, r.symbols, "
                        "r.strength, r.co_changes, r.step_number, r.role"
                    )
                )
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
                graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=rel_type,
                        source=row[0],
                        target=row[1],
                        properties=props,
                    )
                )
        return graph

    def execute_raw(self, query: str, parameters: dict[str, Any] | None = None) -> list[list[Any]]:
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
        """BM25 full-text search using LadybugDB's native FTS extension.

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

    def fuzzy_search(self, query: str, limit: int, max_distance: int = 2) -> list[SearchResult]:
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

        Attempts a batch COPY FROM first (Arrow when pyarrow is installed, else
        CSV), falling back to individual MERGE on failure.
        """
        assert self._conn is not None
        if not embeddings:
            return

        # The HNSW index pins the table: DROP TABLE (bulk path) and SET on
        # the indexed column (fallback path) both fail while it exists.
        self._drop_vector_index()

        if not self._bulk_store_embeddings(embeddings):
            dim = len(embeddings[0].embedding)
            for emb in embeddings:
                try:
                    self._conn.execute(
                        "MERGE (e:Embedding {node_id: $nid}) "
                        f"SET e.vec = CAST($vec AS FLOAT[{dim}]), e.text_sha = $sha",
                        parameters={
                            "nid": emb.node_id,
                            "vec": emb.embedding,
                            "sha": emb.text_sha,
                        },
                    )
                except Exception:
                    logger.debug("store_embeddings failed for node %s", emb.node_id, exc_info=True)

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
            self._conn.execute(f"CALL DROP_VECTOR_INDEX('Embedding', '{_VECTOR_INDEX_NAME}')")
        except Exception:
            pass

    def load_embeddings(self) -> dict[str, tuple[str, list[float]]]:
        """Return ``{node_id: (text_sha, vector)}`` for all stored embeddings.

        Snapshot taken before a full rebuild so vectors for unchanged
        symbols are carried across instead of re-encoded.  Returns ``{}``
        for pre-``text_sha`` schemas — the next store recreates the table
        with the column, so the cost is one full re-encode after upgrade.
        """
        mapping: dict[str, tuple[str, list[float]]] = {}
        with self._read_conn() as conn:
            try:
                rows = self._drain(
                    conn.execute("MATCH (e:Embedding) RETURN e.node_id, e.text_sha, e.vec")
                )
            except Exception:
                logger.debug("load_embeddings failed (pre-text_sha schema?)", exc_info=True)
                return {}
        for row in rows:
            if row[0] and row[1] and row[2]:
                mapping[row[0]] = (row[1], [float(v) for v in row[2]])
        return mapping

    @classmethod
    def _stored_embedding_dim(cls, conn: ladybug.Connection) -> int | None:
        """Return the actual width of stored ``Embedding.vec`` rows.

        ``None`` when the table is empty (or unreadable) — meaning there is
        nothing to compare a query vector's width against, so
        :meth:`vector_search` skips the dimension guard and falls through to
        its normal (empty-result) behavior instead of treating "no vectors
        stored yet" as a mismatch (the W4.1 lazy-embeddings window: hybrid
        search must degrade to keyword-only while the Embedding table is
        still empty, never error).

        Deliberately reads a real row instead of the table's declared column
        type: a freshly created, still-empty Embedding table carries the
        ``EMBEDDING_DIM`` bootstrap placeholder in its schema (see
        :func:`_embedding_ddl`), which says nothing about the width the
        first real :meth:`store_embeddings` call will actually use.
        """
        try:
            rows = cls._drain(conn.execute("MATCH (e:Embedding) RETURN e.vec LIMIT 1"))
        except Exception:
            return None
        if not rows or not rows[0] or rows[0][0] is None:
            return None
        return len(rows[0][0])

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        """Find the closest nodes to *vector* via the HNSW vector index.

        Falls back to a full ``array_cosine_similarity`` scan when the index
        is unavailable (pre-index database or failed index build).  Joins
        with node tables to fetch metadata in a single query.

        Raises:
            RuntimeError: *vector*'s width does not match the stored index's
                actual vector width — e.g. the index was built with the
                "fast" embedding tier (256-dim) but *vector* was encoded
                with "quality" (384-dim), or vice versa (W4.4). Without this
                guard the native FLOAT[N] type-cast fails with an opaque
                binder error on the full-scan fallback, while the HNSW path
                silently swallows the same error and returns as if there
                were no matches at all — neither tells the caller what's
                actually wrong. Never raised when the Embedding table is
                empty (nothing to compare against) — see
                :meth:`_stored_embedding_dim`.
        """
        limit = max(1, int(limit))
        # Vector literals must be inlined — the engine cannot bind a parameter
        # in the index-function argument position, nor distinguish a plain LIST
        # from a fixed-width FLOAT[] for array_cosine_similarity.
        vec_literal = "[" + ", ".join(str(float(v)) for v in vector) + "]"
        dim = len(vector)

        with self._read_conn() as conn:
            stored_dim = self._stored_embedding_dim(conn)
            if stored_dim is not None and stored_dim != dim:
                raise RuntimeError(
                    f"Query embedding is {dim}-dim but the stored index holds "
                    f"{stored_dim}-dim vectors (it was likely built with a "
                    "different --embedding-model tier). Re-run `synaptiq analyze` "
                    "to rebuild the index, or query with the matching tier."
                )
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
        cls, conn: ladybug.Connection, vec_literal: str, dim: int, limit: int
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
        return [(row[0] or "", 1.0 - float(row[1]) if row[1] is not None else 0.0) for row in rows]

    @classmethod
    def _vector_scan_query(
        cls, conn: ladybug.Connection, vec_literal: str, dim: int, limit: int
    ) -> list[tuple[str, float]]:
        """Full-scan cosine similarity over all embeddings.

        Fallback for when the HNSW index is missing or its build failed: the
        ``vec`` column is ``FLOAT[dim]`` but unindexed, so the query vector is
        cast to ``FLOAT[dim]`` to match. There is no legacy ``DOUBLE[]`` path —
        ``bulk_load`` / ``store_embeddings`` always (re)create the Embedding
        table at the current width, so an old column type can't survive a
        reindex (review F9).
        """
        query = (
            f"MATCH (e:Embedding) "
            f"RETURN e.node_id, "
            f"array_cosine_similarity(e.vec, CAST({vec_literal} AS FLOAT[{dim}])) AS sim "
            f"ORDER BY sim DESC LIMIT {limit}"
        )
        try:
            rows = cls._drain(conn.execute(query))
        except Exception:
            logger.debug("vector scan failed", exc_info=True)
            return []
        return [(row[0] or "", float(row[1]) if row[1] is not None else 0.0) for row in rows]

    def get_indexed_files(self) -> dict[str, str]:
        """Return ``{file_path: sha256(content)}`` for all File nodes."""
        mapping: dict[str, str] = {}
        with self._read_conn() as conn:
            try:
                rows = self._drain(conn.execute("MATCH (n:File) RETURN n.file_path, n.content"))
                for row in rows:
                    fp = row[0] or ""
                    content = row[1] or ""
                    mapping[fp] = hashlib.sha256(content.encode()).hexdigest()
            except Exception:
                logger.debug("get_indexed_files failed", exc_info=True)
        return mapping

    @staticmethod
    def _remove_db_files(db_path: Path) -> None:
        """Delete a database and its sibling artifacts.

        LadybugDB stores a database as a single file plus a transient ``.wal``.
        The ``is_dir`` branch also cleans up an index *directory* written by the
        former KuzuDB backend, so :func:`open_with_recovery` heals an upgraded
        install in place. ``.shadow`` is a no-op for LadybugDB (kept as a
        harmless legacy sibling).
        """
        if db_path.is_dir():
            shutil.rmtree(db_path, ignore_errors=True)
        else:
            db_path.unlink(missing_ok=True)
        # str-concat, not with_suffix: the path may already carry a suffix
        # (e.g. a ``.rebuild`` swap file) that with_suffix would replace.
        for suffix in (".wal", ".shadow"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """Replace the entire store with the contents of *graph*.

        Builds into a sibling ``.rebuild`` database first and swaps it in
        only after the load fully succeeds — a failed build (or a crash
        mid-build) leaves the live index untouched.  Rebuilding from
        scratch rather than ``MATCH (n) DETACH DELETE n`` also sidesteps the
        cost and native-layer risk of a large delete.

        Uses COPY FROM for bulk loading nodes and relationships — an in-memory
        Arrow table when pyarrow is installed, otherwise a temporary CSV.  The
        loaders return ``False`` — and this method falls back to individual
        inserts — ONLY when COPY FROM is unavailable (the first COPY fails
        before any rows land).  A COPY that fails *after* partial progress
        RAISES instead (review F2): the ``except BaseException`` below wipes the
        ``.rebuild`` database and propagates, because re-running the row-by-row
        fallback over the whole graph would duplicate the already-COPYed
        relationships (rel inserts are non-idempotent ``CREATE``).

        The swap waits for in-flight reads to drain first: a read whose
        dispatch timed out keeps running in its thread after the RW lock
        is released, and moving database files under a live native query
        risks a crash in the engine's native layer.
        """
        assert self._db_path is not None
        live_path = self._db_path
        tmp_path = live_path.with_name(live_path.name + ".rebuild")

        self._remove_db_files(tmp_path)
        builder = LadybugBackend()
        # Skip building FTS indexes on the empty schema — rebuild_fts_indexes()
        # below builds them over the populated tables right after the COPY, so
        # the empty-table build would be pure waste (~25% of storage-load time).
        builder.initialize(tmp_path, _build_fts_indexes=False)
        try:
            if not builder._bulk_load_nodes(graph):
                builder.add_nodes(list(graph.iter_nodes()))
            if not builder._bulk_load_rels(graph):
                builder.add_relationships(list(graph.iter_relationships()))
            builder.rebuild_fts_indexes()
            # Stamp a fresh incremental-indexing manifest into the builder DB so
            # it rides the swap into the live index (W3.2a). Best-effort — it
            # never raises, so it cannot abort the build.
            builder._write_full_manifest(graph)
        except BaseException:
            builder.close()
            self._remove_db_files(tmp_path)
            raise
        builder.close()

        # Swap: drain readers, close the live handle, move the fresh
        # database into place, reopen.
        self._wait_for_readers()
        self.close()
        self._remove_db_files(live_path)
        tmp_path.replace(live_path)
        for suffix in (".wal", ".shadow"):
            artifact = Path(str(tmp_path) + suffix)
            if artifact.exists():
                artifact.replace(Path(str(live_path) + suffix))
        self.initialize(live_path)

    def rebuild_fts_indexes(self) -> None:
        """Drop and recreate FTS indexes for every searchable node table.

        Must be called after any bulk data change so the BM25 indexes
        reflect the current node contents. Only ``_SEARCHABLE_TABLES`` are
        (re)indexed — Folder/Community/Process are never queried by
        :meth:`fts_search`, :meth:`exact_name_search`, or
        :meth:`fuzzy_search`, so building FTS indexes for them is pure
        waste. Both the DROP and CREATE calls already tolerate failure,
        so this is safe to run against a database whose indexes were
        built by an older version that indexed all node tables — the
        three extra pre-existing indexes are simply left in place,
        un-rebuilt and unqueried.
        """
        assert self._conn is not None
        for table in _SEARCHABLE_TABLES:
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
            # PARALLEL=false is required (verified against LadybugDB): node
            # ``content``/``signature`` and relationship properties carry source
            # code with embedded newlines, and the parallel CSV reader rejects
            # quoted newlines ("Quoted newlines are not supported in parallel CSV
            # reader. Please specify PARALLEL=FALSE in the options."). Without
            # this every COPY of real code fails and bulk_load falls back to
            # row-by-row inserts — ~50x slower on large repos.
            self._conn.execute(f'COPY {table} FROM "{csv_path}" (HEADER=false, PARALLEL=false)')
        finally:
            if csv_path:
                Path(csv_path).unlink(missing_ok=True)

    def _arrow_copy(self, table: str, arrow_tbl: pa.Table) -> None:
        """COPY an in-memory pyarrow Table into *table*.

        LadybugDB resolves the source object via a replacement scan that
        inspects this frame's local variables, so ``arrow_tbl`` MUST stay the
        name both of the parameter and of the identifier in the query — do not
        rename one without the other. A typed Arrow table carries multiline
        strings and ``FLOAT[dim]`` vectors natively, so no ``PARALLEL=false``
        flag, temp file, or per-value stringification is needed. (Unlike kuzu
        0.11.3, LadybugDB's loader needs no ``importlib.util`` shim — see the
        module-level pyarrow import.)
        """
        assert self._conn is not None
        self._conn.execute(f"COPY {table} FROM arrow_tbl")

    def _group_nodes_by_table(self, graph: KnowledgeGraph) -> dict[str, list[GraphNode]]:
        """Bucket nodes into their label tables, deduplicated by ``node.id``.

        The last occurrence of a duplicate id wins. Shared by the CSV and Arrow
        bulk paths so both load a byte-identical row set.
        """
        by_table: dict[str, list[GraphNode]] = {}
        for node in graph.iter_nodes():
            table = _LABEL_TO_TABLE.get(node.label.value)
            if table:
                by_table.setdefault(table, []).append(node)

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
        return by_table

    def _bulk_load_nodes(self, graph: KnowledgeGraph) -> bool:
        """Bulk-load nodes via the fastest available COPY path.

        Prefers the in-memory Arrow path when pyarrow is installed, else the
        temp-CSV path. Returns ``True`` on success; ``False`` signals
        :meth:`bulk_load` to fall back to row-by-row inserts (idempotent MERGE),
        exactly as the CSV path did before.
        """
        if _HAS_PYARROW:
            return self._bulk_load_nodes_arrow(graph)
        return self._bulk_load_nodes_csv(graph)

    def _bulk_load_nodes_csv(self, graph: KnowledgeGraph) -> bool:
        """Load all nodes via temporary CSV files + COPY FROM.

        Same dispatcher contract as :meth:`_bulk_load_rels_csv` (review F2):
        return ``False`` ONLY when the first COPY fails before any rows land
        (COPY unavailable → row-by-row fallback). A failure after any table has
        been COPYed RAISES so ``bulk_load`` aborts and wipes the ``.rebuild``
        database. Node inserts use idempotent ``MERGE`` so the fallback alone
        would not duplicate, but the contract is kept symmetric with the
        relationship path — aborting a partially-loaded rebuild is cleaner and
        avoids re-doing the whole graph row-by-row just to mask a real error.
        """
        mutated = False
        try:
            for table, nodes in self._group_nodes_by_table(graph).items():
                self._csv_copy(
                    table,
                    [
                        [
                            node.id,
                            node.name,
                            node.file_path,
                            node.start_line,
                            node.end_line,
                            node.content,
                            node.signature,
                            node.language,
                            node.class_name,
                            node.is_dead,
                            node.is_entry_point,
                            node.is_exported,
                            _serialize_properties(node.properties),
                        ]
                        for node in nodes
                    ],
                )
                mutated = True
            return True
        except Exception:
            if mutated:
                logger.error(
                    "Node COPY failed after partial progress; aborting rebuild",
                    exc_info=True,
                )
                raise
            logger.warning(
                "CSV COPY for nodes unavailable; falling back to slow row-by-row inserts",
                exc_info=True,
            )
            return False

    def _bulk_load_nodes_arrow(self, graph: KnowledgeGraph) -> bool:
        """Load all nodes via in-memory pyarrow tables + COPY FROM.

        Column order and types mirror ``_NODE_PROPERTIES``. Empty strings are
        coerced to ``NULL`` (see :func:`_arrow_str`) so the result is identical
        to the CSV path, whose reader stores an empty field as ``NULL``.

        Same dispatcher contract as :meth:`_bulk_load_rels_csv` (review F2):
        ``False`` only when the first COPY fails before any rows land; a failure
        after any table has been COPYed RAISES so ``bulk_load`` aborts instead
        of re-loading the whole graph row-by-row.
        """
        mutated = False
        try:
            for table, nodes in self._group_nodes_by_table(graph).items():
                arrow_tbl = pa.table(
                    {
                        "id": pa.array([n.id for n in nodes], type=pa.string()),
                        "name": pa.array([_arrow_str(n.name) for n in nodes], type=pa.string()),
                        "file_path": pa.array(
                            [_arrow_str(n.file_path) for n in nodes], type=pa.string()
                        ),
                        "start_line": pa.array([n.start_line for n in nodes], type=pa.int64()),
                        "end_line": pa.array([n.end_line for n in nodes], type=pa.int64()),
                        "content": pa.array(
                            [_arrow_str(n.content) for n in nodes], type=pa.string()
                        ),
                        "signature": pa.array(
                            [_arrow_str(n.signature) for n in nodes], type=pa.string()
                        ),
                        "language": pa.array(
                            [_arrow_str(n.language) for n in nodes], type=pa.string()
                        ),
                        "class_name": pa.array(
                            [_arrow_str(n.class_name) for n in nodes], type=pa.string()
                        ),
                        "is_dead": pa.array([n.is_dead for n in nodes], type=pa.bool_()),
                        "is_entry_point": pa.array(
                            [n.is_entry_point for n in nodes], type=pa.bool_()
                        ),
                        "is_exported": pa.array([n.is_exported for n in nodes], type=pa.bool_()),
                        "properties_json": pa.array(
                            [_arrow_str(_serialize_properties(n.properties)) for n in nodes],
                            type=pa.string(),
                        ),
                    }
                )
                self._arrow_copy(table, arrow_tbl)
                mutated = True
            return True
        except Exception:
            if mutated:
                logger.error(
                    "Node COPY failed after partial progress; aborting rebuild",
                    exc_info=True,
                )
                raise
            logger.warning(
                "Arrow COPY for nodes unavailable; falling back to slow row-by-row inserts",
                exc_info=True,
            )
            return False

    def _group_rels_by_pair(
        self, graph: KnowledgeGraph
    ) -> dict[tuple[str, str], list[GraphRelationship]]:
        """Bucket relationships by ``(src_table, dst_table)``, deduplicated.

        Deduplication keys on full edge identity — source, target, type, role,
        and step_number — so e.g. USES_TYPE edges with different roles between
        the same pair survive (mirrors the in-memory relationship ID
        semantics). Shared by the CSV and Arrow bulk paths.
        """
        by_pair: dict[tuple[str, str], list[GraphRelationship]] = {}
        for rel in graph.iter_relationships():
            src_table = _table_for_id(rel.source)
            dst_table = _table_for_id(rel.target)
            if src_table and dst_table:
                by_pair.setdefault((src_table, dst_table), []).append(rel)

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
        return by_pair

    def _bulk_load_rels(self, graph: KnowledgeGraph) -> bool:
        """Bulk-load relationships via the fastest available COPY path.

        Arrow when pyarrow is installed, else CSV. ``False`` falls back to
        row-by-row inserts, exactly as the CSV path did before.
        """
        if _HAS_PYARROW:
            return self._bulk_load_rels_arrow(graph)
        return self._bulk_load_rels_csv(graph)

    def _bulk_load_rels_csv(self, graph: KnowledgeGraph) -> bool:
        """Load all relationships via temporary CSV files + COPY FROM.

        Dispatcher contract (review F2): return ``True`` on success and
        ``False`` ONLY when the very first COPY fails before any rows land
        (COPY FROM is unavailable in this environment) — that signals
        :meth:`bulk_load` to fall back to row-by-row inserts. Once ANY pair has
        been COPYed, a later failure RAISES instead of returning ``False``:
        relationship inserts use non-idempotent ``CREATE``, so re-running the
        row-by-row fallback over the whole graph would DUPLICATE every pair
        already COPYed. Raising lets ``bulk_load`` abort and wipe the
        ``.rebuild`` database, leaving the live index untouched (crash-safe).
        """
        mutated = False
        try:
            for (src_table, dst_table), rels in self._group_rels_by_pair(graph).items():
                self._csv_copy(
                    f"CodeRelation_{src_table}_{dst_table}",
                    [
                        [
                            rel.source,
                            rel.target,
                            rel.type.value,
                            float((rel.properties or {}).get("confidence", 1.0)),
                            str((rel.properties or {}).get("role", "")),
                            int((rel.properties or {}).get("step_number", 0)),
                            float((rel.properties or {}).get("strength", 0.0)),
                            int((rel.properties or {}).get("co_changes", 0)),
                            str((rel.properties or {}).get("symbols", "")),
                        ]
                        for rel in rels
                    ],
                )
                mutated = True
            return True
        except Exception:
            if mutated:
                logger.error(
                    "Relationship COPY failed after partial progress; aborting rebuild "
                    "to avoid duplicate edges",
                    exc_info=True,
                )
                raise
            logger.warning(
                "CSV COPY for relationships unavailable; falling back to slow row-by-row inserts",
                exc_info=True,
            )
            return False

    def _bulk_load_rels_arrow(self, graph: KnowledgeGraph) -> bool:
        """Load all relationships via in-memory pyarrow tables + COPY FROM.

        The first two columns (source/target node ids) are matched positionally
        as the rel FROM/TO; the rest mirror the ``_REL_PROPERTIES`` order and
        types. Property coercion matches the CSV path exactly (empty role/symbols
        strings become ``NULL``).

        Same dispatcher contract as :meth:`_bulk_load_rels_csv` (review F2):
        ``False`` only when the first COPY fails before any rows land; a failure
        after any pair has been COPYed RAISES so ``bulk_load`` aborts rather than
        duplicating the already-COPYed edges via the non-idempotent row-by-row
        fallback.
        """
        mutated = False
        try:
            for (src_table, dst_table), rels in self._group_rels_by_pair(graph).items():
                props = [r.properties or {} for r in rels]
                arrow_tbl = pa.table(
                    {
                        "src": pa.array([r.source for r in rels], type=pa.string()),
                        "dst": pa.array([r.target for r in rels], type=pa.string()),
                        "rel_type": pa.array([r.type.value for r in rels], type=pa.string()),
                        "confidence": pa.array(
                            [float(p.get("confidence", 1.0)) for p in props], type=pa.float64()
                        ),
                        "role": pa.array(
                            [_arrow_str(str(p.get("role", ""))) for p in props], type=pa.string()
                        ),
                        "step_number": pa.array(
                            [int(p.get("step_number", 0)) for p in props], type=pa.int64()
                        ),
                        "strength": pa.array(
                            [float(p.get("strength", 0.0)) for p in props], type=pa.float64()
                        ),
                        "co_changes": pa.array(
                            [int(p.get("co_changes", 0)) for p in props], type=pa.int64()
                        ),
                        "symbols": pa.array(
                            [_arrow_str(str(p.get("symbols", ""))) for p in props], type=pa.string()
                        ),
                    }
                )
                self._arrow_copy(f"CodeRelation_{src_table}_{dst_table}", arrow_tbl)
                mutated = True
            return True
        except Exception:
            if mutated:
                logger.error(
                    "Relationship COPY failed after partial progress; aborting rebuild "
                    "to avoid duplicate edges",
                    exc_info=True,
                )
                raise
            logger.warning(
                "Arrow COPY for relationships unavailable; falling back to slow row-by-row inserts",
                exc_info=True,
            )
            return False

    def _bulk_store_embeddings_csv(self, embeddings: list[NodeEmbedding]) -> bool:
        """Store embeddings via temporary CSV + COPY FROM.

        Returns ``True`` on success, ``False`` if COPY FROM is not available.

        Dispatcher contract (review F2): unlike the node/relationship paths this
        is a SINGLE COPY into a freshly DROP+CREATEd table, so there is no
        "partial progress across multiple COPYs" hazard — and
        :meth:`store_embeddings`'s row-by-row fallback is idempotent ``MERGE``
        on the ``node_id`` primary key, into that same recreated table. Both
        properties mean returning ``False`` on any failure can never duplicate a
        vector, so this path keeps the plain return-``False``-on-failure form.
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

            self._csv_copy(
                "Embedding",
                [
                    [emb.node_id, "[" + ",".join(str(v) for v in emb.embedding) + "]", emb.text_sha]
                    for emb in embeddings
                ],
            )
            return True
        except Exception:
            logger.warning(
                "CSV COPY for embeddings failed; falling back to slow row-by-row inserts",
                exc_info=True,
            )
            return False

    def _bulk_store_embeddings(self, embeddings: list[NodeEmbedding]) -> bool:
        """Store embeddings via the fastest available COPY path.

        Arrow when pyarrow is installed, else CSV. ``False`` falls back to the
        row-by-row MERGE path in :meth:`store_embeddings`.
        """
        if _HAS_PYARROW:
            return self._bulk_store_embeddings_arrow(embeddings)
        return self._bulk_store_embeddings_csv(embeddings)

    def _bulk_store_embeddings_arrow(self, embeddings: list[NodeEmbedding]) -> bool:
        """Store embeddings via an in-memory pyarrow Table + COPY FROM.

        The vector column is a ``FLOAT[dim]`` fixed-size list, so vectors are
        copied natively — no per-float ``str()`` and no ``[..]`` string parse.
        Recreates the Embedding table at the actual width first, exactly like
        the CSV path. Returns ``False`` on failure.

        Same dispatcher contract as :meth:`_bulk_store_embeddings_csv` (review
        F2): a single COPY into a freshly recreated table plus an idempotent
        ``MERGE`` fallback means returning ``False`` on failure can never
        duplicate a vector.
        """
        assert self._conn is not None
        try:
            dim = len(embeddings[0].embedding)
            try:
                self._conn.execute("DROP TABLE Embedding")
            except Exception:
                pass
            self._conn.execute(f"CREATE NODE TABLE IF NOT EXISTS Embedding({_embedding_ddl(dim)})")
            arrow_tbl = pa.table(
                {
                    "node_id": pa.array([e.node_id for e in embeddings], type=pa.string()),
                    "vec": pa.array(
                        [e.embedding for e in embeddings], type=pa.list_(pa.float32(), dim)
                    ),
                    "text_sha": pa.array(
                        [_arrow_str(e.text_sha) for e in embeddings], type=pa.string()
                    ),
                }
            )
            self._arrow_copy("Embedding", arrow_tbl)
            return True
        except Exception:
            logger.warning(
                "Arrow COPY for embeddings failed; falling back to slow row-by-row inserts",
                exc_info=True,
            )
            return False

    def _create_schema(self, *, build_fts: bool = True) -> None:
        """Create node/rel/embedding tables and the FTS extension.

        When ``build_fts`` is ``False`` the (empty) FTS indexes are not built —
        see :meth:`initialize`'s ``_build_fts_indexes`` for why the bulk_load
        rebuild path skips them.
        """
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
            f"CREATE REL TABLE GROUP IF NOT EXISTS CodeRelation({pairs_clause}, {_REL_PROPERTIES})"
        )
        try:
            self._conn.execute(rel_stmt)
        except Exception:
            logger.debug("REL TABLE GROUP creation skipped", exc_info=True)

        self._create_manifest_tables()

        if build_fts:
            self._create_fts_indexes()

    def _create_fts_indexes(self) -> None:
        """Create FTS indexes for every searchable node table (idempotent).

        Scoped to ``_SEARCHABLE_TABLES`` rather than all node tables —
        Folder/Community/Process are never queried by any search method,
        so indexing them would waste build time for no benefit.
        """
        assert self._conn is not None
        for table in _SEARCHABLE_TABLES:
            idx_name = f"{table.lower()}_fts"
            try:
                self._conn.execute(
                    f"CALL CREATE_FTS_INDEX('{table}', '{idx_name}', "
                    f"['name', 'content', 'signature'])"
                )
            except Exception:
                # Index may already exist — that's fine.
                pass

    # ------------------------------------------------------------------
    # Incremental-indexing manifest (W3.2a) — DDL + read/write.
    # Storage I/O only; the logical schema, fingerprints, diff, and
    # version/corruption gating live in ``core.ingestion.manifest``.
    # ------------------------------------------------------------------

    def _create_manifest_tables(self) -> None:
        """Create the manifest node tables (idempotent).

        Standalone tables outside ``NodeLabel`` — invisible to the
        ``CodeRelation`` rel group, FTS, ``_migrate_schema``/``_verify_schema``,
        and ``load_graph``. Called from :meth:`_create_schema` so a freshly built
        ``.rebuild`` DB carries them and the manifest survives the swap.
        """
        assert self._conn is not None
        try:
            self._conn.execute(_FILE_MANIFEST_DDL)
            self._conn.execute(_INDEX_MANIFEST_DDL)
        except Exception:
            logger.debug("manifest table creation skipped", exc_info=True)

    def write_manifest(self, manifest: Manifest) -> None:
        """Persist *manifest* into the current DB, replacing any prior manifest.

        Full replace: clears both manifest tables, bulk-writes the per-file rows,
        then writes the index singleton **last** as a completion marker (a
        partial/crashed write leaves no singleton, so :meth:`read_manifest`
        reports "no manifest" and the caller does a full rebuild). Not wrapped in
        an explicit transaction — it mirrors the COPY-based bulk path and is
        crash-safe by riding :meth:`bulk_load`'s ``.rebuild`` swap. Uses the
        write connection, so callers serialize via the external lock.
        """
        assert self._conn is not None
        self._create_manifest_tables()
        self._conn.execute(f"MATCH (m:{_MANIFEST_FILE_TABLE}) DETACH DELETE m")
        self._conn.execute(f"MATCH (im:{_MANIFEST_INDEX_TABLE}) DETACH DELETE im")
        rows = [serialize_file_manifest(fm) for fm in manifest.files.values()]
        if rows:
            self._write_manifest_file_rows(rows)
        version, data = serialize_index_manifest(manifest.index)
        self._conn.execute(
            f"CREATE (im:{_MANIFEST_INDEX_TABLE} {{id: $id, manifest_version: $v, data: $d}})",
            parameters={"id": _MANIFEST_SINGLETON_ID, "v": version, "d": data},
        )

    def _write_manifest_file_rows(self, rows: list[list[Any]]) -> None:
        """Insert FileManifest rows via CSV COPY, falling back to row MERGE.

        Mirrors the node bulk path: COPY is fast at repo scale, and the MERGE
        fallback keeps the write working in COPY-unavailable environments.
        """
        assert self._conn is not None
        try:
            self._csv_copy(_MANIFEST_FILE_TABLE, rows)
            return
        except Exception:
            logger.debug("manifest CSV COPY failed; falling back to row inserts", exc_info=True)
        stmt = (
            f"MERGE (m:{_MANIFEST_FILE_TABLE} {{path: $path}}) "
            "SET m.content_sha = $content_sha, m.language = $language, "
            "m.symbol_ids = $symbol_ids, m.symbol_sigs = $symbol_sigs, "
            "m.out_edges = $out_edges"
        )
        for row in rows:
            self._conn.execute(
                stmt,
                parameters={
                    "path": row[0],
                    "content_sha": row[1],
                    "language": row[2],
                    "symbol_ids": row[3],
                    "symbol_sigs": row[4],
                    "out_edges": row[5],
                },
            )

    def _write_full_manifest(self, graph: KnowledgeGraph) -> None:
        """Build a fresh manifest from *graph* and persist it — best-effort.

        Called by :meth:`bulk_load` on the ``.rebuild`` builder so every full
        build stamps a current manifest that rides the swap into the live DB.
        Fully isolated: any failure is swallowed (the manifest is an optimization
        with a full-rebuild fallback), so a manifest problem can never abort an
        index build. ``git_head`` is left ``None`` here — the graph does not
        carry it; the pipeline/consolidation owner (W3.2e) populates it when it
        computes coupling.
        """
        try:
            from synaptiq import __version__

            self.write_manifest(build_manifest(graph, tool_version=__version__))
        except Exception:
            logger.warning(
                "manifest build/write failed during bulk_load; the next analyze "
                "will fall back to a full rebuild",
                exc_info=True,
            )

    def read_manifest(self) -> Manifest | None:
        """Load the stored manifest, or ``None`` to force a full rebuild.

        Never raises. Returns ``None`` on a missing/empty manifest (§8 trigger
        1), a ``manifest_version`` mismatch (trigger 2), or any unreadable/corrupt
        row (trigger 5) — the caller treats ``None`` as "no trustworthy manifest,
        use the full path". The fingerprint check (trigger 3) is the caller's; it
        needs the current walk (``manifest.compute_fingerprint`` vs the stored
        ``full_fingerprint``).
        """
        try:
            with self._read_conn() as conn:
                index_rows = self._drain(
                    conn.execute(
                        f"MATCH (im:{_MANIFEST_INDEX_TABLE}) RETURN im.manifest_version, im.data"
                    )
                )
                file_rows = self._drain(
                    conn.execute(
                        f"MATCH (m:{_MANIFEST_FILE_TABLE}) RETURN "
                        "m.path, m.content_sha, m.language, m.symbol_ids, "
                        "m.symbol_sigs, m.out_edges"
                    )
                )
        except Exception:
            # Manifest tables absent (pre-manifest / read-only old index) or
            # otherwise unreadable → no trustworthy manifest.
            logger.debug("read_manifest query failed", exc_info=True)
            return None
        index_row = index_rows[0] if index_rows else None
        return load_manifest_from_rows(index_row, file_rows)

    def _get_prepared(self, key: str, query: str) -> Any:
        """Return a cached prepared statement for ``key``, preparing once.

        The cache is bound to the current ``self._conn`` and cleared when
        that connection is (re)created or closed. Preparing once per node
        label / rel-table pair and reusing the statement across every row of
        a batch is the core of the batched-insert speedup — the plan is
        compiled once instead of on every ``execute``.
        """
        assert self._conn is not None
        stmt = self._prepared.get(key)
        if stmt is None:
            # LadybugDB (like kuzu before it — verified against 0.18.1) emits a
            # DeprecationWarning that the separate prepare()+execute() API is
            # deprecated in favour of a single execute() call. We deliberately
            # prepare once and reuse the statement across a batch's rows, which
            # is the whole point of this fast path. Silence the unactionable
            # warning at the call site (reads never prepare, so this narrow
            # window cannot suppress other threads' warnings).
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The use of separate prepare",
                    category=DeprecationWarning,
                )
                stmt = self._conn.prepare(query)
            self._prepared[key] = stmt
        return stmt

    def _insert_node(self, node: GraphNode) -> None:
        """Upsert one node into its label table via a cached prepared statement.

        Uses ``MERGE ... SET`` (keyed on the primary ``id``) rather than
        ``CREATE`` so the insert is idempotent. The incremental re-index path
        (:func:`~synaptiq.core.ingestion.pipeline.apply_reindex`) deletes only
        the changed file's own nodes and then re-inserts the freshly parsed
        graph, which re-includes *persistent* structural nodes — e.g. ancestor
        ``Folder`` nodes whose ``file_path`` is the directory, not the changed
        file, and so survive :meth:`remove_nodes_by_file`. ``CREATE`` raised a
        duplicate-primary-key error on those every time (previously swallowed
        per row, but fatal to a single batched transaction); ``MERGE`` matches
        the existing node and refreshes its properties instead. On a fresh
        database (the ``bulk_load`` fallback) every ``MERGE`` is a create, so
        behaviour there is unchanged.

        Runs inside :meth:`_write_batch`'s transaction; an unexpected execution
        error propagates so the batch rolls back atomically. Nodes with an
        unknown label are skipped (logged) exactly as before.
        """
        assert self._conn is not None
        table = _LABEL_TO_TABLE.get(node.label.value)
        if table is None:
            logger.warning("Unknown label %s for node %s", node.label, node.id)
            return

        stmt = self._get_prepared(
            f"node:{table}",
            f"MERGE (n:{table} {{id: $id}}) "
            f"SET n.name = $name, n.file_path = $file_path, "
            f"n.start_line = $start_line, n.end_line = $end_line, "
            f"n.content = $content, n.signature = $signature, "
            f"n.language = $language, n.class_name = $class_name, "
            f"n.is_dead = $is_dead, n.is_entry_point = $is_entry_point, "
            f"n.is_exported = $is_exported, n.properties_json = $properties_json",
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
        self._conn.execute(stmt, parameters=params)

    def _insert_relationship(self, rel: GraphRelationship) -> None:
        """MATCH source and target, then MERGE the rel via a prepared statement.

        Uses ``MERGE ... SET`` keyed on the endpoint pair **and** the logical
        ``rel_type`` property — the rel-table group stores the logical kind in
        ``rel_type`` (not as a label), so the merge key is ``(source, target,
        rel_type)`` and MERGE matches-or-creates on that full triple (verified
        against LadybugDB: a distinct ``rel_type`` between the same endpoints is
        a distinct edge; re-inserting the same triple refreshes it in place
        rather than duplicating). This makes edge insertion **idempotent**,
        fixing a latent duplication bug in the scoped re-index path: re-parsing
        a file re-emits persistent structural edges — e.g. folder→folder /
        folder→file ``CONTAINS`` whose endpoints survive
        :meth:`remove_nodes_by_file` (their ``file_path`` is the directory, not
        the changed file) — which the old ``CREATE`` duplicated on every reindex
        and which would accumulate permanently once periodic full wipes stop. On
        a fresh database (the ``bulk_load`` fallback) every MERGE is a create,
        so behaviour there is unchanged.

        Runs inside :meth:`_write_batch`'s transaction; an execution error
        propagates so the batch can roll back atomically. A relationship
        whose endpoints are not present simply matches nothing and creates
        nothing (no error), and one whose ids don't resolve to a table is
        skipped (logged) — both exactly as before.
        """
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

        stmt = self._get_prepared(
            f"rel:{src_table}:{tgt_table}",
            f"MATCH (a:{src_table}), (b:{tgt_table}) "
            f"WHERE a.id = $src AND b.id = $tgt "
            f"MERGE (a)-[r:CodeRelation {{rel_type: $rel_type}}]->(b) "
            f"SET r.confidence = $confidence, "
            f"r.role = $role, "
            f"r.step_number = $step_number, "
            f"r.strength = $strength, "
            f"r.co_changes = $co_changes, "
            f"r.symbols = $symbols",
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
        self._conn.execute(stmt, parameters=params)

    def _query_nodes(self, query: str, parameters: dict[str, Any] | None = None) -> list[GraphNode]:
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
