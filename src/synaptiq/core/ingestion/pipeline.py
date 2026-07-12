"""Pipeline orchestrator for Synaptiq.

Runs all ingestion phases in sequence, populates an in-memory knowledge graph,
bulk-loads it into a storage backend, and returns a summary of the results.

Phases executed:
    0. Incremental diff (reserved -- not yet implemented)
    1. File walking
    2. Structure processing (File/Folder nodes + CONTAINS edges)
    3. Code parsing (symbol nodes + DEFINES edges)
    4. Import resolution (IMPORTS edges)
    5. Call tracing (CALLS edges)
    6. Heritage extraction (EXTENDS / IMPLEMENTS edges)
    7. Type analysis (USES_TYPE edges)
    8. Community detection (COMMUNITY nodes + MEMBER_OF edges)
    9. Process detection (PROCESS nodes + STEP_IN_PROCESS edges)
    10. Dead code detection (flags unreachable symbols)
    11. Change coupling (COUPLED_WITH edges from git history)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from synaptiq.config.ignore import load_gitignore
from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import NodeLabel
from synaptiq.core.ingestion.calls import process_calls
from synaptiq.core.ingestion.community import process_communities
from synaptiq.core.ingestion.coupling import (
    collect_coupling_commits,
    process_coupling,
)
from synaptiq.core.ingestion.dead_code import process_dead_code
from synaptiq.core.ingestion.heritage import process_heritage
from synaptiq.core.ingestion.imports import process_imports
from synaptiq.core.ingestion.parser_phase import process_parsing
from synaptiq.core.ingestion.processes import process_processes
from synaptiq.core.ingestion.rest_linking import process_rest_linking
from synaptiq.core.ingestion.structure import process_structure
from synaptiq.core.ingestion.types import process_types
from synaptiq.core.ingestion.walker import FileEntry, walk_repo
from synaptiq.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a pipeline run."""

    files: int = 0
    symbols: int = 0
    relationships: int = 0
    clusters: int = 0
    processes: int = 0
    dead_code: int = 0
    coupled_pairs: int = 0
    rest_links: int = 0
    embeddings: int = 0
    duration_seconds: float = 0.0
    incremental: bool = False
    changed_files: int = 0
    # Wall-clock seconds per phase, keyed by the same names passed to
    # progress_callback (plus "Loading to storage" / "Generating embeddings"
    # when a storage backend is supplied). Values approximately sum to
    # duration_seconds.
    phase_timings: dict[str, float] = field(default_factory=dict)


_SYMBOL_LABELS: frozenset[NodeLabel] = frozenset(NodeLabel) - {
    NodeLabel.FILE,
    NodeLabel.FOLDER,
    NodeLabel.COMMUNITY,
    NodeLabel.PROCESS,
}


def run_pipeline(
    repo_path: Path,
    storage: StorageBackend | None = None,
    full: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
    skip_embeddings: bool = False,
) -> tuple[KnowledgeGraph, PipelineResult]:
    """Run phases 1-11 of the ingestion pipeline.

    When *storage* is provided the graph is bulk-loaded into it after
    all phases complete.  When ``None``, only the in-memory graph is
    returned (useful for branch comparison snapshots).

    Parameters
    ----------
    repo_path:
        Root directory of the repository to analyse.
    storage:
        An already-initialised :class:`StorageBackend` to persist the graph.
        Pass ``None`` to skip storage loading.
    full:
        When ``True``, skip incremental-diff logic (Phase 0) and force a full
        re-index.  Currently Phase 0 is a no-op regardless of this flag.
    progress_callback:
        Optional ``(phase_name, progress)`` callback where *progress* is a
        float in ``[0.0, 1.0]``.
    skip_embeddings:
        When ``True``, skip embedding generation after storage loading.
        Useful for faster iteration when vector search is not needed.

    Returns
    -------
    tuple[KnowledgeGraph, PipelineResult]
        The populated graph and a summary dataclass with counts and timings.
    """
    start = time.monotonic()
    result = PipelineResult()

    def report(phase: str, pct: float) -> None:
        if progress_callback is not None:
            progress_callback(phase, pct)

    @contextmanager
    def timed_phase(phase: str):
        """Bracket *phase* with the existing report(0.0)/report(1.0) calls
        (contract unchanged) and record its wall time on ``result``."""
        report(phase, 0.0)
        t0 = time.monotonic()
        try:
            yield
        finally:
            result.phase_timings[phase] = time.monotonic() - t0
            report(phase, 1.0)

    # W2.4 (G11): coupling's `git log` is a single GIL-releasing subprocess
    # that otherwise sits serially at the tail of the pipeline. Run its
    # "collect" half on a dedicated single-use worker thread so the wait
    # overlaps with every CPU phase between the walk and "Analyzing git
    # history". A concurrent.futures Future gives a clean lifecycle plus native
    # error propagation: Future.result() re-raises the worker's exception with
    # its original traceback (no error box, no masked-empty branch — F4b), and
    # the try/finally below guarantees the executor is shut down on ANY phase
    # failure so the worker thread is never orphaned.
    #
    # Fork-safety: this thread can be alive while process_parsing fans parsing
    # out to a process pool, so that pool MUST stay on the 'spawn' start method
    # — forking after a live thread leaves locks in an unknown state in the
    # child. See parser_phase._parse_with_processes.
    git_log_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synaptiq-gitlog")
    git_future: Future[list[list[str]]] | None = None
    try:
        with timed_phase("Walking files"):
            gitignore = load_gitignore(repo_path)
            files = walk_repo(repo_path, gitignore)
            result.files = len(files)

            # graph_files reproduces the filter process_coupling applies today
            # (`{n.file_path for n in graph.get_nodes_by_label(NodeLabel.FILE)}`)
            # without waiting for "Processing structure" to run: structure.py
            # creates exactly one File node per walked FileEntry, unfiltered and
            # undeduplicated, so the two sets are always identical. Snapshot and
            # submit inside this timed block so thread-creation overhead lands
            # in "Walking files" instead of an untimed gap. The worker only runs
            # collect_coupling_commits (pure subprocess + str parsing); it never
            # touches `graph`, which has no internal locking.
            graph_files = {f.path for f in files}
            git_future = git_log_executor.submit(
                collect_coupling_commits, repo_path, graph_files
            )

        graph = KnowledgeGraph()

        with timed_phase("Processing structure"):
            process_structure(files, graph)

        with timed_phase("Parsing code"):
            parse_data = process_parsing(files, graph)

        with timed_phase("Resolving imports"):
            process_imports(parse_data, graph)

        with timed_phase("Tracing calls"):
            process_calls(parse_data, graph)

        with timed_phase("Linking REST endpoints"):
            result.rest_links = process_rest_linking(parse_data, graph)

        with timed_phase("Extracting heritage"):
            process_heritage(parse_data, graph)

        with timed_phase("Analyzing types"):
            process_types(parse_data, graph)

        with timed_phase("Detecting communities"):
            result.clusters = process_communities(graph)

        with timed_phase("Detecting execution flows"):
            result.processes = process_processes(graph)

        with timed_phase("Finding dead code"):
            result.dead_code = process_dead_code(graph)

        with timed_phase("Analyzing git history"):
            # Join the background collector (see the "Walking files" block).
            # Future.result() blocks until the git-log subprocess finishes (0
            # extra wait if it already completed during the CPU phases above)
            # and re-raises any UNEXPECTED collection failure with its original
            # traceback. The common "not a git repo" case never reaches here --
            # parse_git_log catches it and yields commits=[] instead. This
            # phase's recorded time is that (usually already-elapsed) wait plus
            # process_coupling's apply cost, so phase_timings stays honest: it
            # still sums to ~duration_seconds, just smaller overall because the
            # subprocess wait is no longer serial.
            commits = git_future.result()
            result.coupled_pairs = process_coupling(graph, repo_path, commits=commits)
    finally:
        # Reap the single-use executor. On the happy path the future is already
        # done, so this is cheap. On a phase failure the collect may still be
        # running: cancel the (possibly not-yet-started) future and shut down
        # WITHOUT blocking — shutdown(wait=False) lets the finite git-log
        # subprocess finish in the background, and because the executor is
        # per-invocation (one worker, not one per failure) nothing accumulates
        # unboundedly: the Future is dropped and the executor reaped when that
        # worker ends.
        if git_future is not None:
            git_future.cancel()
        git_log_executor.shutdown(wait=False, cancel_futures=True)

    if storage is not None:
        # Snapshot before bulk_load — it resets the whole database,
        # embedding table included.
        previous_embeddings = (
            load_previous_embeddings(storage) if not skip_embeddings else {}
        )

        with timed_phase("Loading to storage"):
            storage.bulk_load(graph)

        if not skip_embeddings:
            with timed_phase("Generating embeddings"):
                from synaptiq.core.embeddings.embedder import embed_graph

                embeddings = embed_graph(graph, previous=previous_embeddings)
                if embeddings:
                    storage.store_embeddings(embeddings)
                result.embeddings = len(embeddings)

    result.symbols = sum(1 for n in graph.iter_nodes() if n.label in _SYMBOL_LABELS)
    result.relationships = graph.relationship_count
    result.duration_seconds = time.monotonic() - start

    return graph, result


def parse_files(
    file_entries: list[FileEntry],
    repo_path: Path,
) -> KnowledgeGraph:
    """Parse files through phases 2-7 without touching storage.

    This is the CPU-intensive part of re-indexing that does NOT require
    database access and can run without holding any locks.

    Returns the partial in-memory graph.
    """
    graph = KnowledgeGraph()

    process_structure(file_entries, graph)
    parse_data = process_parsing(file_entries, graph)
    process_imports(parse_data, graph)
    process_calls(parse_data, graph)
    process_heritage(parse_data, graph)
    process_types(parse_data, graph)

    return graph


def apply_reindex(
    file_entries: list[FileEntry],
    storage: StorageBackend,
    graph: KnowledgeGraph,
) -> None:
    """Apply a pre-parsed graph to storage (delete old, insert new).

    This is the I/O-intensive part that requires database access and
    MUST be called under an exclusive write lock.

    Does NOT rebuild FTS (BM25) indexes.  ``rebuild_fts_indexes()`` drops and
    recreates every searchable index from scratch (``DROP_FTS_INDEX`` +
    ``CREATE_FTS_INDEX`` per table) -- an O(whole corpus) operation, so
    calling it here would pay that cost on every single-file save (G3)
    instead of O(one file).  A full rebuild always still happens at the
    watcher's next debounced global phase: ``commit_full_index`` ->
    ``storage.bulk_load()`` unconditionally rebuilds every FTS index from the
    fresh graph (see ``KuzuBackend.bulk_load``), so no separate "FTS dirty"
    flag is needed -- the global phase's existing change-tracking in
    ``_GlobalPhaseScheduler`` (``watcher.py``) already gates it, and every
    non-skipped rebuild it triggers rebuilds FTS as a side effect of
    ``bulk_load``.

    Staleness window and bound: between a save and that next global rebuild,
    BM25 results are not actively refreshed by this function, so they may
    lag behind the graph -- which IS updated immediately here, so exact-name
    search and graph traversal are unaffected regardless of FTS freshness.
    (In practice, ``QUERY_FTS_INDEX`` on the pinned ``kuzu==0.11.3`` already
    reflects rows inserted/deleted on the same connection without an
    explicit rebuild -- verified empirically, not a documented Kuzu
    guarantee, so this function does not rely on it either way.)
    ``KuzuBackend.fts_search`` catches query failures per-table, so a stale
    or mid-mutation FTS index degrades results but never raises -- hybrid
    search cannot error because of this.  The window is bounded by the same
    ceiling that already governs embeddings (which have likewise never been
    refreshed per-save): ``GLOBAL_PHASE_INTERVAL`` seconds of edit quiescence
    (default 30s), or ``MAX_STALENESS_SECONDS`` under continuous churn that
    never quiesces (default 600s) -- see ``watcher.py``.

    Skip-if-clean interaction: ``_GlobalPhaseScheduler`` may skip a rebuild
    whose accumulated-change fingerprint matches the last *committed*
    fingerprint -- i.e. the pending changes reproduce content already
    reflected in the last successful ``bulk_load``.  That's safe for FTS
    too: the matching commit already rebuilt the indexes for that exact
    content, so skipping repeats no stale state.  Any save whose content
    actually differs from what was last committed yields a different
    fingerprint, which always drives a real rebuild (and thus a real FTS
    refresh).
    """
    for entry in file_entries:
        storage.remove_nodes_by_file(entry.path)

    storage.add_nodes(list(graph.iter_nodes()))
    storage.add_relationships(list(graph.iter_relationships()))


def build_graph(repo_path: Path) -> KnowledgeGraph:
    """Run phases 1-11 and return the in-memory graph (no storage load).

    This is used by branch comparison to build a graph snapshot without
    needing a storage backend.
    """
    graph, _ = run_pipeline(repo_path)
    return graph


def load_previous_embeddings(storage: StorageBackend) -> dict:
    """Snapshot stored embeddings so unchanged symbols skip re-encoding.

    Optional capability: backends without ``load_embeddings`` (or with a
    pre-``text_sha`` schema) yield ``{}``, which falls back to encoding
    everything.  Safe to call concurrently with readers — it is a plain
    read and needs no lock.
    """
    loader = getattr(storage, "load_embeddings", None)
    if loader is None:
        return {}
    try:
        return loader() or {}
    except Exception:
        logger.debug("Could not load previous embeddings", exc_info=True)
        return {}


def build_full_index(
    repo_path: Path,
    *,
    full: bool = True,
    skip_embeddings: bool = False,
    previous_embeddings: dict | None = None,
):
    """CPU phase of a full rebuild: pipeline + embeddings, no DB access.

    Runs WITHOUT any lock so concurrent queries keep flowing while the
    pipeline crunches.  Embedding failures (model unavailable, offline)
    degrade gracefully — the graph is still usable without fresh vectors.

    *previous_embeddings* (``{node_id: (text_sha, vector)}``, from
    :func:`load_previous_embeddings`) lets the embed step reuse vectors
    for symbols whose text did not change.

    Returns ``(graph, embeddings, result)``.  Shared by the watcher's
    global phase and the server-side reindex so the build/commit split
    has exactly one implementation.
    """
    graph, result = run_pipeline(repo_path, None, full=full)

    embeddings: list = []
    if not skip_embeddings:
        try:
            from synaptiq.core.embeddings.embedder import embed_graph

            embeddings = embed_graph(graph, previous=previous_embeddings)
            result.embeddings = len(embeddings)
        except Exception:
            logger.warning(
                "Embedding generation failed; vector search will be stale "
                "until it succeeds",
                exc_info=True,
            )
    return graph, embeddings, result


def commit_full_index(storage: StorageBackend, graph: KnowledgeGraph, embeddings: list) -> None:
    """I/O phase of a full rebuild — caller must hold the exclusive write lock.

    ``bulk_load`` resets the entire database including the embedding table,
    so embeddings must be re-stored here or vector search silently degrades
    to keyword-only.
    """
    storage.bulk_load(graph)
    if embeddings:
        storage.store_embeddings(embeddings)


def write_meta(data_dir: Path, repo_path: Path, result: PipelineResult) -> None:
    """Write ``meta.json`` with index stats and the indexing timestamp."""
    from synaptiq import __version__

    meta = {
        "version": __version__,
        "name": repo_path.name,
        "path": str(repo_path),
        "stats": {
            "files": result.files,
            "symbols": result.symbols,
            "relationships": result.relationships,
            "clusters": result.clusters,
            "flows": result.processes,
            "dead_code": result.dead_code,
            "coupled_pairs": result.coupled_pairs,
            "embeddings": result.embeddings,
            "phase_timings": result.phase_timings,
        },
        "last_indexed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
