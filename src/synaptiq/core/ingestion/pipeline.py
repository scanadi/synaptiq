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
from synaptiq.core.embeddings.embedder import DEFAULT_TIER_NAME
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
from synaptiq.core.ingestion.incremental import (
    REASON_INCREMENTAL,
    REASON_NO_MANIFEST,
    IncrementalPlan,
    build_current_manifest,
    plan_incremental,
    should_consolidate,
)
from synaptiq.core.ingestion.incremental_build import build_incremental_delta
from synaptiq.core.ingestion.manifest import (
    IndexPending,
    Manifest,
    build_manifest,
    content_sha,
)
from synaptiq.core.ingestion.parser_phase import process_parsing
from synaptiq.core.ingestion.processes import process_processes
from synaptiq.core.ingestion.rest_linking import process_rest_linking
from synaptiq.core.ingestion.structure import process_structure
from synaptiq.core.ingestion.types import process_types
from synaptiq.core.ingestion.walker import FileEntry, walk_repo
from synaptiq.core.storage.base import GraphDelta, StorageBackend

logger = logging.getLogger(__name__)

#: Force a full rebuild instead of an incremental delta (e.g. ``analyze --full``).
REASON_FORCED_FULL = "forced_full"

# Node id label prefixes that are per-file *symbols* (not File/Folder/global
# artifacts) — used to count "symbols updated" from a delta's upserts.
_SYMBOL_ID_PREFIXES: tuple[str, ...] = tuple(
    f"{label.value}:"
    for label in NodeLabel
    if label not in (NodeLabel.FILE, NodeLabel.FOLDER, NodeLabel.COMMUNITY, NodeLabel.PROCESS)
)


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
    # Embedding tier name ("quality" / "fast") this run encoded (or would
    # encode) with, set unconditionally from run_pipeline's `embedding_tier`
    # argument (default "quality") — persisted into meta.json's
    # stats.embedding_model by write_meta so the query side and later
    # rebuilds know which model produced the stored vectors (W4.4). Set even
    # on a run that skipped embedding generation entirely (e.g. `analyze
    # --embeddings off`): the Embedding table ends up empty in that case, so
    # there is nothing stored to mismatch it against.
    embedding_model: str = ""
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
    embedding_tier: str = DEFAULT_TIER_NAME,
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
    embedding_tier:
        Embedding tier name (``"quality"`` or ``"fast"``, see
        ``core/embeddings/embedder.py``) to encode with when *storage* is
        given and *skip_embeddings* is ``False``. Ignored otherwise.
        Recorded into ``result.embedding_model`` regardless, so
        :func:`write_meta` persists the caller's requested tier even on a
        run that didn't itself encode (e.g. ``--embeddings off``) — the
        Embedding table ends up empty in that case, so there is nothing to
        mismatch it against.

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
            git_future = git_log_executor.submit(collect_coupling_commits, repo_path, graph_files)

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

    result.embedding_model = embedding_tier

    if storage is not None:
        # Snapshot before bulk_load — it resets the whole database,
        # embedding table included.
        previous_embeddings = load_previous_embeddings(storage) if not skip_embeddings else {}

        with timed_phase("Loading to storage"):
            storage.bulk_load(graph)

        if not skip_embeddings:
            with timed_phase("Generating embeddings"):
                from synaptiq.core.embeddings.embedder import embed_graph
                from synaptiq.core.embeddings.lazy_worker import stamp_inline_complete

                embeddings = embed_graph(graph, tier=embedding_tier, previous=previous_embeddings)
                if embeddings:
                    storage.store_embeddings(embeddings)
                result.embeddings = len(embeddings)
                # 2.0.4 (BUG 2): this is an inline (non-lazy-worker) embed-store —
                # a stale deferred/failed/encoding sentinel from an earlier lazy
                # worker run must not outlive the vectors it was waiting for.
                data_dir = getattr(storage, "data_dir", None)
                if data_dir is not None:
                    stamp_inline_complete(data_dir, result.embeddings)

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
    fresh graph (see ``LadybugBackend.bulk_load``), so no separate "FTS dirty"
    flag is needed -- the global phase's existing change-tracking in
    ``_GlobalPhaseScheduler`` (``watcher.py``) already gates it, and every
    non-skipped rebuild it triggers rebuilds FTS as a side effect of
    ``bulk_load``.

    Staleness window and bound: between a save and that next global rebuild,
    BM25 results are not actively refreshed by this function, so they may
    lag behind the graph -- which IS updated immediately here, so exact-name
    search and graph traversal are unaffected regardless of FTS freshness.
    (An earlier ``QUERY_FTS_INDEX`` reflecting rows inserted/deleted on the
    same connection without an explicit rebuild was observed empirically on
    kuzu 0.11.3 -- NOT re-verified on LadybugDB, never a documented guarantee,
    and this function does not rely on it either way.)
    ``LadybugBackend.fts_search`` catches query failures per-table, so a stale
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
    tier: str | None = None,
):
    """CPU phase of a full rebuild: pipeline + embeddings, no DB access.

    Runs WITHOUT any lock so concurrent queries keep flowing while the
    pipeline crunches.  Embedding failures (model unavailable, offline)
    degrade gracefully — the graph is still usable without fresh vectors.

    Daemon path: ``serve``/``watch`` global rebuilds embed **synchronously**
    here (warm + cheap via ``text_sha`` reuse).  The lazy background worker
    (``analyze --embeddings lazy``, see ``core/embeddings/lazy_worker.py``) is a
    CLI-``analyze`` concept only — two background encoders colliding with a
    daemon's own rebuild would be worse than a short synchronous re-embed.

    *previous_embeddings* (``{node_id: (text_sha, vector)}``, from
    :func:`load_previous_embeddings`) lets the embed step reuse vectors
    for symbols whose text did not change.

    *tier* is the embedding tier name to encode with. ``None`` (default)
    re-derives it from ``repo_path/.synaptiq/meta.json`` via
    :func:`~synaptiq.core.embeddings.embedder.tier_from_meta`: the daemon
    rebuild paths that call this (the watcher's global phase, the primary's
    socket ``reindex`` handler) have no per-cycle CLI flag to take it from,
    so they must keep following whatever tier the index already claims
    rather than drifting back to the default on every rebuild — the same
    "re-derive from meta, don't guess" rule the lazy worker follows. Pass an
    explicit tier to override (e.g. tests).

    Returns ``(graph, embeddings, result)``.  Shared by the watcher's
    global phase and the server-side reindex so the build/commit split
    has exactly one implementation.
    """
    graph, result = run_pipeline(repo_path, None, full=full)

    resolved_tier = tier
    if resolved_tier is None:
        from synaptiq.core.embeddings.embedder import tier_from_meta

        resolved_tier = tier_from_meta(repo_path / ".synaptiq").name
    result.embedding_model = resolved_tier

    embeddings: list = []
    if not skip_embeddings:
        try:
            from synaptiq.core.embeddings.embedder import embed_graph

            embeddings = embed_graph(graph, tier=resolved_tier, previous=previous_embeddings)
            result.embeddings = len(embeddings)
        except Exception:
            logger.warning(
                "Embedding generation failed; vector search will be stale until it succeeds",
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


def write_meta(
    data_dir: Path,
    repo_path: Path,
    result: PipelineResult,
    *,
    mode: str = "full",
    reason: str = "",
    changed_files: int = 0,
    dependents: int = 0,
    symbols_updated: int = 0,
) -> None:
    """Write ``meta.json`` with index stats and the indexing timestamp.

    W3.2e: also records ``manifest_version`` and a ``last_index`` block (the path
    the last update took — ``incremental``/``full``, and why) so ``synaptiq
    status`` can show "last analyze: incremental (3 files) / full". Full builds
    default ``mode="full"``; the incremental path passes its scoped counts.
    ``last_indexed_at`` is unchanged, so the W4.5 freshness trailer is unaffected.
    """
    from synaptiq import __version__
    from synaptiq.core.ingestion.manifest import CURRENT_MANIFEST_VERSION

    meta = {
        "version": __version__,
        "name": repo_path.name,
        "path": str(repo_path),
        "manifest_version": CURRENT_MANIFEST_VERSION,
        "stats": {
            "files": result.files,
            "symbols": result.symbols,
            "relationships": result.relationships,
            "clusters": result.clusters,
            "flows": result.processes,
            "dead_code": result.dead_code,
            "coupled_pairs": result.coupled_pairs,
            "embeddings": result.embeddings,
            # Tier ("quality" / "fast") the stored vectors were encoded with
            # (W4.4) — read back by tier_from_meta() so the query side and
            # later rebuilds always match the model that produced them.
            "embedding_model": result.embedding_model,
            "phase_timings": result.phase_timings,
        },
        "last_index": {
            "mode": mode,
            "reason": reason,
            "changed_files": changed_files,
            "dependents": dependents,
            "symbols_updated": symbols_updated,
        },
        "last_indexed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


# ===========================================================================
# Incremental indexing orchestration (W3.2e)
# ===========================================================================


@dataclass
class IncrementalOutcome:
    """Result of the incremental decision and — when taken — the scoped apply.

    When :attr:`full_rebuild_required` the caller runs its OWN full-build path
    (the ``run_pipeline`` / ``build_full_index`` it already owns, with its own
    embedding handling); :func:`run_incremental` only *decides* and never triggers
    a full build itself, so the full + embedding paths stay in one place. Otherwise
    the scoped delta has already been applied to storage and the manifest persisted,
    and the scoped fields describe what happened for the output/status line.
    """

    full_rebuild_required: bool
    reason: str
    plan: IncrementalPlan | None = None
    previous: Manifest | None = None
    walk: list[FileEntry] = field(default_factory=list)
    # Populated only on the incremental branch:
    delta: GraphDelta | None = None
    new_manifest: Manifest | None = None
    changed_files: int = 0
    dependents: int = 0
    symbols_updated: int = 0
    phase_timings: dict[str, float] = field(default_factory=dict)

    def upsert_graph(self) -> KnowledgeGraph:
        """A graph of just the delta's upserted nodes + fresh edges.

        The input the embedding partition needs to find *which* changed symbols to
        re-encode — unchanged symbols keep their stored vectors (``apply_graph_delta``
        never wipes the Embedding table), so only these need attention.
        """
        graph = KnowledgeGraph()
        if self.delta is None:
            return graph
        for node in self.delta.nodes_upsert:
            graph.add_node(node)
        for rel in self.delta.edges_add:
            graph.add_relationship(rel)
        return graph


def _reparse_changed_files(walk: list[FileEntry], previous: Manifest) -> KnowledgeGraph:
    """Structure+parse (phases 2-3) only the content-changed / added files.

    Enough for :func:`build_current_manifest` to give the planner fresh
    ``symbol_sigs`` (body-only vs identity classification). No resolution — the
    planner reads only ``symbol_sigs`` off the current manifest; edges are
    re-resolved later, inside ``build_incremental_delta``, over the full scope.
    """
    prev_sha = {path: fm.content_sha for path, fm in previous.files.items()}
    changed = [entry for entry in walk if content_sha(entry.content) != prev_sha.get(entry.path)]
    graph = KnowledgeGraph()
    process_structure(changed, graph)
    process_parsing(changed, graph)
    return graph


def _accumulate_pending(new_manifest: Manifest, previous: Manifest, plan: IncrementalPlan) -> None:
    """Fold this apply into the since-consolidation counters + carry coupling.

    An incremental apply defers every global phase, so all ``*_dirty`` flags go
    ``True`` and the affected-symbol / changed-file / applies counters accumulate
    (the consolidation gates read them). ``consolidated_at`` and ``git_head`` carry
    forward unchanged — no consolidation happened and coupling was not recomputed.
    """
    prev = previous.index.pending
    new_manifest.index.pending = IndexPending(
        affected_symbols=prev.affected_symbols + plan.affected_symbol_count,
        changed_files=prev.changed_files + plan.changed_file_count,
        applies_since_consolidation=prev.applies_since_consolidation + 1,
        fts_dirty=True,
        hnsw_dirty=True,
        community_dirty=True,
        process_dirty=True,
    )
    new_manifest.index.consolidated_at = previous.index.consolidated_at
    new_manifest.index.git_head = previous.index.git_head


def stamp_full_manifest(
    storage: StorageBackend,
    graph: KnowledgeGraph,
    *,
    tool_version: str = "",
    git_head: str | None = None,
) -> None:
    """Overwrite the stored manifest with a fresh one carrying *git_head*.

    ``bulk_load`` stamps the manifest as a by-product but cannot see git (it holds
    only the graph), so it leaves ``git_head=None``. W3.2e callers run this right
    after a full build to populate ``git_head`` — the D8 coupling-gate baseline for
    the next incremental decision — and to reset the pending/consolidated_at stamp
    to fresh. Best-effort: a manifest write failure must never fail an otherwise
    successful index (the manifest is an optimization with a full-rebuild fallback).
    """
    try:
        storage.write_manifest(build_manifest(graph, tool_version=tool_version, git_head=git_head))
    except Exception:
        logger.warning(
            "stamp_full_manifest failed; the git-HEAD consolidation gate will fall "
            "back to a full rebuild on the next update",
            exc_info=True,
        )


def _count_manifest_symbols(manifest: Manifest) -> int:
    """Count per-file *symbol* ids (excludes File/Folder/global-artifact ids)."""
    return sum(
        1
        for fm in manifest.files.values()
        for sid in fm.symbol_ids
        if sid.startswith(_SYMBOL_ID_PREFIXES)
    )


def build_incremental_result(data_dir: Path, outcome: IncrementalOutcome) -> PipelineResult:
    """A :class:`PipelineResult` for ``meta.json`` after an incremental apply.

    File/symbol counts come fresh from the new manifest; the global-phase stats
    (clusters/flows/dead_code/coupled_pairs/embeddings/relationships) are carried
    from the prior ``meta.json`` — they are deferred on the incremental path and
    stay bounded-stale until the next consolidation, so reporting the last known
    value is more honest than zeroing them.
    """
    result = PipelineResult(incremental=True, changed_files=outcome.changed_files)
    result.files = len(outcome.walk)
    if outcome.new_manifest is not None:
        result.symbols = _count_manifest_symbols(outcome.new_manifest)
    result.phase_timings = dict(outcome.phase_timings)

    stats: dict = {}
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        try:
            stats = json.loads(meta_path.read_text(encoding="utf-8")).get("stats", {}) or {}
        except (ValueError, OSError):
            stats = {}
    result.relationships = int(stats.get("relationships", 0) or 0)
    result.clusters = int(stats.get("clusters", 0) or 0)
    result.processes = int(stats.get("flows", 0) or 0)
    result.dead_code = int(stats.get("dead_code", 0) or 0)
    result.coupled_pairs = int(stats.get("coupled_pairs", 0) or 0)
    result.embeddings = int(stats.get("embeddings", 0) or 0)
    result.embedding_model = str(stats.get("embedding_model", "") or "")
    return result


def run_incremental(
    repo_path: Path,
    storage: StorageBackend,
    *,
    tool_version: str = "",
    repair_inbound: bool = False,
    force_full: bool = False,
    check_consolidation: bool = True,
    previous: Manifest | None = None,
    git_head: str | None = None,
    now: float | None = None,
    apply: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
) -> IncrementalOutcome:
    """Decide full-vs-incremental and, when incremental, apply the scoped delta.

    The shared orchestration behind ``analyze`` (default incremental) and the
    watcher's global phase (D10). Reads the stored manifest (unless *previous* is
    supplied — the watcher passes its resident copy), walks + reparses only the
    content-changed / added files to build the *current* manifest, plans, and:

    * ``force_full`` / no manifest / plan verdict / consolidation gate ⇒ returns
      ``full_rebuild_required=True`` with a reason; the caller runs its own full
      build (so the full path and its embedding handling live in one place).
    * otherwise ⇒ builds the :class:`GraphDelta`, applies it in one storage
      transaction, accumulates the pending/staleness counters, carries
      ``git_head`` + ``consolidated_at`` forward, and persists the new manifest —
      all here — returning the scoped counts. Embeddings are the caller's (deferred
      per mode), since ``apply_graph_delta`` leaves stored vectors intact.

    *git_head* (current ``git rev-parse HEAD``) drives the D8 coupling gate; *now*
    (``time.time()``) the staleness gate. Both are the caller's to compute so this
    stays testable. Phase timings are recorded under incremental-specific names so
    ``--profile`` stays honest about which path ran.

    *apply* controls the final storage mutation: ``True`` (analyze, single-process
    file lock) applies the delta + persists the manifest inline; ``False`` (the
    watcher) does all the read-only compute here — walk, reparse, plan, resolve —
    and returns the delta + already-accumulated manifest on the outcome for the
    caller to apply under its RW write lock, so the heavy compute never blocks
    readers.
    """
    timings: dict[str, float] = {}

    def report(phase: str, pct: float) -> None:
        if progress_callback is not None:
            progress_callback(phase, pct)

    @contextmanager
    def timed(phase: str):
        report(phase, 0.0)
        t0 = time.monotonic()
        try:
            yield
        finally:
            timings[phase] = time.monotonic() - t0
            report(phase, 1.0)

    if previous is None:
        previous = storage.read_manifest()

    with timed("Walking files"):
        gitignore = load_gitignore(repo_path)
        walk = walk_repo(repo_path, gitignore)

    if force_full:
        return IncrementalOutcome(True, REASON_FORCED_FULL, previous=previous, walk=walk,
                                  phase_timings=timings)
    if previous is None:
        return IncrementalOutcome(True, REASON_NO_MANIFEST, walk=walk, phase_timings=timings)

    with timed("Reparsing changed files"):
        reparse_graph = _reparse_changed_files(walk, previous)
        current = build_current_manifest(
            walk, reparse_graph, tool_version=tool_version, git_head=git_head
        )

    with timed("Planning incremental scope"):
        plan = plan_incremental(previous, current, repair_inbound=repair_inbound)

    if plan.full_rebuild_required:
        return IncrementalOutcome(True, plan.reason, plan=plan, previous=previous, walk=walk,
                                  phase_timings=timings)

    if check_consolidation:
        git_head_moved = (
            git_head is not None
            and previous.index.git_head is not None
            and previous.index.git_head != git_head
        )
        consolidate, why = should_consolidate(
            previous, plan, git_head_moved=git_head_moved, now=now
        )
        if consolidate:
            return IncrementalOutcome(True, why, plan=plan, previous=previous, walk=walk,
                                      phase_timings=timings)

    with timed("Resolving incremental delta"):
        delta, new_manifest = build_incremental_delta(
            plan, previous, walk, tool_version=tool_version, git_head=previous.index.git_head
        )
        _accumulate_pending(new_manifest, previous, plan)

    if apply:
        with timed("Applying delta to storage"):
            storage.apply_graph_delta(delta)
            storage.write_manifest(new_manifest)

    symbols_updated = sum(1 for n in delta.nodes_upsert if n.id.startswith(_SYMBOL_ID_PREFIXES))
    return IncrementalOutcome(
        full_rebuild_required=False,
        reason=REASON_INCREMENTAL,
        plan=plan,
        previous=previous,
        walk=walk,
        delta=delta,
        new_manifest=new_manifest,
        changed_files=plan.changed_file_count,
        dependents=len(plan.dependents),
        symbols_updated=symbols_updated,
        phase_timings=timings,
    )
