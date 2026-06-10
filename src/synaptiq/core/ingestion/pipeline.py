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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from synaptiq.config.ignore import load_gitignore
from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import NodeLabel
from synaptiq.core.ingestion.calls import process_calls
from synaptiq.core.ingestion.community import process_communities
from synaptiq.core.ingestion.coupling import process_coupling
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

    report("Walking files", 0.0)
    gitignore = load_gitignore(repo_path)
    files = walk_repo(repo_path, gitignore)
    result.files = len(files)
    report("Walking files", 1.0)

    graph = KnowledgeGraph()

    report("Processing structure", 0.0)
    process_structure(files, graph)
    report("Processing structure", 1.0)

    report("Parsing code", 0.0)
    parse_data = process_parsing(files, graph)
    report("Parsing code", 1.0)

    report("Resolving imports", 0.0)
    process_imports(parse_data, graph)
    report("Resolving imports", 1.0)

    report("Tracing calls", 0.0)
    process_calls(parse_data, graph)
    report("Tracing calls", 1.0)

    report("Linking REST endpoints", 0.0)
    result.rest_links = process_rest_linking(parse_data, graph)
    report("Linking REST endpoints", 1.0)

    report("Extracting heritage", 0.0)
    process_heritage(parse_data, graph)
    report("Extracting heritage", 1.0)

    report("Analyzing types", 0.0)
    process_types(parse_data, graph)
    report("Analyzing types", 1.0)

    report("Detecting communities", 0.0)
    result.clusters = process_communities(graph)
    report("Detecting communities", 1.0)

    report("Detecting execution flows", 0.0)
    result.processes = process_processes(graph)
    report("Detecting execution flows", 1.0)

    report("Finding dead code", 0.0)
    result.dead_code = process_dead_code(graph)
    report("Finding dead code", 1.0)

    report("Analyzing git history", 0.0)
    result.coupled_pairs = process_coupling(graph, repo_path)
    report("Analyzing git history", 1.0)

    if storage is not None:
        report("Loading to storage", 0.0)
        storage.bulk_load(graph)
        report("Loading to storage", 1.0)

        if not skip_embeddings:
            report("Generating embeddings", 0.0)
            from synaptiq.core.embeddings.embedder import embed_graph

            embeddings = embed_graph(graph)
            if embeddings:
                storage.store_embeddings(embeddings)
            result.embeddings = len(embeddings)
            report("Generating embeddings", 1.0)

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
    """
    for entry in file_entries:
        storage.remove_nodes_by_file(entry.path)

    storage.add_nodes(list(graph.iter_nodes()))
    storage.add_relationships(list(graph.iter_relationships()))
    storage.rebuild_fts_indexes()


def build_graph(repo_path: Path) -> KnowledgeGraph:
    """Run phases 1-11 and return the in-memory graph (no storage load).

    This is used by branch comparison to build a graph snapshot without
    needing a storage backend.
    """
    graph, _ = run_pipeline(repo_path)
    return graph


logger = logging.getLogger(__name__)


def build_full_index(
    repo_path: Path,
    *,
    full: bool = True,
    skip_embeddings: bool = False,
):
    """CPU phase of a full rebuild: pipeline + embeddings, no DB access.

    Runs WITHOUT any lock so concurrent queries keep flowing while the
    pipeline crunches.  Embedding failures (model unavailable, offline)
    degrade gracefully — the graph is still usable without fresh vectors.

    Returns ``(graph, embeddings, result)``.  Shared by the watcher's
    global phase and the server-side reindex so the build/commit split
    has exactly one implementation.
    """
    graph, result = run_pipeline(repo_path, None, full=full)

    embeddings: list = []
    if not skip_embeddings:
        try:
            from synaptiq.core.embeddings.embedder import embed_graph

            embeddings = embed_graph(graph)
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
        },
        "last_indexed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
