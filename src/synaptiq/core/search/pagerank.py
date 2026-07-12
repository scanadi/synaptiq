"""Personalized PageRank for focus-file biased search ranking.

When an agent provides ``focus_files``, this module builds an igraph
directed graph from the knowledge graph's CALLS, IMPORTS, and USES_TYPE
edges and runs Personalized PageRank with the restart vector concentrated
on symbols defined in those files.

The graph projection (edge dump + igraph construction) is the expensive
part and is independent of the focus set, so it is cached per storage
*generation* — many concurrent agents asking about different files share
one projection, and a watcher reindex (which bumps the generation)
invalidates it.  Only the cheap reset-vector PPR runs per focus set.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import NamedTuple

from synaptiq.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_EDGE_TYPES = ("calls", "imports", "uses_type")


class _GraphProjection(NamedTuple):
    """An igraph view of the code graph plus lookup indexes."""

    node_list: list[str]
    idx_by_file: dict[str, list[int]]
    graph: object  # ig.Graph — typed loosely so igraph stays an optional import


def personalized_pagerank(
    storage: StorageBackend,
    focus_files: list[str],
    damping: float = 0.85,
) -> dict[str, float]:
    """Compute Personalized PageRank biased toward *focus_files*.

    Parameters
    ----------
    storage:
        An initialised storage backend.  Backends exposing a
        ``generation`` counter (e.g. :class:`LadybugBackend`) get correct
        cache invalidation across reindexes; others fall back to
        caching for the lifetime of the storage object.
    focus_files:
        File paths whose symbols receive elevated restart probability.
    damping:
        PageRank damping factor (default 0.85).

    Returns
    -------
    dict[str, float]
        Mapping of node IDs to their PPR scores.
    """
    focus_key = frozenset(focus_files)
    generation = getattr(storage, "generation", 0)
    return _cached_ppr(storage, generation, focus_key, damping)


@lru_cache(maxsize=16)
def _cached_ppr(
    storage: StorageBackend,
    generation: int,
    focus_key: frozenset[str],
    damping: float,
) -> dict[str, float]:
    """PPR scores for one focus set, computed over the shared projection."""
    projection = _cached_projection(storage, generation)
    if projection is None:
        return {}

    focus_indices: list[int] = []
    for file_path in focus_key:
        focus_indices.extend(projection.idx_by_file.get(file_path, ()))
    if not focus_indices:
        return {}

    reset = [0.0] * len(projection.node_list)
    weight = 1.0 / len(focus_indices)
    for idx in focus_indices:
        reset[idx] = weight

    try:
        scores = projection.graph.personalized_pagerank(damping=damping, reset=reset)
    except Exception:
        logger.debug("PPR computation failed", exc_info=True)
        return {}

    return {projection.node_list[i]: s for i, s in enumerate(scores) if s > 0}


@lru_cache(maxsize=2)
def _cached_projection(
    storage: StorageBackend, generation: int
) -> _GraphProjection | None:
    """Build the igraph projection once per storage generation."""
    try:
        import igraph as ig
    except ImportError:
        logger.warning("igraph not available — skipping Personalized PageRank")
        return None

    edges = _get_edges(storage, _EDGE_TYPES)
    if not edges:
        return None

    node_set: set[str] = set()
    for src, tgt in edges:
        node_set.add(src)
        node_set.add(tgt)
    node_list = sorted(node_set)
    id_to_idx = {nid: i for i, nid in enumerate(node_list)}

    g = ig.Graph(directed=True)
    g.add_vertices(len(node_list))
    g.add_edges([(id_to_idx[s], id_to_idx[t]) for s, t in edges])

    # Node IDs are formatted as label:file_path:symbol_name — index the
    # path segment so per-call focus lookup is a dict hit, not a scan.
    idx_by_file: dict[str, list[int]] = {}
    for nid, idx in id_to_idx.items():
        parts = nid.split(":", 2)
        if len(parts) >= 2:
            idx_by_file.setdefault(parts[1], []).append(idx)

    return _GraphProjection(node_list=node_list, idx_by_file=idx_by_file, graph=g)


def _get_edges(
    storage: StorageBackend, edge_types: tuple[str, ...]
) -> list[tuple[str, str]]:
    """Extract (source_id, target_id) pairs for the given rel types."""
    if hasattr(storage, "get_all_edges"):
        return storage.get_all_edges(edge_types)

    # Fallback: query via raw Cypher.
    types_str = ", ".join(f"'{t}'" for t in edge_types)
    rows = (
        storage.execute_raw(
            f"MATCH (a)-[r:CodeRelation]->(b) "
            f"WHERE r.rel_type IN [{types_str}] "
            f"RETURN a.id, b.id"
        )
        or []
    )
    return [(r[0], r[1]) for r in rows if r[0] and r[1]]
