"""Personalized PageRank for focus-file biased search ranking.

When an agent provides ``focus_files``, this module builds an igraph
directed graph from the knowledge graph's CALLS, IMPORTS, and USES_TYPE
edges and runs Personalized PageRank with the restart vector concentrated
on symbols defined in those files.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from synaptiq.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)


def personalized_pagerank(
    storage: StorageBackend,
    focus_files: list[str],
    damping: float = 0.85,
) -> dict[str, float]:
    """Compute Personalized PageRank biased toward *focus_files*.

    Parameters
    ----------
    storage:
        An initialised storage backend.
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
    return _cached_ppr(storage, focus_key, damping)


@lru_cache(maxsize=16)
def _cached_ppr(
    storage: StorageBackend,
    focus_key: frozenset[str],
    damping: float,
) -> dict[str, float]:
    """Cached PPR computation keyed on the focus file set.

    Note: the cache keys on ``id(storage)`` (default object hash).
    Results are stale if the graph changes while the storage object
    lives.  This is acceptable for a single MCP session — the cache
    is small (maxsize=16) and clears on server restart.
    """
    try:
        import igraph as ig
    except ImportError:
        logger.warning("igraph not available — skipping Personalized PageRank")
        return {}

    # Build adjacency from the graph.
    edge_types = ("calls", "imports", "uses_type")
    edges = _get_edges(storage, edge_types)
    if not edges:
        return {}

    # Map node IDs to igraph vertex indices.
    node_set: set[str] = set()
    for src, tgt in edges:
        node_set.add(src)
        node_set.add(tgt)
    node_list = sorted(node_set)
    id_to_idx = {nid: i for i, nid in enumerate(node_list)}

    g = ig.Graph(directed=True)
    g.add_vertices(len(node_list))
    edge_list = [
        (id_to_idx[s], id_to_idx[t]) for s, t in edges if s in id_to_idx and t in id_to_idx
    ]
    g.add_edges(edge_list)

    # Build restart vector: concentrate on symbols in focus files.
    focus_indices: set[int] = set()
    for nid, idx in id_to_idx.items():
        # Node IDs are formatted as label:file_path:symbol_name
        parts = nid.split(":", 2)
        if len(parts) >= 2:
            file_path = parts[1]
            if file_path in focus_key:
                focus_indices.add(idx)

    if not focus_indices:
        return {}

    reset = [0.0] * len(node_list)
    weight = 1.0 / len(focus_indices)
    for idx in focus_indices:
        reset[idx] = weight

    try:
        scores = g.personalized_pagerank(damping=damping, reset=reset)
    except Exception:
        logger.debug("PPR computation failed", exc_info=True)
        return {}

    return {node_list[i]: s for i, s in enumerate(scores) if s > 0}


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
