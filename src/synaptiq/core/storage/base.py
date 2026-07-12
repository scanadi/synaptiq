"""Storage backend abstraction for Synaptiq.

Defines the :class:`StorageBackend` protocol that all concrete storage
implementations (LadybugDB, Neo4j, in-memory, etc.) must satisfy, along with
supporting data classes for search results and embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, GraphRelationship


@dataclass
class SearchResult:
    """A single result from a full-text or vector search."""

    node_id: str
    score: float
    node_name: str = ""
    file_path: str = ""
    label: str = ""
    snippet: str = ""

@dataclass
class NodeEmbedding:
    """An embedding vector associated with a graph node.

    ``text_sha`` is the SHA-256 of the natural-language text the vector
    was generated from — the reuse key that lets full rebuilds skip
    re-encoding symbols whose text did not change.
    """

    node_id: str
    embedding: list[float] = field(default_factory=list)
    text_sha: str = ""


@dataclass
class EdgeRef:
    """A relationship addressed by its endpoints and logical kind.

    The delete key for an incremental delta. The rel-table group stores the
    logical kind in the ``rel_type`` property (not as a label), so an edge is
    identified by the ``(rel_type, source, target)`` triple.
    """

    rel_type: str
    source: str
    target: str


@dataclass
class GraphDelta:
    """A surgical, scoped change set applied to storage in one transaction.

    Produced by the incremental resolver (W3.2c) and consumed by
    :meth:`StorageBackend.apply_graph_delta` (W3.2d). This is a **frozen
    interface** — other sub-packages build against these field names and types
    (incremental-indexing design §5.5).

    Fields:

    * ``nodes_upsert`` — added / body-only-changed / identity-changed nodes,
      applied as an idempotent ``MERGE`` upsert (a body-only edit becomes a
      property refresh with no edge impact).
    * ``nodes_remove`` — ids of genuinely-removed symbols (and deleted-file
      symbols). Removed surgically by id so surviving symbols keep their
      inbound edges; removing a symbol cascades only its own dangling edges.
    * ``edges_add`` — freshly resolved edges, inserted idempotently (``MERGE``).
    * ``edges_remove`` — edges the re-resolved files previously contributed,
      deleted before re-insert so the apply is idempotent without a global
      ``DETACH DELETE``.
    * ``dead_recount`` — symbol ids whose incoming-CALLS in-degree may have
      crossed zero (targets of added/removed CALLS plus upserted/removed
      nodes); ``is_dead`` is recomputed locally for each.
    """

    nodes_upsert: list[GraphNode] = field(default_factory=list)
    nodes_remove: list[str] = field(default_factory=list)
    edges_add: list[GraphRelationship] = field(default_factory=list)
    edges_remove: list[EdgeRef] = field(default_factory=list)
    dead_recount: set[str] = field(default_factory=set)


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol that every Synaptiq storage backend must implement.

    Covers the full lifecycle of graph persistence: initialisation,
    CRUD operations on nodes and relationships, querying, full-text
    search, vector search, and incremental re-indexing support.
    """

    def initialize(self, path: Path) -> None:
        """Open or create the backing store at *path*."""
        ...

    def close(self) -> None:
        """Release resources held by the backend."""
        ...

    def add_nodes(self, nodes: list[GraphNode]) -> None:
        """Insert or upsert a batch of nodes."""
        ...

    def add_relationships(self, rels: list[GraphRelationship]) -> None:
        """Insert or upsert a batch of relationships."""
        ...

    def remove_nodes_by_file(self, file_path: str) -> int:
        """Remove all nodes originating from *file_path*.

        Returns:
            The number of nodes removed.
        """
        ...

    def remove_nodes_by_id(self, node_ids: list[str]) -> int:
        """Surgically remove only the nodes with the given ids.

        Unlike :meth:`remove_nodes_by_file`, this leaves other symbols from the
        same file — and their inbound edges from unchanged files — in place. It
        is the scoped-removal primitive used by the incremental delta path.

        Returns:
            The number of nodes actually removed.
        """
        ...

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return a single node by ID, or ``None`` if not found."""
        ...

    def get_callers(self, node_id: str) -> list[GraphNode]:
        """Return nodes that call the node identified by *node_id*."""
        ...

    def get_callees(self, node_id: str) -> list[GraphNode]:
        """Return nodes called by the node identified by *node_id*."""
        ...

    def get_type_refs(self, node_id: str) -> list[GraphNode]:
        """Return nodes that reference the type identified by *node_id*."""
        ...

    def traverse(self, start_id: str, depth: int, direction: str = "callers") -> list[GraphNode]:
        """Breadth-first traversal up to *depth* hops from *start_id*.

        Args:
            direction: ``"callers"`` follows incoming CALLS (blast radius),
                       ``"callees"`` follows outgoing CALLS (dependencies).
        """
        ...

    def traverse_with_depth(
        self, start_id: str, depth: int, direction: str = "callers"
    ) -> list[tuple[GraphNode, int]]:
        """Like :meth:`traverse` but returns ``(node, hop_distance)`` pairs."""
        ...

    def get_callers_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return callers paired with the CALLS edge confidence score."""
        ...

    def get_callees_with_confidence(self, node_id: str) -> list[tuple[GraphNode, float]]:
        """Return callees paired with the CALLS edge confidence score."""
        ...

    def get_process_memberships(self, node_ids: list[str]) -> dict[str, str]:
        """Return ``{node_id: process_name}`` for nodes that belong to a process."""
        ...

    def load_graph(self) -> KnowledgeGraph:
        """Load the full graph into an in-memory KnowledgeGraph."""
        ...

    def execute_raw(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Execute a raw backend-specific query string with optional parameters."""
        ...

    def fts_search(self, query: str, limit: int) -> list[SearchResult]:
        """Full-text search across indexed node content."""
        ...

    def fuzzy_search(
        self, query: str, limit: int, max_distance: int = 2
    ) -> list[SearchResult]:
        """Fuzzy name search by edit distance."""
        ...

    def store_embeddings(self, embeddings: list[NodeEmbedding]) -> None:
        """Persist embedding vectors for the given nodes."""
        ...

    def vector_search(self, vector: list[float], limit: int) -> list[SearchResult]:
        """Find the closest nodes to *vector* by cosine similarity."""
        ...

    def get_indexed_files(self) -> dict[str, str]:
        """Return a mapping of ``{file_path: content_hash}`` for all indexed files."""
        ...

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """Replace the entire store contents with *graph*."""
        ...

    def apply_graph_delta(self, delta: GraphDelta) -> None:
        """Apply a scoped :class:`GraphDelta` atomically in one transaction.

        Applies, in order: ``edges_remove``, ``nodes_remove``, ``nodes_upsert``,
        ``edges_add``, then the scoped ``is_dead`` recount over ``dead_recount``.
        A mid-delta failure rolls the whole change set back. Global artifacts
        (full-text index, vector index, communities, processes) are left stale
        and reconciled later at consolidation — no rebuild happens here.
        """
        ...
