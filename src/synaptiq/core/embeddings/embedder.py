"""Batch embedding pipeline for Synaptiq knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable symbol node, encodes them using *fastembed*, and returns a
list of :class:`NodeEmbedding` objects ready for storage.

Only code-level symbol nodes are embedded.  Structural nodes (Folder,
Community, Process) are deliberately skipped — they lack the semantic
richness that makes embedding worthwhile.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from synaptiq.core.embeddings.text import build_class_method_index, generate_text
from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import NodeLabel
from synaptiq.core.storage.base import NodeEmbedding

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_model(model_name: str) -> TextEmbedding:
    from fastembed import TextEmbedding

    from synaptiq.core.resources import current_limits

    # current_limits() caps ONNX intra-op threads per the active profile:
    # strict server caps for daemons, the polite max(2, cores - 2) default
    # for interactive commands, or an explicit `analyze --jobs` override.
    # 0 → fastembed/ONNX default (all cores).
    threads = current_limits().embed_threads
    return TextEmbedding(model_name=model_name, threads=threads or None)


# Labels worth embedding — skip Folder, Community, Process (structural only).
EMBEDDABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.FILE,
        NodeLabel.FUNCTION,
        NodeLabel.CLASS,
        NodeLabel.METHOD,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
        NodeLabel.ENUM,
    }
)


def embeddable_node_count(graph: KnowledgeGraph) -> int:
    """Number of nodes :func:`embed_graph` would encode for *graph*.

    Shared by ``analyze --embeddings lazy`` (to tell the user how many
    vectors are being encoded in the background) and the lazy worker (to
    size its progress totals) so both report the same ``N``.
    """
    return sum(1 for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS)


def _partition_texts(
    graph: KnowledgeGraph,
    previous: dict[str, tuple[str, list[float]]] | None,
) -> tuple[list, list[str], list[str], list[NodeEmbedding | None], list[int]]:
    """Compute nodes/texts/``text_sha``s and split into reused vs. pending.

    Single implementation shared by :func:`embed_graph` (which goes on to
    encode the pending indices) and :func:`partition_embeddings` (which only
    needs the split) — the reused/pending decision can never drift between
    the two callers.

    Returns:
        ``(nodes, texts, shas, results, pending)`` where *results* has one
        slot per node — a :class:`NodeEmbedding` built straight from
        *previous* when its ``text_sha`` matches, else ``None`` — and
        *pending* lists the indices that still need the ONNX model.
    """
    nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]
    if not nodes:
        return [], [], [], [], []

    class_method_idx = build_class_method_index(graph)
    texts = [generate_text(node, graph, class_method_idx) for node in nodes]
    shas = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

    results: list[NodeEmbedding | None] = [None] * len(nodes)
    pending: list[int] = []
    for i, (node, sha) in enumerate(zip(nodes, shas)):
        prev = previous.get(node.id) if previous else None
        if prev is not None and prev[0] == sha:
            results[i] = NodeEmbedding(node_id=node.id, embedding=list(prev[1]), text_sha=sha)
        else:
            pending.append(i)
    return nodes, texts, shas, results, pending


def partition_embeddings(
    graph: KnowledgeGraph,
    previous: dict[str, tuple[str, list[float]]] | None = None,
) -> tuple[list[NodeEmbedding], int]:
    """Split *graph*'s embeddable nodes into reused vectors and a pending count.

    Runs the exact same text/``text_sha`` comparison :func:`embed_graph` uses
    internally to decide what needs re-encoding, but never loads the ONNX
    model. Callers that only need to know what changed — e.g.
    ``analyze --embeddings lazy``, which stores the reused vectors
    immediately and hands the rest to a background worker — get the answer
    at generate-text cost instead of encode cost.

    Args:
        graph: The knowledge graph whose nodes would be embedded.
        previous: ``{node_id: (text_sha, vector)}`` from the prior index
            (see ``LadybugBackend.load_embeddings``).  ``None`` or ``{}``
            makes everything pending.

    Returns:
        ``(reused, pending_count)``: *reused* holds one :class:`NodeEmbedding`
        per node whose freshly generated text hashes to the same
        ``text_sha`` as *previous*; *pending_count* is how many embeddable
        nodes still need encoding.
    """
    _, _, _, results, pending = _partition_texts(graph, previous)
    return [r for r in results if r is not None], len(pending)


def embed_graph(
    graph: KnowledgeGraph,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    previous: dict[str, tuple[str, list[float]]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[NodeEmbedding]:
    """Generate embeddings for all embeddable nodes in the graph.

    Each embeddable node is converted to a natural-language description
    via :func:`generate_text`.  When *previous* carries a vector whose
    ``text_sha`` matches the freshly generated text, that vector is
    reused as-is — only new or changed symbols go through the ONNX
    model, and the model is not even loaded when nothing changed.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        model_name: The fastembed model identifier.  Defaults to
            ``"BAAI/bge-small-en-v1.5"``.
        batch_size: Number of texts to encode per batch.  Defaults to 64.
        previous: ``{node_id: (text_sha, vector)}`` from the prior index
            (see ``LadybugBackend.load_embeddings``).  ``None`` or ``{}``
            encodes everything.
        progress_callback: Optional ``(done, total)`` callback invoked after
            each encoded batch (and once up-front with the reused count) where
            *total* is the embeddable-node count and *done* counts reused +
            encoded so far.  Used by the lazy background worker to publish
            per-batch progress to ``embeddings_state.json``.

    Returns:
        A list of :class:`NodeEmbedding` instances, one per embeddable node,
        each carrying the node's ID, its vector, and the SHA-256 of the
        text that produced it.
    """
    nodes, texts, shas, results, pending = _partition_texts(graph, previous)
    if not nodes:
        return []

    reused = len(nodes) - len(pending)
    if progress_callback is not None:
        progress_callback(reused, len(nodes))

    if pending:
        model = _get_model(model_name)
        done = reused
        # Stream the generator so progress can be published per batch instead
        # of only after every vector is materialized (the encode is the slow
        # part — minutes on a cold index — so mid-flight progress matters).
        vectors_iter = model.embed([texts[i] for i in pending], batch_size=batch_size)
        for pos, vector in enumerate(vectors_iter):
            i = pending[pos]
            results[i] = NodeEmbedding(
                node_id=nodes[i].id, embedding=vector.tolist(), text_sha=shas[i]
            )
            done += 1
            if progress_callback is not None and (done % batch_size == 0 or done == len(nodes)):
                progress_callback(done, len(nodes))

    if reused:
        logger.info(
            "Embeddings: %d reused, %d encoded (of %d symbols)",
            reused,
            len(pending),
            len(nodes),
        )

    return [r for r in results if r is not None]
