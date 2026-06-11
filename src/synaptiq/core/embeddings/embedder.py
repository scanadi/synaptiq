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

    # Under the server profile this caps ONNX intra-op threads so watcher
    # rebuilds can't saturate the machine; 0 → fastembed/ONNX default.
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

def embed_graph(
    graph: KnowledgeGraph,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    previous: dict[str, tuple[str, list[float]]] | None = None,
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
            (see ``KuzuBackend.load_embeddings``).  ``None`` or ``{}``
            encodes everything.

    Returns:
        A list of :class:`NodeEmbedding` instances, one per embeddable node,
        each carrying the node's ID, its vector, and the SHA-256 of the
        text that produced it.
    """
    nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]

    if not nodes:
        return []

    class_method_idx = build_class_method_index(graph)
    texts = [generate_text(node, graph, class_method_idx) for node in nodes]
    shas = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

    results: list[NodeEmbedding | None] = [None] * len(nodes)
    pending: list[int] = []
    for i, (node, sha) in enumerate(zip(nodes, shas)):
        prev = previous.get(node.id) if previous else None
        if prev is not None and prev[0] == sha:
            results[i] = NodeEmbedding(
                node_id=node.id, embedding=list(prev[1]), text_sha=sha
            )
        else:
            pending.append(i)

    if pending:
        model = _get_model(model_name)
        vectors = list(
            model.embed([texts[i] for i in pending], batch_size=batch_size)
        )
        for i, vector in zip(pending, vectors):
            results[i] = NodeEmbedding(
                node_id=nodes[i].id, embedding=vector.tolist(), text_sha=shas[i]
            )

    reused = len(nodes) - len(pending)
    if reused:
        logger.info(
            "Embeddings: %d reused, %d encoded (of %d symbols)",
            reused,
            len(pending),
            len(nodes),
        )

    return [r for r in results if r is not None]
