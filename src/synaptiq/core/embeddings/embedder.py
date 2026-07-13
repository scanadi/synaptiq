"""Batch embedding pipeline for Synaptiq knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable symbol node, encodes them using one of two named embedding
*tiers* (see :data:`MODEL_TIERS`), and returns a list of :class:`NodeEmbedding`
objects ready for storage.

Only code-level symbol nodes are embedded.  Structural nodes (Folder,
Community, Process) are deliberately skipped — they lack the semantic
richness that makes embedding worthwhile.

Embedding tiers (W4.4)
-----------------------
Two named tiers trade encode speed for representation quality:

* ``"quality"`` (default) — BAAI/bge-small-en-v1.5 via fastembed/ONNX,
  384-dim. A true transformer forward pass per text; the highest-quality
  vectors, ~235 texts/sec on CPU (measured).
* ``"fast"`` — minishlab/potion-base-8M via `model2vec
  <https://github.com/MinishLab/model2vec>`_, 256-dim. A distilled *static*
  embedding (token lookup + mean-pool, no transformer, no ONNX) — measured
  ~180x faster to encode (~43k texts/sec on CPU) at some quality cost.
  Optional dependency: ``synaptiq[fast-embeddings]``.

The two tiers produce vectors of different width and are NOT
interchangeable — a "fast"-built index must be queried with the "fast"
model (see ``LadybugBackend.vector_search``'s dimension guard) and its
``text_sha`` cache keys are salted with the tier's model id (see
:func:`_partition_texts`) so switching tiers on a rebuild always forces a
full re-encode instead of silently mixing 256-dim and 384-dim vectors.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from synaptiq.core.embeddings.text import build_class_method_index, generate_text
from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import NodeLabel
from synaptiq.core.storage.base import NodeEmbedding

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTier:
    """A named embedding tier: which model, what vector width, which backend."""

    name: str
    model_id: str
    dim: int
    backend: str  # "fastembed" | "model2vec"


# Named tiers, selectable per-repo via `analyze --embedding-model`. Keep each
# tier's `name` equal to its dict key — code elsewhere (meta.json's
# stats.embedding_model, the CLI flag's choices) assumes they match.
MODEL_TIERS: dict[str, ModelTier] = {
    "quality": ModelTier(
        name="quality", model_id="BAAI/bge-small-en-v1.5", dim=384, backend="fastembed"
    ),
    "fast": ModelTier(
        name="fast", model_id="minishlab/potion-base-8M", dim=256, backend="model2vec"
    ),
}
DEFAULT_TIER_NAME = "quality"

_MODEL2VEC_INSTALL_HINT = (
    "requires the 'model2vec' package, which is not installed. Install it with: "
    "pip install 'synaptiq[fast-embeddings]'  (or: uv sync --extra fast-embeddings)"
)


def resolve_tier(name: str | None) -> ModelTier:
    """Look up a :class:`ModelTier` by name; ``None``/``""`` means the default.

    Raises:
        ValueError: *name* is non-empty and not a registered tier — callers
            that take the name from user input (the CLI flag) should let
            this surface as a clear "unknown tier" error rather than
            silently falling back to a different tier than requested.
    """
    if not name:
        return MODEL_TIERS[DEFAULT_TIER_NAME]
    tier = MODEL_TIERS.get(name)
    if tier is None:
        raise ValueError(
            f"Unknown embedding tier {name!r}. Choose one of: {', '.join(sorted(MODEL_TIERS))}."
        )
    return tier


def tier_from_meta(data_dir: Path | None) -> ModelTier:
    """Resolve the tier a repo's index was built with, from ``meta.json``.

    Reads ``stats.embedding_model`` as written by
    :func:`~synaptiq.core.ingestion.pipeline.write_meta`. Never raises — a
    missing *data_dir*, a missing/corrupt ``meta.json``, or an unrecognised
    tier name (an index from a pre-W4.4 synaptiq, or a newer one with a tier
    this build doesn't know) all degrade to the historical default
    ("quality") rather than breaking the caller.

    This is the single source of truth the query side (MCP tools, hybrid
    search, `synaptiq query`) and the daemon/lazy-worker rebuild paths all
    use to stay consistent with whatever tier actually produced the stored
    vectors — see the module docstring.
    """
    if data_dir is None:
        return MODEL_TIERS[DEFAULT_TIER_NAME]
    try:
        meta = json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))
        name = meta.get("stats", {}).get("embedding_model")
    except Exception:
        return MODEL_TIERS[DEFAULT_TIER_NAME]
    return MODEL_TIERS.get(name, MODEL_TIERS[DEFAULT_TIER_NAME])


def _check_model2vec_available(tier_name: str) -> None:
    """Raise a clear, actionable ``ImportError`` if model2vec isn't installed.

    Called both eagerly (:func:`ensure_tier_available`, so `analyze
    --embedding-model fast` fails fast before running the pipeline) and
    lazily (:func:`_get_model`, so every other path that ends up encoding —
    the lazy worker, a daemon rebuild — gets the same clear message instead
    of a bare ``ModuleNotFoundError`` several frames down).
    """
    try:
        import model2vec  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"The {tier_name!r} embedding tier {_MODEL2VEC_INSTALL_HINT}") from exc


def ensure_tier_available(tier_name: str) -> None:
    """Raise early if *tier_name*'s backend package isn't installed.

    fastembed (the "quality" backend) is a core dependency, so there is
    nothing to check for it. Intended for callers like `analyze
    --embedding-model` that want to fail fast — before running the
    (potentially long) pipeline — rather than discovering a missing
    optional dependency only after minutes of indexing.
    """
    tier = resolve_tier(tier_name)
    if tier.backend == "model2vec":
        _check_model2vec_available(tier.name)


@lru_cache(maxsize=4)
def _get_model(tier_name: str) -> Any:
    """Load (and cache) the encoder for *tier_name* — dispatches by backend.

    Cached per tier name so repeated calls — one per background-worker
    batch, one per MCP query, or across a long-running `serve`/`mcp`
    process's lifetime — reuse the already-loaded model instead of
    reloading (or re-downloading) it every time.
    """
    tier = resolve_tier(tier_name)
    if tier.backend == "fastembed":
        from fastembed import TextEmbedding

        from synaptiq.core.resources import current_limits

        # current_limits() caps ONNX intra-op threads per the active profile:
        # strict server caps for daemons, the polite max(2, cores - 2) default
        # for interactive commands, or an explicit `analyze --jobs` override.
        # 0 → fastembed/ONNX default (all cores).
        threads = current_limits().embed_threads
        return TextEmbedding(model_name=tier.model_id, threads=threads or None)
    if tier.backend == "model2vec":
        _check_model2vec_available(tier.name)
        from model2vec import StaticModel

        return StaticModel.from_pretrained(tier.model_id)
    raise ValueError(  # pragma: no cover - guarded by the MODEL_TIERS registry
        f"Unknown embedding backend {tier.backend!r} for tier {tier.name!r}"
    )


def _encode_batches(
    tier: ModelTier, model: Any, texts: list[str], batch_size: int
) -> Iterator[list[float]]:
    """Yield one embedding vector (a plain ``list[float]``) per text, in order.

    Backend-specific: fastembed's ``TextEmbedding.embed`` is a true
    streaming generator (one ONNX forward pass per batch, yielded
    incrementally), which is why :func:`embed_graph` can publish mid-flight
    progress on a slow cold index. model2vec's ``StaticModel.encode``
    returns the whole batch as one ndarray in a single call — there is no
    transformer forward pass to stream, and it is fast enough (measured
    ~180x fastembed's rate on this project's text shapes) that chunking
    here only preserves the same progress-callback cadence, not latency.
    ``use_multiprocessing=False`` keeps it single-process and
    deterministic: model2vec is fast enough on one core alone that
    spawning workers would cost more than it saves, and it sidesteps
    fork/spawn edge cases inside the detached lazy-embedding worker
    process (``core/embeddings/lazy_worker.py``).
    """
    if tier.backend == "fastembed":
        for vector in model.embed(texts, batch_size=batch_size):
            yield vector.tolist()
    elif tier.backend == "model2vec":
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            vectors = model.encode(chunk, batch_size=batch_size, use_multiprocessing=False)
            for row in vectors:
                yield row.tolist()
    else:  # pragma: no cover - guarded by the MODEL_TIERS registry
        raise ValueError(f"Unknown embedding backend {tier.backend!r} for tier {tier.name!r}")


def encode_query(tier_name: str, text: str) -> list[float]:
    """Encode a single query string with *tier_name*'s model.

    Used by the query side (``mcp.tools._get_query_embedding``, and
    transitively hybrid search / `synaptiq query`) to embed a search query
    with the SAME model an index was built with — see :func:`tier_from_meta`.
    """
    tier = resolve_tier(tier_name)
    model = _get_model(tier.name)
    return next(iter(_encode_batches(tier, model, [text], batch_size=1)))


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
    tier: ModelTier,
) -> tuple[list, list[str], list[str], list[NodeEmbedding | None], list[int]]:
    """Compute nodes/texts/``text_sha``s and split into reused vs. pending.

    Single implementation shared by :func:`embed_graph` (which goes on to
    encode the pending indices) and :func:`partition_embeddings` (which only
    needs the split) — the reused/pending decision can never drift between
    the two callers.

    ``text_sha`` is salted with *tier*'s model id (not just the generated
    text), so a tier switch invalidates reuse for every node even when the
    text itself is unchanged: a "quality"-encoded 384-dim vector is never
    mistaken for a valid cache hit when the caller now wants "fast"'s
    256-dim space (or vice versa) — this is what makes
    :func:`partition_embeddings` return zero reused vectors right after a
    tier switch instead of mixing vector widths in the same store. The salt
    is the concrete model id rather than the abstract tier name so a
    hypothetical future change to what model a tier maps to also
    invalidates old vectors, the same way. ``text_sha`` is a pure cache key
    (never displayed, never compared against anything external — see
    ``NodeEmbedding``), so folding the model identity into it is safe.

    Returns:
        ``(nodes, texts, shas, results, pending)`` where *results* has one
        slot per node — a :class:`NodeEmbedding` built straight from
        *previous* when its ``text_sha`` matches, else ``None`` — and
        *pending* lists the indices that still need the model.
    """
    nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]
    if not nodes:
        return [], [], [], [], []

    class_method_idx = build_class_method_index(graph)
    texts = [generate_text(node, graph, class_method_idx) for node in nodes]
    salt = tier.model_id.encode("utf-8")
    shas = [hashlib.sha256(salt + b"\x00" + text.encode("utf-8")).hexdigest() for text in texts]

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
    tier: str = DEFAULT_TIER_NAME,
) -> tuple[list[NodeEmbedding], int]:
    """Split *graph*'s embeddable nodes into reused vectors and a pending count.

    Runs the exact same text/``text_sha`` comparison :func:`embed_graph` uses
    internally to decide what needs re-encoding, but never loads the
    embedding model. Callers that only need to know what changed — e.g.
    ``analyze --embeddings lazy``, which stores the reused vectors
    immediately and hands the rest to a background worker — get the answer
    at generate-text cost instead of encode cost.

    Args:
        graph: The knowledge graph whose nodes would be embedded.
        previous: ``{node_id: (text_sha, vector)}`` from the prior index
            (see ``LadybugBackend.load_embeddings``).  ``None`` or ``{}``
            makes everything pending.
        tier: Embedding tier name (``"quality"`` or ``"fast"``) the caller
            intends to encode with. Must match the tier *previous* was built
            with for any reuse to be found — see :func:`_partition_texts`;
            passing a different tier than *previous* used makes everything
            pending (a tier switch always forces a full re-encode, never a
            silent mix of vector widths).

    Returns:
        ``(reused, pending_count)``: *reused* holds one :class:`NodeEmbedding`
        per node whose freshly generated text hashes to the same
        ``text_sha`` as *previous*; *pending_count* is how many embeddable
        nodes still need encoding.
    """
    resolved = resolve_tier(tier)
    _, _, _, results, pending = _partition_texts(graph, previous, resolved)
    return [r for r in results if r is not None], len(pending)


def embed_graph(
    graph: KnowledgeGraph,
    tier: str = DEFAULT_TIER_NAME,
    batch_size: int = 64,
    previous: dict[str, tuple[str, list[float]]] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[NodeEmbedding]:
    """Generate embeddings for all embeddable nodes in the graph.

    Each embeddable node is converted to a natural-language description
    via :func:`generate_text`.  When *previous* carries a vector whose
    ``text_sha`` matches the freshly generated text, that vector is
    reused as-is — only new or changed symbols go through the model, and
    the model is not even loaded when nothing changed.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        tier: Embedding tier name — ``"quality"`` (default, BAAI/bge-small-
            en-v1.5, 384-dim, fastembed/ONNX) or ``"fast"`` (minishlab/
            potion-base-8M, 256-dim, model2vec static embeddings). See the
            module docstring. Raises :class:`ValueError` for an unknown name.
        batch_size: Number of texts to encode per batch.  Defaults to 64.
        previous: ``{node_id: (text_sha, vector)}`` from the prior index
            (see ``LadybugBackend.load_embeddings``).  ``None`` or ``{}``
            encodes everything. Vectors built with a *different* tier than
            *tier* are never reused (see :func:`_partition_texts`) — they
            simply become pending like any other changed text.
        progress_callback: Optional ``(done, total)`` callback invoked after
            each encoded batch (and once up-front with the reused count) where
            *total* is the embeddable-node count and *done* counts reused +
            encoded so far.  Used by the lazy background worker to publish
            per-batch progress to ``embeddings_state.json``.

    Returns:
        A list of :class:`NodeEmbedding` instances, one per embeddable node,
        each carrying the node's ID, its vector, and the SHA-256 of the
        (tier-salted) text that produced it.
    """
    resolved = resolve_tier(tier)
    nodes, texts, shas, results, pending = _partition_texts(graph, previous, resolved)
    if not nodes:
        return []

    reused = len(nodes) - len(pending)
    if progress_callback is not None:
        progress_callback(reused, len(nodes))

    if pending:
        model = _get_model(resolved.name)
        done = reused
        # Stream so progress can be published per batch instead of only
        # after every vector is materialized. Encoding is the slow part of
        # a cold index on the "quality" tier (minutes), so mid-flight
        # progress matters there; the "fast" tier is quick enough that this
        # loop is mostly bookkeeping.
        pending_texts = [texts[i] for i in pending]
        vectors_iter = _encode_batches(resolved, model, pending_texts, batch_size)
        for pos, vector in enumerate(vectors_iter):
            i = pending[pos]
            results[i] = NodeEmbedding(node_id=nodes[i].id, embedding=vector, text_sha=shas[i])
            done += 1
            if progress_callback is not None and (done % batch_size == 0 or done == len(nodes)):
                progress_callback(done, len(nodes))

    if reused:
        logger.info(
            "Embeddings: %d reused, %d encoded (of %d symbols, tier=%s)",
            reused,
            len(pending),
            len(nodes),
            resolved.name,
        )

    return [r for r in results if r is not None]
