"""Tests for the batch embedding pipeline (Task 24) and the W4.4 tier registry.

Verifies that ``embed_graph`` correctly:
- Filters nodes to only embeddable labels (skipping Folder, Community, Process)
- Generates text via ``generate_text`` for each eligible node
- Passes texts through the selected tier's model in batches
- Returns properly structured ``NodeEmbedding`` objects
- Handles edge cases: empty graphs, tier selection, batch sizes

IMPORTANT: All tests mock ``TextEmbedding``/``StaticModel`` to avoid slow
model downloads, EXCEPT the ``TestFastTierRealModel2Vec`` class, which
exercises the real (lightweight, no-ONNX) model2vec package end-to-end —
see its docstring.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synaptiq.core.embeddings.embedder import (
    DEFAULT_TIER_NAME,
    EMBEDDABLE_LABELS,
    MODEL_TIERS,
    _check_model2vec_available,
    _get_model,
    embed_graph,
    encode_query,
    ensure_tier_available,
    partition_embeddings,
    resolve_tier,
    tier_from_meta,
)
from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel
from synaptiq.core.storage.base import NodeEmbedding


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Clear the lru_cache on _get_model before each test so mocks work."""
    _get_model.cache_clear()
    yield
    _get_model.cache_clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_graph() -> KnowledgeGraph:
    """Graph with two embeddable nodes (function, class) and one non-embeddable (folder)."""
    graph = KnowledgeGraph()
    graph.add_node(
        GraphNode(
            id="function:src/a.py:foo",
            label=NodeLabel.FUNCTION,
            name="foo",
            file_path="src/a.py",
        )
    )
    graph.add_node(
        GraphNode(
            id="class:src/a.py:Bar",
            label=NodeLabel.CLASS,
            name="Bar",
            file_path="src/a.py",
        )
    )
    graph.add_node(
        GraphNode(
            id="folder::src",
            label=NodeLabel.FOLDER,
            name="src",
        )
    )
    return graph


@pytest.fixture
def all_label_graph() -> KnowledgeGraph:
    """Graph containing one node of every label for completeness testing."""
    graph = KnowledgeGraph()
    nodes = [
        GraphNode(id="file:src/a.py:", label=NodeLabel.FILE, name="a.py", file_path="src/a.py"),
        GraphNode(
            id="function:src/a.py:foo",
            label=NodeLabel.FUNCTION,
            name="foo",
            file_path="src/a.py",
        ),
        GraphNode(
            id="class:src/a.py:Bar",
            label=NodeLabel.CLASS,
            name="Bar",
            file_path="src/a.py",
        ),
        GraphNode(
            id="method:src/a.py:baz",
            label=NodeLabel.METHOD,
            name="baz",
            file_path="src/a.py",
            class_name="Bar",
        ),
        GraphNode(
            id="interface:src/types.ts:IFoo",
            label=NodeLabel.INTERFACE,
            name="IFoo",
            file_path="src/types.ts",
        ),
        GraphNode(
            id="type_alias:src/types.py:UserID",
            label=NodeLabel.TYPE_ALIAS,
            name="UserID",
            file_path="src/types.py",
        ),
        GraphNode(
            id="enum:src/enums.py:Color",
            label=NodeLabel.ENUM,
            name="Color",
            file_path="src/enums.py",
        ),
        # Non-embeddable labels:
        GraphNode(id="folder::src", label=NodeLabel.FOLDER, name="src"),
        GraphNode(id="community::auth", label=NodeLabel.COMMUNITY, name="auth"),
        GraphNode(id="process::login", label=NodeLabel.PROCESS, name="login"),
    ]
    for n in nodes:
        graph.add_node(n)
    return graph


# ---------------------------------------------------------------------------
# Tests — EMBEDDABLE_LABELS constant
# ---------------------------------------------------------------------------


class TestEmbeddableLabels:
    """Verify the EMBEDDABLE_LABELS constant."""

    def test_contains_expected_labels(self) -> None:
        expected = {
            NodeLabel.FILE,
            NodeLabel.FUNCTION,
            NodeLabel.CLASS,
            NodeLabel.METHOD,
            NodeLabel.INTERFACE,
            NodeLabel.TYPE_ALIAS,
            NodeLabel.ENUM,
        }
        assert EMBEDDABLE_LABELS == expected

    def test_excludes_structural_labels(self) -> None:
        assert NodeLabel.FOLDER not in EMBEDDABLE_LABELS
        assert NodeLabel.COMMUNITY not in EMBEDDABLE_LABELS
        assert NodeLabel.PROCESS not in EMBEDDABLE_LABELS

    def test_is_frozenset(self) -> None:
        assert isinstance(EMBEDDABLE_LABELS, frozenset)


# ---------------------------------------------------------------------------
# Tests — Basic embedding
# ---------------------------------------------------------------------------


class TestEmbedGraphBasic:
    """Core functionality of embed_graph."""

    @patch("fastembed.TextEmbedding")
    def test_returns_node_embeddings(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """embed_graph returns a list of NodeEmbedding objects for embeddable nodes."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        assert len(results) == 2  # function + class; folder is skipped
        assert all(isinstance(r, NodeEmbedding) for r in results)

    @patch("fastembed.TextEmbedding")
    def test_embedding_vectors_are_lists_of_float(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """Embedding vectors are plain Python lists, not numpy arrays."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        for r in results:
            assert isinstance(r.embedding, list)
            assert all(isinstance(v, float) for v in r.embedding)

    @patch("fastembed.TextEmbedding")
    def test_embedding_values_match(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """Embedding values from the model are correctly mapped to NodeEmbedding objects."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        # We should get two results with the two mock vectors
        embeddings = [r.embedding for r in results]
        assert [0.1, 0.2, 0.3] in embeddings or pytest.approx([0.1, 0.2, 0.3]) in embeddings
        assert [0.4, 0.5, 0.6] in embeddings or pytest.approx([0.4, 0.5, 0.6]) in embeddings

    @patch("fastembed.TextEmbedding")
    def test_node_ids_are_correct(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """NodeEmbedding objects carry the correct node IDs."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        node_ids = {r.node_id for r in results}
        assert "function:src/a.py:foo" in node_ids
        assert "class:src/a.py:Bar" in node_ids


# ---------------------------------------------------------------------------
# Tests — Filtering non-embeddable
# ---------------------------------------------------------------------------


class TestEmbedGraphFiltering:
    """Filtering of non-embeddable nodes."""

    @patch("fastembed.TextEmbedding")
    def test_skips_folder_nodes(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """Folder nodes are excluded from embedding."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(sample_graph)

        node_ids = {r.node_id for r in results}
        assert "folder::src" not in node_ids

    @patch("fastembed.TextEmbedding")
    def test_skips_community_and_process(
        self, mock_te_cls: MagicMock, all_label_graph: KnowledgeGraph
    ) -> None:
        """Community and Process nodes are excluded from embedding."""
        embeddable_count = 7  # FILE, FUNCTION, CLASS, METHOD, INTERFACE, TYPE_ALIAS, ENUM
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]) for _ in range(embeddable_count)]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(all_label_graph)

        assert len(results) == embeddable_count
        node_ids = {r.node_id for r in results}
        assert "folder::src" not in node_ids
        assert "community::auth" not in node_ids
        assert "process::login" not in node_ids

    @patch("fastembed.TextEmbedding")
    def test_all_embeddable_labels_included(
        self, mock_te_cls: MagicMock, all_label_graph: KnowledgeGraph
    ) -> None:
        """All embeddable label types produce NodeEmbedding objects."""
        embeddable_count = 7
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]) for _ in range(embeddable_count)]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(all_label_graph)

        node_ids = {r.node_id for r in results}
        assert "file:src/a.py:" in node_ids
        assert "function:src/a.py:foo" in node_ids
        assert "class:src/a.py:Bar" in node_ids
        assert "method:src/a.py:baz" in node_ids
        assert "interface:src/types.ts:IFoo" in node_ids
        assert "type_alias:src/types.py:UserID" in node_ids
        assert "enum:src/enums.py:Color" in node_ids


# ---------------------------------------------------------------------------
# Tests — Empty graph
# ---------------------------------------------------------------------------


class TestEmbedGraphEmpty:
    """Edge case: empty graph."""

    @patch("fastembed.TextEmbedding")
    def test_empty_graph_returns_empty_list(self, mock_te_cls: MagicMock) -> None:
        """An empty graph produces no embeddings."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([])
        mock_te_cls.return_value = mock_model

        graph = KnowledgeGraph()
        results = embed_graph(graph)

        assert results == []

    @patch("fastembed.TextEmbedding")
    def test_graph_with_only_non_embeddable_returns_empty(self, mock_te_cls: MagicMock) -> None:
        """A graph containing only non-embeddable nodes returns an empty list."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter([])
        mock_te_cls.return_value = mock_model

        graph = KnowledgeGraph()
        graph.add_node(
            GraphNode(id="folder::src", label=NodeLabel.FOLDER, name="src")
        )
        graph.add_node(
            GraphNode(id="community::auth", label=NodeLabel.COMMUNITY, name="auth")
        )

        results = embed_graph(graph)

        assert results == []


# ---------------------------------------------------------------------------
# Tests — Model configuration
# ---------------------------------------------------------------------------


def _expected_threads() -> int | None:
    """The threads value _get_model forwards to TextEmbedding.

    Mirrors ``_get_model``'s own ``limits.embed_threads or None`` logic —
    since W1.4 the interactive profile resolves a polite ``max(2, cores - 2)``
    default rather than 0/None (the exact math is covered in
    test_resources.py; here we only assert faithful forwarding).
    """
    from synaptiq.core.resources import current_limits

    return current_limits().embed_threads or None


class TestEmbedGraphModelConfig:
    """Model name and batch size configuration."""

    @patch("fastembed.TextEmbedding")
    def test_default_model_name(self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph) -> None:
        """Default model is BAAI/bge-small-en-v1.5."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph)

        mock_te_cls.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5", threads=_expected_threads()
        )

    @patch("model2vec.StaticModel")
    def test_fast_tier_loads_model2vec(
        self, mock_static_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """tier="fast" dispatches to model2vec.StaticModel, not fastembed."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.4, 0.5]])
        mock_static_cls.from_pretrained.return_value = mock_model

        embed_graph(sample_graph, tier="fast")

        mock_static_cls.from_pretrained.assert_called_once_with("minishlab/potion-base-8M")

    def test_unknown_tier_raises(self, sample_graph: KnowledgeGraph) -> None:
        """An unrecognised tier name raises ValueError rather than silently
        falling back to a different tier than the caller asked for."""
        with pytest.raises(ValueError, match="Unknown embedding tier"):
            embed_graph(sample_graph, tier="ultra-quality")

    @patch("fastembed.TextEmbedding")
    def test_custom_batch_size_passed_to_embed(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """The batch_size parameter is forwarded to model.embed()."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph, batch_size=32)

        # Verify batch_size was passed to embed()
        embed_call = mock_model.embed.call_args
        assert embed_call.kwargs.get("batch_size") == 32 or (
            len(embed_call.args) > 1 and embed_call.args[1] == 32
        )


# ---------------------------------------------------------------------------
# Tests — generate_text integration
# ---------------------------------------------------------------------------


class TestEmbedGraphTextGeneration:
    """Verifies generate_text is called for each embeddable node."""

    @patch("synaptiq.core.embeddings.embedder.generate_text")
    @patch("fastembed.TextEmbedding")
    def test_generate_text_called_for_each_node(
        self,
        mock_te_cls: MagicMock,
        mock_gen_text: MagicMock,
        sample_graph: KnowledgeGraph,
    ) -> None:
        """generate_text is called once per embeddable node."""
        mock_gen_text.return_value = "mock text"
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph)

        # generate_text should be called twice (function + class, not folder)
        assert mock_gen_text.call_count == 2

    @patch("synaptiq.core.embeddings.embedder.generate_text")
    @patch("fastembed.TextEmbedding")
    def test_generated_texts_passed_to_model(
        self,
        mock_te_cls: MagicMock,
        mock_gen_text: MagicMock,
        sample_graph: KnowledgeGraph,
    ) -> None:
        """Texts from generate_text are forwarded to model.embed()."""
        mock_gen_text.side_effect = ["text for foo", "text for Bar"]
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph)

        # The texts list passed to model.embed should contain both texts
        embed_call_args = mock_model.embed.call_args
        texts_arg = (
            embed_call_args.args[0]
            if embed_call_args.args
            else embed_call_args.kwargs.get("documents", [])
        )
        assert "text for foo" in texts_arg
        assert "text for Bar" in texts_arg


# ---------------------------------------------------------------------------
# Tests — Batch processing
# ---------------------------------------------------------------------------


class TestEmbedGraphBatchProcessing:
    """Verifies batch processing behaviour with larger graphs."""

    @patch("fastembed.TextEmbedding")
    def test_many_nodes_all_embedded(self, mock_te_cls: MagicMock) -> None:
        """A graph with many embeddable nodes produces one embedding per node."""
        graph = KnowledgeGraph()
        count = 100
        for i in range(count):
            graph.add_node(
                GraphNode(
                    id=f"function:src/mod.py:fn_{i}",
                    label=NodeLabel.FUNCTION,
                    name=f"fn_{i}",
                    file_path="src/mod.py",
                )
            )

        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([float(i), float(i + 1), float(i + 2)]) for i in range(count)]
        )
        mock_te_cls.return_value = mock_model

        results = embed_graph(graph, batch_size=16)

        assert len(results) == count
        # Each embedding should have 3 dimensions
        assert all(len(r.embedding) == 3 for r in results)

    @patch("fastembed.TextEmbedding")
    def test_default_batch_size_is_64(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """When batch_size is not specified, 64 is used by default."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph)

        embed_call = mock_model.embed.call_args
        assert embed_call.kwargs.get("batch_size") == 64 or (
            len(embed_call.args) > 1 and embed_call.args[1] == 64
        )


class TestIncrementalReuse:
    """embed_graph(previous=...) reuses vectors for unchanged texts."""

    @patch("fastembed.TextEmbedding")
    def test_full_reuse_never_loads_model(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """When every text is unchanged, the ONNX model is not even created."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        first = embed_graph(sample_graph)
        assert all(e.text_sha for e in first)

        previous = {e.node_id: (e.text_sha, e.embedding) for e in first}
        _get_model.cache_clear()
        mock_te_cls.reset_mock()

        second = embed_graph(sample_graph, previous=previous)

        mock_te_cls.assert_not_called()
        assert {e.node_id: e.embedding for e in second} == {
            e.node_id: e.embedding for e in first
        }
        assert {e.node_id: e.text_sha for e in second} == {
            e.node_id: e.text_sha for e in first
        }

    @patch("fastembed.TextEmbedding")
    def test_changed_text_reencodes_only_that_symbol(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """A stale text_sha re-encodes that symbol; the rest are reused."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        first = embed_graph(sample_graph)
        previous = {e.node_id: (e.text_sha, e.embedding) for e in first}
        changed_id = first[0].node_id
        previous[changed_id] = ("stale-sha", previous[changed_id][1])

        _get_model.cache_clear()
        fresh_model = MagicMock()
        fresh_model.embed.return_value = iter([np.array([0.7, 0.8, 0.9])])
        mock_te_cls.reset_mock()
        mock_te_cls.return_value = fresh_model

        second = embed_graph(sample_graph, previous=previous)

        # Exactly one text went through the model.
        texts_encoded = fresh_model.embed.call_args.args[0]
        assert len(texts_encoded) == 1
        by_id = {e.node_id: e for e in second}
        assert by_id[changed_id].embedding == [0.7, 0.8, 0.9]
        assert by_id[changed_id].text_sha != "stale-sha"
        for e in first[1:]:
            assert by_id[e.node_id].embedding == e.embedding


# ---------------------------------------------------------------------------
# Tests — partition_embeddings (W4.1b)
# ---------------------------------------------------------------------------


class TestPartitionEmbeddings:
    """partition_embeddings splits reused vs. pending without touching ONNX.

    It must always agree with the reused/pending split embed_graph computes
    internally (they share one implementation, ``_partition_texts``) — these
    tests pin that equivalence on a fixture rather than just re-testing
    embed_graph's own behaviour.
    """

    @patch("fastembed.TextEmbedding")
    def test_never_loads_model(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """Even with nothing to reuse, partition_embeddings never creates the
        ONNX model — that's the whole point of exposing it separately from
        embed_graph."""
        reused, pending_count = partition_embeddings(sample_graph)

        mock_te_cls.assert_not_called()
        assert reused == []
        assert pending_count == 2  # function + class; folder is skipped

    def test_empty_graph_returns_empty(self) -> None:
        graph = KnowledgeGraph()
        reused, pending_count = partition_embeddings(graph)
        assert reused == []
        assert pending_count == 0

    def test_graph_with_only_non_embeddable_returns_empty(self) -> None:
        graph = KnowledgeGraph()
        graph.add_node(GraphNode(id="folder::src", label=NodeLabel.FOLDER, name="src"))
        reused, pending_count = partition_embeddings(graph)
        assert reused == []
        assert pending_count == 0

    def test_no_previous_makes_everything_pending(self, sample_graph: KnowledgeGraph) -> None:
        reused, pending_count = partition_embeddings(sample_graph, previous=None)
        assert reused == []
        assert pending_count == 2

    @patch("fastembed.TextEmbedding")
    def test_matches_embed_graph_full_reuse_split(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """Full reuse: partition_embeddings' reused set is exactly what
        embed_graph(previous=...) would itself reuse (same ids, vectors, shas)."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        first = embed_graph(sample_graph)
        previous = {e.node_id: (e.text_sha, e.embedding) for e in first}

        reused, pending_count = partition_embeddings(sample_graph, previous)

        assert pending_count == 0
        assert {r.node_id: r.embedding for r in reused} == {
            e.node_id: e.embedding for e in first
        }
        assert {r.node_id: r.text_sha for r in reused} == {
            e.node_id: e.text_sha for e in first
        }

    @patch("fastembed.TextEmbedding")
    def test_matches_embed_graph_partial_reuse_split(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """One stale text_sha: partition_embeddings and embed_graph agree on
        exactly which node is pending vs. reused, and on the reused vectors."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        first = embed_graph(sample_graph)
        previous = {e.node_id: (e.text_sha, e.embedding) for e in first}
        changed_id = first[0].node_id
        previous[changed_id] = ("stale-sha", previous[changed_id][1])

        reused, pending_count = partition_embeddings(sample_graph, previous)

        assert pending_count == 1
        reused_ids = {r.node_id for r in reused}
        assert changed_id not in reused_ids
        assert reused_ids == {e.node_id for e in first if e.node_id != changed_id}

        # Ask embed_graph to do the real encode with the same `previous` —
        # it must re-encode exactly the one node partition_embeddings flagged
        # pending, and its own reused vectors must match ours exactly.
        _get_model.cache_clear()
        fresh_model = MagicMock()
        fresh_model.embed.return_value = iter([np.array([0.7, 0.8, 0.9])])
        mock_te_cls.reset_mock()
        mock_te_cls.return_value = fresh_model

        second = embed_graph(sample_graph, previous=previous)
        texts_encoded = fresh_model.embed.call_args.args[0]
        assert len(texts_encoded) == pending_count

        by_id = {e.node_id: e for e in second}
        for r in reused:
            assert by_id[r.node_id].embedding == r.embedding
            assert by_id[r.node_id].text_sha == r.text_sha

    @patch("fastembed.TextEmbedding")
    def test_reused_plus_pending_equals_total_nodes(
        self, mock_te_cls: MagicMock, all_label_graph: KnowledgeGraph
    ) -> None:
        embeddable_count = 7
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]) for _ in range(embeddable_count)]
        )
        mock_te_cls.return_value = mock_model

        first = embed_graph(all_label_graph)
        previous = {e.node_id: (e.text_sha, e.embedding) for e in first}
        # Invalidate two of the seven so the split is neither 0 nor total.
        stale_ids = [first[0].node_id, first[3].node_id]
        for nid in stale_ids:
            previous[nid] = ("stale-sha", previous[nid][1])

        reused, pending_count = partition_embeddings(all_label_graph, previous)

        assert pending_count == 2
        assert len(reused) == embeddable_count - 2
        assert len(reused) + pending_count == embeddable_count


# ---------------------------------------------------------------------------
# Tests — model tier registry (W4.4)
# ---------------------------------------------------------------------------


class TestModelTierRegistry:
    """resolve_tier / MODEL_TIERS / DEFAULT_TIER_NAME."""

    def test_default_tier_name_is_quality(self) -> None:
        assert DEFAULT_TIER_NAME == "quality"

    def test_registry_has_quality_and_fast(self) -> None:
        assert set(MODEL_TIERS) == {"quality", "fast"}

    def test_tier_name_matches_its_own_dict_key(self) -> None:
        """Registry invariant other code relies on — e.g. meta.json's
        stats.embedding_model stores tier.name directly as the dict key,
        so a mismatch here would silently break tier_from_meta lookups."""
        for key, tier in MODEL_TIERS.items():
            assert tier.name == key

    def test_quality_tier_shape(self) -> None:
        tier = MODEL_TIERS["quality"]
        assert tier.model_id == "BAAI/bge-small-en-v1.5"
        assert tier.dim == 384
        assert tier.backend == "fastembed"

    def test_fast_tier_shape(self) -> None:
        tier = MODEL_TIERS["fast"]
        assert tier.model_id == "minishlab/potion-base-8M"
        assert tier.dim == 256
        assert tier.backend == "model2vec"

    def test_quality_and_fast_have_different_dims(self) -> None:
        """The whole point of the dimension guard downstream — if these
        ever matched, LadybugBackend.vector_search's mismatch check would
        never fire and a tier switch could silently mix vector widths."""
        assert MODEL_TIERS["quality"].dim != MODEL_TIERS["fast"].dim

    def test_resolve_none_returns_default(self) -> None:
        assert resolve_tier(None).name == DEFAULT_TIER_NAME

    def test_resolve_empty_string_returns_default(self) -> None:
        assert resolve_tier("").name == DEFAULT_TIER_NAME

    def test_resolve_known_name(self) -> None:
        assert resolve_tier("fast").name == "fast"
        assert resolve_tier("quality").name == "quality"

    def test_resolve_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding tier"):
            resolve_tier("nonexistent")

    def test_resolve_unknown_name_lists_choices(self) -> None:
        with pytest.raises(ValueError, match="fast") as exc_info:
            resolve_tier("nonexistent")
        assert "quality" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests — tier_from_meta (reading the persisted tier back from meta.json)
# ---------------------------------------------------------------------------


class TestTierFromMeta:
    """Resolves the tier an index was built with — never raises."""

    def test_none_data_dir_returns_default(self) -> None:
        assert tier_from_meta(None).name == DEFAULT_TIER_NAME

    def test_missing_meta_json_returns_default(self, tmp_path) -> None:
        assert tier_from_meta(tmp_path).name == DEFAULT_TIER_NAME

    def test_nonexistent_data_dir_returns_default(self, tmp_path) -> None:
        assert tier_from_meta(tmp_path / "does-not-exist").name == DEFAULT_TIER_NAME

    def test_reads_persisted_fast_tier(self, tmp_path) -> None:
        (tmp_path / "meta.json").write_text(json.dumps({"stats": {"embedding_model": "fast"}}))
        assert tier_from_meta(tmp_path).name == "fast"

    def test_reads_persisted_quality_tier(self, tmp_path) -> None:
        (tmp_path / "meta.json").write_text(
            json.dumps({"stats": {"embedding_model": "quality"}})
        )
        assert tier_from_meta(tmp_path).name == "quality"

    def test_unknown_tier_in_meta_falls_back_to_default(self, tmp_path) -> None:
        """A tier name from a future synaptiq version this build doesn't
        know about (or a hand-edited meta.json) degrades gracefully rather
        than crashing the query path."""
        (tmp_path / "meta.json").write_text(
            json.dumps({"stats": {"embedding_model": "ultra-2027"}})
        )
        assert tier_from_meta(tmp_path).name == DEFAULT_TIER_NAME

    def test_missing_stats_key_returns_default(self, tmp_path) -> None:
        (tmp_path / "meta.json").write_text(json.dumps({"version": "1.0"}))
        assert tier_from_meta(tmp_path).name == DEFAULT_TIER_NAME

    def test_corrupt_json_returns_default(self, tmp_path) -> None:
        (tmp_path / "meta.json").write_text("{not valid json")
        assert tier_from_meta(tmp_path).name == DEFAULT_TIER_NAME


# ---------------------------------------------------------------------------
# Tests — tier switch invalidates text_sha reuse (W4.4)
# ---------------------------------------------------------------------------


class TestTierSwitchInvalidatesReuse:
    """A different tier must never reuse another tier's vectors — they have
    different widths, so "reusing" one across a tier switch would corrupt
    the Embedding table's single FLOAT[dim] column (the store path always
    (re)creates that column at the width of whatever it's given — see
    ladybug_backend.store_embeddings). embedder._partition_texts salts
    text_sha with the tier's model id specifically to prevent this.
    """

    @patch("fastembed.TextEmbedding")
    def test_switching_quality_to_fast_reuses_nothing(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        quality_first = embed_graph(sample_graph, tier="quality")
        previous = {e.node_id: (e.text_sha, e.embedding) for e in quality_first}

        reused, pending_count = partition_embeddings(sample_graph, previous, tier="fast")

        assert reused == []
        assert pending_count == len(quality_first) == 2

    @patch("model2vec.StaticModel")
    def test_switching_fast_to_quality_reuses_nothing(
        self, mock_static_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.4, 0.5]])
        mock_static_cls.from_pretrained.return_value = mock_model

        fast_first = embed_graph(sample_graph, tier="fast")
        previous = {e.node_id: (e.text_sha, e.embedding) for e in fast_first}

        reused, pending_count = partition_embeddings(sample_graph, previous, tier="quality")

        assert reused == []
        assert pending_count == len(fast_first) == 2

    @patch("fastembed.TextEmbedding")
    def test_same_tier_still_reuses(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """Regression guard: the tier salt must not break normal same-tier
        reuse — only an actual tier CHANGE should invalidate it."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        first = embed_graph(sample_graph, tier="quality")
        previous = {e.node_id: (e.text_sha, e.embedding) for e in first}

        reused, pending_count = partition_embeddings(sample_graph, previous, tier="quality")

        assert pending_count == 0
        assert len(reused) == len(first)

    @patch("fastembed.TextEmbedding")
    def test_default_tier_arg_matches_explicit_quality(
        self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph
    ) -> None:
        """Omitting `tier` (defaults to "quality") and passing tier="quality"
        explicitly must salt identically — the default is not a distinct
        pseudo-tier that also invalidates reuse."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        first = embed_graph(sample_graph)  # no tier= at all
        previous = {e.node_id: (e.text_sha, e.embedding) for e in first}

        reused, pending_count = partition_embeddings(sample_graph, previous, tier="quality")

        assert pending_count == 0
        assert len(reused) == len(first)


# ---------------------------------------------------------------------------
# Tests — missing optional dependency (model2vec) error path (W4.4)
# ---------------------------------------------------------------------------


class TestMissingModel2VecDependency:
    """Simulates model2vec not being installed via ``sys.modules``.

    Setting a module name to ``None`` in ``sys.modules`` is the standard way
    to make ``import X`` raise ``ImportError`` even though the package is
    actually installed (see the ``importlib`` docs) — so these tests
    exercise the real "not installed" branch without needing a separate,
    dependency-stripped virtualenv.
    """

    def test_ensure_tier_available_quality_never_raises(self) -> None:
        """fastembed (quality's backend) is a core dependency — nothing to
        check, regardless of whether model2vec happens to be installed."""
        ensure_tier_available("quality")  # must not raise

    def test_ensure_tier_available_fast_raises_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "model2vec", None)
        with pytest.raises(ImportError, match="fast-embeddings"):
            ensure_tier_available("fast")

    def test_get_model_fast_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "model2vec", None)
        with pytest.raises(ImportError, match="fast-embeddings"):
            _get_model("fast")

    def test_embed_graph_fast_raises_when_missing(
        self, monkeypatch: pytest.MonkeyPatch, sample_graph: KnowledgeGraph
    ) -> None:
        monkeypatch.setitem(sys.modules, "model2vec", None)
        with pytest.raises(ImportError, match="fast-embeddings"):
            embed_graph(sample_graph, tier="fast")

    def test_error_message_is_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "model2vec", None)
        with pytest.raises(ImportError) as exc_info:
            _check_model2vec_available("fast")
        message = str(exc_info.value)
        assert "pip install" in message
        assert "synaptiq[fast-embeddings]" in message
        assert "'fast'" in message

    def test_partition_embeddings_never_touches_model2vec(
        self, monkeypatch: pytest.MonkeyPatch, sample_graph: KnowledgeGraph
    ) -> None:
        """partition_embeddings only compares text_sha — it must not load
        any model, so a missing model2vec must not break it even when
        tier="fast" is requested (the caller only wants to know what
        changed, not to encode anything yet)."""
        monkeypatch.setitem(sys.modules, "model2vec", None)
        reused, pending_count = partition_embeddings(sample_graph, tier="fast")
        assert reused == []
        assert pending_count == 2

    def test_recovers_after_dependency_becomes_available(self) -> None:
        """lru_cache must not poison _get_model with a cached failure —
        confirms a subsequent call can still succeed (e.g. a later test, or
        a `pip install` performed mid-session)."""
        _get_model.cache_clear()
        with patch("model2vec.StaticModel") as mock_static_cls:
            mock_static_cls.from_pretrained.return_value = MagicMock()
            _get_model("fast")
            mock_static_cls.from_pretrained.assert_called_once_with("minishlab/potion-base-8M")


# ---------------------------------------------------------------------------
# Tests — real model2vec end-to-end (no mocking) — W4.4 spike verification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_fast_model() -> None:
    """Skip this class when potion-base-8M can't be loaded (offline CI, no
    cached weights) — mirrors tests/e2e/test_lazy_embeddings.py's
    `embedding_model` fixture, which does the same for the "quality" tier.
    """
    try:
        from model2vec import StaticModel

        StaticModel.from_pretrained("minishlab/potion-base-8M")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"model2vec fast-tier model unavailable: {exc}")


class TestFastTierRealModel2Vec:
    """Exercises the real model2vec package end-to-end — no mocking.

    model2vec is pure Python + numpy (no torch, no onnxruntime — confirmed
    by the W4.4 spike), so unlike fastembed/ONNX it is cheap enough to run
    for real in the unit test suite instead of always mocking it.
    """

    def test_real_encode_returns_256dim_normalized_vectors(
        self, real_fast_model: None, sample_graph: KnowledgeGraph
    ) -> None:
        results = embed_graph(sample_graph, tier="fast")

        assert len(results) == 2  # function + class in sample_graph
        for r in results:
            assert len(r.embedding) == 256
            norm = sum(v * v for v in r.embedding) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-3)

    def test_real_encode_is_deterministic(
        self, real_fast_model: None, sample_graph: KnowledgeGraph
    ) -> None:
        """Same text -> same vector: static embeddings have no dropout or
        sampling, unlike a stateful ONNX session this needs no seeding."""
        first = embed_graph(sample_graph, tier="fast")
        _get_model.cache_clear()
        second = embed_graph(sample_graph, tier="fast")

        assert {e.node_id: e.embedding for e in first} == {
            e.node_id: e.embedding for e in second
        }

    def test_real_encode_query_matches_registered_dim(self, real_fast_model: None) -> None:
        vector = encode_query("fast", "a search query about authentication")
        assert len(vector) == MODEL_TIERS["fast"].dim == 256
