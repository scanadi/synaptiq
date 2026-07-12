"""Tests for the batch embedding pipeline (Task 24).

Verifies that ``embed_graph`` correctly:
- Filters nodes to only embeddable labels (skipping Folder, Community, Process)
- Generates text via ``generate_text`` for each eligible node
- Passes texts through fastembed's ``TextEmbedding`` model in batches
- Returns properly structured ``NodeEmbedding`` objects
- Handles edge cases: empty graphs, custom model names, batch sizes

IMPORTANT: All tests mock ``TextEmbedding`` to avoid slow model downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synaptiq.core.embeddings.embedder import EMBEDDABLE_LABELS, _get_model, embed_graph
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

    @patch("fastembed.TextEmbedding")
    def test_custom_model_name(self, mock_te_cls: MagicMock, sample_graph: KnowledgeGraph) -> None:
        """A custom model name is forwarded to TextEmbedding."""
        mock_model = MagicMock()
        mock_model.embed.return_value = iter(
            [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        )
        mock_te_cls.return_value = mock_model

        embed_graph(sample_graph, model_name="BAAI/bge-base-en-v1.5")

        mock_te_cls.assert_called_once_with(
            model_name="BAAI/bge-base-en-v1.5", threads=_expected_threads()
        )

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
