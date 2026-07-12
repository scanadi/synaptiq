"""Tests for the per-generation SCC cache behind ``handle_cycles`` (B2).

``handle_cycles`` used to call ``storage.load_graph()`` and recompute
strongly connected components on every MCP call, even though the result
only changes when the index is rebuilt. These tests exercise the memoized
path via a lightweight storage stub exposing a ``generation`` counter --
the same style as ``tests/core/test_pagerank.py``, which caches the
Personalized PageRank projection the same way.
"""

from __future__ import annotations

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from synaptiq.mcp.tools import _cached_scc_groups, handle_cycles


class FakeStorage:
    """Storage stub exposing load_graph plus a generation counter."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.generation = 1
        self.load_graph_calls = 0
        self._graph = graph

    def load_graph(self) -> KnowledgeGraph:
        self.load_graph_calls += 1
        return self._graph


def _graph_with_cycle() -> KnowledgeGraph:
    """A two-function graph where ``a`` and ``b`` call each other."""
    g = KnowledgeGraph()
    g.add_node(
        GraphNode(
            id="function:src/a.py:a",
            label=NodeLabel.FUNCTION,
            name="a",
            file_path="src/a.py",
            start_line=1,
            end_line=2,
            content="def a(): b()",
            signature="def a()",
        )
    )
    g.add_node(
        GraphNode(
            id="function:src/b.py:b",
            label=NodeLabel.FUNCTION,
            name="b",
            file_path="src/b.py",
            start_line=1,
            end_line=2,
            content="def b(): a()",
            signature="def b()",
        )
    )
    g.add_relationship(
        GraphRelationship(
            id="calls:1",
            type=RelType.CALLS,
            source="function:src/a.py:a",
            target="function:src/b.py:b",
            properties={"confidence": 1.0},
        )
    )
    g.add_relationship(
        GraphRelationship(
            id="calls:2",
            type=RelType.CALLS,
            source="function:src/b.py:b",
            target="function:src/a.py:a",
            properties={"confidence": 1.0},
        )
    )
    return g


@pytest.fixture(autouse=True)
def clear_cache():
    _cached_scc_groups.cache_clear()
    yield
    _cached_scc_groups.cache_clear()


def test_two_consecutive_calls_load_graph_once():
    storage = FakeStorage(_graph_with_cycle())

    handle_cycles(storage)
    handle_cycles(storage)

    assert storage.load_graph_calls == 1


def test_generation_bump_triggers_recompute():
    storage = FakeStorage(_graph_with_cycle())

    handle_cycles(storage)
    storage.generation = 2
    handle_cycles(storage)

    assert storage.load_graph_calls == 2


def test_cached_and_fresh_output_are_equal():
    storage = FakeStorage(_graph_with_cycle())

    fresh = handle_cycles(storage)
    cached = handle_cycles(storage)

    assert storage.load_graph_calls == 1
    assert fresh == cached
    assert "Circular Dependencies" in fresh
    assert " a (Function)" in fresh
    assert " b (Function)" in fresh


def test_different_min_size_still_shares_cache():
    """The SCC decomposition is independent of min_size, so varying it
    across calls at the same generation must not trigger a reload."""
    storage = FakeStorage(_graph_with_cycle())

    default_result = handle_cycles(storage)
    narrow_result = handle_cycles(storage, min_size=3)

    assert storage.load_graph_calls == 1
    assert "Circular Dependencies" in default_result
    assert narrow_result == "No circular dependencies detected."
