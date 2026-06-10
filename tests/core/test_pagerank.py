"""Tests for Personalized PageRank projection caching."""

from __future__ import annotations

import pytest

from synaptiq.core.search.pagerank import (
    _cached_ppr,
    _cached_projection,
    personalized_pagerank,
)


class FakeStorage:
    """Storage stub exposing the Cypher fallback path and a generation counter."""

    def __init__(self) -> None:
        self.generation = 1
        self.edge_queries = 0

    def execute_raw(self, query, parameters=None):
        self.edge_queries += 1
        return [
            ["function:src/a.py:fa", "function:src/b.py:fb"],
            ["function:src/b.py:fb", "function:src/c.py:fc"],
            ["class:src/c.py:C", "function:src/a.py:fa"],
        ]


@pytest.fixture(autouse=True)
def clear_caches():
    _cached_ppr.cache_clear()
    _cached_projection.cache_clear()
    yield
    _cached_ppr.cache_clear()
    _cached_projection.cache_clear()


def test_scores_biased_toward_focus_file():
    storage = FakeStorage()
    scores = personalized_pagerank(storage, ["src/a.py"])
    assert scores
    assert "function:src/a.py:fa" in scores


def test_projection_shared_across_focus_sets():
    storage = FakeStorage()
    s1 = personalized_pagerank(storage, ["src/a.py"])
    s2 = personalized_pagerank(storage, ["src/b.py"])
    assert s1 and s2
    # The expensive edge dump ran once; only the reset vector differed.
    assert storage.edge_queries == 1


def test_generation_bump_invalidates_projection():
    storage = FakeStorage()
    personalized_pagerank(storage, ["src/a.py"])
    storage.generation = 2
    personalized_pagerank(storage, ["src/a.py"])
    assert storage.edge_queries == 2


def test_repeated_call_hits_result_cache():
    storage = FakeStorage()
    s1 = personalized_pagerank(storage, ["src/a.py"])
    s2 = personalized_pagerank(storage, ["src/a.py"])
    assert s1 is s2
    assert storage.edge_queries == 1


def test_unknown_focus_file_returns_empty():
    storage = FakeStorage()
    assert personalized_pagerank(storage, ["src/zzz.py"]) == {}
