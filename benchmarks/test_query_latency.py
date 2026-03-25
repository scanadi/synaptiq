"""Benchmark: Query latency measurements.

Measures p50/p95/p99 latency for search and tool handler operations
on a pre-indexed synthetic repository.
"""

from __future__ import annotations

import statistics
import time

import pytest

from synaptiq.mcp.tools import (
    handle_context,
    handle_explain,
    handle_impact,
    handle_query,
)


def _measure_latency(fn, iterations: int = 50) -> dict[str, float]:
    """Run *fn* multiple times and return percentile stats."""
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)

    timings.sort()
    return {
        "p50": timings[len(timings) // 2],
        "p95": timings[int(len(timings) * 0.95)],
        "p99": timings[int(len(timings) * 0.99)],
        "mean": statistics.mean(timings),
        "min": min(timings),
        "max": max(timings),
    }


@pytest.mark.benchmark
class TestQueryLatency:
    """Query latency benchmarks on a pre-indexed repo."""

    def test_hybrid_search_latency(self, indexed_small_repo):
        """Measure hybrid search latency (synaptiq_query)."""
        _, storage, _, _ = indexed_small_repo

        stats = _measure_latency(lambda: handle_query(storage, "process"))

        print("\n[hybrid_search]")
        for k, v in stats.items():
            print(f"  {k}: {v * 1000:.1f}ms")

        assert stats["p95"] < 2.0, f"p95 latency too high: {stats['p95'] * 1000:.0f}ms"

    def test_context_latency(self, indexed_small_repo):
        """Measure context lookup latency (synaptiq_context)."""
        _, storage, _, _ = indexed_small_repo

        stats = _measure_latency(lambda: handle_context(storage, "process_0"))

        print("\n[context]")
        for k, v in stats.items():
            print(f"  {k}: {v * 1000:.1f}ms")

        assert stats["p95"] < 2.0

    def test_impact_latency(self, indexed_small_repo):
        """Measure impact analysis latency (synaptiq_impact)."""
        _, storage, _, _ = indexed_small_repo

        stats = _measure_latency(
            lambda: handle_impact(storage, "process_0", depth=3),
            iterations=30,
        )

        print("\n[impact]")
        for k, v in stats.items():
            print(f"  {k}: {v * 1000:.1f}ms")

        assert stats["p95"] < 5.0

    def test_explain_latency(self, indexed_small_repo):
        """Measure explain latency (synaptiq_explain)."""
        _, storage, _, _ = indexed_small_repo

        stats = _measure_latency(lambda: handle_explain(storage, "Service_0"))

        print("\n[explain]")
        for k, v in stats.items():
            print(f"  {k}: {v * 1000:.1f}ms")

        assert stats["p95"] < 2.0
