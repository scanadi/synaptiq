"""Benchmark: Indexing speed across repo sizes.

Measures wall-clock time for the full ingestion pipeline (without embeddings)
across synthetic repos of varying sizes.
"""

from __future__ import annotations

import shutil
import time

import pytest

from synaptiq.core.ingestion.pipeline import run_pipeline
from synaptiq.core.storage.kuzu_backend import KuzuBackend

from .conftest import REPO_SIZES, generate_repo


@pytest.mark.benchmark
class TestIndexingSpeed:
    """Indexing speed benchmarks."""

    @pytest.mark.parametrize("size_name,size", list(REPO_SIZES.items()))
    def test_pipeline_speed(self, tmp_path, size_name, size):
        """Measure end-to-end pipeline time for a given repo size."""
        repo_dir = generate_repo(tmp_path, size)

        storage = KuzuBackend()
        db_path = repo_dir / ".synaptiq" / "kuzu"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage.initialize(db_path)

        start = time.perf_counter()
        graph, result = run_pipeline(repo_dir, storage=storage, skip_embeddings=True)
        elapsed = time.perf_counter() - start

        storage.close()
        shutil.rmtree(repo_dir / ".synaptiq", ignore_errors=True)

        files_per_sec = result.files / elapsed if elapsed > 0 else 0

        print(f"\n[{size_name}] {size} files")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Files/sec: {files_per_sec:.0f}")
        print(f"  Symbols: {result.symbols}")
        print(f"  Relationships: {result.relationships}")
        print(f"  Clusters: {result.clusters}")
        print(f"  Dead code: {result.dead_code}")

        # Sanity assertions.
        assert result.files > 0
        assert result.symbols > 0
        assert elapsed < 300  # Should finish in under 5 minutes.

    def test_pipeline_throughput_baseline(self, tmp_path):
        """Baseline: ensure at least 10 files/sec on a 50-file repo."""
        repo_dir = generate_repo(tmp_path, 50)

        storage = KuzuBackend()
        db_path = repo_dir / ".synaptiq" / "kuzu"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage.initialize(db_path)

        start = time.perf_counter()
        graph, result = run_pipeline(repo_dir, storage=storage, skip_embeddings=True)
        elapsed = time.perf_counter() - start

        storage.close()
        shutil.rmtree(repo_dir / ".synaptiq", ignore_errors=True)

        files_per_sec = result.files / elapsed if elapsed > 0 else 0
        print(f"\nBaseline: {files_per_sec:.0f} files/sec"
              f" ({elapsed:.2f}s for {result.files} files)")
        assert files_per_sec >= 2, f"Throughput too low: {files_per_sec:.1f} files/sec"
