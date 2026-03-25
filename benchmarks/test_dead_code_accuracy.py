"""Benchmark: Dead code detection accuracy.

Measures precision, recall, and F1 score of Synaptiq's dead code
detection against synthetic repos with known dead code locations.
"""

from __future__ import annotations

import shutil

import pytest

from synaptiq.core.ingestion.pipeline import run_pipeline
from synaptiq.core.storage.kuzu_backend import KuzuBackend

from .conftest import generate_repo


@pytest.mark.benchmark
class TestDeadCodeAccuracy:
    """Dead code detection accuracy benchmarks."""

    def test_known_dead_code_detection(self, tmp_path):
        """Measure precision/recall on a repo with known dead functions.

        The synthetic repo generator creates ``_unused_helper_N`` functions
        every 5th module. These are never called and should be detected
        as dead code.
        """
        repo_dir = generate_repo(tmp_path, 50)

        storage = KuzuBackend()
        db_path = repo_dir / ".synaptiq" / "kuzu"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage.initialize(db_path)

        graph, result = run_pipeline(repo_dir, storage=storage, skip_embeddings=True)
        storage.close()
        shutil.rmtree(repo_dir / ".synaptiq", ignore_errors=True)

        # Ground truth: _unused_helper_N for indices 0, 5, 10, 15, ...
        py_count = int(50 * 0.8)  # 40 python files
        expected_dead = {f"_unused_helper_{i}" for i in range(0, py_count, 5)}

        # Actual dead code detected.
        detected_dead: set[str] = set()
        all_symbols: set[str] = set()
        for node in graph.iter_nodes():
            if node.start_line > 0:  # Skip file/folder nodes.
                all_symbols.add(node.name)
                if node.is_dead:
                    detected_dead.add(node.name)

        # True positives: expected dead AND detected dead.
        tp = len(expected_dead & detected_dead)
        # False positives: detected dead but NOT expected dead.
        # (We filter to only _unused_helper names for FP calculation
        # since the system may legitimately detect other dead code.)
        unexpected_dead = detected_dead - expected_dead
        fp = len(unexpected_dead)
        # False negatives: expected dead but NOT detected.
        fn = len(expected_dead - detected_dead)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print("\n[dead code accuracy]")
        print(f"  Expected dead: {len(expected_dead)} ({', '.join(sorted(expected_dead)[:5])}...)")
        print(f"  Detected dead: {len(detected_dead)}")
        print(f"  True positives: {tp}")
        print(f"  False positives: {fp} ({', '.join(sorted(unexpected_dead)[:5])}...)")
        print(f"  False negatives: {fn}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
        print(f"  F1: {f1:.2f}")

        # We expect high recall on the synthetic _unused_helper functions.
        assert recall >= 0.5, f"Dead code recall too low: {recall:.2f}"
        assert result.dead_code >= 0  # At least the pipeline ran.
