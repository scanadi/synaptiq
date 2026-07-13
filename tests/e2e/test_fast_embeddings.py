"""End-to-end tests for the "fast" embedding tier (W4.4).

Exercises the REAL model2vec package (minishlab/potion-base-8M) — no
mocking. Unlike the "quality" tier's fastembed/ONNX path, model2vec is
pure Python + numpy (no torch, no onnxruntime — confirmed by the W4.4
spike), so it's cheap enough to run for real here rather than only in
mocked unit tests. See tests/core/test_embedder.py's
``TestFastTierRealModel2Vec`` for the model-level (no CLI, no storage)
real-model coverage this file builds on, and tests/cli/test_main.py's
``TestAnalyzeEmbeddingModelFlag`` for the CLI-flag behavior covered with a
fast, deterministic fake model instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from synaptiq.cli.main import app
from synaptiq.core.storage.ladybug_backend import LadybugBackend

runner = CliRunner()


@pytest.fixture(scope="module")
def fast_tier_model() -> None:
    """Skip this module when potion-base-8M can't be loaded (offline CI, no
    cached weights) — mirrors test_lazy_embeddings.py's `embedding_model`
    fixture, which does the same for the "quality" tier."""
    try:
        from model2vec import StaticModel

        StaticModel.from_pretrained("minishlab/potion-base-8M")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"model2vec fast-tier model unavailable: {exc}")


@pytest.fixture(scope="module")
def quality_tier_model() -> None:
    """Skip tests needing the real "quality" model too (offline CI)."""
    try:
        from fastembed import TextEmbedding

        TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"quality-tier embedding model unavailable: {exc}")


def _tiny_repo(repo: Path) -> None:
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "auth.py").write_text(
        "def validate_user(user):\n"
        "    return check_credentials(user)\n"
        "\n"
        "\n"
        "def check_credentials(user):\n"
        "    return user.is_active\n",
        encoding="utf-8",
    )
    (src / "util.py").write_text(
        "def format_name(name):\n    return name.strip().title()\n",
        encoding="utf-8",
    )


class TestFastTierCliAnalyze:
    """`analyze --embedding-model fast --embeddings sync` end to end."""

    def test_stores_real_256dim_normalized_vectors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_tier_model: None
    ) -> None:
        repo = tmp_path / "repo"
        _tiny_repo(repo)
        monkeypatch.chdir(repo)

        result = runner.invoke(
            app, ["analyze", ".", "--embeddings", "sync", "--embedding-model", "fast"]
        )
        assert result.exit_code == 0, result.output

        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta["stats"]["embedding_model"] == "fast"
        assert meta["stats"]["embeddings"] > 0

        storage = LadybugBackend()
        storage.initialize(repo / ".synaptiq" / "kuzu", read_only=True)
        try:
            stored = storage.load_embeddings()
            assert stored
            for _sha, vector in stored.values():
                assert len(vector) == 256
                norm = sum(v * v for v in vector) ** 0.5
                assert norm == pytest.approx(1.0, abs=1e-2)
        finally:
            storage.close()

    def test_query_against_fast_tier_index_returns_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_tier_model: None
    ) -> None:
        """The query side (mcp.tools._get_query_embedding) must resolve and
        use the SAME "fast" model the index was built with — a "quality"
        (384-dim) query vector against this 256-dim index would hit the
        LadybugBackend dimension guard and fail instead of returning
        results."""
        repo = tmp_path / "repo"
        _tiny_repo(repo)
        monkeypatch.chdir(repo)

        analyze_result = runner.invoke(
            app, ["analyze", ".", "--embeddings", "sync", "--embedding-model", "fast"]
        )
        assert analyze_result.exit_code == 0, analyze_result.output

        query_result = runner.invoke(app, ["query", "validate_user"])
        assert query_result.exit_code == 0, query_result.output
        assert "validate_user" in query_result.output
        # No dimension-mismatch error text leaked into the results.
        assert "dim but the stored index" not in query_result.output

    def test_tier_switch_from_quality_reencodes_at_256dim_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fast_tier_model: None,
        quality_tier_model: None,
    ) -> None:
        """A repo first indexed with the real "quality" model, then
        switched to "fast", ends up with ONLY 256-dim vectors in storage —
        none of the old 384-dim ones survive as "reused" (the tier-salted
        text_sha in embedder._partition_texts forces a full re-encode)."""
        repo = tmp_path / "repo"
        _tiny_repo(repo)
        monkeypatch.chdir(repo)

        first = runner.invoke(app, ["analyze", ".", "--embeddings", "sync"])
        assert first.exit_code == 0, first.output
        first_meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert first_meta["stats"]["embedding_model"] == "quality"

        second = runner.invoke(
            app, ["analyze", ".", "--embeddings", "sync", "--embedding-model", "fast"]
        )
        assert second.exit_code == 0, second.output
        second_meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert second_meta["stats"]["embedding_model"] == "fast"
        # Every symbol re-encoded fresh, none carried over from "quality".
        assert second_meta["stats"]["embeddings"] == first_meta["stats"]["embeddings"]

        storage = LadybugBackend()
        storage.initialize(repo / ".synaptiq" / "kuzu", read_only=True)
        try:
            stored = storage.load_embeddings()
            assert stored
            widths = {len(vector) for _sha, vector in stored.values()}
            assert widths == {256}
        finally:
            storage.close()
