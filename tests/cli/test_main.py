"""Tests for the Synaptiq CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from synaptiq import __version__
from synaptiq.cli.main import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Normalize Rich CLI output for substring assertions.

    CI terminals differ from dev machines in two ways that break naive
    ``in`` checks: Rich emits ANSI style spans (which can split a token
    like ``--jobs`` into separately-styled ``--`` and ``jobs`` runs), and
    narrow widths wrap text across panel-border ``│`` characters. Strip
    both and collapse whitespace so assertions are terminal-independent.
    """
    return " ".join(_ANSI_RE.sub("", output).replace("│", " ").split())


@pytest.fixture(autouse=True)
def _no_running_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI tests hermetic: a real `synaptiq serve` running in the
    developer's cwd must not capture the commands under test."""
    monkeypatch.setattr("synaptiq.cli.main._healthy_server_socket", lambda _d: None)


@pytest.fixture(autouse=True)
def _terminal_independent_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CLI output assertions terminal-independent.

    CI terminals enable color (Rich then splits tokens like ``--jobs`` or
    ``v2.0.0`` across ANSI style spans, breaking substring asserts) and use
    narrow widths (wrapping text mid-sentence). ``NO_COLOR`` wins over
    ``FORCE_COLOR`` in Rich, and a wide ``COLUMNS`` prevents wrapping; the
    ``_plain()`` helper stays as defense in depth for panel borders.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("COLUMNS", "200")


class TestVersion:
    """Tests for the --version flag."""

    def test_version_long_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"Synaptiq v{__version__}" in _plain(result.output)

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert f"Synaptiq v{__version__}" in _plain(result.output)

    def test_version_string_format(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert f"Synaptiq v{__version__}" in _plain(result.output)


class TestHelp:
    """Tests for the --help flag."""

    def test_help_exit_code(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_app_name(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "Synaptiq" in _plain(result.output)

    def test_help_lists_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        expected_commands = [
            "analyze",
            "status",
            "list",
            "clean",
            "query",
            "context",
            "impact",
            "dead-code",
            "cypher",
            "setup",
            "watch",
            "diff",
            "mcp",
        ]
        for cmd in expected_commands:
            assert cmd in _plain(result.output), f"Command '{cmd}' not found in --help output"


class TestAnalyzeProfile:
    """Tests for `analyze --profile`."""

    @staticmethod
    def _write_tiny_repo(repo: Path) -> None:
        src = repo / "src"
        src.mkdir(parents=True)
        (src / "main.py").write_text(
            "def main():\n    return helper()\n\n\ndef helper():\n    return 42\n",
            encoding="utf-8",
        )

    def test_analyze_profile_prints_timing_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        self._write_tiny_repo(repo)
        monkeypatch.chdir(repo)

        result = runner.invoke(app, ["analyze", "--profile", "--no-embeddings"])

        assert result.exit_code == 0, result.output
        assert "Phase timings" in _plain(result.output)
        assert "Walking files" in _plain(result.output)
        assert "% of total" in _plain(result.output)

    def test_analyze_without_profile_omits_timing_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        self._write_tiny_repo(repo)
        monkeypatch.chdir(repo)

        result = runner.invoke(app, ["analyze", "--no-embeddings"])

        assert result.exit_code == 0, result.output
        assert "Phase timings" not in _plain(result.output)

    def test_analyze_always_records_phase_timings_in_meta(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """phase_timings lands in meta.json regardless of --profile — it's a
        display-only flag; the timing data itself is always captured."""
        repo = tmp_path / "repo"
        self._write_tiny_repo(repo)
        monkeypatch.chdir(repo)

        result = runner.invoke(app, ["analyze", "--no-embeddings"])
        assert result.exit_code == 0, result.output

        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert "phase_timings" in meta["stats"]
        assert "Walking files" in meta["stats"]["phase_timings"]


class TestAnalyzeJobsFlag:
    """Tests for `analyze --jobs` (W1.4)."""

    def test_analyze_help_mentions_jobs_flag(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--jobs" in _plain(result.output)

    def test_analyze_help_documents_precedence(self) -> None:
        """--help must document flag > env vars > profile defaults precedence.

        Rich wraps the help text inside a panel whose `│` border characters
        land mid-sentence at line breaks — strip them before collapsing
        whitespace or the assertion depends on terminal width.
        """
        result = runner.invoke(app, ["analyze", "--help"])
        output = _plain(result.output)
        assert "Precedence: --jobs > env vars > profile defaults." in output

    def test_analyze_rejects_negative_jobs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["analyze", "--jobs", "-1"])
        assert result.exit_code == 1
        assert "--jobs must be >= 0" in _plain(result.output)

    def test_analyze_jobs_sets_process_wide_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """analyze --jobs N calls set_jobs(N) before storage/pipeline creation.

        `analyze` imports `run_pipeline` locally (inside the function body),
        so the spy must patch the source module — a patch on
        `synaptiq.cli.main` would never be seen.
        """
        repo = tmp_path / "repo"
        src = repo / "src"
        src.mkdir(parents=True)
        (src / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
        monkeypatch.chdir(repo)

        from synaptiq.core.ingestion import pipeline as pipeline_module
        from synaptiq.core.resources import current_limits

        captured: dict = {}
        real_run_pipeline = pipeline_module.run_pipeline

        def _spy_run_pipeline(*args, **kwargs):
            captured["limits"] = current_limits()
            return real_run_pipeline(*args, **kwargs)

        monkeypatch.setattr(pipeline_module, "run_pipeline", _spy_run_pipeline)

        result = runner.invoke(app, ["analyze", "--jobs", "2", "--no-embeddings"])

        assert result.exit_code == 0, result.output
        assert captured["limits"].db_threads == 2
        assert captured["limits"].embed_threads == 2
        assert captured["limits"].pool_workers == 2


class _FakeEmbedModel:
    """Deterministic fastembed stand-in so CLI tests never download ONNX."""

    def embed(self, texts, batch_size: int = 64):
        import hashlib

        import numpy as np

        for text in texts:
            seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
            yield np.random.default_rng(seed).random(384).astype(np.float32)


class TestAnalyzeEmbeddingsFlag:
    """Tests for `analyze --embeddings lazy|sync|off` (W4.1)."""

    @staticmethod
    def _tiny_repo(repo: Path) -> None:
        src = repo / "src"
        src.mkdir(parents=True)
        (src / "main.py").write_text(
            "def main():\n    return helper()\n\n\ndef helper():\n    return 42\n",
            encoding="utf-8",
        )

    def test_help_lists_embeddings_modes(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        output = _plain(result.output)
        assert "--embeddings" in output
        assert "lazy" in output and "sync" in output and "off" in output

    def test_lazy_is_default_and_spawns_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)

        spawned: dict = {}
        monkeypatch.setattr(
            "synaptiq.core.embeddings.lazy_worker.spawn_lazy_worker",
            lambda rp: spawned.setdefault("repo", rp) or 9999,
        )

        result = runner.invoke(app, ["analyze"])  # no --embeddings → lazy
        assert result.exit_code == 0, result.output
        assert "Index ready" in _plain(result.output)
        assert "in the background" in _plain(result.output)
        assert spawned["repo"] == repo.resolve()

    def test_off_skips_embeddings_and_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)

        calls = {"n": 0}
        monkeypatch.setattr(
            "synaptiq.core.embeddings.lazy_worker.spawn_lazy_worker",
            lambda rp: calls.__setitem__("n", calls["n"] + 1),
        )

        result = runner.invoke(app, ["analyze", ".", "--embeddings", "off"])
        assert result.exit_code == 0, result.output
        assert "Indexing complete" in _plain(result.output)
        assert "in the background" not in _plain(result.output)
        assert calls["n"] == 0
        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta["stats"]["embeddings"] == 0

    def test_sync_embeds_inline_without_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)

        monkeypatch.setattr(
            "synaptiq.core.embeddings.embedder._get_model",
            lambda *a, **k: _FakeEmbedModel(),
        )
        calls = {"n": 0}
        monkeypatch.setattr(
            "synaptiq.core.embeddings.lazy_worker.spawn_lazy_worker",
            lambda rp: calls.__setitem__("n", calls["n"] + 1),
        )

        result = runner.invoke(app, ["analyze", ".", "--embeddings", "sync"])
        assert result.exit_code == 0, result.output
        assert "Indexing complete" in _plain(result.output)
        assert calls["n"] == 0  # sync never spawns a background worker
        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta["stats"]["embeddings"] > 0  # vectors stored inline

    def test_no_embeddings_alias_warns_and_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)

        calls = {"n": 0}
        monkeypatch.setattr(
            "synaptiq.core.embeddings.lazy_worker.spawn_lazy_worker",
            lambda rp: calls.__setitem__("n", calls["n"] + 1),
        )

        result = runner.invoke(app, ["analyze", ".", "--no-embeddings"])
        assert result.exit_code == 0, result.output
        assert "deprecated" in _plain(result.output)
        assert calls["n"] == 0  # behaves like off
        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta["stats"]["embeddings"] == 0


class _FakeStaticModel:
    """Deterministic model2vec.StaticModel stand-in for CLI tests.

    Unlike fastembed's streaming-generator `.embed()`, model2vec's
    `.encode()` returns the whole batch as one array — see
    embedder._encode_batches, which dispatches on this shape difference.
    """

    def encode(self, texts, batch_size: int = 64, use_multiprocessing: bool = True):
        import hashlib

        import numpy as np

        return np.array(
            [
                np.random.default_rng(int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "big"))
                .random(256)
                .astype(np.float32)
                for t in texts
            ]
        )


class TestAnalyzeEmbeddingModelFlag:
    """Tests for `analyze --embedding-model quality|fast` (W4.4)."""

    @staticmethod
    def _tiny_repo(repo: Path) -> None:
        src = repo / "src"
        src.mkdir(parents=True)
        (src / "main.py").write_text(
            "def main():\n    return helper()\n\n\ndef helper():\n    return 42\n",
            encoding="utf-8",
        )

    def test_help_lists_embedding_model_choices(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        output = _plain(result.output)
        assert "--embedding-model" in output
        assert "quality" in output and "fast" in output

    def test_default_tier_is_quality(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(
            "synaptiq.core.embeddings.embedder._get_model", lambda *a, **k: _FakeEmbedModel()
        )

        result = runner.invoke(app, ["analyze", ".", "--embeddings", "sync"])

        assert result.exit_code == 0, result.output
        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta["stats"]["embedding_model"] == "quality"

    def test_fast_tier_persisted_and_stores_256dim_vectors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The encode itself uses _FakeStaticModel, but analyze's eager
        # ensure_tier_available() does a real find_spec("model2vec") that a
        # monkeypatch can't satisfy — skip in environments without the extra.
        pytest.importorskip("model2vec")
        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)
        monkeypatch.setattr(
            "synaptiq.core.embeddings.embedder._get_model", lambda *a, **k: _FakeStaticModel()
        )

        result = runner.invoke(
            app, ["analyze", ".", "--embeddings", "sync", "--embedding-model", "fast"]
        )

        assert result.exit_code == 0, result.output
        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta["stats"]["embedding_model"] == "fast"
        assert meta["stats"]["embeddings"] > 0

        from synaptiq.core.storage.ladybug_backend import LadybugBackend

        storage = LadybugBackend()
        storage.initialize(repo / ".synaptiq" / "kuzu", read_only=True)
        try:
            stored = storage.load_embeddings()
            assert stored
            assert len(next(iter(stored.values()))[1]) == 256
        finally:
            storage.close()

    def test_missing_dependency_fails_fast_before_indexing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """model2vec "not installed" (simulated) — analyze must fail with a
        clear, actionable error BEFORE running the pipeline at all, not
        after minutes of indexing."""
        import sys

        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)
        monkeypatch.setitem(sys.modules, "model2vec", None)

        result = runner.invoke(
            app, ["analyze", ".", "--embeddings", "sync", "--embedding-model", "fast"]
        )

        assert result.exit_code == 1
        # The exact bracketed install hint, not just a loose substring: Rich
        # console markup parses an unescaped "[fast-embeddings]" as a style
        # tag and silently drops it, so this specific assertion catches that
        # regression class (it did, once, during development of this flag).
        assert "synaptiq[fast-embeddings]" in _plain(result.output)
        # Failed before any indexing started — no .synaptiq directory at all.
        assert not (repo / ".synaptiq").exists()

    def test_off_mode_skips_missing_dependency_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--embeddings off never encodes anything, so a missing fast-tier
        dependency must not block indexing even with --embedding-model fast."""
        import sys

        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)
        monkeypatch.setitem(sys.modules, "model2vec", None)

        result = runner.invoke(
            app, ["analyze", ".", "--embeddings", "off", "--embedding-model", "fast"]
        )

        assert result.exit_code == 0, result.output
        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta["stats"]["embeddings"] == 0

    def test_tier_switch_forces_full_reencode_sync_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repo indexed with "quality" then re-analyzed with "fast" must
        re-encode every symbol fresh rather than reusing the
        differently-sized old vectors (embedder._partition_texts' tier-
        salted text_sha)."""
        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)

        def _dispatch(tier_name):
            return _FakeEmbedModel() if tier_name == "quality" else _FakeStaticModel()

        monkeypatch.setattr("synaptiq.core.embeddings.embedder._get_model", _dispatch)

        first = runner.invoke(app, ["analyze", ".", "--embeddings", "sync"])
        assert first.exit_code == 0, first.output
        meta_after_first = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta_after_first["stats"]["embedding_model"] == "quality"
        quality_count = meta_after_first["stats"]["embeddings"]
        assert quality_count > 0

        second = runner.invoke(
            app, ["analyze", ".", "--embeddings", "sync", "--embedding-model", "fast"]
        )
        assert second.exit_code == 0, second.output
        meta_after_second = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta_after_second["stats"]["embedding_model"] == "fast"
        # Every symbol was re-encoded fresh (same total count as before) —
        # a partial/leftover reuse from the old tier would have either
        # errored (mixed FLOAT[dim] widths in one COPY) or under-counted.
        assert meta_after_second["stats"]["embeddings"] == quality_count

        from synaptiq.core.storage.ladybug_backend import LadybugBackend

        storage = LadybugBackend()
        storage.initialize(repo / ".synaptiq" / "kuzu", read_only=True)
        try:
            stored = storage.load_embeddings()
            assert len(next(iter(stored.values()))[1]) == 256
        finally:
            storage.close()

    def test_tier_switch_in_lazy_mode_shows_zero_reuse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lazy mode's console output explicitly distinguishes "N reused"
        from "encoding N" — a tier switch must show the latter (nothing
        reused), matching partition_embeddings returning zero reused."""
        repo = tmp_path / "repo"
        self._tiny_repo(repo)
        monkeypatch.chdir(repo)

        def _dispatch(tier_name):
            return _FakeEmbedModel() if tier_name == "quality" else _FakeStaticModel()

        monkeypatch.setattr("synaptiq.core.embeddings.embedder._get_model", _dispatch)

        first = runner.invoke(app, ["analyze", ".", "--embeddings", "sync"])
        assert first.exit_code == 0, first.output

        spawned: dict = {}
        monkeypatch.setattr(
            "synaptiq.core.embeddings.lazy_worker.spawn_lazy_worker",
            lambda rp: spawned.setdefault("called", True) or 9999,
        )

        second = runner.invoke(
            app, ["analyze", ".", "--embeddings", "lazy", "--embedding-model", "fast"]
        )

        assert second.exit_code == 0, second.output
        assert "reused" not in second.output.lower()
        assert "in the background" in second.output
        assert spawned.get("called") is True


class TestAnalyzeLazyReuse:
    """`analyze --embeddings lazy` reuses vectors across rebuilds (W4.1b).

    Uses the deterministic `_FakeEmbedModel` (no ONNX, no subprocess) so these
    stay fast and race-free — the real detached-worker path is covered
    end-to-end in tests/e2e/test_lazy_embeddings.py.
    """

    @staticmethod
    def _repo_with_two_files(repo: Path) -> None:
        src = repo / "src"
        src.mkdir(parents=True)
        (src / "main.py").write_text(
            "def main():\n    return helper()\n\n\ndef helper():\n    return 42\n",
            encoding="utf-8",
        )
        (src / "util.py").write_text(
            "def format_name(name):\n    return name.strip().title()\n",
            encoding="utf-8",
        )

    @staticmethod
    def _seed_with_sync_analyze(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Populate a full embedding table quickly via the fake model — no
        ONNX, no lazy worker involved."""
        monkeypatch.setattr(
            "synaptiq.core.embeddings.embedder._get_model",
            lambda *a, **k: _FakeEmbedModel(),
        )
        result = runner.invoke(app, ["analyze", str(repo), "--embeddings", "sync"])
        assert result.exit_code == 0, result.output

    def test_unchanged_repo_reuses_everything_and_skips_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from synaptiq.core.storage.ladybug_backend import LadybugBackend

        repo = tmp_path / "repo"
        self._repo_with_two_files(repo)
        self._seed_with_sync_analyze(repo, monkeypatch)

        before = LadybugBackend()
        before.initialize(repo / ".synaptiq" / "kuzu", read_only=True)
        try:
            stored_before = before.load_embeddings()
        finally:
            before.close()
        assert stored_before  # sanity: the sync run actually stored vectors

        monkeypatch.chdir(repo)
        spawned = {"n": 0}
        monkeypatch.setattr(
            "synaptiq.core.embeddings.lazy_worker.spawn_lazy_worker",
            lambda rp: spawned.__setitem__("n", spawned["n"] + 1),
        )

        result = runner.invoke(app, ["analyze"])  # lazy is the default, repo unchanged
        assert result.exit_code == 0, result.output
        assert "Index ready" in _plain(result.output)
        assert "vectors reused" in _plain(result.output)
        assert "in the background" not in _plain(result.output)  # nothing pending
        assert spawned["n"] == 0  # zero-change analyze must not spawn a no-op worker

        after = LadybugBackend()
        after.initialize(repo / ".synaptiq" / "kuzu", read_only=True)
        try:
            stored_after = after.load_embeddings()
        finally:
            after.close()
        assert len(stored_after) == len(stored_before)

        meta = json.loads((repo / ".synaptiq" / "meta.json").read_text())
        assert meta["stats"]["embeddings"] == len(stored_before)

    def test_partial_change_stores_reused_synchronously_before_spawning_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from synaptiq.core.storage.ladybug_backend import LadybugBackend

        repo = tmp_path / "repo"
        self._repo_with_two_files(repo)
        self._seed_with_sync_analyze(repo, monkeypatch)

        before = LadybugBackend()
        before.initialize(repo / ".synaptiq" / "kuzu", read_only=True)
        try:
            total_before = len(before.load_embeddings())
        finally:
            before.close()

        # Touch ONE file: append a brand-new, self-contained function. Its own
        # text is new (no previous entry -> pending) and it changes util.py's
        # FILE node `defines:` text -> that FILE node also goes pending.
        # Nothing calls or is called by it, so every other symbol's generated
        # text — and therefore its text_sha — is untouched: exactly 2 of the
        # (total_before + 1) embeddable nodes should end up pending.
        (repo / "src" / "util.py").write_text(
            "def format_name(name):\n    return name.strip().title()\n\n\n"
            "def extra():\n    return 1\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(repo)

        call_order: list[str] = []
        original_store = LadybugBackend.store_embeddings

        def _spy_store(self, embeddings):
            call_order.append(f"store:{len(embeddings)}")
            return original_store(self, embeddings)

        monkeypatch.setattr(LadybugBackend, "store_embeddings", _spy_store)

        spawned: dict = {}

        def _spy_spawn(rp):
            call_order.append("spawn")
            spawned["repo"] = rp
            return 4242

        monkeypatch.setattr("synaptiq.core.embeddings.lazy_worker.spawn_lazy_worker", _spy_spawn)

        result = runner.invoke(app, ["analyze"])  # lazy is the default
        assert result.exit_code == 0, result.output
        assert "Index ready" in _plain(result.output)
        assert "vectors reused" in _plain(result.output)
        assert "in the background" in _plain(result.output)
        assert spawned.get("repo") == repo.resolve()

        # Reused vectors are stored synchronously, BEFORE the worker spawns.
        assert call_order == [f"store:{total_before - 1}", "spawn"]

        # The worker never actually ran (spawn is mocked to a no-op), so the
        # DB must hold EXACTLY the reused set right now — strictly less than
        # the full node count, proving the store above was partial, not full.
        after = LadybugBackend()
        after.initialize(repo / ".synaptiq" / "kuzu", read_only=True)
        try:
            stored_after = after.load_embeddings()
        finally:
            after.close()
        assert len(stored_after) == total_before - 1


class TestStatus:
    """Tests for the status command."""

    def test_status_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Status should error when no .synaptiq directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert "No index found" in _plain(result.output)

    def test_status_with_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Status should display stats from meta.json."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / ".synaptiq"
        data_dir.mkdir()
        meta = {
            "version": "0.1.0",
            "stats": {
                "files": 10,
                "symbols": 42,
                "relationships": 100,
                "clusters": 3,
                "flows": 0,
                "dead_code": 5,
                "coupled_pairs": 0,
            },
            "last_indexed_at": "2025-01-15T10:00:00+00:00",
        }
        (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Index status for" in _plain(result.output)
        assert "0.1.0" in _plain(result.output)
        assert "10" in _plain(result.output)  # files
        assert "42" in _plain(result.output)  # symbols
        assert "100" in _plain(result.output)  # relationships

    @staticmethod
    def _write_index(tmp_path: Path, state: dict | None, *, embeddings: int = 0) -> Path:
        data_dir = tmp_path / ".synaptiq"
        data_dir.mkdir(exist_ok=True)
        meta = {
            "version": "1.0.0",
            "stats": {"files": 1, "symbols": 2, "relationships": 3, "embeddings": embeddings},
            "last_indexed_at": "2026-07-12T10:00:00+00:00",
        }
        (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        if state is not None:
            (data_dir / "embeddings_state.json").write_text(json.dumps(state), encoding="utf-8")
        return data_dir

    def test_status_shows_encoding_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        self._write_index(tmp_path, {"state": "encoding", "done": 12431, "total": 26203})
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "encoding 12,431/26,203" in _plain(result.output)

    def test_status_shows_complete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._write_index(tmp_path, {"state": "complete", "done": 42, "total": 42}, embeddings=42)
        result = runner.invoke(app, ["status"])
        assert "42 (complete)" in _plain(result.output)

    def test_status_shows_failed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._write_index(tmp_path, {"state": "failed", "error": "model offline"})
        result = runner.invoke(app, ["status"])
        assert "failed" in _plain(result.output)
        assert "model offline" in _plain(result.output)

    def test_status_shows_failed_error_with_brackets_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker failure whose message contains a literal bracketed
        substring (e.g. the fast tier's missing-dependency install hint,
        W4.4) must render in full — unescaped, Rich console markup parses
        "[fast-embeddings]" as a style tag and silently drops it."""
        monkeypatch.chdir(tmp_path)
        self._write_index(
            tmp_path,
            {
                "state": "failed",
                "error": "The 'fast' embedding tier requires the 'model2vec' package, "
                "which is not installed. Install it with: pip install "
                "'synaptiq[fast-embeddings]'.",
            },
        )
        result = runner.invoke(app, ["status"])
        assert "synaptiq[fast-embeddings]" in _plain(result.output)

    def test_status_shows_deferred(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._write_index(tmp_path, {"state": "deferred", "detail": "index locked"})
        result = runner.invoke(app, ["status"])
        assert "deferred" in _plain(result.output)

    def test_status_falls_back_to_meta_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no state file, the stored embedding count is shown."""
        monkeypatch.chdir(tmp_path)
        self._write_index(tmp_path, None, embeddings=17)
        result = runner.invoke(app, ["status"])
        assert "Embeddings:" in _plain(result.output)
        assert "17" in _plain(result.output)


class TestListRepos:
    """Tests for the list command."""

    def test_list_calls_handle_list_repos(self) -> None:
        """List should call handle_list_repos and print the result."""
        with patch(
            "synaptiq.mcp.tools.handle_list_repos",
            return_value="Indexed repositories (1):\n\n  1. my-project",
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "my-project" in _plain(result.output)

    def test_list_no_repos(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """List should show 'no repos' message when none are indexed."""
        monkeypatch.chdir(tmp_path)
        # Patch the global registry to a non-existent dir so the fallback also fails
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # handle_list_repos returns "No indexed repositories found." when nothing found
        assert (
            "No indexed repositories found" in _plain(result.output)
            or "repositories" in _plain(result.output).lower()
        )


class TestClean:
    """Tests for the clean command."""

    def test_clean_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clean should error when no .synaptiq directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["clean", "--force"])
        assert result.exit_code == 1
        assert "No index found" in _plain(result.output)

    def test_clean_with_force(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clean with --force should delete .synaptiq without confirmation."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / ".synaptiq"
        data_dir.mkdir()
        (data_dir / "meta.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(app, ["clean", "--force"])
        assert result.exit_code == 0
        assert "Deleted" in _plain(result.output)
        assert not data_dir.exists()

    def test_clean_aborted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clean should abort when user says no."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / ".synaptiq"
        data_dir.mkdir()
        (data_dir / "meta.json").write_text("{}", encoding="utf-8")

        runner.invoke(app, ["clean"], input="n\n")
        assert data_dir.exists()  # Not deleted


class TestQuery:
    """Tests for the query command."""

    def test_query_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Query should error when no .synaptiq/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["query", "find classes"])
        assert result.exit_code == 1
        assert "No index found" in _plain(result.output)

    def test_query_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Query should call handle_query with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("synaptiq.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "synaptiq.mcp.server.handle_query",
                return_value="1. MyClass (Class) -- src/main.py",
            ):
                result = runner.invoke(app, ["query", "find classes"])
        assert result.exit_code == 0
        assert "MyClass" in _plain(result.output)


class TestContext:
    """Tests for the context command."""

    def test_context_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Context should error when no .synaptiq/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["context", "MyClass"])
        assert result.exit_code == 1
        assert "No index found" in _plain(result.output)

    def test_context_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Context should call handle_context with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("synaptiq.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "synaptiq.mcp.server.handle_context",
                return_value="Symbol: MyClass (Class)\nFile: src/main.py:1-50",
            ):
                result = runner.invoke(app, ["context", "MyClass"])
        assert result.exit_code == 0
        assert "MyClass" in _plain(result.output)


class TestImpact:
    """Tests for the impact command."""

    def test_impact_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Impact should error when no .synaptiq/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["impact", "MyClass.method"])
        assert result.exit_code == 1
        assert "No index found" in _plain(result.output)

    def test_impact_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Impact should call handle_impact with loaded storage and depth."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("synaptiq.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "synaptiq.mcp.server.handle_impact",
                return_value="Impact analysis for: MyClass.method",
            ):
                result = runner.invoke(app, ["impact", "MyClass.method", "--depth", "5"])
        assert result.exit_code == 0
        assert "Impact analysis" in _plain(result.output)

    def test_impact_default_depth(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Impact without --depth should use default depth of 3."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("synaptiq.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "synaptiq.mcp.server.handle_impact",
                return_value="Impact analysis for: foo",
            ):
                result = runner.invoke(app, ["impact", "foo"])
        assert result.exit_code == 0


class TestDeadCode:
    """Tests for the dead-code command."""

    def test_dead_code_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dead-code should error when no .synaptiq/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["dead-code"])
        assert result.exit_code == 1
        assert "No index found" in _plain(result.output)

    def test_dead_code_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dead-code should call handle_dead_code with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("synaptiq.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "synaptiq.mcp.server.handle_dead_code",
                return_value="No dead code detected.",
            ):
                result = runner.invoke(app, ["dead-code"])
        assert result.exit_code == 0
        assert "No dead code detected" in _plain(result.output)


class TestCypher:
    """Tests for the cypher command."""

    def test_cypher_no_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cypher should error when no .synaptiq/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["cypher", "MATCH (n) RETURN n"])
        assert result.exit_code == 1
        assert "No index found" in _plain(result.output)

    def test_cypher_with_storage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cypher should call handle_cypher with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("synaptiq.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "synaptiq.mcp.server.handle_cypher",
                return_value="Results (3 rows):\n\n  1. foo",
            ):
                result = runner.invoke(app, ["cypher", "MATCH (n) RETURN n"])
        assert result.exit_code == 0
        assert "Results" in _plain(result.output)


class TestSetup:
    """Tests for the setup command."""

    def test_setup_no_flags_shows_both(self) -> None:
        """Setup with no flags should show config for both Claude and Cursor."""
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "Claude Code" in _plain(result.output)
        assert "Cursor" in _plain(result.output)
        assert '"synaptiq"' in _plain(result.output)

    def test_setup_claude_only(self) -> None:
        """Setup with --claude should show only Claude config."""
        result = runner.invoke(app, ["setup", "--claude"])
        assert result.exit_code == 0
        assert "Claude Code" in _plain(result.output)
        assert "Cursor" not in _plain(result.output)

    def test_setup_cursor_only(self) -> None:
        """Setup with --cursor should show only Cursor config."""
        result = runner.invoke(app, ["setup", "--cursor"])
        assert result.exit_code == 0
        assert "Cursor" in _plain(result.output)
        assert "Claude Code" not in _plain(result.output)

    def test_setup_both_flags(self) -> None:
        """Setup with both flags should show both configs."""
        result = runner.invoke(app, ["setup", "--claude", "--cursor"])
        assert result.exit_code == 0
        assert "Claude Code" in _plain(result.output)
        assert "Cursor" in _plain(result.output)


class TestMcp:
    """Tests for the mcp command."""

    def test_mcp_command_exists(self) -> None:
        """The mcp command should be registered."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP server" in _plain(result.output) or "stdio" in _plain(result.output).lower()

    def test_mcp_calls_server_main(self) -> None:
        """MCP command should call asyncio.run(mcp_main())."""
        with patch("synaptiq.cli.main.asyncio", create=True):
            with patch("synaptiq.mcp.server.main"):
                # We need to mock at the import level inside the function
                import asyncio as real_asyncio

                with patch.object(real_asyncio, "run") as mock_run:
                    runner.invoke(app, ["mcp"])
                    mock_run.assert_called_once()


class TestServe:
    """Tests for the serve command."""

    def test_serve_command_exists(self) -> None:
        """The serve command should be registered."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "watch" in _plain(result.output).lower()

    def test_serve_without_watch_delegates_to_mcp(self) -> None:
        """serve without --watch should behave like synaptiq mcp."""
        import asyncio as real_asyncio

        with patch.object(real_asyncio, "run") as mock_run:
            runner.invoke(app, ["serve"])
            mock_run.assert_called_once()


class TestWatch:
    """Tests for the watch command."""

    def test_watch_command_exists(self) -> None:
        """The watch command should be registered."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch mode" in _plain(result.output) or "re-index" in _plain(result.output).lower()

    def test_diff_command_exists(self) -> None:
        """The diff command should be registered."""
        result = runner.invoke(app, ["diff", "--help"])
        assert result.exit_code == 0
        assert "branch" in _plain(result.output).lower()
