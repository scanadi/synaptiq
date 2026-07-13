"""End-to-end tests for lazy background embeddings (W4.1).

These exercise the *real* detached worker: ``analyze --embeddings lazy`` spawns
an actual ``python -m synaptiq _embed-worker`` subprocess, and the test polls
``embeddings_state.json`` (foreground, bounded) until it settles.  Worker
processes are killed deterministically by the PID recorded in the state file.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from synaptiq.cli.main import app
from synaptiq.core.embeddings import lazy_worker
from synaptiq.core.storage.base import NodeEmbedding
from synaptiq.core.storage.ladybug_backend import LadybugBackend, is_lock_error

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_model() -> None:
    """Skip this module when the ONNX model cannot be loaded (offline CI)."""
    try:
        from fastembed import TextEmbedding

        TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"embedding model unavailable: {exc}")


def _tiny_repo(repo: Path) -> None:
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


def _build_committed_index(repo: Path) -> tuple[Path, Path]:
    """Create a real, committed index with embeddings skipped (the lazy start)."""
    from synaptiq.core.ingestion.pipeline import run_pipeline, write_meta
    from synaptiq.core.storage.ladybug_backend import open_with_recovery

    _tiny_repo(repo)
    data_dir = repo / ".synaptiq"
    data_dir.mkdir()
    db_path = data_dir / "kuzu"
    storage = open_with_recovery(db_path, data_dir / "meta.json", build_fts_indexes=False)
    _, result = run_pipeline(repo, storage, full=True, skip_embeddings=True)
    write_meta(data_dir, repo, result)
    storage.close()
    return data_dir, db_path


def _poll_state(
    data_dir: Path, *, timeout: float = 120.0, expect_pid: int | None = None
) -> dict | None:
    """Poll ``embeddings_state.json`` until a terminal state settles.

    When *expect_pid* is given, a terminal state whose ``pid`` doesn't match
    it is a stale leftover from an EARLIER worker (e.g. a previous analyze
    cycle in the same test) and polling continues — otherwise, on a repo with
    two lazy cycles, this could return the first cycle's "complete" before
    the second cycle's worker has even started overwriting the file.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = lazy_worker.read_state(data_dir)
        if (
            state is not None
            and state.get("state") in ("complete", "failed", "deferred")
            and (expect_pid is None or state.get("pid") == expect_pid)
        ):
            return state
        time.sleep(0.5)
    return lazy_worker.read_state(data_dir)


def _kill(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _query_with_retry(db_path: Path, query: str, *, timeout: float = 20.0):
    """Read-only hybrid query, retrying past the worker's brief store window."""
    from synaptiq.core.search.hybrid import hybrid_search

    deadline = time.time() + timeout
    last_exc: Exception | None = None
    while time.time() < deadline:
        storage = LadybugBackend()
        try:
            storage.initialize(db_path, read_only=True)
        except Exception as exc:  # the worker may hold the write lock momentarily
            storage.close()
            last_exc = exc
            if is_lock_error(exc):
                time.sleep(0.3)
                continue
            raise
        try:
            return hybrid_search(query, storage, limit=10)
        finally:
            storage.close()
    raise AssertionError(f"index not queryable within {timeout}s: {last_exc}")


# ---------------------------------------------------------------------------
# Full lazy round-trip
# ---------------------------------------------------------------------------


class TestLazyAnalyzeEndToEnd:
    def test_index_ready_fast_then_vectors_fill_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedding_model
    ) -> None:
        repo = tmp_path / "repo"
        _tiny_repo(repo)
        monkeypatch.chdir(repo)
        data_dir = repo / ".synaptiq"

        worker_pid: int | None = None
        try:
            result = runner.invoke(app, ["analyze", ".", "--embeddings", "lazy"])
            assert result.exit_code == 0, result.output
            # Returned WITHOUT encoding inline: no embedding count in the summary,
            # and an explicit background-worker notice instead.
            assert "Index ready" in result.output
            assert "in the background" in result.output

            # The committed index is queryable immediately (BM25 + fuzzy), even
            # while the worker is still encoding vectors.
            results = _query_with_retry(data_dir / "kuzu", "helper")
            assert any(r.node_name == "helper" for r in results)

            # Wait for the detached worker to finish (foreground bounded poll).
            state = _poll_state(data_dir)
            assert state is not None, "worker never published a state file"
            worker_pid = state.get("pid")
            assert state["state"] == "complete", state
            assert state["total"] == state["done"] > 0

            # Vectors are persisted and the HNSW index answers queries.
            storage = LadybugBackend()
            storage.initialize(data_dir / "kuzu", read_only=True)
            try:
                stored = storage.load_embeddings()
                assert len(stored) == state["total"]
                probe = list(next(iter(stored.values()))[1])
                assert storage.vector_search(probe, limit=5)
            finally:
                storage.close()

            # meta.json carries the final embedding count.
            meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
            assert meta["stats"]["embeddings"] == state["total"]
        finally:
            _kill(worker_pid)


# ---------------------------------------------------------------------------
# Cross-rebuild vector reuse (W4.1b)
# ---------------------------------------------------------------------------


class TestLazyCrossRebuildReuse:
    """Lazy mode reuses vectors across rebuilds instead of re-encoding the
    full set every time.

    Closes W4.1's documented deviation: ``bulk_load`` used to wipe the
    Embedding table before the background worker ever got a chance to
    snapshot it, so every lazy ``analyze`` — even a no-op rebuild — re-encoded
    every symbol in the background. These exercise the real detached worker
    end-to-end, like ``TestLazyAnalyzeEndToEnd`` above.
    """

    def test_partial_change_reuses_most_vectors_synchronously(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedding_model
    ) -> None:
        repo = tmp_path / "repo"
        _tiny_repo(repo)
        monkeypatch.chdir(repo)
        data_dir = repo / ".synaptiq"

        # Spy (not mock) spawn_lazy_worker so both cycles' real PIDs are
        # known — _poll_state needs the SECOND cycle's PID to avoid reading
        # the first cycle's still-"complete" state as if it were fresh (the
        # second worker can take a couple of seconds just to start up and
        # overwrite the file).
        spawned_pids: list[int] = []
        original_spawn = lazy_worker.spawn_lazy_worker

        def _spy_spawn(rp):
            pid = original_spawn(rp)
            if pid is not None:
                spawned_pids.append(pid)
            return pid

        monkeypatch.setattr("synaptiq.core.embeddings.lazy_worker.spawn_lazy_worker", _spy_spawn)

        worker_pid: int | None = None
        try:
            # First full lazy cycle: commit, then wait for the worker so the
            # index holds a complete, real embedding table to reuse from.
            result = runner.invoke(app, ["analyze", ".", "--embeddings", "lazy"])
            assert result.exit_code == 0, result.output
            assert len(spawned_pids) == 1
            state = _poll_state(data_dir, expect_pid=spawned_pids[0])
            assert state is not None and state["state"] == "complete", state
            worker_pid = state.get("pid")
            first_total = state["total"]
            assert first_total > 0

            # Touch ONE file: append a brand-new, self-contained function.
            # Every existing symbol's generated text is untouched except
            # util.py's FILE node (its `defines:` list gains a name) — a
            # small, deterministic, mostly-reused delta.
            (repo / "src" / "util.py").write_text(
                "def format_name(name):\n    return name.strip().title()\n\n\n"
                "def extra():\n    return 1\n",
                encoding="utf-8",
            )

            result = runner.invoke(app, ["analyze", ".", "--embeddings", "lazy"])
            assert result.exit_code == 0, result.output
            assert "Index ready" in result.output
            assert "vectors reused" in result.output
            assert len(spawned_pids) == 2
            assert spawned_pids[1] != spawned_pids[0]

            # Reused vectors are stored SYNCHRONOUSLY — already readable, and
            # already searchable via a real vector query, before the (maybe
            # still-running) worker has touched anything. helper()'s text
            # never changed, so its vector must already be exactly the one
            # from the first cycle.
            storage = LadybugBackend()
            storage.initialize(data_dir / "kuzu", read_only=True)
            try:
                stored_now = storage.load_embeddings()
                # At minimum every unchanged node from the first cycle is
                # already here — the worker can only ever ADD to this count,
                # so this holds no matter how the two processes interleave.
                assert len(stored_now) >= first_total - 1
                helper_id = "function:src/main.py:helper"
                assert helper_id in stored_now

                from synaptiq.core.search.hybrid import hybrid_search

                probe = list(stored_now[helper_id][1])
                results = hybrid_search("helper", storage, query_embedding=probe, limit=5)
                assert any(r.node_id == helper_id for r in results)
            finally:
                storage.close()

            # The background worker (handling the small delta) finishes and
            # settles the index at the new, larger total. Its progress state
            # reports the PENDING delta it actually encoded (here: util.py's
            # FILE node, whose `defines:` list changed, plus the brand-new
            # function) — NOT the full embeddable count. Before the 2.0.3
            # honest-progress fix this showed "encoding 0/<full count>" for a
            # tiny delta; `first_total` (a cold start, nothing reusable) keeps
            # pending == full count, so its meaning above is unchanged.
            state = _poll_state(data_dir, expect_pid=spawned_pids[1])
            assert state is not None, "worker never published a state file"
            worker_pid = state.get("pid", worker_pid)
            assert state["state"] == "complete", state
            assert state["total"] == state["done"]
            assert 0 < state["total"] < first_total, (
                "worker progress must report the pending delta, not the full embeddable count"
            )

            storage = LadybugBackend()
            storage.initialize(data_dir / "kuzu", read_only=True)
            try:
                assert len(storage.load_embeddings()) == first_total + 1
            finally:
                storage.close()
        finally:
            _kill(worker_pid)

    def test_unchanged_repo_second_analyze_needs_no_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embedding_model
    ) -> None:
        repo = tmp_path / "repo"
        _tiny_repo(repo)
        monkeypatch.chdir(repo)
        data_dir = repo / ".synaptiq"

        worker_pid: int | None = None
        try:
            result = runner.invoke(app, ["analyze", ".", "--embeddings", "lazy"])
            assert result.exit_code == 0, result.output
            state = _poll_state(data_dir)
            assert state is not None and state["state"] == "complete", state
            worker_pid = state.get("pid")
            first_total = state["total"]

            # No file changes at all — every text_sha matches the first cycle.
            result = runner.invoke(app, ["analyze", ".", "--embeddings", "lazy"])
            assert result.exit_code == 0, result.output
            assert "Index ready" in result.output
            assert "vectors reused" in result.output
            # Zero-change analyze must not spawn a no-op background worker.
            assert "in the background" not in result.output

            storage = LadybugBackend()
            storage.initialize(data_dir / "kuzu", read_only=True)
            try:
                assert len(storage.load_embeddings()) == first_total
            finally:
                storage.close()
        finally:
            _kill(worker_pid)


# ---------------------------------------------------------------------------
# Deferral under a real cross-process write lock
# ---------------------------------------------------------------------------


_HOLDER_SCRIPT = """
import time
from pathlib import Path
from synaptiq.core.storage.ladybug_backend import LadybugBackend
s = LadybugBackend()
s.initialize(Path({db!r}), read_only=False)
Path({ready!r}).write_text("held")
time.sleep(30)
"""


class TestLazyStoreDeferral:
    def test_store_defers_when_db_locked_by_another_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        data_dir, db_path = _build_committed_index(repo)
        # Keep the test fast — don't actually wait the production backoff.
        monkeypatch.setattr(lazy_worker, "_STORE_RETRY_BACKOFF", (0.2, 0.2))

        ready = data_dir / "holder.ready"
        holder = subprocess.Popen(
            [sys.executable, "-c", _HOLDER_SCRIPT.format(db=str(db_path), ready=str(ready))]
        )
        try:
            for _ in range(200):  # wait up to ~20s for the holder to grab the lock
                if ready.exists():
                    break
                time.sleep(0.1)
            assert ready.exists(), "holder process never acquired the DB write lock"

            fake = [
                NodeEmbedding(
                    node_id="function:src/main.py:main", embedding=[0.1] * 8, text_sha="s"
                )
            ]
            # The write lock is held by another process — the worker must give up.
            assert lazy_worker._store_with_retry(db_path, fake) is False
        finally:
            holder.terminate()
            try:
                holder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                holder.kill()

        # The index is still healthy after the contention clears.
        storage = LadybugBackend()
        storage.initialize(db_path, read_only=True)
        try:
            graph = storage.load_graph()
            assert sum(1 for _ in graph.iter_nodes()) > 0
        finally:
            storage.close()

    def test_worker_writes_deferred_state_when_store_gives_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worker surfaces a locked store as ``deferred`` (index untouched)."""
        repo = tmp_path / "repo"
        data_dir, db_path = _build_committed_index(repo)

        # Fake the encode (no ONNX) and force the store to report a lock give-up.
        import hashlib

        import numpy as np

        class _Fake:
            def embed(self, texts, batch_size: int = 64):
                for t in texts:
                    seed = int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "big")
                    yield np.random.default_rng(seed).random(384).astype(np.float32)

        monkeypatch.setattr("synaptiq.core.embeddings.embedder._get_model", lambda *a, **k: _Fake())
        monkeypatch.setattr(lazy_worker, "_store_with_retry", lambda *a, **k: False)

        assert lazy_worker.run_lazy_embedding_worker(repo) == 0
        state = lazy_worker.read_state(data_dir)
        assert state["state"] == "deferred"
        # No vectors landed; the stored count stays at zero.
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["stats"]["embeddings"] == 0
