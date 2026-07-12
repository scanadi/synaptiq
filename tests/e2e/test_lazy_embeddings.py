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


def _poll_state(data_dir: Path, *, timeout: float = 120.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = lazy_worker.read_state(data_dir)
        if state is not None and state.get("state") in ("complete", "failed", "deferred"):
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
