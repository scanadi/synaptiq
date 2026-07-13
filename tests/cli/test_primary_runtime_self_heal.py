"""Integration test: daemon-startup embedding self-heal (2.0.4, BUG 3b).

`self_heal_pending_embeddings` itself is unit-tested exhaustively in
tests/core/test_lazy_worker.py::TestSelfHeal (in-process, no daemon
scaffolding). This file proves the OTHER half of BUG 3b — that
`_PrimaryRuntime.start()` (the real `serve --watch` primary/proxy-promotion
entry point) actually wires it in and runs it to completion, under the real
async rwlock, before the runtime is considered started — using the same
deterministic fake embedding model as the rest of the lazy-embeddings test
suite (no ONNX download, no subprocess).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from synaptiq.cli.main import _PrimaryRuntime
from synaptiq.core.daemon.lock import LockManager
from synaptiq.core.embeddings import lazy_worker
from synaptiq.core.embeddings.embedder import embeddable_node_count
from synaptiq.core.ingestion.pipeline import run_pipeline, write_meta
from synaptiq.core.storage.ladybug_backend import open_with_recovery


class _FakeEmbedModel:
    """Deterministic stand-in for fastembed's ``TextEmbedding`` (matches
    tests/core/test_lazy_worker.py's fixture)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts, batch_size: int = 64):
        for text in texts:
            seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
            yield np.random.default_rng(seed).random(self.dim).astype(np.float32)


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "synaptiq.core.embeddings.embedder._get_model",
        lambda *a, **k: _FakeEmbedModel(),
    )


def _build_indexed_repo(tmp_path: Path) -> tuple[Path, Path, int]:
    """A real, committed index WITHOUT embeddings — mirrors a lazy `analyze`
    whose worker never got to store anything. Returns (repo, data_dir,
    embeddable_total)."""
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "main.py").write_text(
        "def main():\n    return helper()\n\n\ndef helper():\n    return 42\n",
        encoding="utf-8",
    )
    data_dir = repo / ".synaptiq"
    data_dir.mkdir()
    db_path = data_dir / "kuzu"

    storage = open_with_recovery(db_path, data_dir / "meta.json", build_fts_indexes=False)
    graph, result = run_pipeline(repo, storage, full=True, skip_embeddings=True)
    total = embeddable_node_count(graph)
    write_meta(data_dir, repo, result)
    storage.close()
    return repo, data_dir, total


class TestPrimaryRuntimeSelfHeal:
    async def test_start_heals_a_stale_deferred_state_before_returning(
        self, tmp_path: Path
    ) -> None:
        repo, data_dir, expected_total = _build_indexed_repo(tmp_path)
        assert expected_total > 0  # sanity: the fixture repo has embeddable nodes

        # Seed the exact field-verified scenario: a dead worker's `deferred`
        # sentinel, with meta.json still short of what it was waiting for.
        lazy_worker.write_state(
            data_dir,
            "deferred",
            total=expected_total,
            detail="index locked; re-run `synaptiq analyze` to encode",
            pid=99999999,
        )
        meta_before = json.loads((data_dir / "meta.json").read_text())
        assert meta_before["stats"]["embeddings"] == 0

        lock_mgr = LockManager(data_dir)
        assert lock_mgr.try_acquire() is not None
        runtime = _PrimaryRuntime(repo, data_dir, lock_mgr)
        stop_event = asyncio.Event()

        try:
            await runtime.start(stop_event)

            # The self-heal ran to completion as part of start() itself —
            # by the time start() returns, the sentinel is honest and the
            # vectors are actually stored.
            state = lazy_worker.read_state(data_dir)
            assert state is not None
            assert state["state"] == "complete"

            stored = runtime.storage.load_embeddings()
            assert len(stored) == expected_total

            meta_after = json.loads((data_dir / "meta.json").read_text())
            assert meta_after["stats"]["embeddings"] == expected_total
        finally:
            stop_event.set()
            await runtime.stop()
            lock_mgr.release()

    async def test_start_is_a_noop_when_no_state_file_exists(self, tmp_path: Path) -> None:
        """The common case (no prior lazy worker ever ran) must start
        cleanly with no self-heal side effects."""
        repo, data_dir, _ = _build_indexed_repo(tmp_path)
        assert not (data_dir / "embeddings_state.json").exists()

        lock_mgr = LockManager(data_dir)
        assert lock_mgr.try_acquire() is not None
        runtime = _PrimaryRuntime(repo, data_dir, lock_mgr)
        stop_event = asyncio.Event()

        try:
            await runtime.start(stop_event)
            assert not (data_dir / "embeddings_state.json").exists()
            assert runtime.storage.load_embeddings() == {}
        finally:
            stop_event.set()
            await runtime.stop()
            lock_mgr.release()


class TestSocketReindexStampsComplete:
    """2.0.4 (BUG 2): the daemon's socket-delegated reindex path
    (`_reindex_async` -> `commit_full_index`, a synchronous embed-store)
    must stamp its own `complete` sentinel too. Isolated from the startup
    self-heal above by seeding the stale sentinel AFTER `start()` — at
    startup there is no state file yet, so that self-heal is a no-op and
    only the reindex's own stamp is under test here.
    """

    async def test_reindex_clears_a_stale_sentinel_left_by_an_earlier_run(
        self, tmp_path: Path
    ) -> None:
        from synaptiq.core.daemon.socket_client import SocketClient

        repo, data_dir, expected_total = _build_indexed_repo(tmp_path)

        lock_mgr = LockManager(data_dir)
        assert lock_mgr.try_acquire() is not None
        runtime = _PrimaryRuntime(repo, data_dir, lock_mgr)
        stop_event = asyncio.Event()

        try:
            await runtime.start(stop_event)  # no state file yet -> startup self-heal is a no-op

            # An unrelated earlier lazy worker left a stale sentinel behind.
            lazy_worker.write_state(data_dir, "deferred", total=999, detail="unrelated stale entry")

            client = SocketClient(lock_mgr.socket_path)
            await client.connect()
            try:
                response = await client.reindex(full=True)
            finally:
                await client.close()

            stats = json.loads(response)["stats"]
            assert stats["files"] >= 1

            state = lazy_worker.read_state(data_dir)
            assert state is not None
            assert state["state"] == "complete"
            assert state["total"] == expected_total  # NOT the stale 999
        finally:
            stop_event.set()
            await runtime.stop()
            lock_mgr.release()
