"""Tests for the lazy background embedding worker (W4.1).

The worker encodes vectors for an already-committed index, out of ``analyze``'s
critical path.  These tests run the worker *in-process* with a deterministic
fake embedding model (no ONNX download) and a real LadybugDB index, so they
cover the full load → encode → staleness-guard → store → meta-update loop
without spawning subprocesses.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from synaptiq.core.embeddings import lazy_worker
from synaptiq.core.storage.base import NodeEmbedding
from synaptiq.core.storage.ladybug_backend import LadybugBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeEmbedModel:
    """Deterministic stand-in for fastembed's ``TextEmbedding``.

    Yields a stable pseudo-random 384-d vector per text so no model download
    happens and results are reproducible within a test run.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts, batch_size: int = 64):  # noqa: D401 - mimics fastembed API
        for text in texts:
            seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
            yield np.random.default_rng(seed).random(self.dim).astype(np.float32)


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ONNX model factory so encoding never downloads anything."""
    monkeypatch.setattr(
        "synaptiq.core.embeddings.embedder._get_model",
        lambda *a, **k: _FakeEmbedModel(),
    )


@pytest.fixture()
def indexed_repo(tmp_path: Path):
    """Build a real, committed index WITHOUT embeddings — the lazy start state.

    Mirrors what ``analyze --embeddings lazy`` leaves on disk: a populated
    graph, ``meta.json`` with ``embeddings: 0``, and an empty Embedding table.
    """
    from synaptiq.core.ingestion.pipeline import run_pipeline, write_meta
    from synaptiq.core.storage.ladybug_backend import open_with_recovery

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
    _, result = run_pipeline(repo, storage, full=True, skip_embeddings=True)
    write_meta(data_dir, repo, result)
    storage.close()
    return repo, data_dir, db_path


def _read_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------


class TestStateFile:
    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        lazy_worker.write_state(tmp_path, "encoding", done=5, total=10, started_at="t0")
        state = lazy_worker.read_state(tmp_path)
        assert state is not None
        assert state["state"] == "encoding"
        assert state["done"] == 5
        assert state["total"] == 10
        assert state["started_at"] == "t0"
        assert state["pid"] == os.getpid()

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        assert lazy_worker.read_state(tmp_path / "absent") is None

    def test_write_is_atomic(self, tmp_path: Path) -> None:
        """A completed write leaves no ``.tmp`` sibling behind."""
        lazy_worker.write_state(tmp_path, "complete", done=3, total=3)
        leftovers = list(tmp_path.glob(f"{lazy_worker.STATE_FILENAME}.tmp"))
        assert leftovers == []


# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------


class TestSingleInstanceLock:
    def test_second_acquire_returns_none(self, tmp_path: Path) -> None:
        fd = lazy_worker._acquire_single_instance(tmp_path)
        assert fd is not None
        try:
            assert lazy_worker._acquire_single_instance(tmp_path) is None
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_worker_exits_quietly_when_lock_held(self, indexed_repo) -> None:
        """A second worker must exit 0 and NOT touch the index or state file."""
        repo, data_dir, _ = indexed_repo
        fd = lazy_worker._acquire_single_instance(data_dir)
        assert fd is not None
        try:
            rc = lazy_worker.run_lazy_embedding_worker(repo)
            assert rc == 0
            # It never reached the encode stage, so no state file was written.
            assert lazy_worker.read_state(data_dir) is None
            assert _read_meta(data_dir)["stats"]["embeddings"] == 0
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# ---------------------------------------------------------------------------
# Happy path: encode → store → meta update
# ---------------------------------------------------------------------------


class TestEncodeAndStore:
    def test_worker_encodes_stores_and_updates_meta(self, indexed_repo) -> None:
        repo, data_dir, db_path = indexed_repo
        rc = lazy_worker.run_lazy_embedding_worker(repo)
        assert rc == 0

        state = lazy_worker.read_state(data_dir)
        assert state is not None
        assert state["state"] == "complete"
        assert state["total"] == state["done"] > 0

        # meta.json embedding count reflects what was stored.
        assert _read_meta(data_dir)["stats"]["embeddings"] == state["total"]

        # Vectors are persisted and the HNSW vector index answers queries.
        storage = LadybugBackend()
        storage.initialize(db_path, read_only=True)
        try:
            stored = storage.load_embeddings()
            assert len(stored) == state["total"]
            probe = list(next(iter(stored.values()))[1])
            assert len(probe) == 384
            hits = storage.vector_search(probe, limit=5)
            assert hits, "vector search should return results after the worker stores vectors"
        finally:
            storage.close()

    def test_encoding_progress_is_published(self, indexed_repo, monkeypatch) -> None:
        """The progress callback surfaces intermediate ``encoding`` states, and
        ``total`` is the PENDING count (3 here), not the full embeddable count —
        embed_graph reports done against the full count and the worker re-bases
        it onto the pending delta (see W4.1 progress-honesty fix)."""
        repo, data_dir, _ = indexed_repo
        seen: list[dict] = []

        # Force a known split: 0 reused, 3 pending — regardless of the real graph.
        monkeypatch.setattr(
            "synaptiq.core.embeddings.embedder.partition_embeddings",
            lambda graph, previous=None, tier="quality": ([], 3),
        )

        def fake_embed(graph, tier=None, previous=None, progress_callback=None):
            # embed_graph's first callback carries the reused count (0), then one
            # per batch: done = reused + encoded against the FULL count (3 here).
            progress_callback(0, 3)
            progress_callback(1, 3)
            seen.append(lazy_worker.read_state(data_dir))
            progress_callback(3, 3)
            return []

        monkeypatch.setattr("synaptiq.core.embeddings.embedder.embed_graph", fake_embed)
        lazy_worker.run_lazy_embedding_worker(repo)

        assert seen and seen[0]["state"] == "encoding"
        assert seen[0]["done"] == 1
        assert seen[0]["total"] == 3
        assert lazy_worker.read_state(data_dir)["state"] == "complete"
        assert lazy_worker.read_state(data_dir)["total"] == 3

    def test_total_is_pending_not_full_embeddable(self, indexed_repo, monkeypatch) -> None:
        """The bug: with mostly-reused vectors, ``total`` must be the small pending
        delta (honest ``synaptiq status``), while meta reflects ALL stored vectors."""
        repo, data_dir, _ = indexed_repo

        reused = [
            NodeEmbedding(
                node_id=f"function:src/x.py:f{i}", embedding=[0.1] * 384, text_sha=f"s{i}"
            )
            for i in range(25)
        ]
        # 25 reused, 2 pending — the honest progress denominator is 2, not 27.
        monkeypatch.setattr(
            "synaptiq.core.embeddings.embedder.partition_embeddings",
            lambda graph, previous=None, tier="quality": (list(reused), 2),
        )

        def fake_embed(graph, tier=None, previous=None, progress_callback=None):
            progress_callback(25, 27)  # reused marker (done=reused, total=full)
            progress_callback(27, 27)  # both pending encoded
            new = [
                NodeEmbedding(
                    node_id=f"function:src/y.py:g{i}", embedding=[0.2] * 384, text_sha=f"n{i}"
                )
                for i in range(2)
            ]
            return list(reused) + new

        monkeypatch.setattr("synaptiq.core.embeddings.embedder.embed_graph", fake_embed)
        lazy_worker.run_lazy_embedding_worker(repo)

        state = lazy_worker.read_state(data_dir)
        assert state["state"] == "complete"
        assert state["total"] == 2, "progress total must be the pending delta, not full count"
        assert state["done"] == 2
        # meta reflects every stored vector (reused + newly encoded), not just pending.
        assert _read_meta(data_dir)["stats"]["embeddings"] == 27

    def test_complete_zero_when_nothing_embeddable(self, indexed_repo, monkeypatch) -> None:
        from synaptiq.core.graph.graph import KnowledgeGraph

        repo, data_dir, _ = indexed_repo
        monkeypatch.setattr(
            lazy_worker, "_load_graph_and_previous", lambda db: (KnowledgeGraph(), {})
        )
        lazy_worker.run_lazy_embedding_worker(repo)
        state = lazy_worker.read_state(data_dir)
        assert state["state"] == "complete"
        assert state["total"] == 0
        assert _read_meta(data_dir)["stats"]["embeddings"] == 0


# ---------------------------------------------------------------------------
# Embedding tier resolution (W4.4) — the worker is a detached subprocess
# with no CLI arg, so it must always re-derive the tier from meta.json.
# ---------------------------------------------------------------------------


class _FakeStaticModel:
    """Deterministic stand-in for model2vec's ``StaticModel.encode()`` shape
    (a single batched call returning a 2D array, unlike fastembed's
    streaming-generator ``.embed()``) — see embedder._encode_batches."""

    def encode(self, texts, batch_size: int = 64, use_multiprocessing: bool = True):
        return np.array(
            [
                np.random.default_rng(int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "big"))
                .random(256)
                .astype(np.float32)
                for t in texts
            ]
        )


class TestWorkerTierResolution:
    def test_worker_passes_meta_tier_to_embed_graph(self, indexed_repo, monkeypatch) -> None:
        """indexed_repo's meta.json was written by a plain run_pipeline call
        (no --embedding-model), so it defaults to "quality"."""
        repo, data_dir, _ = indexed_repo
        seen_tiers: list = []

        def spy_embed(graph, tier=None, previous=None, progress_callback=None):
            seen_tiers.append(tier)
            return []

        monkeypatch.setattr("synaptiq.core.embeddings.embedder.embed_graph", spy_embed)
        lazy_worker.run_lazy_embedding_worker(repo)

        assert seen_tiers == ["quality"]

    def test_worker_honors_fast_tier_recorded_in_meta(self, indexed_repo, monkeypatch) -> None:
        repo, data_dir, _ = indexed_repo
        meta = _read_meta(data_dir)
        meta["stats"]["embedding_model"] = "fast"
        (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        seen_tiers: list = []

        def spy_embed(graph, tier=None, previous=None, progress_callback=None):
            seen_tiers.append(tier)
            return []

        monkeypatch.setattr("synaptiq.core.embeddings.embedder.embed_graph", spy_embed)
        lazy_worker.run_lazy_embedding_worker(repo)

        assert seen_tiers == ["fast"]

    def test_worker_encodes_and_stores_256dim_vectors_for_fast_tier(
        self, indexed_repo, monkeypatch
    ) -> None:
        """Full loop through the REAL embed_graph/tier dispatch (not spied):
        a repo whose meta.json says "fast" ends up with real 256-dim vectors
        in storage, proving the tier name round-trips correctly end to end
        (meta.json -> tier_from_meta -> embed_graph -> _encode_batches ->
        store_embeddings' actual-width table)."""
        repo, data_dir, db_path = indexed_repo
        meta = _read_meta(data_dir)
        meta["stats"]["embedding_model"] = "fast"
        (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        # Override the module's autouse fastembed-shaped fake (_fake_model,
        # above) with a model2vec-shaped one for this test only —
        # _encode_batches dispatches on tier.backend ("model2vec" here, from
        # meta.json), not by introspecting the model instance, so this fake
        # only needs to match .encode()'s shape for the real dispatch path
        # to exercise it.
        monkeypatch.setattr(
            "synaptiq.core.embeddings.embedder._get_model", lambda *a, **k: _FakeStaticModel()
        )

        rc = lazy_worker.run_lazy_embedding_worker(repo)
        assert rc == 0

        state = lazy_worker.read_state(data_dir)
        assert state["state"] == "complete"

        storage = LadybugBackend()
        storage.initialize(db_path, read_only=True)
        try:
            stored = storage.load_embeddings()
            assert stored, "worker should have stored at least one vector"
            sample_vec = next(iter(stored.values()))[1]
            assert len(sample_vec) == 256
        finally:
            storage.close()


# ---------------------------------------------------------------------------
# Staleness guard
# ---------------------------------------------------------------------------


class TestStalenessGuard:
    def test_reencodes_when_index_changes_mid_encode(self, indexed_repo, monkeypatch) -> None:
        """A concurrent re-index bumps meta; the worker re-encodes the new graph."""
        from synaptiq.core.embeddings.embedder import embed_graph as real_embed_graph

        repo, data_dir, _ = indexed_repo
        calls = {"n": 0}

        def flaky_embed(graph, tier=None, previous=None, progress_callback=None):
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate another `analyze` committing a fresh graph mid-encode.
                meta = _read_meta(data_dir)
                meta["last_indexed_at"] = "2099-01-01T00:00:00+00:00"
                (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            return real_embed_graph(
                graph, tier=tier, previous=previous, progress_callback=progress_callback
            )

        monkeypatch.setattr("synaptiq.core.embeddings.embedder.embed_graph", flaky_embed)
        rc = lazy_worker.run_lazy_embedding_worker(repo)
        assert rc == 0
        assert calls["n"] == 2  # first result discarded as stale, re-encoded
        assert lazy_worker.read_state(data_dir)["state"] == "complete"

    def test_defers_when_index_keeps_changing(self, indexed_repo, monkeypatch) -> None:
        repo, data_dir, _ = indexed_repo
        monkeypatch.setattr(lazy_worker, "_MAX_GENERATIONS", 2)
        counter = {"n": 0}

        def always_stale(graph, tier=None, previous=None, progress_callback=None):
            counter["n"] += 1
            meta = _read_meta(data_dir)
            meta["last_indexed_at"] = f"2099-02-0{counter['n']}T00:00:00+00:00"
            (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            return []

        monkeypatch.setattr("synaptiq.core.embeddings.embedder.embed_graph", always_stale)
        rc = lazy_worker.run_lazy_embedding_worker(repo)
        assert rc == 0
        assert counter["n"] == 2  # exhausted the generation budget
        assert lazy_worker.read_state(data_dir)["state"] == "deferred"


# ---------------------------------------------------------------------------
# Failure / deferral paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_failed_state_on_encode_error(self, indexed_repo, monkeypatch) -> None:
        repo, data_dir, _ = indexed_repo

        def boom(*a, **k):
            raise RuntimeError("onnx exploded")

        monkeypatch.setattr("synaptiq.core.embeddings.embedder.embed_graph", boom)
        rc = lazy_worker.run_lazy_embedding_worker(repo)
        assert rc == 0
        state = lazy_worker.read_state(data_dir)
        assert state["state"] == "failed"
        assert "onnx exploded" in state["error"]

    def test_deferred_when_store_fails(self, indexed_repo, monkeypatch) -> None:
        repo, data_dir, _ = indexed_repo
        monkeypatch.setattr(lazy_worker, "_store_with_retry", lambda *a, **k: False)
        rc = lazy_worker.run_lazy_embedding_worker(repo)
        assert rc == 0
        assert lazy_worker.read_state(data_dir)["state"] == "deferred"
        # Index remains healthy: no vectors stored, count untouched.
        assert _read_meta(data_dir)["stats"]["embeddings"] == 0

    def test_deferred_when_index_unreadable(self, indexed_repo, monkeypatch) -> None:
        repo, data_dir, _ = indexed_repo
        monkeypatch.setattr(lazy_worker, "_load_graph_and_previous", lambda db: None)
        rc = lazy_worker.run_lazy_embedding_worker(repo)
        assert rc == 0
        assert lazy_worker.read_state(data_dir)["state"] == "deferred"

    def test_noop_without_synaptiq_dir(self, tmp_path: Path) -> None:
        assert lazy_worker.run_lazy_embedding_worker(tmp_path / "absent") == 0


# ---------------------------------------------------------------------------
# Store-with-retry (lock handling)
# ---------------------------------------------------------------------------


class TestStoreWithRetry:
    def test_gives_up_after_persistent_lock(self, indexed_repo, monkeypatch) -> None:
        repo, data_dir, db_path = indexed_repo
        monkeypatch.setattr(lazy_worker, "_STORE_RETRY_BACKOFF", (0.0, 0.0))
        attempts = {"n": 0}

        def locked_init(self, path, *, read_only=False, _build_fts_indexes=True):
            attempts["n"] += 1
            raise RuntimeError("Could not set lock on file: busy")

        monkeypatch.setattr(LadybugBackend, "initialize", locked_init)
        emb = [NodeEmbedding(node_id="x", embedding=[0.1] * 4, text_sha="s")]
        assert lazy_worker._store_with_retry(db_path, emb) is False
        assert attempts["n"] == 3  # initial + two backoff retries

    def test_succeeds_after_transient_lock(self, indexed_repo, monkeypatch) -> None:
        repo, data_dir, db_path = indexed_repo
        monkeypatch.setattr(lazy_worker, "_STORE_RETRY_BACKOFF", (0.0, 0.0))
        real_init = LadybugBackend.initialize
        state = {"n": 0}

        def flaky_init(self, path, *, read_only=False, _build_fts_indexes=True):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("Lock is held by PID 123")
            return real_init(self, path, read_only=read_only, _build_fts_indexes=_build_fts_indexes)

        monkeypatch.setattr(LadybugBackend, "initialize", flaky_init)
        emb = [
            NodeEmbedding(node_id="function:src/main.py:main", embedding=[0.1] * 384, text_sha="s")
        ]
        assert lazy_worker._store_with_retry(db_path, emb) is True
        assert state["n"] == 2


# ---------------------------------------------------------------------------
# meta.json embedding-count update
# ---------------------------------------------------------------------------


class TestMetaUpdate:
    def test_updates_when_anchor_matches(self, indexed_repo) -> None:
        _, data_dir, _ = indexed_repo
        anchor = _read_meta(data_dir)["last_indexed_at"]
        lazy_worker._update_meta_embeddings(data_dir, 7, expect_anchor=anchor)
        assert _read_meta(data_dir)["stats"]["embeddings"] == 7

    def test_skips_when_anchor_changed(self, indexed_repo) -> None:
        _, data_dir, _ = indexed_repo
        anchor = _read_meta(data_dir)["last_indexed_at"]
        lazy_worker._update_meta_embeddings(data_dir, 7, expect_anchor=anchor)
        # A newer analyze owns the count now; stale update must be skipped.
        lazy_worker._update_meta_embeddings(data_dir, 999, expect_anchor="stale-anchor")
        assert _read_meta(data_dir)["stats"]["embeddings"] == 7


# ---------------------------------------------------------------------------
# Spawn helper
# ---------------------------------------------------------------------------


class TestSpawn:
    def test_returns_pid_and_detaches(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / ".synaptiq").mkdir()
        captured: dict = {}

        class _FakeProc:
            pid = 4242

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.delenv("SYNAPTIQ_EMBED_THREADS", raising=False)
        monkeypatch.setattr(lazy_worker.subprocess, "Popen", fake_popen)

        pid = lazy_worker.spawn_lazy_worker(tmp_path)
        assert pid == 4242
        assert captured["cmd"][:4] == [sys.executable, "-m", "synaptiq", "_embed-worker"]
        assert captured["cmd"][4] == str(tmp_path)
        assert captured["kwargs"]["start_new_session"] is True
        # Background politeness: a thread cap is injected when unset.
        assert "SYNAPTIQ_EMBED_THREADS" in captured["kwargs"]["env"]

    def test_respects_existing_thread_override(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / ".synaptiq").mkdir()
        captured: dict = {}

        class _FakeProc:
            pid = 1

        monkeypatch.setenv("SYNAPTIQ_EMBED_THREADS", "1")
        monkeypatch.setattr(
            lazy_worker.subprocess,
            "Popen",
            lambda cmd, **kw: captured.update(env=kw.get("env")) or _FakeProc(),
        )
        lazy_worker.spawn_lazy_worker(tmp_path)
        assert captured["env"]["SYNAPTIQ_EMBED_THREADS"] == "1"

    def test_returns_none_when_spawn_fails(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / ".synaptiq").mkdir()

        def boom(*a, **k):
            raise OSError("no exec")

        monkeypatch.setattr(lazy_worker.subprocess, "Popen", boom)
        assert lazy_worker.spawn_lazy_worker(tmp_path) is None


# ---------------------------------------------------------------------------
# pid_alive (2.0.4, BUG 1 / BUG 3b)
# ---------------------------------------------------------------------------


class TestPidAlive:
    def test_current_process_is_alive(self) -> None:
        assert lazy_worker.pid_alive(os.getpid()) is True

    def test_dead_pid_is_not_alive(self) -> None:
        assert lazy_worker.pid_alive(99999999) is False  # very unlikely to be a real pid

    def test_none_is_not_alive(self) -> None:
        assert lazy_worker.pid_alive(None) is False

    def test_non_int_is_not_alive(self) -> None:
        assert lazy_worker.pid_alive("not-a-pid") is False

    def test_zero_or_negative_is_not_alive(self) -> None:
        assert lazy_worker.pid_alive(0) is False
        assert lazy_worker.pid_alive(-5) is False


