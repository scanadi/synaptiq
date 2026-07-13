"""Detached background embedding worker for ``analyze --embeddings lazy``.

The CLI ``analyze`` command commits the graph first (``bulk_load`` with
embeddings skipped) so the index is queryable in seconds, then spawns this
worker as a *detached* process to encode vectors behind the user's back.
Production data (lvlp-app: pipeline+storage 11s, ONNX encode 620s) shows the
encode is 98% of a cold index — moving it off the critical path is the whole
point of W4.1.

Lifecycle (one worker process)::

    analyze  ──bulk_load, write_meta, close DB──▶ spawn_lazy_worker() ──▶ exits
                                                        │ (python -m synaptiq
                                                        │  _embed-worker <repo>)
                                                        ▼
    worker: flock(embed_worker.lock, non-blocking)
              │ held? → exit 0 (another worker is on it)
              ▼
            anchor = meta.last_indexed_at                       (staleness anchor)
            tier   = meta.stats.embedding_model  (W4.4 — "quality"/"fast"; never a CLI arg)
              ▼
      ┌── open DB read-only → load_graph + snapshot vectors → CLOSE
      │     │ open fails (daemon holds it / corrupt) → abort quietly, never wipe
      │     ▼
      │   encode (tier's model, polite thread cap) → embeddings_state.json per batch
      │     ▼
      │   meta.last_indexed_at changed since anchor?
      └────── yes → re-encode the newer graph (bounded) ; no ↓
              ▼
            open DB read-write (retry on lock → state="deferred") → store_embeddings
              ▼
            update meta.stats.embeddings ; state="complete" ; release lock ; exit

The single-writer store respects a live ``serve``/``watch`` daemon: it never
fights the lock — it retries a few times, then defers to the next rebuild.

Daemons (``serve``/``watch``) do NOT use this worker: their global rebuilds
keep embeddings **synchronous** (warm + cheap via ``text_sha`` reuse; two
background encoders colliding is worse). Lazy is a CLI-``analyze`` concept.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Files the worker owns inside ``.synaptiq/``.
STATE_FILENAME = "embeddings_state.json"
WORKER_LOCK_FILENAME = "embed_worker.lock"
WORKER_LOG_FILENAME = "embed_worker.log"

# On-disk index path — kept as ``kuzu`` for back-compat with indexes written by
# the former KuzuDB backend (see ``open_with_recovery``); ``analyze`` uses the
# same name.
_DB_DIRNAME = "kuzu"

# How many times to re-encode when a newer ``analyze`` commits a fresh graph
# mid-encode before giving up and deferring (pathological continuous churn).
_MAX_GENERATIONS = 5

# Read-write open retry schedule (seconds) when the DB is locked by a daemon or
# a racing ``analyze``.  Give up after the last one with state="deferred".
_STORE_RETRY_BACKOFF = (2.0, 5.0)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write *payload* as JSON via temp-file + ``os.replace`` (atomic).

    Readers (``synaptiq status``, tests polling the state file) therefore
    never observe a half-written file.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_state(
    data_dir: Path,
    state: str,
    *,
    done: int = 0,
    total: int = 0,
    started_at: str | None = None,
    error: str | None = None,
    detail: str | None = None,
    pid: int | None = None,
) -> None:
    """Publish the worker's progress to ``embeddings_state.json`` (atomic).

    ``state`` is one of ``encoding`` / ``complete`` / ``deferred`` / ``failed``.
    """
    payload: dict = {
        "state": state,
        "done": done,
        "total": total,
        "pid": pid if pid is not None else os.getpid(),
        "updated_at": _now_iso(),
    }
    if started_at is not None:
        payload["started_at"] = started_at
    if error is not None:
        payload["error"] = error
    if detail is not None:
        payload["detail"] = detail
    try:
        _atomic_write_json(data_dir / STATE_FILENAME, payload)
    except OSError:
        logger.debug("Could not write %s", STATE_FILENAME, exc_info=True)


def read_state(data_dir: Path) -> dict | None:
    """Return the parsed ``embeddings_state.json`` or ``None`` if absent/bad."""
    try:
        return json.loads((data_dir / STATE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_meta_timestamp(data_dir: Path) -> str | None:
    """Return ``meta.json``'s ``last_indexed_at`` — the staleness anchor."""
    try:
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return meta.get("last_indexed_at")


def _update_meta_embeddings(data_dir: Path, count: int, *, expect_anchor: str | None) -> None:
    """Set ``meta.stats.embeddings`` = *count*, preserving everything else.

    Skips the update if ``last_indexed_at`` no longer matches *expect_anchor*
    — a newer ``analyze`` has rewritten meta and owns the count now, so we must
    not stamp a stale number over it.
    """
    meta_path = data_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if expect_anchor is not None and meta.get("last_indexed_at") != expect_anchor:
        return
    meta.setdefault("stats", {})["embeddings"] = count
    try:
        _atomic_write_json(meta_path, meta)
    except OSError:
        logger.debug("Could not update meta.json embedding count", exc_info=True)


# ---------------------------------------------------------------------------
# Spawn side (called by the CLI ``analyze`` command)
# ---------------------------------------------------------------------------


def _polite_embed_threads() -> int:
    """A background-friendly ONNX thread cap: ``max(2, cores // 2)``.

    Politeness matters MORE for a background process than for a foreground
    ``analyze`` (the interactive default is ``max(2, cores - 2)``), so the
    worker leaves half the machine for the user's real work.
    """
    cores = os.cpu_count() or 4
    return max(2, cores // 2)


def spawn_lazy_worker(repo_path: Path) -> int | None:
    """Spawn the detached embedding worker for *repo_path*; return its PID.

    The child is fully detached (``start_new_session=True``) so it outlives the
    parent CLI and a closed terminal, with stdout/stderr redirected to
    ``.synaptiq/embed_worker.log``.  Returns ``None`` if the spawn itself fails
    (the index is still fully usable — only vectors are missing).
    """
    data_dir = repo_path / ".synaptiq"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / WORKER_LOG_FILENAME

    env = os.environ.copy()
    # Respect an explicit user override; otherwise stay polite in the background.
    env.setdefault("SYNAPTIQ_EMBED_THREADS", str(_polite_embed_threads()))

    cmd = [sys.executable, "-m", "synaptiq", "_embed-worker", str(repo_path)]
    try:
        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    except OSError:
        logger.warning("Could not open embed worker log at %s", log_path, exc_info=True)
        return None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
            cwd=str(repo_path),
        )
    except OSError:
        logger.warning("Could not spawn embed worker", exc_info=True)
        return None
    finally:
        # The child holds its own dup of the fd; the parent can release it.
        log_file.close()
    return proc.pid


# ---------------------------------------------------------------------------
# Worker side (runs in the detached ``_embed-worker`` process)
# ---------------------------------------------------------------------------


def _acquire_single_instance(data_dir: Path) -> int | None:
    """Non-blocking ``flock`` on ``embed_worker.lock``; ``None`` if held.

    Returns the open fd on success (kept open for the worker's lifetime so the
    OS releases the lock automatically when the process exits/crashes).
    """
    lock_path = data_dir / WORKER_LOCK_FILENAME
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _load_graph_and_previous(db_path: Path):
    """Open the DB read-only, load the graph + prior vectors, then close.

    Returns ``(graph, previous)`` or ``None`` on ANY failure (a daemon holding
    the DB read-write, a corrupt/missing index, ...).  Uses a plain backend —
    never ``open_with_recovery`` — so the worker can never wipe the index.
    """
    from synaptiq.core.ingestion.pipeline import load_previous_embeddings
    from synaptiq.core.storage.ladybug_backend import LadybugBackend

    storage = LadybugBackend()
    try:
        storage.initialize(db_path, read_only=True)
    except Exception:
        storage.close()
        logger.info("Embed worker: could not open index read-only; aborting", exc_info=True)
        return None
    try:
        graph = storage.load_graph()
        previous = load_previous_embeddings(storage)
        return graph, previous
    except Exception:
        logger.info("Embed worker: failed to load graph; aborting", exc_info=True)
        return None
    finally:
        storage.close()


def _store_with_retry(db_path: Path, embeddings: list) -> bool:
    """Open the DB read-write and store *embeddings*, retrying on lock.

    A live ``serve``/``watch`` daemon (or a racing ``analyze``) may hold the
    single-writer lock.  We do NOT fight it: retry a few times with backoff,
    then give up so the caller can mark the run ``deferred`` (the next
    ``analyze``/daemon rebuild encodes).  Uses a plain backend — never
    ``open_with_recovery`` — so a lock error can never trigger a wipe.
    """
    from synaptiq.core.storage.ladybug_backend import LadybugBackend, is_lock_error

    attempt = 0
    while True:
        storage = LadybugBackend()
        try:
            storage.initialize(db_path, read_only=False)
        except Exception as exc:
            storage.close()
            if is_lock_error(exc) and attempt < len(_STORE_RETRY_BACKOFF):
                wait = _STORE_RETRY_BACKOFF[attempt]
                logger.info("Embed worker: index locked, retrying store in %.0fs", wait)
                time.sleep(wait)
                attempt += 1
                continue
            logger.warning("Embed worker: could not open index to store vectors", exc_info=True)
            return False
        try:
            storage.store_embeddings(embeddings)
            return True
        except Exception:
            logger.warning("Embed worker: store_embeddings failed", exc_info=True)
            return False
        finally:
            storage.close()


def run_lazy_embedding_worker(repo_path: Path) -> int:
    """Encode embeddings for the already-committed index at *repo_path*.

    Entry point for the hidden ``synaptiq _embed-worker`` command.  Always
    returns ``0`` — a background worker failing must never look like a broken
    ``analyze`` (the index is fully usable without fresh vectors); failures are
    surfaced through ``embeddings_state.json`` instead.
    """
    repo_path = Path(repo_path).resolve()
    data_dir = repo_path / ".synaptiq"
    db_path = data_dir / _DB_DIRNAME

    if not data_dir.is_dir():
        logger.info("Embed worker: no .synaptiq at %s; nothing to do", repo_path)
        return 0

    lock_fd = _acquire_single_instance(data_dir)
    if lock_fd is None:
        logger.info("Embed worker: another worker holds the lock; exiting")
        return 0

    started_at = _now_iso()
    try:
        _encode_and_store(data_dir, db_path, started_at)
    except Exception as exc:  # never let a crash escape as a nonzero worker exit
        logger.warning("Embed worker: unexpected failure", exc_info=True)
        write_state(data_dir, "failed", started_at=started_at, error=str(exc))
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)
    return 0


def _encode_and_store(data_dir: Path, db_path: Path, started_at: str) -> None:
    """The staleness-guarded encode → store loop (holds the single-instance lock)."""
    from synaptiq.core.embeddings.embedder import (
        embed_graph,
        embeddable_node_count,
        partition_embeddings,
        tier_from_meta,
    )

    for generation in range(_MAX_GENERATIONS):
        anchor = _read_meta_timestamp(data_dir)  # (b) staleness anchor
        # W4.4: re-derive the embedding tier from meta.json every generation
        # rather than taking it as a CLI arg — this is a detached subprocess
        # with no CLI args to take it from, and `analyze` already stamps the
        # tier into meta.json (stats.embedding_model) before spawning this
        # worker. Reading it alongside the staleness anchor means a racing
        # `analyze --embedding-model` that changes tier mid-encode is picked
        # up on the worker's next generation too, not just the timestamp.
        tier = tier_from_meta(data_dir)

        loaded = _load_graph_and_previous(db_path)  # (c) read-only load, then close
        if loaded is None:
            # Could not read the index (daemon owns it / corrupt). Abort
            # quietly — the daemon's own rebuild, or the next analyze, encodes.
            write_state(
                data_dir,
                "deferred",
                started_at=started_at,
                detail="index unavailable for read; a later rebuild will encode",
            )
            return
        graph, previous = loaded

        if embeddable_node_count(graph) == 0:
            _update_meta_embeddings(data_dir, 0, expect_anchor=anchor)
            write_state(data_dir, "complete", done=0, total=0, started_at=started_at)
            return

        # Only the PENDING delta is actually encoded — embed_graph reuses every
        # unchanged vector and runs the model on just the new/changed symbols. So
        # report progress against the pending count, not the full embeddable-node
        # count: a one-file change must read "encoding 3/12", never
        # "encoding 0/26,909". partition_embeddings runs the same split
        # embed_graph does but never loads the model (generate-text cost), so this
        # is cheap next to the encode it precedes.
        reused_vecs, pending = partition_embeddings(graph, previous, tier=tier.name)
        reused = len(reused_vecs)
        if pending == 0:
            # Nothing new to encode (everything reused). Still store the reused
            # set so meta reflects the true vector count, but there is no work to
            # show progress for.
            embeddings = reused_vecs
            if _read_meta_timestamp(data_dir) != anchor:
                continue
            if _store_with_retry(db_path, embeddings):
                _update_meta_embeddings(data_dir, len(embeddings), expect_anchor=anchor)
                write_state(data_dir, "complete", done=0, total=0, started_at=started_at)
            else:
                write_state(
                    data_dir,
                    "deferred",
                    done=0,
                    total=0,
                    started_at=started_at,
                    detail="index locked; re-run `synaptiq analyze` to encode",
                )
            return

        write_state(data_dir, "encoding", done=0, total=pending, started_at=started_at)

        def _on_progress(done: int, _full: int) -> None:
            # embed_graph reports done = reused + encoded-so-far against the full
            # embeddable count; re-base onto the pending delta for honest status.
            write_state(
                data_dir,
                "encoding",
                done=max(0, min(done - reused, pending)),
                total=pending,
                started_at=started_at,
            )

        embeddings = embed_graph(
            graph, tier=tier.name, previous=previous, progress_callback=_on_progress
        )  # (d)

        # (e) Staleness guard: a newer analyze committed a fresh graph while we
        # encoded. Don't store stale vectors — re-encode the current graph.
        if _read_meta_timestamp(data_dir) != anchor:
            logger.info(
                "Embed worker: index changed during encode (generation %d); re-encoding",
                generation + 1,
            )
            continue

        # (f) Store under the single-writer lock (retry → deferred). Progress is
        # reported against the pending delta (see above), so a completed encode
        # reads "pending/pending".
        if _store_with_retry(db_path, embeddings):
            _update_meta_embeddings(data_dir, len(embeddings), expect_anchor=anchor)
            write_state(data_dir, "complete", done=pending, total=pending, started_at=started_at)
        else:
            write_state(
                data_dir,
                "deferred",
                done=pending,
                total=pending,
                started_at=started_at,
                detail="index locked; re-run `synaptiq analyze` to encode",
            )
        return

    # Ran out of generations — the index kept changing under us.
    logger.info(
        "Embed worker: index kept changing across %d generations; deferring", _MAX_GENERATIONS
    )
    write_state(
        data_dir,
        "deferred",
        started_at=started_at,
        detail="index changed repeatedly during encoding; re-run `synaptiq analyze`",
    )
