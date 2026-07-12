"""Watch mode for Synaptiq — re-indexes on file changes.

Uses ``watchfiles`` (Rust-backed) for efficient file system monitoring with
native debouncing.  Changes are processed in tiers:

- **File-local** (immediate): Parse without lock, then write under write lock.
  Updates the graph only — full-text (BM25) indexes are not actively
  refreshed here, exactly like embeddings, which this tier has also never
  refreshed per-save (see ``apply_reindex`` in ``pipeline.py`` for the exact
  staleness contract and its bound).
- **Global** (debounced): Build the full graph and re-embed symbols without
  the lock, then ``bulk_load`` (which unconditionally rebuilds every FTS
  index as part of the swap, see ``KuzuBackend.bulk_load``) +
  ``store_embeddings`` under the write lock.  This is the only place FTS is
  rebuilt — there is no separate "FTS dirty" flag; the scheduler's own
  change-tracking below already gates it, since every non-skipped rebuild it
  triggers goes through ``bulk_load``.
  The global phase is governed by a scheduler that:

  * **debounces to quiescence** — a rebuild fires only after
    ``GLOBAL_PHASE_INTERVAL`` seconds with *no new change*; any change during
    the wait resets the timer (so sustained editing no longer rebuilds the
    whole repo every ``30s + build_time``);
  * enforces a **max-staleness ceiling** — if churn never quiesces, a rebuild
    is forced once ``MAX_STALENESS_SECONDS`` have elapsed since the first
    un-built change, so the index can never stay stale forever;
  * **skips clean rebuilds** — if the accumulated changes since the last
    commit would reproduce the already-committed graph (fingerprint match),
    the pipeline is skipped entirely;
  * runs **single-flight** — the actual build+commit goes through a shared
    :class:`RebuildCoordinator`, so it can never run concurrently with a
    socket-delivered ``reindex`` in the same process.

The dirty state is cleared *after* a successful commit, and changes that land
mid-build mark the *next* build rather than being lost.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from watchfiles import DefaultFilter

from synaptiq.config.ignore import load_gitignore, should_ignore
from synaptiq.config.languages import is_supported
from synaptiq.core.daemon.rwlock import AsyncRWLock
from synaptiq.core.ingestion.walker import FileEntry, read_file
from synaptiq.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Seconds of quiescence (no new change) before the global analysis phases run.
GLOBAL_PHASE_INTERVAL = 30

# Hard ceiling on index staleness: even under continuous churn that never
# quiesces, force a global rebuild once this many seconds have elapsed since
# the first un-built change.  Env-overridable for ops tuning and tests.
MAX_STALENESS_SECONDS = float(os.environ.get("SYNAPTIQ_MAX_STALENESS_SECONDS", "600"))


class _SynaptiqFilter(DefaultFilter):
    """watchfiles' DefaultFilter (ignores .git, __pycache__, node_modules,
    editor swap files, ...) extended with Synaptiq's own data directory.

    Subclassing matters: passing a bare callable as ``watch_filter`` would
    REPLACE the default filter, so every ``.git/index.lock`` create/delete
    from routine git activity would reach the reindex path.
    """

    def __call__(self, change: object, path: str) -> bool:
        return (
            super().__call__(change, path)
            and "/.synaptiq/" not in path
            and not path.endswith("/.synaptiq")
        )


def _content_hash(content: str) -> str:
    """Stable content hash used to fingerprint changed files for skip-if-clean."""
    return hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()


def _fingerprint(changed: dict[str, str | None]) -> str:
    """Order-independent fingerprint of the changes accumulated since the last
    successful commit.

    ``changed`` maps a repo-relative path to its content hash, or ``None`` for
    a deletion.  Two repository states that would rebuild to the same graph
    share a fingerprint, so a match means the pipeline can be skipped.
    """
    digest = hashlib.sha256()
    for path in sorted(changed):
        value = changed[path]
        digest.update(path.encode("utf-8", "surrogatepass"))
        digest.update(b"\x00")
        digest.update(b"\x01" if value is None else value.encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


class RebuildCoordinator:
    """Process-wide single-flight guard for full-index rebuilds.

    The watcher's global phase and the socket ``reindex`` handler both run
    their build+commit through :meth:`run`, so two full CPU rebuilds can never
    execute concurrently in one server process (the G10 fix).  The two callers
    build with *different* options (the watcher always embeds; a socket reindex
    may ``skip_embeddings`` or pass ``full=False``), so they are mutually
    excluded rather than folded into one shared build — coalescing them would
    hand a caller the wrong graph.

    Repeated *watcher* triggers coalesce upstream: the global-phase loop runs
    exactly one build at a time, so a burst of edits collapses into a single
    follow-up rebuild instead of a queue.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @property
    def busy(self) -> bool:
        """True while a rebuild currently holds the guard."""
        return self._lock.locked()

    async def run(self, build: Callable[[], Awaitable[T]]) -> T:
        """Run *build* under the single-flight guard, waiting if another
        rebuild is already in progress."""
        async with self._lock:
            return await build()


class _GlobalPhaseScheduler:
    """Debounce + max-staleness + skip-if-clean scheduler for the watcher's
    global rebuild, kept separate from :func:`watch_repo` so its timing logic
    is unit-testable without the filesystem watcher.

    The actual build+commit is injected as ``on_build`` (an async callable
    returning ``True`` once the commit lands, ``False`` if shutdown interrupted
    it) and is always run through the shared :class:`RebuildCoordinator`.
    """

    def __init__(
        self,
        coordinator: RebuildCoordinator,
        on_build: Callable[[], Awaitable[bool]],
        *,
        stopped: Callable[[], bool],
    ) -> None:
        self._coordinator = coordinator
        self._on_build = on_build
        self._stopped = stopped
        # path -> content hash (None = deleted), accumulated since last commit.
        self._changed: dict[str, str | None] = {}
        # Monotonically increasing generation, bumped on every processed batch;
        # lets a commit detect changes that landed while it was building.
        self._change_gen = 0
        self._first_change_at: float | None = None
        self._last_change_at = 0.0
        self._committed_fp: str | None = None
        self._wake = asyncio.Event()

    def notify(self, changed: dict[str, str | None]) -> None:
        """Record a processed change batch and reset the quiescence timer.

        Cheap and non-blocking so the watch loop never stalls.  ``changed``
        maps repo-relative path to content hash (``None`` = deleted).
        """
        if not changed:
            return
        now = asyncio.get_running_loop().time()
        self._changed.update(changed)
        self._change_gen += 1
        if self._first_change_at is None:
            self._first_change_at = now
        self._last_change_at = now
        self._wake.set()

    async def run(self) -> None:
        """Drive global rebuilds forever (until cancelled or stopped)."""
        while not self._stopped():
            # Idle until at least one un-built change is pending.
            while self._first_change_at is None:
                self._wake.clear()
                if self._first_change_at is None:
                    await self._wake.wait()
                if self._stopped():
                    return
            if not await self._await_quiescence():
                return
            await self._maybe_build()

    async def _await_quiescence(self) -> bool:
        """Block until the repo has been quiet for ``GLOBAL_PHASE_INTERVAL`` or
        the max-staleness ceiling is reached.  Returns ``False`` on shutdown.

        ``_last_change_at`` is the source of truth for the debounce; the wake
        event is only an optimisation to avoid over-sleeping, so a missed wake
        self-corrects on the next time comparison.
        """
        loop = asyncio.get_running_loop()
        while True:
            self._wake.clear()
            if self._stopped():
                return False
            if self._first_change_at is None:
                return True
            now = loop.time()
            quiet_for = now - self._last_change_at
            staleness = now - self._first_change_at
            if quiet_for >= GLOBAL_PHASE_INTERVAL or staleness >= MAX_STALENESS_SECONDS:
                return True
            delay = min(
                GLOBAL_PHASE_INTERVAL - quiet_for,
                MAX_STALENESS_SECONDS - staleness,
            )
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=delay)

    async def _maybe_build(self) -> None:
        """Run one rebuild, skipping it if the change set is clean."""
        if not self._changed:
            return
        built_gen = self._change_gen
        fingerprint = _fingerprint(self._changed)
        if fingerprint == self._committed_fp:
            logger.info(
                "Skipping global rebuild — repository already matches the "
                "last committed index"
            )
            self._mark_committed(built_gen, fingerprint)
            return
        try:
            committed = await self._coordinator.run(self._on_build)
        except Exception:
            # A persistent build failure must not spin a rebuild loop: stop
            # self-retriggering but keep the accumulated changes so the next
            # file event re-arms the build (no update is ever dropped).
            logger.exception(
                "Global analysis phase failed; will retry after the next change"
            )
            self._first_change_at = None
            return
        if committed:
            self._mark_committed(built_gen, fingerprint)

    def _mark_committed(self, built_gen: int, fingerprint: str) -> None:
        """Clear the dirty state AFTER a successful commit (or skip).

        Changes that landed mid-build (detected via the generation counter)
        mark the NEXT build instead of being lost.
        """
        self._committed_fp = fingerprint
        if self._change_gen == built_gen:
            # Fully caught up — nothing changed during the build.
            self._changed.clear()
            self._first_change_at = None
        else:
            # Changes arrived mid-build; they belong to a single follow-up
            # build.  Re-anchor the staleness clock to now.
            self._first_change_at = asyncio.get_running_loop().time()


async def _run_under_write_lock(
    rwlock: AsyncRWLock | None, fn: object, *args: object
) -> None:
    """Run *fn* in a thread, optionally under an exclusive write lock."""
    if rwlock is not None:
        async with rwlock.writer():
            await asyncio.to_thread(fn, *args)
    else:
        await asyncio.to_thread(fn, *args)

def _prepare_entries(
    changed_paths: list[Path],
    repo_path: Path,
    gitignore_patterns: list[str] | None = None,
) -> tuple[list[FileEntry], list[str]]:
    """Prepare file entries and deleted paths from changed paths.

    Returns (entries_to_parse, deleted_relative_paths).
    """
    entries: list[FileEntry] = []
    deleted: list[str] = []

    for abs_path in changed_paths:
        try:
            relative = str(abs_path.relative_to(repo_path))
        except (ValueError, OSError):
            continue

        # Deleted paths get the same filtering as live ones — without it,
        # transient files (lock files, build artifacts) would trigger
        # storage deletions and global rebuilds.
        if should_ignore(relative, gitignore_patterns):
            continue
        if not is_supported(abs_path):
            continue

        if not abs_path.is_file():
            # Source file was deleted — record for removal from storage.
            deleted.append(relative)
            continue

        entry = read_file(repo_path, abs_path)
        if entry is not None:
            entries.append(entry)

    return entries, deleted


def _apply_deletions(
    storage: StorageBackend,
    deleted_paths: list[str],
) -> None:
    """Remove nodes for deleted files from storage."""
    for rel_path in deleted_paths:
        try:
            storage.remove_nodes_by_file(rel_path)
        except Exception:
            logger.debug("Failed to remove nodes for %s", rel_path, exc_info=True)


async def watch_repo(
    repo_path: Path,
    storage: StorageBackend,
    *,
    stop_event: asyncio.Event | None = None,
    rwlock: AsyncRWLock | None = None,
    rebuild_coordinator: RebuildCoordinator | None = None,
) -> None:
    """Main watch loop — monitor files and re-index on changes.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to watch.
    storage:
        An already-initialised storage backend.
    stop_event:
        Optional event to signal shutdown (useful for testing).
        When set, the watch loop exits gracefully.
    rwlock:
        Optional RWLock for coordinating storage access with
        concurrent readers (e.g. the MCP server in combined mode).
    rebuild_coordinator:
        Optional shared single-flight guard.  Pass the same instance used by
        the socket ``reindex`` handler so the watcher's global phase and a
        socket-delivered reindex can never run two full builds concurrently.
        Defaults to a private guard when watching standalone.
    """
    import watchfiles

    from synaptiq.core.ingestion.pipeline import (
        apply_reindex,
        build_full_index,
        commit_full_index,
        load_previous_embeddings,
        parse_files,
        write_meta,
    )

    gitignore = load_gitignore(repo_path)
    files_changed = 0
    coordinator = rebuild_coordinator if rebuild_coordinator is not None else RebuildCoordinator()

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    async def _on_build() -> bool:
        """Build the full graph lock-free, then commit under the write lock.

        Returns True once the commit lands, False if shutdown interrupted it.
        Runs inside the shared :class:`RebuildCoordinator`, so it is guaranteed
        single-flight against the socket reindex handler.
        """
        logger.info("Running global analysis phases...")
        # Snapshot current vectors first (plain read, no lock) so only changed
        # symbols are re-encoded.
        previous = await asyncio.to_thread(load_previous_embeddings, storage)
        full_graph, embeddings, result = await asyncio.to_thread(
            build_full_index, repo_path, previous_embeddings=previous
        )
        if _stopped():
            return False
        await _run_under_write_lock(
            rwlock, commit_full_index, storage, full_graph, embeddings
        )
        write_meta(repo_path / ".synaptiq", repo_path, result)
        logger.info("Global phases completed")
        return True

    scheduler = _GlobalPhaseScheduler(coordinator, _on_build, stopped=_stopped)
    global_task = asyncio.create_task(scheduler.run())

    logger.info("Watching %s for changes...", repo_path)

    try:
        async for changes in watchfiles.awatch(
            repo_path,
            rust_timeout=500,
            stop_event=stop_event,
            watch_filter=_SynaptiqFilter(),
        ):
            try:
                changed_paths: list[Path] = []
                seen: set[str] = set()
                for _change_type, path_str in changes:
                    if path_str not in seen:
                        seen.add(path_str)
                        changed_paths.append(Path(path_str))

                if not changed_paths:
                    continue

                # Step 1: Prepare entries and identify deletions (no lock needed).
                entries, deleted = await asyncio.to_thread(
                    _prepare_entries, changed_paths, repo_path, gitignore
                )

                # Step 2: Handle deletions under write lock.
                if deleted:
                    await _run_under_write_lock(rwlock, _apply_deletions, storage, deleted)

                # Step 3: Parse files WITHOUT lock (CPU-intensive, no DB access).
                if entries:
                    graph = await asyncio.to_thread(parse_files, entries, repo_path)

                    # Step 4: Apply to storage UNDER write lock (I/O only).
                    # Graph-only — apply_reindex intentionally leaves FTS stale
                    # (see its docstring); Step 5's notify() below is what
                    # eventually triggers the FTS refresh, via the global
                    # phase's bulk_load.
                    await _run_under_write_lock(
                        rwlock, apply_reindex, entries, storage, graph
                    )

                    files_changed += len(entries)
                    logger.info("Reindexed %d file(s)", len(entries))

                # Step 5: Record the batch for the global phase.  Content
                # hashes drive skip-if-clean; the notify resets the quiescence
                # debounce so the global rebuild waits for editing to settle.
                if entries or deleted:
                    hashed: dict[str, str | None] = {
                        entry.path: _content_hash(entry.content) for entry in entries
                    }
                    for rel_path in deleted:
                        hashed[rel_path] = None
                    scheduler.notify(hashed)
            except Exception:
                # One bad batch must not kill the watcher — a dead watcher
                # silently serves an ever-staler index.
                logger.exception("Failed to process change batch; continuing to watch")

            # Check stop event between batches.
            if _stopped():
                break
    finally:
        global_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await global_task

    logger.info("Watch stopped. Total files reindexed: %d", files_changed)
