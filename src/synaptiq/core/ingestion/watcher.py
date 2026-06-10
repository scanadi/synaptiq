"""Watch mode for Synaptiq — re-indexes on file changes.

Uses ``watchfiles`` (Rust-backed) for efficient file system monitoring with
native debouncing.  Changes are processed in tiers:

- **File-local** (immediate): Parse without lock, then write under write lock.
- **Global** (timer, 30s after a change): Build the full graph and re-embed
  symbols without the lock, then ``bulk_load`` + ``store_embeddings`` under
  the write lock.  The timer runs independently of further file events, so a
  single edit followed by silence still triggers the global rebuild.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from watchfiles import DefaultFilter

from synaptiq.config.ignore import load_gitignore, should_ignore
from synaptiq.config.languages import is_supported
from synaptiq.core.daemon.rwlock import AsyncRWLock
from synaptiq.core.ingestion.walker import FileEntry, read_file
from synaptiq.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Seconds after a change before the global analysis phases run.
GLOBAL_PHASE_INTERVAL = 30


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
    """
    import watchfiles

    from synaptiq.core.ingestion.pipeline import (
        apply_reindex,
        build_full_index,
        commit_full_index,
        parse_files,
        write_meta,
    )

    gitignore = load_gitignore(repo_path)
    dirty_event = asyncio.Event()
    files_changed = 0

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    async def _global_phase_loop() -> None:
        """Run global analyses on a timer after changes — independent of
        further file events, so staleness is bounded even when editing stops."""
        while True:
            await dirty_event.wait()
            await asyncio.sleep(GLOBAL_PHASE_INTERVAL)
            if _stopped():
                return
            dirty_event.clear()
            try:
                logger.info("Running global analysis phases...")
                full_graph, embeddings, result = await asyncio.to_thread(
                    build_full_index, repo_path
                )
                if _stopped():
                    return
                await _run_under_write_lock(
                    rwlock, commit_full_index, storage, full_graph, embeddings
                )
                write_meta(repo_path / ".synaptiq", repo_path, result)
                logger.info("Global phases completed")
            except Exception:
                # Retry on the next change rather than self-rescheduling —
                # a persistent failure must not rebuild the repo in a loop.
                logger.exception(
                    "Global analysis phase failed; will retry after the next change"
                )

    global_task = asyncio.create_task(_global_phase_loop())

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
                    dirty_event.set()

                # Step 3: Parse files WITHOUT lock (CPU-intensive, no DB access).
                if entries:
                    graph = await asyncio.to_thread(parse_files, entries, repo_path)

                    # Step 4: Apply to storage UNDER write lock (I/O only).
                    await _run_under_write_lock(
                        rwlock, apply_reindex, entries, storage, graph
                    )

                    files_changed += len(entries)
                    dirty_event.set()
                    logger.info("Reindexed %d file(s)", len(entries))
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
