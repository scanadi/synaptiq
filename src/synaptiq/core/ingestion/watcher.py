"""Watch mode for Synaptiq — re-indexes on file changes.

Uses ``watchfiles`` (Rust-backed) for efficient file system monitoring with
native debouncing.  Changes are processed in tiers:

- **File-local** (immediate): Parse without lock, then write under write lock.
- **Global** (30s batch): Build full graph without lock, then bulk_load under
  write lock. Only the storage write holds the lock, not the 11-phase pipeline.
- **Embeddings** (60s batch): Re-embed changed symbols.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from synaptiq.config.ignore import load_gitignore, should_ignore
from synaptiq.config.languages import is_supported
from synaptiq.core.daemon.rwlock import AsyncRWLock
from synaptiq.core.ingestion.walker import FileEntry, read_file
from synaptiq.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Timer thresholds (seconds).
GLOBAL_PHASE_INTERVAL = 30
EMBEDDING_INTERVAL = 60


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
        if not abs_path.is_file():
            # File was deleted — record for removal from storage.
            try:
                relative = str(abs_path.relative_to(repo_path))
                deleted.append(relative)
            except (ValueError, OSError):
                pass
            continue

        try:
            relative = str(abs_path.relative_to(repo_path))
        except ValueError:
            continue

        if should_ignore(relative, gitignore_patterns):
            continue

        if not is_supported(abs_path):
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
            pass


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

    from synaptiq.core.ingestion.pipeline import apply_reindex, parse_files, run_pipeline

    gitignore = load_gitignore(repo_path)
    dirty = False
    last_global = time.monotonic()
    files_changed = 0

    logger.info("Watching %s for changes...", repo_path)

    async for changes in watchfiles.awatch(
        repo_path,
        rust_timeout=500,
        stop_event=stop_event,
    ):
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
            await _run_under_write_lock(rwlock, apply_reindex, entries, storage, graph)

            files_changed += len(entries)
            dirty = True
            logger.info("Reindexed %d file(s)", len(entries))

        # Check stop event between batches.
        if stop_event is not None and stop_event.is_set():
            break

        now = time.monotonic()
        if dirty and (now - last_global) >= GLOBAL_PHASE_INTERVAL:
            logger.info("Running global analysis phases...")

            # Build full graph WITHOUT lock (CPU-intensive 11-phase pipeline).
            full_graph, _ = await asyncio.to_thread(
                run_pipeline, repo_path, None, True
            )

            # Check stop event before the dangerous bulk_load write.
            if stop_event is not None and stop_event.is_set():
                logger.info("Shutdown requested, skipping bulk_load")
                break

            # Apply to storage UNDER write lock (only the bulk_load step).
            await _run_under_write_lock(rwlock, storage.bulk_load, full_graph)

            dirty = False
            last_global = time.monotonic()
            logger.info("Global phases completed")

    logger.info("Watch stopped. Total files reindexed: %d", files_changed)
