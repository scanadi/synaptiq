"""Async readers-writer lock for coordinating concurrent access.

Implements a write-preferring RWLock: when a writer is waiting, new readers
block until the writer completes. This prevents writer starvation under
heavy read load (e.g. many agents querying while the watcher needs to write).

Usage::

    rwlock = AsyncRWLock()

    # Multiple readers can hold the lock concurrently
    async with rwlock.reader():
        data = storage.get_node(node_id)

    # Only one writer at a time; blocks all readers
    async with rwlock.writer():
        storage.bulk_load(graph)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

# Default timeout for acquiring locks (seconds).
DEFAULT_TIMEOUT = 60.0


class AsyncRWLock:
    """Async readers-writer lock with write-preference.

    - Multiple readers can hold the lock concurrently.
    - Only one writer at a time; a writer waits for all readers to release.
    - Pending writers block new readers (write-preferring) to prevent
      writer starvation.
    """

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._readers: int = 0
        self._writer: bool = False
        self._pending_writers: int = 0

    @asynccontextmanager
    async def reader(self, timeout: float = DEFAULT_TIMEOUT) -> AsyncIterator[None]:
        """Acquire a read lock. Multiple readers may hold this concurrently.

        Raises ``asyncio.TimeoutError`` if the lock cannot be acquired
        within *timeout* seconds.
        """
        await asyncio.wait_for(self._acquire_read(), timeout=timeout)
        try:
            yield
        finally:
            await self._release_read()

    @asynccontextmanager
    async def writer(self, timeout: float = DEFAULT_TIMEOUT) -> AsyncIterator[None]:
        """Acquire an exclusive write lock.

        Raises ``asyncio.TimeoutError`` if the lock cannot be acquired
        within *timeout* seconds.
        """
        await asyncio.wait_for(self._acquire_write(), timeout=timeout)
        try:
            yield
        finally:
            await self._release_write()

    # ------------------------------------------------------------------
    # Internal acquisition / release
    # ------------------------------------------------------------------

    async def _acquire_read(self) -> None:
        async with self._cond:
            # Wait while a writer holds the lock or writers are pending
            while self._writer or self._pending_writers > 0:
                await self._cond.wait()
            self._readers += 1

    async def _release_read(self) -> None:
        async with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    async def _acquire_write(self) -> None:
        async with self._cond:
            self._pending_writers += 1
            try:
                while self._writer or self._readers > 0:
                    await self._cond.wait()
            except BaseException:
                self._pending_writers -= 1
                self._cond.notify_all()
                raise
            self._pending_writers -= 1
            self._writer = True

    async def _release_write(self) -> None:
        async with self._cond:
            self._writer = False
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Introspection (for monitoring / debugging)
    # ------------------------------------------------------------------

    @property
    def readers(self) -> int:
        """Number of active readers."""
        return self._readers

    @property
    def writing(self) -> bool:
        """Whether a writer currently holds the lock."""
        return self._writer

    @property
    def pending_writers(self) -> int:
        """Number of writers waiting to acquire the lock."""
        return self._pending_writers
