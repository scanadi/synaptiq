"""Tests for synaptiq.core.daemon.rwlock — AsyncRWLock."""

from __future__ import annotations

import asyncio

import pytest

from synaptiq.core.daemon.rwlock import AsyncRWLock

# ======================================================================
# Basic semantics
# ======================================================================


class TestRWLockBasics:
    """Core reader/writer semantics."""

    async def test_single_reader(self) -> None:
        rw = AsyncRWLock()
        async with rw.reader():
            assert rw.readers == 1
        assert rw.readers == 0

    async def test_single_writer(self) -> None:
        rw = AsyncRWLock()
        async with rw.writer():
            assert rw.writing is True
        assert rw.writing is False

    async def test_concurrent_readers(self) -> None:
        """Multiple readers can hold the lock at the same time."""
        rw = AsyncRWLock()
        entered = asyncio.Event()
        hold = asyncio.Event()

        async def reader_task() -> None:
            async with rw.reader():
                entered.set()
                await hold.wait()

        task = asyncio.create_task(reader_task())
        await entered.wait()

        # Second reader should acquire immediately while first holds.
        async with rw.reader():
            assert rw.readers == 2

        hold.set()
        await task

    async def test_writer_excludes_readers(self) -> None:
        """While writer holds lock, readers must wait."""
        rw = AsyncRWLock()
        writer_acquired = asyncio.Event()
        reader_acquired = asyncio.Event()
        release_writer = asyncio.Event()

        async def writer_task() -> None:
            async with rw.writer():
                writer_acquired.set()
                await release_writer.wait()

        async def reader_task() -> None:
            async with rw.reader():
                reader_acquired.set()

        wt = asyncio.create_task(writer_task())
        await writer_acquired.wait()

        # Reader should NOT acquire while writer holds.
        rt = asyncio.create_task(reader_task())
        await asyncio.sleep(0.05)
        assert not reader_acquired.is_set()

        # Release writer → reader should now proceed.
        release_writer.set()
        await asyncio.wait_for(reader_acquired.wait(), timeout=2.0)
        await wt
        await rt

    async def test_writer_excludes_writers(self) -> None:
        """Only one writer at a time."""
        rw = AsyncRWLock()
        first_acquired = asyncio.Event()
        second_acquired = asyncio.Event()
        release_first = asyncio.Event()

        async def first_writer() -> None:
            async with rw.writer():
                first_acquired.set()
                await release_first.wait()

        async def second_writer() -> None:
            async with rw.writer():
                second_acquired.set()

        t1 = asyncio.create_task(first_writer())
        await first_acquired.wait()

        t2 = asyncio.create_task(second_writer())
        await asyncio.sleep(0.05)
        assert not second_acquired.is_set()

        release_first.set()
        await asyncio.wait_for(second_acquired.wait(), timeout=2.0)
        await t1
        await t2


# ======================================================================
# Write-preference
# ======================================================================


class TestWritePreference:
    """Pending writers block new readers (prevents starvation)."""

    async def test_pending_writer_blocks_new_readers(self) -> None:
        rw = AsyncRWLock()
        reader_holding = asyncio.Event()
        writer_waiting = asyncio.Event()
        late_reader_acquired = asyncio.Event()
        release_reader = asyncio.Event()

        async def initial_reader() -> None:
            async with rw.reader():
                reader_holding.set()
                await release_reader.wait()

        async def blocked_writer() -> None:
            writer_waiting.set()
            async with rw.writer():
                pass  # Just needs to acquire and release.

        async def late_reader() -> None:
            # Wait until writer is queued.
            await writer_waiting.wait()
            await asyncio.sleep(0.05)
            async with rw.reader():
                late_reader_acquired.set()

        t_reader = asyncio.create_task(initial_reader())
        await reader_holding.wait()

        t_writer = asyncio.create_task(blocked_writer())
        t_late = asyncio.create_task(late_reader())

        # Give time for the writer and late reader to queue up.
        await asyncio.sleep(0.1)

        # Late reader should NOT acquire because a writer is pending.
        assert not late_reader_acquired.is_set()
        assert rw.pending_writers == 1

        # Release initial reader → writer runs → then late reader runs.
        release_reader.set()
        await asyncio.wait_for(
            asyncio.gather(t_reader, t_writer, t_late), timeout=3.0
        )
        assert late_reader_acquired.is_set()


# ======================================================================
# Timeouts
# ======================================================================


class TestRWLockTimeout:
    """Timeout behavior."""

    async def test_reader_timeout(self) -> None:
        """Reader times out when writer holds the lock."""
        rw = AsyncRWLock()
        hold = asyncio.Event()

        async def long_writer() -> None:
            async with rw.writer():
                await hold.wait()

        task = asyncio.create_task(long_writer())
        await asyncio.sleep(0.02)

        with pytest.raises(asyncio.TimeoutError):
            async with rw.reader(timeout=0.05):
                pass  # Should not reach here.

        hold.set()
        await task

    async def test_writer_timeout(self) -> None:
        """Writer times out when readers hold the lock."""
        rw = AsyncRWLock()
        hold = asyncio.Event()

        async def long_reader() -> None:
            async with rw.reader():
                await hold.wait()

        task = asyncio.create_task(long_reader())
        await asyncio.sleep(0.02)

        with pytest.raises(asyncio.TimeoutError):
            async with rw.writer(timeout=0.05):
                pass

        hold.set()
        await task

    async def test_timeout_does_not_corrupt_state(self) -> None:
        """After a timeout, the lock is still usable."""
        rw = AsyncRWLock()
        hold = asyncio.Event()

        async def long_writer() -> None:
            async with rw.writer():
                await hold.wait()

        task = asyncio.create_task(long_writer())
        await asyncio.sleep(0.02)

        # This should time out.
        with pytest.raises(asyncio.TimeoutError):
            async with rw.writer(timeout=0.05):
                pass

        # Release the original writer.
        hold.set()
        await task

        # Lock should still work normally.
        async with rw.reader():
            assert rw.readers == 1
        async with rw.writer():
            assert rw.writing is True
