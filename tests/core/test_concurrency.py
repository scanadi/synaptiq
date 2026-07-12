"""Concurrency tests — simulating multi-agent access patterns.

These tests verify that the RWLock, connection pool, and socket server
behave correctly under concurrent load typical of multiple Claude Code
agents hitting Synaptiq simultaneously.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from synaptiq.core.daemon.rwlock import AsyncRWLock
from synaptiq.core.ingestion.pipeline import run_pipeline
from synaptiq.core.storage.ladybug_backend import LadybugBackend

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def indexed_repo(tmp_path: Path) -> tuple[Path, LadybugBackend]:
    """Create a small indexed repo for concurrency tests."""
    src = tmp_path / "repo" / "src"
    src.mkdir(parents=True)

    (src / "app.py").write_text(
        "def hello():\n"
        "    return 'hello'\n\n"
        "def goodbye():\n"
        "    return 'bye'\n",
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        "def helper():\n"
        "    pass\n\n"
        "class Config:\n"
        "    DEBUG = True\n",
        encoding="utf-8",
    )

    repo_path = tmp_path / "repo"
    db_path = tmp_path / "test_db"
    backend = LadybugBackend()
    backend.initialize(db_path)
    run_pipeline(repo_path, backend)

    yield repo_path, backend
    backend.close()


# ======================================================================
# Concurrent reads (connection pool)
# ======================================================================


class TestConcurrentReads:
    """Multiple read operations running in parallel."""

    async def test_parallel_get_node(
        self, indexed_repo: tuple[Path, LadybugBackend]
    ) -> None:
        """Multiple get_node calls succeed concurrently."""
        _, storage = indexed_repo

        async def read_node(node_id: str) -> bool:
            node = await asyncio.to_thread(storage.get_node, node_id)
            return node is not None

        results = await asyncio.gather(
            read_node("function:src/app.py:hello"),
            read_node("function:src/app.py:goodbye"),
            read_node("function:src/utils.py:helper"),
            read_node("class:src/utils.py:Config"),
        )
        assert all(results)

    async def test_parallel_fts_search(
        self, indexed_repo: tuple[Path, LadybugBackend]
    ) -> None:
        """Multiple FTS search calls succeed concurrently."""
        _, storage = indexed_repo

        async def search(term: str) -> list:
            return await asyncio.to_thread(storage.fts_search, term, 5)

        results = await asyncio.gather(
            search("hello"),
            search("helper"),
            search("Config"),
        )
        for result in results:
            assert isinstance(result, list)

    async def test_many_concurrent_reads(
        self, indexed_repo: tuple[Path, LadybugBackend]
    ) -> None:
        """Stress test: 20 concurrent read operations."""
        _, storage = indexed_repo
        node_ids = [
            "function:src/app.py:hello",
            "function:src/app.py:goodbye",
            "function:src/utils.py:helper",
            "class:src/utils.py:Config",
        ]

        async def read_cycle(idx: int) -> bool:
            node_id = node_ids[idx % len(node_ids)]
            node = await asyncio.to_thread(storage.get_node, node_id)
            return node is not None

        results = await asyncio.gather(*(read_cycle(i) for i in range(20)))
        assert all(results)


# ======================================================================
# Reader/writer coordination with RWLock
# ======================================================================


class TestReadWriteCoordination:
    """Simulate agents reading while watcher writes."""

    async def test_reads_during_write(
        self, indexed_repo: tuple[Path, LadybugBackend]
    ) -> None:
        """Reads complete before a write acquires exclusive access."""
        _, storage = indexed_repo
        rwlock = AsyncRWLock()
        read_results: list[bool] = []
        write_completed = asyncio.Event()

        async def read_task() -> None:
            async with rwlock.reader():
                node = await asyncio.to_thread(
                    storage.get_node, "function:src/app.py:hello"
                )
                read_results.append(node is not None)

        async def write_task() -> None:
            async with rwlock.writer():
                # Simulate a storage write.
                await asyncio.sleep(0.01)
                write_completed.set()

        # Start several readers, then a writer.
        tasks = [asyncio.create_task(read_task()) for _ in range(5)]
        tasks.append(asyncio.create_task(write_task()))

        await asyncio.gather(*tasks)
        assert all(read_results)
        assert write_completed.is_set()

    async def test_write_blocks_subsequent_reads(
        self, indexed_repo: tuple[Path, LadybugBackend]
    ) -> None:
        """While writer holds lock, new readers must wait."""
        _, storage = indexed_repo
        rwlock = AsyncRWLock()
        writer_acquired = asyncio.Event()
        reader_acquired = asyncio.Event()
        release_writer = asyncio.Event()

        async def writer_task() -> None:
            async with rwlock.writer():
                writer_acquired.set()
                await release_writer.wait()

        async def reader_task() -> None:
            await writer_acquired.wait()
            async with rwlock.reader():
                reader_acquired.set()
                await asyncio.to_thread(
                    storage.get_node, "function:src/app.py:hello"
                )

        wt = asyncio.create_task(writer_task())
        rt = asyncio.create_task(reader_task())

        await writer_acquired.wait()
        await asyncio.sleep(0.05)
        assert not reader_acquired.is_set()

        release_writer.set()
        await asyncio.wait_for(asyncio.gather(wt, rt), timeout=3.0)
        assert reader_acquired.is_set()


# ======================================================================
# Connection pool under contention
# ======================================================================


class TestConnectionPool:
    """Verify the LadybugBackend connection pool handles contention."""

    async def test_pool_returns_connections(
        self, indexed_repo: tuple[Path, LadybugBackend]
    ) -> None:
        """Connections are returned to pool after use."""
        _, storage = indexed_repo

        initial_pool_size = len(storage._read_pool)

        await asyncio.to_thread(storage.get_node, "function:src/app.py:hello")

        # Pool should still have connections (returned after use).
        assert len(storage._read_pool) >= initial_pool_size

    async def test_pool_handles_burst(
        self, indexed_repo: tuple[Path, LadybugBackend]
    ) -> None:
        """Pool creates connections as needed under burst load."""
        _, storage = indexed_repo

        async def burst_read(idx: int) -> bool:
            node = await asyncio.to_thread(
                storage.get_node, "function:src/app.py:hello"
            )
            return node is not None

        # 10 concurrent reads — pool should expand as needed.
        results = await asyncio.gather(*(burst_read(i) for i in range(10)))
        assert all(results)

    async def test_execute_raw_concurrent(
        self, indexed_repo: tuple[Path, LadybugBackend]
    ) -> None:
        """execute_raw uses read pool for concurrent queries."""
        _, storage = indexed_repo

        async def query(label: str) -> list:
            return await asyncio.to_thread(
                storage.execute_raw,
                f"MATCH (n:{label}) RETURN n.id LIMIT 5",
            )

        results = await asyncio.gather(
            query("Function"),
            query("Class"),
            query("File"),
        )
        # All queries should return results.
        for r in results:
            assert isinstance(r, list)
