"""End-to-end tests for multi-instance primary/proxy coordination.

Exercises the full flow: LockManager acquires the lock, SocketServer starts,
SocketClient connects as a proxy, and queries are dispatched correctly.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from synaptiq.core.daemon.lock import LockManager
from synaptiq.core.daemon.socket_client import SocketClient
from synaptiq.core.daemon.socket_server import SocketServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _test_dispatch(method: str, params: dict) -> str:
    """Simple dispatch function used by the socket server in tests."""
    if method == "ping":
        return "pong"
    if method == "tool":
        name = params.get("name", "")
        if name == "synaptiq_list_repos":
            return "No indexed repositories found."
        return f"result:{name}"
    if method == "resource":
        return f"resource:{params.get('uri', '')}"
    return "unknown"


def _make_short_data_dir() -> tuple[str, Path]:
    """Create a temp directory short enough for AF_UNIX sockets on macOS.

    macOS limits AF_UNIX paths to 104 bytes. ``tempfile.mkdtemp()`` produces
    short paths under /tmp which stay well within the limit.

    Returns (tmpdir_str, data_dir Path) so the caller can clean up tmpdir.
    """
    tmpdir = tempfile.mkdtemp()
    data_dir = Path(tmpdir) / ".synaptiq"
    data_dir.mkdir()
    return tmpdir, data_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiInstanceCoordination:
    """Integration tests for primary + proxy coordination through lock + socket."""

    @pytest.mark.asyncio
    async def test_primary_serves_proxy_queries(self) -> None:
        """Full flow: primary acquires lock, starts socket server, proxy connects and queries."""
        tmpdir, data_dir = _make_short_data_dir()
        try:
            # 1. Instance 1 (primary): acquire lock, start socket server
            primary_lock = LockManager(data_dir)
            lock_info = primary_lock.try_acquire()
            assert lock_info is not None, "Primary should acquire the lock"
            assert lock_info.pid == os.getpid()

            socket_path = Path(lock_info.socket)
            server = SocketServer(socket_path, _test_dispatch)
            await server.start()

            try:
                # 2. Instance 2 (proxy): fail to acquire lock, read existing, connect
                proxy_lock = LockManager(data_dir)
                proxy_acquire = proxy_lock.try_acquire()
                assert proxy_acquire is None, "Proxy should fail to acquire the lock"

                existing = proxy_lock.read_existing()
                assert existing is not None
                assert existing.pid == os.getpid()
                assert existing.socket == str(socket_path)

                client = SocketClient(Path(existing.socket))
                await client.connect()

                try:
                    # 3. Proxy pings -> True
                    assert await client.ping() is True

                    # 4. Proxy calls tool -> gets result
                    result = await client.call_tool("synaptiq_list_repos", {})
                    assert result == "No indexed repositories found."

                    result2 = await client.call_tool("some_tool", {"x": 1})
                    assert result2 == "result:some_tool"

                    # Also test resource reads
                    res = await client.read_resource("file:///hello.txt")
                    assert res == "resource:file:///hello.txt"
                finally:
                    await client.close()
            finally:
                await server.stop()
                primary_lock.release()
        finally:
            # Cleanup temp directory
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_stale_primary_recovery(self) -> None:
        """A new instance detects and recovers from a crashed primary."""
        tmpdir, data_dir = _make_short_data_dir()
        try:
            # 1. Write a lock file with a dead PID
            lock_path = data_dir / "synaptiq.lock"
            dead_pid = 99999999
            lock_data = {
                "pid": dead_pid,
                "socket": str(data_dir / "synaptiq.sock"),
                "started_at": "2025-01-01T00:00:00+00:00",
            }
            lock_path.write_text(json.dumps(lock_data))

            # 2. New instance reads existing lock info and detects staleness
            new_lock = LockManager(data_dir)
            existing = new_lock.read_existing()
            assert existing is not None
            assert existing.pid == dead_pid
            assert existing.is_stale() is True

            # 3. force_cleanup removes stale files, then try_acquire succeeds
            new_lock.force_cleanup()
            lock_info = new_lock.try_acquire()
            assert lock_info is not None, "Should acquire lock after cleaning up stale files"
            assert lock_info.pid == os.getpid()

            new_lock.release()
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_multiple_proxies_concurrent(self) -> None:
        """Multiple proxy clients can query the primary simultaneously."""
        tmpdir, data_dir = _make_short_data_dir()
        try:
            # 1. Primary with socket server
            primary_lock = LockManager(data_dir)
            lock_info = primary_lock.try_acquire()
            assert lock_info is not None

            socket_path = Path(lock_info.socket)
            server = SocketServer(socket_path, _test_dispatch)
            await server.start()

            try:
                # 2. Create 3 proxy clients
                clients: list[SocketClient] = []
                for _ in range(3):
                    c = SocketClient(socket_path)
                    await c.connect()
                    clients.append(c)

                try:
                    # 3. All query concurrently via asyncio.gather
                    async def proxy_workflow(client: SocketClient, idx: int) -> dict:
                        ping_ok = await client.ping()
                        tool_result = await client.call_tool(f"tool_{idx}", {})
                        resource_result = await client.read_resource(f"res://{idx}")
                        return {
                            "ping": ping_ok,
                            "tool": tool_result,
                            "resource": resource_result,
                        }

                    results = await asyncio.gather(
                        proxy_workflow(clients[0], 0),
                        proxy_workflow(clients[1], 1),
                        proxy_workflow(clients[2], 2),
                    )

                    # Verify all got correct results
                    for idx, res in enumerate(results):
                        assert res["ping"] is True, f"Client {idx} ping failed"
                        assert res["tool"] == f"result:tool_{idx}", (
                            f"Client {idx} tool result wrong: {res['tool']}"
                        )
                        assert res["resource"] == f"resource:res://{idx}", (
                            f"Client {idx} resource result wrong: {res['resource']}"
                        )
                finally:
                    for c in clients:
                        await c.close()
            finally:
                await server.stop()
                primary_lock.release()
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
