"""Tests for synaptiq.core.daemon.socket_client — SocketClient."""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from synaptiq.core.daemon.lock import LockInfo
from synaptiq.core.daemon.socket_client import PrimaryPromotedError, SocketClient
from synaptiq.core.daemon.socket_server import SocketServer

# ======================================================================
# Fixtures
# ======================================================================


def mock_dispatch(method: str, params: dict) -> str:
    """Simple echo-style dispatch for tests."""
    if method == "ping":
        return "pong"
    if method == "tool":
        return f"result for {params.get('name', '?')}"
    if method == "resource":
        return f"resource: {params.get('uri', '?')}"
    return "unknown"


@pytest.fixture()
def socket_path() -> Path:
    """Return a short socket path to avoid macOS AF_UNIX length limits."""
    sock = Path(tempfile.mkdtemp()) / "ax.sock"
    yield sock  # type: ignore[misc]
    if sock.exists():
        sock.unlink()
    sock.parent.rmdir()


@pytest.fixture()
async def server(socket_path: Path) -> SocketServer:
    """Start a SocketServer and tear it down after the test."""
    srv = SocketServer(socket_path, mock_dispatch)
    await srv.start()
    yield srv  # type: ignore[misc]
    await srv.stop()


@pytest.fixture()
async def client(socket_path: Path, server: SocketServer) -> SocketClient:
    """Connect a SocketClient to the running server."""
    cli = SocketClient(socket_path)
    await cli.connect()
    yield cli  # type: ignore[misc]
    await cli.close()


# ======================================================================
# Tests
# ======================================================================


class TestSocketClient:
    """Tests for the SocketClient class."""

    async def test_ping(self, client: SocketClient) -> None:
        """Connect to server, ping, get True."""
        assert await client.ping() is True

    async def test_call_tool(self, client: SocketClient) -> None:
        """Forward tool call, get result."""
        result = await client.call_tool("grep", {"pattern": "foo"})
        assert result == "result for grep"

    async def test_read_resource(self, client: SocketClient) -> None:
        """Forward resource read, get result."""
        result = await client.read_resource("file:///tmp/data.txt")
        assert result == "resource: file:///tmp/data.txt"

    async def test_connect_failure(self) -> None:
        """Connect to nonexistent socket, get ConnectionError."""
        bad_path = Path(tempfile.mkdtemp()) / "no.sock"
        cli = SocketClient(bad_path)
        with pytest.raises(ConnectionError):
            await cli.connect()
        bad_path.parent.rmdir()

    async def test_multiple_sequential_calls(self, client: SocketClient) -> None:
        """Multiple calls on the same connection all work."""
        assert await client.ping() is True
        r1 = await client.call_tool("grep", {})
        assert r1 == "result for grep"
        r2 = await client.read_resource("uri://a")
        assert r2 == "resource: uri://a"
        assert await client.ping() is True


def _alive_lock_info(socket: Path) -> LockInfo:
    return LockInfo(
        pid=os.getpid(),
        socket=str(socket),
        started_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def _dead_lock_info(socket: Path) -> LockInfo:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return LockInfo(
        pid=proc.pid,
        socket=str(socket),
        started_at=datetime.now(tz=timezone.utc).isoformat(),
    )


class TestLockAwareRecovery:
    """Reconnect recovery via the lock file after the primary moves or dies."""

    async def test_follows_moved_primary(
        self, socket_path: Path, server: SocketServer
    ) -> None:
        """Dead cached socket + healthy lock at a new path → follow the lock."""
        dead_path = socket_path.parent / "dead.sock"
        cli = SocketClient(
            dead_path,
            lock_reader=lambda: _alive_lock_info(socket_path),
        )
        try:
            assert await cli.ping() is True
        finally:
            await cli.close()

    async def test_promotes_when_lock_is_stale(self, socket_path: Path) -> None:
        """Stale lock (dead PID) + winning takeover → PrimaryPromotedError."""
        dead_path = socket_path.parent / "dead.sock"
        promoted: list[bool] = []

        async def _on_primary_lost() -> bool:
            promoted.append(True)
            return True

        cli = SocketClient(
            dead_path,
            lock_reader=lambda: _dead_lock_info(dead_path),
            on_primary_lost=_on_primary_lost,
        )
        with pytest.raises(PrimaryPromotedError):
            await cli.ping()
        assert promoted == [True]

    async def test_promotes_when_lock_is_missing(self, socket_path: Path) -> None:
        """Missing lock file behaves like a stale one — takeover is attempted."""
        dead_path = socket_path.parent / "dead.sock"

        async def _on_primary_lost() -> bool:
            return True

        cli = SocketClient(
            dead_path,
            lock_reader=lambda: None,
            on_primary_lost=_on_primary_lost,
        )
        with pytest.raises(PrimaryPromotedError):
            await cli.ping()

    async def test_lost_takeover_race_surfaces_connection_error(
        self, socket_path: Path
    ) -> None:
        """Callback returning False (lost the race) → plain ConnectionError."""
        dead_path = socket_path.parent / "dead.sock"

        async def _on_primary_lost() -> bool:
            return False

        cli = SocketClient(
            dead_path,
            lock_reader=lambda: None,
            on_primary_lost=_on_primary_lost,
        )
        with pytest.raises(ConnectionError) as excinfo:
            await cli.ping()
        assert not isinstance(excinfo.value, PrimaryPromotedError)


class TestLargeResponses:
    """Responses must survive asyncio's default 64KB StreamReader limit."""

    async def test_large_response_round_trips(self, socket_path: Path) -> None:
        """A response well past 64KB arrives intact instead of killing the connection."""
        payload = "x" * (256 * 1024)

        def fat_dispatch(method: str, params: dict) -> str:
            return payload

        srv = SocketServer(socket_path, fat_dispatch)
        await srv.start()
        cli = SocketClient(socket_path)
        await cli.connect()
        try:
            result = await cli.call_tool("anything", {})
            assert result == payload
        finally:
            await cli.close()
            await srv.stop()
