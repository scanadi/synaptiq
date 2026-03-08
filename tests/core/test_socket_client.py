"""Tests for synaptiq.core.daemon.socket_client — SocketClient."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from synaptiq.core.daemon.socket_client import SocketClient
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
