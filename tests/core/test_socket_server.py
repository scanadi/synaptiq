"""Tests for synaptiq.core.daemon.socket_server — SocketServer."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest

from synaptiq.core.daemon.socket_server import SocketServer

# ======================================================================
# Fixtures
# ======================================================================


def mock_dispatch(method: str, params: dict) -> str:
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
    import tempfile

    sock = Path(tempfile.mkdtemp()) / "ax.sock"
    yield sock  # type: ignore[misc]
    if sock.exists():
        sock.unlink()
    sock.parent.rmdir()


# ======================================================================
# Helpers
# ======================================================================


async def send_request(
    socket_path: Path,
    method: str,
    params: dict | None = None,
) -> dict:
    """Open a connection, send one JSON-RPC-ish request, return the response."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    req = {
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params or {},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return json.loads(line)


async def send_raw(socket_path: Path, raw: str) -> dict:
    """Send a raw string (possibly invalid JSON) and return the parsed response."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    writer.write(raw.encode())
    await writer.drain()

    line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return json.loads(line)


# ======================================================================
# Tests
# ======================================================================


class TestSocketServer:
    """Tests for the SocketServer class."""

    async def test_starts_and_stops(self, socket_path: Path) -> None:
        """Server starts, socket file exists, stops cleanly."""
        server = SocketServer(socket_path, mock_dispatch)
        await server.start()
        try:
            assert socket_path.exists()
        finally:
            await server.stop()
        assert not socket_path.exists()

    async def test_handles_ping(self, socket_path: Path) -> None:
        """Send a ping request, get 'pong' response."""
        server = SocketServer(socket_path, mock_dispatch)
        await server.start()
        try:
            resp = await send_request(socket_path, "ping")
            assert resp["result"] == "pong"
            assert resp["id"] is not None
            assert "error" not in resp
        finally:
            await server.stop()

    async def test_handles_tool_call(self, socket_path: Path) -> None:
        """Send a tool request, get dispatched result."""
        server = SocketServer(socket_path, mock_dispatch)
        await server.start()
        try:
            resp = await send_request(socket_path, "tool", {"name": "grep"})
            assert resp["result"] == "result for grep"
        finally:
            await server.stop()

    async def test_handles_malformed_json(self, socket_path: Path) -> None:
        """Send invalid JSON, get error response (server should not crash)."""
        server = SocketServer(socket_path, mock_dispatch)
        await server.start()
        try:
            resp = await send_raw(socket_path, "NOT JSON\n")
            assert "error" in resp
            assert resp["error"]["code"] == -1
            assert "Malformed JSON" in resp["error"]["message"]

            # Server still works after the bad request.
            resp2 = await send_request(socket_path, "ping")
            assert resp2["result"] == "pong"
        finally:
            await server.stop()

    async def test_multiple_clients(self, socket_path: Path) -> None:
        """3 concurrent ping clients all get responses."""
        server = SocketServer(socket_path, mock_dispatch)
        await server.start()
        try:
            results = await asyncio.gather(
                send_request(socket_path, "ping"),
                send_request(socket_path, "ping"),
                send_request(socket_path, "ping"),
            )
            assert len(results) == 3
            for resp in results:
                assert resp["result"] == "pong"
        finally:
            await server.stop()
