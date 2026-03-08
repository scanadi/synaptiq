# Multi-Instance Concurrency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `synaptiq serve --watch` support multiple concurrent instances by having the first instance become the primary (owns DB + watcher) and subsequent instances become proxies (forward queries over Unix socket).

**Architecture:** Primary/proxy pattern using `fcntl.flock()` for lock acquisition and `asyncio` Unix domain sockets for inter-process communication. The same command self-detects its role at startup.

**Tech Stack:** Python stdlib only — `fcntl`, `asyncio`, `json`, `os`, `uuid`. No new dependencies.

**Design doc:** `docs/plans/2026-02-23-multi-instance-concurrency-design.md`

---

### Task 1: Lock File Manager (`lock.py`)

**Files:**
- Create: `src/synaptiq/core/daemon/__init__.py`
- Create: `src/synaptiq/core/daemon/lock.py`
- Test: `tests/core/test_lock.py`

**Step 1: Write the failing tests**

Create `tests/core/test_lock.py`:

```python
"""Tests for the daemon lock file manager."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from synaptiq.core.daemon.lock import LockManager, LockInfo


class TestLockManagerAcquire:
    """LockManager.try_acquire() gets an exclusive flock."""

    def test_acquire_creates_lock_file(self, tmp_path: Path) -> None:
        mgr = LockManager(tmp_path / ".synaptiq")
        info = mgr.try_acquire()

        assert info is not None
        assert info.pid == os.getpid()
        assert (tmp_path / ".synaptiq" / "synaptiq.lock").exists()

        mgr.release()

    def test_acquire_writes_valid_json(self, tmp_path: Path) -> None:
        mgr = LockManager(tmp_path / ".synaptiq")
        info = mgr.try_acquire()

        raw = (tmp_path / ".synaptiq" / "synaptiq.lock").read_text()
        data = json.loads(raw)
        assert data["pid"] == os.getpid()
        assert "socket" in data
        assert "started_at" in data

        mgr.release()

    def test_second_acquire_fails(self, tmp_path: Path) -> None:
        mgr1 = LockManager(tmp_path / ".synaptiq")
        mgr2 = LockManager(tmp_path / ".synaptiq")

        info1 = mgr1.try_acquire()
        assert info1 is not None

        info2 = mgr2.try_acquire()
        assert info2 is None

        mgr1.release()

    def test_release_removes_lock_and_socket(self, tmp_path: Path) -> None:
        mgr = LockManager(tmp_path / ".synaptiq")
        mgr.try_acquire()
        mgr.release()

        assert not (tmp_path / ".synaptiq" / "synaptiq.lock").exists()

    def test_creates_synaptiq_dir_if_missing(self, tmp_path: Path) -> None:
        mgr = LockManager(tmp_path / ".synaptiq")
        info = mgr.try_acquire()

        assert info is not None
        assert (tmp_path / ".synaptiq").is_dir()

        mgr.release()


class TestLockManagerRead:
    """LockManager.read_existing() reads lock info from another process."""

    def test_read_existing_returns_info(self, tmp_path: Path) -> None:
        mgr = LockManager(tmp_path / ".synaptiq")
        mgr.try_acquire()

        reader = LockManager(tmp_path / ".synaptiq")
        info = reader.read_existing()

        assert info is not None
        assert info.pid == os.getpid()

        mgr.release()

    def test_read_existing_returns_none_when_no_lock(self, tmp_path: Path) -> None:
        reader = LockManager(tmp_path / ".synaptiq")
        info = reader.read_existing()
        assert info is None

    def test_is_stale_detects_dead_pid(self, tmp_path: Path) -> None:
        synaptiq_dir = tmp_path / ".synaptiq"
        synaptiq_dir.mkdir(parents=True)
        lock_data = {
            "pid": 99999999,  # Almost certainly not running
            "socket": str(synaptiq_dir / "synaptiq.sock"),
            "started_at": "2026-01-01T00:00:00Z",
        }
        (synaptiq_dir / "synaptiq.lock").write_text(json.dumps(lock_data))

        reader = LockManager(synaptiq_dir)
        info = reader.read_existing()

        assert info is not None
        assert info.is_stale()

    def test_is_stale_false_for_live_pid(self, tmp_path: Path) -> None:
        mgr = LockManager(tmp_path / ".synaptiq")
        mgr.try_acquire()

        reader = LockManager(tmp_path / ".synaptiq")
        info = reader.read_existing()

        assert info is not None
        assert not info.is_stale()

        mgr.release()
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/core/test_lock.py -v`
Expected: ModuleNotFoundError (synaptiq.core.daemon.lock does not exist yet)

**Step 3: Write the implementation**

Create `src/synaptiq/core/daemon/__init__.py`:

```python
"""Daemon utilities for multi-instance coordination."""
```

Create `src/synaptiq/core/daemon/lock.py`:

```python
"""Lock file manager for primary/proxy coordination.

Uses ``fcntl.flock()`` for atomic lock acquisition.  The lock file at
``.synaptiq/synaptiq.lock`` contains JSON metadata (PID, socket path, timestamp)
so other instances can decide whether to become a proxy or take over.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LockInfo:
    """Parsed lock file contents."""

    pid: int
    socket: str
    started_at: str

    def is_stale(self) -> bool:
        """Return True if the process that created this lock is no longer running."""
        try:
            os.kill(self.pid, 0)
            return False
        except OSError:
            return True

    def to_dict(self) -> dict:
        return {"pid": self.pid, "socket": self.socket, "started_at": self.started_at}


class LockManager:
    """Manage the ``.synaptiq/synaptiq.lock`` file for primary/proxy coordination."""

    def __init__(self, synaptiq_dir: Path) -> None:
        self._synaptiq_dir = synaptiq_dir
        self._lock_path = synaptiq_dir / "synaptiq.lock"
        self._socket_path = synaptiq_dir / "synaptiq.sock"
        self._fd: int | None = None

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def try_acquire(self) -> LockInfo | None:
        """Try to acquire the exclusive lock.

        Returns LockInfo on success, None if another process holds the lock.
        """
        self._synaptiq_dir.mkdir(parents=True, exist_ok=True)

        try:
            fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                os.close(fd)
            except Exception:
                pass
            return None

        self._fd = fd

        info = LockInfo(
            pid=os.getpid(),
            socket=str(self._socket_path),
            started_at=datetime.now(tz=timezone.utc).isoformat(),
        )

        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(info.to_dict()).encode())

        return info

    def read_existing(self) -> LockInfo | None:
        """Read lock info written by another process.

        Returns None if the lock file does not exist or is unreadable.
        """
        if not self._lock_path.exists():
            return None
        try:
            data = json.loads(self._lock_path.read_text())
            return LockInfo(
                pid=data["pid"],
                socket=data["socket"],
                started_at=data.get("started_at", ""),
            )
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def release(self) -> None:
        """Release the lock and clean up files."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        self._lock_path.unlink(missing_ok=True)
        self._socket_path.unlink(missing_ok=True)

    def force_cleanup(self) -> None:
        """Remove stale lock and socket files without holding the lock."""
        self._lock_path.unlink(missing_ok=True)
        self._socket_path.unlink(missing_ok=True)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/core/test_lock.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/synaptiq/core/daemon/__init__.py src/synaptiq/core/daemon/lock.py tests/core/test_lock.py
git commit -m "feat: add lock file manager for multi-instance coordination"
```

---

### Task 2: Socket Server (`socket_server.py`)

**Files:**
- Create: `src/synaptiq/core/daemon/socket_server.py`
- Test: `tests/core/test_socket_server.py`

**Step 1: Write the failing tests**

Create `tests/core/test_socket_server.py`:

```python
"""Tests for the daemon socket server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synaptiq.core.daemon.socket_server import SocketServer


@pytest.fixture
def mock_dispatch():
    """A mock dispatch function that returns a canned response."""
    def dispatch(method: str, params: dict) -> str:
        if method == "ping":
            return "pong"
        if method == "tool":
            return f"result for {params.get('name', '?')}"
        if method == "resource":
            return f"resource: {params.get('uri', '?')}"
        return "unknown"
    return dispatch


class TestSocketServerStartStop:
    @pytest.mark.asyncio
    async def test_starts_and_stops(self, tmp_path: Path, mock_dispatch) -> None:
        sock_path = tmp_path / "test.sock"
        server = SocketServer(sock_path, mock_dispatch)

        await server.start()
        assert sock_path.exists()

        await server.stop()

    @pytest.mark.asyncio
    async def test_handles_ping(self, tmp_path: Path, mock_dispatch) -> None:
        sock_path = tmp_path / "test.sock"
        server = SocketServer(sock_path, mock_dispatch)
        await server.start()

        reader, writer = await asyncio.open_unix_connection(str(sock_path))
        request = json.dumps({"id": "1", "method": "ping", "params": {}}) + "\n"
        writer.write(request.encode())
        await writer.drain()

        line = await reader.readline()
        response = json.loads(line)
        assert response["id"] == "1"
        assert response["result"] == "pong"

        writer.close()
        await writer.wait_closed()
        await server.stop()

    @pytest.mark.asyncio
    async def test_handles_tool_call(self, tmp_path: Path, mock_dispatch) -> None:
        sock_path = tmp_path / "test.sock"
        server = SocketServer(sock_path, mock_dispatch)
        await server.start()

        reader, writer = await asyncio.open_unix_connection(str(sock_path))
        request = json.dumps({
            "id": "2",
            "method": "tool",
            "params": {"name": "synaptiq_query", "arguments": {"query": "auth"}},
        }) + "\n"
        writer.write(request.encode())
        await writer.drain()

        line = await reader.readline()
        response = json.loads(line)
        assert response["id"] == "2"
        assert "synaptiq_query" in response["result"]

        writer.close()
        await writer.wait_closed()
        await server.stop()

    @pytest.mark.asyncio
    async def test_handles_malformed_json(self, tmp_path: Path, mock_dispatch) -> None:
        sock_path = tmp_path / "test.sock"
        server = SocketServer(sock_path, mock_dispatch)
        await server.start()

        reader, writer = await asyncio.open_unix_connection(str(sock_path))
        writer.write(b"not valid json\n")
        await writer.drain()

        line = await reader.readline()
        response = json.loads(line)
        assert "error" in response

        writer.close()
        await writer.wait_closed()
        await server.stop()

    @pytest.mark.asyncio
    async def test_multiple_clients(self, tmp_path: Path, mock_dispatch) -> None:
        sock_path = tmp_path / "test.sock"
        server = SocketServer(sock_path, mock_dispatch)
        await server.start()

        async def send_ping(client_id: str) -> dict:
            reader, writer = await asyncio.open_unix_connection(str(sock_path))
            request = json.dumps({"id": client_id, "method": "ping", "params": {}}) + "\n"
            writer.write(request.encode())
            await writer.drain()
            line = await reader.readline()
            writer.close()
            await writer.wait_closed()
            return json.loads(line)

        results = await asyncio.gather(send_ping("a"), send_ping("b"), send_ping("c"))
        assert all(r["result"] == "pong" for r in results)

        await server.stop()
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/core/test_socket_server.py -v`
Expected: ModuleNotFoundError

**Step 3: Write the implementation**

Create `src/synaptiq/core/daemon/socket_server.py`:

```python
"""Unix domain socket server for the primary synaptiq daemon.

Accepts line-delimited JSON requests from proxy instances and dispatches
them to the storage backend through a caller-provided dispatch function.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class SocketServer:
    """Async Unix socket server that dispatches JSON-RPC-style requests."""

    def __init__(
        self,
        socket_path: Path,
        dispatch: Callable[[str, dict], str],
    ) -> None:
        self._socket_path = socket_path
        self._dispatch = dispatch
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        """Start listening on the Unix socket."""
        self._socket_path.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(self._socket_path)
        )
        logger.info("Socket server listening on %s", self._socket_path)

    async def stop(self) -> None:
        """Stop the server and clean up the socket file."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._socket_path.unlink(missing_ok=True)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single client connection (one line = one request)."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    response = {"id": None, "error": {"code": -1, "message": "Invalid JSON"}}
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
                    continue

                req_id = request.get("id")
                method = request.get("method", "")
                params = request.get("params", {})

                try:
                    result = self._dispatch(method, params)
                    response = {"id": req_id, "result": result}
                except Exception as exc:
                    response = {"id": req_id, "error": {"code": -1, "message": str(exc)}}

                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Client handler error", exc_info=True)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/core/test_socket_server.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/synaptiq/core/daemon/socket_server.py tests/core/test_socket_server.py
git commit -m "feat: add Unix socket server for primary daemon"
```

---

### Task 3: Socket Client (`socket_client.py`)

**Files:**
- Create: `src/synaptiq/core/daemon/socket_client.py`
- Test: `tests/core/test_socket_client.py`

**Step 1: Write the failing tests**

Create `tests/core/test_socket_client.py`:

```python
"""Tests for the daemon socket client."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from synaptiq.core.daemon.socket_client import SocketClient
from synaptiq.core.daemon.socket_server import SocketServer


def _echo_dispatch(method: str, params: dict) -> str:
    if method == "ping":
        return "pong"
    if method == "tool":
        return f"tool:{params.get('name', '?')}"
    if method == "resource":
        return f"resource:{params.get('uri', '?')}"
    return "unknown"


@pytest.fixture
async def server_and_path(tmp_path: Path):
    """Start a socket server and yield its path."""
    sock_path = tmp_path / "test.sock"
    server = SocketServer(sock_path, _echo_dispatch)
    await server.start()
    yield sock_path
    await server.stop()


class TestSocketClient:
    @pytest.mark.asyncio
    async def test_ping(self, server_and_path: Path) -> None:
        client = SocketClient(server_and_path)
        await client.connect()
        result = await client.ping()
        assert result is True
        await client.close()

    @pytest.mark.asyncio
    async def test_call_tool(self, server_and_path: Path) -> None:
        client = SocketClient(server_and_path)
        await client.connect()
        result = await client.call_tool("synaptiq_query", {"query": "auth"})
        assert "synaptiq_query" in result
        await client.close()

    @pytest.mark.asyncio
    async def test_read_resource(self, server_and_path: Path) -> None:
        client = SocketClient(server_and_path)
        await client.connect()
        result = await client.read_resource("synaptiq://overview")
        assert "overview" in result
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_failure(self, tmp_path: Path) -> None:
        client = SocketClient(tmp_path / "nonexistent.sock")
        with pytest.raises(ConnectionError):
            await client.connect()

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(self, server_and_path: Path) -> None:
        client = SocketClient(server_and_path)
        await client.connect()
        r1 = await client.call_tool("synaptiq_query", {"query": "a"})
        r2 = await client.call_tool("synaptiq_context", {"symbol": "b"})
        assert "synaptiq_query" in r1
        assert "synaptiq_context" in r2
        await client.close()
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/core/test_socket_client.py -v`
Expected: ModuleNotFoundError

**Step 3: Write the implementation**

Create `src/synaptiq/core/daemon/socket_client.py`:

```python
"""Unix domain socket client for proxy synaptiq instances.

Connects to the primary daemon's socket server and forwards tool/resource
calls as line-delimited JSON requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class SocketClient:
    """Async client that connects to the primary daemon's Unix socket."""

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Connect to the primary daemon's socket.

        Raises ConnectionError if the socket is not available.
        """
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                str(self._socket_path)
            )
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            raise ConnectionError(f"Cannot connect to {self._socket_path}: {exc}") from exc

    async def close(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

    async def _request(self, method: str, params: dict) -> str:
        """Send a request and wait for the response."""
        if self._reader is None or self._writer is None:
            raise ConnectionError("Not connected")

        req_id = str(uuid.uuid4())
        request = {"id": req_id, "method": method, "params": params}
        self._writer.write((json.dumps(request) + "\n").encode())
        await self._writer.drain()

        line = await self._reader.readline()
        if not line:
            raise ConnectionError("Connection closed by primary")

        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"]["message"])
        return response["result"]

    async def ping(self) -> bool:
        """Health check — returns True if the primary is responsive."""
        try:
            result = await self._request("ping", {})
            return result == "pong"
        except Exception:
            return False

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Forward a tool call to the primary."""
        return await self._request("tool", {"name": name, "arguments": arguments})

    async def read_resource(self, uri: str) -> str:
        """Forward a resource read to the primary."""
        return await self._request("resource", {"uri": uri})
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/core/test_socket_client.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/synaptiq/core/daemon/socket_client.py tests/core/test_socket_client.py
git commit -m "feat: add Unix socket client for proxy instances"
```

---

### Task 4: Integrate Primary/Proxy into MCP Server

**Files:**
- Modify: `src/synaptiq/mcp/server.py`
- Test: `tests/mcp/test_server_proxy.py`

**Step 1: Write the failing tests**

Create `tests/mcp/test_server_proxy.py`:

```python
"""Tests for proxy mode in the MCP server."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from synaptiq.mcp.server import call_tool, read_resource, set_proxy_client


class TestProxyDispatch:
    """When a proxy client is set, tool calls forward through it."""

    @pytest.mark.asyncio
    async def test_call_tool_uses_proxy(self) -> None:
        mock_client = AsyncMock()
        mock_client.call_tool.return_value = "proxied result"

        set_proxy_client(mock_client)
        try:
            result = await call_tool("synaptiq_query", {"query": "test"})
            assert result[0].text == "proxied result"
            mock_client.call_tool.assert_called_once_with("synaptiq_query", {"query": "test"})
        finally:
            set_proxy_client(None)

    @pytest.mark.asyncio
    async def test_read_resource_uses_proxy(self) -> None:
        mock_client = AsyncMock()
        mock_client.read_resource.return_value = "proxied overview"

        set_proxy_client(mock_client)
        try:
            result = await read_resource("synaptiq://overview")
            assert result == "proxied overview"
            mock_client.read_resource.assert_called_once_with("synaptiq://overview")
        finally:
            set_proxy_client(None)

    @pytest.mark.asyncio
    async def test_call_tool_falls_back_to_local_when_no_proxy(self) -> None:
        """Without a proxy client, tools dispatch locally (existing behavior)."""
        set_proxy_client(None)
        # This should not raise — it falls through to the normal path
        # (which may fail because no storage is initialized, but that's fine)
        result = await call_tool("synaptiq_list_repos", {})
        assert result[0].text is not None  # Returns some string, even error
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/mcp/test_server_proxy.py -v`
Expected: ImportError (set_proxy_client does not exist yet)

**Step 3: Modify the MCP server**

Modify `src/synaptiq/mcp/server.py` — add proxy client support:

Add after the `_lock` global (line 44):

```python
_proxy_client: "SocketClient | None" = None


def set_proxy_client(client) -> None:
    """Inject a socket client for proxy mode."""
    global _proxy_client  # noqa: PLW0603
    _proxy_client = client
```

Modify `call_tool` (replace lines 213-224):

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    if _proxy_client is not None:
        result = await _proxy_client.call_tool(name, arguments)
        return [TextContent(type="text", text=result)]

    storage = _get_storage()

    if _lock is not None:
        async with _lock:
            result = await asyncio.to_thread(_dispatch_tool, name, arguments, storage)
    else:
        result = _dispatch_tool(name, arguments, storage)

    return [TextContent(type="text", text=result)]
```

Modify `read_resource` (replace lines 261-270):

```python
@server.read_resource()
async def read_resource(uri) -> str:
    """Read the contents of an Synaptiq resource."""
    if _proxy_client is not None:
        return await _proxy_client.read_resource(str(uri))

    storage = _get_storage()
    uri_str = str(uri)

    if _lock is not None:
        async with _lock:
            return await asyncio.to_thread(_dispatch_resource, uri_str, storage)
    return _dispatch_resource(uri_str, storage)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/mcp/test_server_proxy.py -v`
Expected: All 3 tests PASS

**Step 5: Run all existing MCP tests to check for regressions**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/mcp/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/synaptiq/mcp/server.py tests/mcp/test_server_proxy.py
git commit -m "feat: add proxy client support to MCP server"
```

---

### Task 5: Wire Up the `serve` Command

**Files:**
- Modify: `src/synaptiq/cli/main.py:334-388`
- Test: `tests/cli/test_serve_modes.py`

**Step 1: Write the failing tests**

Create `tests/cli/test_serve_modes.py`:

```python
"""Tests for the serve command's primary/proxy mode detection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from synaptiq.core.daemon.lock import LockManager


class TestServeModeDetection:
    """The serve command should auto-detect primary vs proxy mode."""

    def test_first_instance_becomes_primary(self, tmp_path: Path) -> None:
        """When no lock exists, the instance should acquire the lock."""
        mgr = LockManager(tmp_path / ".synaptiq")
        info = mgr.try_acquire()

        assert info is not None
        assert info.pid == os.getpid()

        mgr.release()

    def test_second_instance_detects_existing_primary(self, tmp_path: Path) -> None:
        """When a lock exists with a live PID, the instance should become a proxy."""
        mgr1 = LockManager(tmp_path / ".synaptiq")
        info1 = mgr1.try_acquire()
        assert info1 is not None

        mgr2 = LockManager(tmp_path / ".synaptiq")
        info2 = mgr2.try_acquire()
        assert info2 is None  # Can't acquire — must be proxy

        existing = mgr2.read_existing()
        assert existing is not None
        assert not existing.is_stale()

        mgr1.release()

    def test_stale_lock_gets_cleaned_up(self, tmp_path: Path) -> None:
        """When lock has a dead PID, the new instance should take over."""
        synaptiq_dir = tmp_path / ".synaptiq"
        synaptiq_dir.mkdir(parents=True)
        lock_data = {
            "pid": 99999999,
            "socket": str(synaptiq_dir / "synaptiq.sock"),
            "started_at": "2026-01-01T00:00:00Z",
        }
        (synaptiq_dir / "synaptiq.lock").write_text(json.dumps(lock_data))

        mgr = LockManager(synaptiq_dir)
        existing = mgr.read_existing()
        assert existing is not None
        assert existing.is_stale()

        # After cleanup, should be able to acquire
        mgr.force_cleanup()
        info = mgr.try_acquire()
        assert info is not None

        mgr.release()
```

**Step 2: Run tests to verify they fail or pass (these test the lock module, so should pass)**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/cli/test_serve_modes.py -v`
Expected: All 3 PASS (using already-built lock module)

**Step 3: Rewrite the `serve` command**

Modify `src/synaptiq/cli/main.py` — replace the `serve` function (lines 333-388):

```python
@app.command()
def serve(
    watch: bool = typer.Option(False, "--watch", "-w", help="Enable file watching with auto-reindex."),
) -> None:
    """Start MCP server, optionally with live file watching."""
    import asyncio
    import sys

    from synaptiq.mcp.server import main as mcp_main, set_lock, set_storage, set_proxy_client

    if not watch:
        asyncio.run(mcp_main())
        return

    from synaptiq.core.daemon.lock import LockManager

    repo_path = Path.cwd().resolve()
    synaptiq_dir = repo_path / ".synaptiq"
    synaptiq_dir.mkdir(parents=True, exist_ok=True)

    lock_mgr = LockManager(synaptiq_dir)
    lock_info = lock_mgr.try_acquire()

    if lock_info is None:
        # Another instance is primary — check if it's healthy
        existing = lock_mgr.read_existing()
        if existing is not None and existing.is_stale():
            lock_mgr.force_cleanup()
            lock_info = lock_mgr.try_acquire()

    if lock_info is not None:
        # We are the PRIMARY
        _serve_primary(repo_path, synaptiq_dir, lock_mgr)
    else:
        # We are a PROXY
        existing = lock_mgr.read_existing()
        if existing is None:
            print("Error: cannot read lock info from primary", file=sys.stderr)
            raise typer.Exit(code=1)
        _serve_proxy(existing.socket)


def _serve_primary(repo_path: Path, synaptiq_dir: Path, lock_mgr: "LockManager") -> None:
    """Run as primary: DB + watcher + MCP + socket server."""
    import asyncio
    import sys

    from synaptiq.core.ingestion.pipeline import run_pipeline
    from synaptiq.core.ingestion.watcher import watch_repo
    from synaptiq.core.storage.kuzu_backend import KuzuBackend
    from synaptiq.core.daemon.socket_server import SocketServer
    from synaptiq.mcp.server import server as mcp_server, set_lock, set_storage, _dispatch_tool, _dispatch_resource

    db_path = synaptiq_dir / "kuzu"

    storage = KuzuBackend()
    storage.initialize(db_path)

    if not (synaptiq_dir / "meta.json").exists():
        print("Running initial index...", file=sys.stderr)
        run_pipeline(repo_path, storage, full=True)

    lock = asyncio.Lock()
    set_storage(storage)
    set_lock(lock)

    def dispatch(method: str, params: dict) -> str:
        if method == "ping":
            return "pong"
        if method == "tool":
            return _dispatch_tool(params.get("name", ""), params.get("arguments", {}), storage)
        if method == "resource":
            return _dispatch_resource(params.get("uri", ""), storage)
        return f"Unknown method: {method}"

    socket_server = SocketServer(lock_mgr.socket_path, dispatch)

    async def _run() -> None:
        from mcp.server.stdio import stdio_server

        stop = asyncio.Event()
        await socket_server.start()

        try:
            async with stdio_server() as (read, write):
                async def _mcp_then_stop():
                    await mcp_server.run(read, write, mcp_server.create_initialization_options())
                    stop.set()

                await asyncio.gather(
                    _mcp_then_stop(),
                    watch_repo(repo_path, storage, stop_event=stop, lock=lock),
                )
        finally:
            await socket_server.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        storage.close()
        lock_mgr.release()


def _serve_proxy(socket_path: str) -> None:
    """Run as proxy: MCP over stdio, forwarding to the primary via socket."""
    import asyncio
    import sys

    from synaptiq.core.daemon.socket_client import SocketClient
    from synaptiq.mcp.server import main as mcp_main, set_proxy_client

    client = SocketClient(Path(socket_path))

    async def _run() -> None:
        from mcp.server.stdio import stdio_server
        from synaptiq.mcp.server import server as mcp_server

        await client.connect()
        set_proxy_client(client)

        try:
            async with stdio_server() as (read, write):
                await mcp_server.run(read, write, mcp_server.create_initialization_options())
        finally:
            await client.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
```

**Step 4: Run all tests**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/ -v`
Expected: All tests PASS (including existing CLI tests)

**Step 5: Commit**

```bash
git add src/synaptiq/cli/main.py tests/cli/test_serve_modes.py
git commit -m "feat: serve command auto-detects primary vs proxy mode"
```

---

### Task 6: Integration Test — Multiple Instances

**Files:**
- Create: `tests/e2e/test_multi_instance.py`

**Step 1: Write the integration test**

Create `tests/e2e/test_multi_instance.py`:

```python
"""End-to-end test: multiple synaptiq instances coordinate via lock + socket."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from synaptiq.core.daemon.lock import LockManager
from synaptiq.core.daemon.socket_client import SocketClient
from synaptiq.core.daemon.socket_server import SocketServer


def _test_dispatch(method: str, params: dict) -> str:
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


class TestMultiInstanceCoordination:
    """Simulate primary + proxy coordination through lock + socket."""

    @pytest.mark.asyncio
    async def test_primary_serves_proxy_queries(self, tmp_path: Path) -> None:
        synaptiq_dir = tmp_path / ".synaptiq"

        # Instance 1: acquire lock, start socket server
        mgr1 = LockManager(synaptiq_dir)
        info = mgr1.try_acquire()
        assert info is not None

        server = SocketServer(mgr1.socket_path, _test_dispatch)
        await server.start()

        # Instance 2: fail to acquire lock, connect as proxy
        mgr2 = LockManager(synaptiq_dir)
        assert mgr2.try_acquire() is None

        existing = mgr2.read_existing()
        assert existing is not None
        assert not existing.is_stale()

        client = SocketClient(Path(existing.socket))
        await client.connect()

        # Proxy queries go through socket to primary
        assert await client.ping() is True
        result = await client.call_tool("synaptiq_list_repos", {})
        assert "No indexed repositories" in result

        # Cleanup
        await client.close()
        await server.stop()
        mgr1.release()

    @pytest.mark.asyncio
    async def test_stale_primary_recovery(self, tmp_path: Path) -> None:
        synaptiq_dir = tmp_path / ".synaptiq"
        synaptiq_dir.mkdir(parents=True)

        # Simulate a crashed primary (dead PID in lock file)
        lock_data = {
            "pid": 99999999,
            "socket": str(synaptiq_dir / "synaptiq.sock"),
            "started_at": "2026-01-01T00:00:00Z",
        }
        (synaptiq_dir / "synaptiq.lock").write_text(json.dumps(lock_data))

        # New instance detects stale lock and takes over
        mgr = LockManager(synaptiq_dir)
        existing = mgr.read_existing()
        assert existing is not None
        assert existing.is_stale()

        mgr.force_cleanup()
        info = mgr.try_acquire()
        assert info is not None

        mgr.release()

    @pytest.mark.asyncio
    async def test_multiple_proxies_concurrent(self, tmp_path: Path) -> None:
        synaptiq_dir = tmp_path / ".synaptiq"

        mgr = LockManager(synaptiq_dir)
        mgr.try_acquire()

        server = SocketServer(mgr.socket_path, _test_dispatch)
        await server.start()

        # Spawn 3 concurrent proxy clients
        async def proxy_query(n: int) -> str:
            client = SocketClient(mgr.socket_path)
            await client.connect()
            result = await client.call_tool("synaptiq_query", {"query": f"test_{n}"})
            await client.close()
            return result

        results = await asyncio.gather(proxy_query(1), proxy_query(2), proxy_query(3))
        assert all("synaptiq_query" in r for r in results)

        await server.stop()
        mgr.release()
```

**Step 2: Run the integration tests**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/e2e/test_multi_instance.py -v`
Expected: All 3 tests PASS

**Step 3: Run the full test suite**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/e2e/test_multi_instance.py
git commit -m "test: add e2e tests for multi-instance coordination"
```

---

### Task 7: Update `.gitignore` and Docs

**Files:**
- Modify: `.gitignore` (add `synaptiq.sock` and `synaptiq.lock` patterns)
- Modify: `docs/plans/2026-02-23-multi-instance-concurrency-design.md` (mark as implemented)

**Step 1: Update `.gitignore`**

The `.synaptiq/` directory is already gitignored, so `synaptiq.lock` and `synaptiq.sock` within it are already covered. No change needed.

Verify: `grep ".synaptiq" .gitignore` should show `.synaptiq/`.

**Step 2: Mark design doc as implemented**

Change line 3 in design doc from `**Status:** Approved` to `**Status:** Implemented`.

**Step 3: Run full test suite one final time**

Run: `cd /Users/stevicacanadi/projects/levelup/synaptiq && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add docs/plans/2026-02-23-multi-instance-concurrency-design.md
git commit -m "docs: mark multi-instance concurrency design as implemented"
```
