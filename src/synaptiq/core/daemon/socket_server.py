"""Async Unix domain socket server for the primary synaptiq daemon.

Accepts line-delimited JSON requests and dispatches them through a
caller-provided function.  Used by the primary instance to serve
queries from proxy instances.

Read operations acquire a shared read lock so multiple agents can
query concurrently.  Write operations (via the watcher) acquire an
exclusive write lock.

Protocol
--------
Request:  ``{"id": "<uuid>", "method": "<method>", "params": {...}}\n``
Response: ``{"id": "<uuid>", "result": "..."}\n``
     or:  ``{"id": "<uuid>", "error": {"code": -1, "message": "..."}}\n``
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable

from synaptiq.core.daemon.rwlock import AsyncRWLock

logger = logging.getLogger(__name__)

# Timeout for dispatching a single request (seconds).
DISPATCH_TIMEOUT = 120.0
WRITE_DISPATCH_TIMEOUT = 600.0

# Maximum number of concurrent socket dispatch operations.
MAX_CONCURRENT_DISPATCHES = 16


class SocketServer:
    """Async Unix domain socket server for inter-process communication."""

    def __init__(
        self,
        socket_path: Path,
        dispatch: Callable[[str, dict], str],
        *,
        rwlock: AsyncRWLock | None = None,
        write_methods: set[str] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._dispatch = dispatch
        self._rwlock = rwlock
        self._write_methods = write_methods or set()
        self._server: asyncio.AbstractServer | None = None
        # Semaphore to limit concurrent dispatches.
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_DISPATCHES)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start listening on the Unix socket."""
        # Ensure the parent directory exists.
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove a stale socket file if it exists.
        if self._socket_path.exists():
            self._socket_path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )
        logger.info("Socket server listening on %s", self._socket_path)

    async def stop(self) -> None:
        """Stop the server and clean up the socket file."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._socket_path.exists():
            self._socket_path.unlink()
            logger.info("Removed socket file %s", self._socket_path)

    # ------------------------------------------------------------------
    # Client handling
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one client connection.  Each line is one JSON request."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break  # EOF — client disconnected

                response = await self._process_line(line)
                writer.write(response.encode("utf-8"))
                await writer.drain()
        except Exception:
            logger.exception("Error handling client connection")
        finally:
            writer.close()
            await writer.wait_closed()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_line(self, raw: bytes) -> str:
        """Parse one line, dispatch in a thread, and return a JSON response."""
        try:
            request = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return json.dumps({
                "id": None,
                "error": {"code": -1, "message": f"Malformed JSON: {exc}"},
            }) + "\n"

        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            async with self._semaphore:
                coro = asyncio.to_thread(self._dispatch, method, params)
                is_write = method in self._write_methods
                timeout = WRITE_DISPATCH_TIMEOUT if is_write else DISPATCH_TIMEOUT
                if self._rwlock is not None:
                    if is_write:
                        async with self._rwlock.writer():
                            result = await asyncio.wait_for(coro, timeout=timeout)
                    else:
                        async with self._rwlock.reader():
                            result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await asyncio.wait_for(coro, timeout=timeout)
            return json.dumps({"id": req_id, "result": result}) + "\n"
        except asyncio.TimeoutError:
            return json.dumps({
                "id": req_id,
                "error": {"code": -2, "message": "Request timed out"},
            }) + "\n"
        except Exception as exc:
            return json.dumps({
                "id": req_id,
                "error": {"code": -1, "message": str(exc)},
            }) + "\n"
