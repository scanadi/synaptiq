"""Async Unix domain socket server for the primary synaptiq daemon.

Accepts line-delimited JSON requests and dispatches them through a
caller-provided function.  Used by the primary instance to serve
queries from proxy instances.

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

logger = logging.getLogger(__name__)


class SocketServer:
    """Async Unix domain socket server for inter-process communication."""

    def __init__(
        self,
        socket_path: Path,
        dispatch: Callable[[str, dict], str],
        *,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._dispatch = dispatch
        self._lock = lock
        self._server: asyncio.AbstractServer | None = None

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
            if self._lock is not None:
                async with self._lock:
                    result = await asyncio.to_thread(self._dispatch, method, params)
            else:
                result = await asyncio.to_thread(self._dispatch, method, params)
            return json.dumps({"id": req_id, "result": result}) + "\n"
        except Exception as exc:
            return json.dumps({
                "id": req_id,
                "error": {"code": -1, "message": str(exc)},
            }) + "\n"
