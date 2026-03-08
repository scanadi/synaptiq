"""Async Unix domain socket client for proxy synaptiq instances.

Connects to the primary daemon's Unix socket and forwards MCP
tool/resource calls using the same line-delimited JSON protocol
that :class:`~synaptiq.core.daemon.socket_server.SocketServer` speaks.

Concurrency-safe: uses a pending-futures dict keyed by request ID so
multiple concurrent ``call_tool`` / ``read_resource`` calls on the same
connection do not get their responses mixed up.

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
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Timeout for individual requests (seconds).
REQUEST_TIMEOUT = 60.0

# Reconnection settings.
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 0.5


class SocketClient:
    """Async Unix domain socket client for inter-process communication.

    Supports concurrent requests via request-ID multiplexing and
    automatic reconnection on broken connections.
    """

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # Pending responses keyed by request ID.
        self._pending: dict[str, asyncio.Future[dict]] = {}
        # Lock to serialize writes (reads are demuxed by the reader task).
        self._write_lock = asyncio.Lock()
        # Background task that reads responses and dispatches to futures.
        self._reader_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the primary daemon.

        Raises :class:`ConnectionError` if the socket is unreachable.
        """
        try:
            self._reader, self._writer = await asyncio.open_unix_connection(
                str(self._socket_path),
            )
        except (OSError, ConnectionRefusedError) as exc:
            raise ConnectionError(
                f"Cannot connect to {self._socket_path}: {exc}"
            ) from exc
        # Start the background reader task for response demultiplexing.
        self._reader_task = asyncio.create_task(self._read_loop())
        logger.info("Connected to primary daemon at %s", self._socket_path)

    async def _teardown(self) -> None:
        """Clean up reader task and writer — shared by close() and _reconnect()."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

    async def close(self) -> None:
        """Close the connection."""
        await self._teardown()

        # Cancel any pending futures.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Connection closed"))
        self._pending.clear()

        logger.info("Disconnected from %s", self._socket_path)

    # ------------------------------------------------------------------
    # Background reader (response demultiplexer)
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Continuously read responses and dispatch to pending futures."""
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break  # EOF — server disconnected

                try:
                    response = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning("Malformed response from server: %s", line[:100])
                    continue

                req_id = response.get("id")
                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(response)
                else:
                    logger.debug("Unexpected response ID: %s", req_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Reader loop error")
        finally:
            # Signal all pending futures that the connection is broken.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Connection lost"))
            self._pending.clear()

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the primary daemon."""
        await self._teardown()

        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(
                    str(self._socket_path),
                )
                self._reader_task = asyncio.create_task(self._read_loop())
                logger.info("Reconnected to primary daemon (attempt %d)", attempt + 1)
                return
            except (OSError, ConnectionRefusedError):
                if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                    await asyncio.sleep(RECONNECT_DELAY * (attempt + 1))

        raise ConnectionError(
            f"Failed to reconnect to {self._socket_path} "
            f"after {MAX_RECONNECT_ATTEMPTS} attempts"
        )

    # ------------------------------------------------------------------
    # Low-level request/response
    # ------------------------------------------------------------------

    async def _request(self, method: str, params: dict) -> str:
        """Send a JSON request and wait for the matching response.

        Uses request-ID multiplexing so concurrent calls are safe.
        Returns the ``result`` string from the server response.

        Raises :class:`ConnectionError` if not connected,
        :class:`RuntimeError` if the server returns an error payload,
        and :class:`asyncio.TimeoutError` if the request times out.
        """
        for _attempt in range(2):  # retry once on connection failure
            if self._reader is None or self._writer is None:
                try:
                    await self._reconnect()
                except ConnectionError:
                    raise

            req_id = str(uuid.uuid4())
            req = {
                "id": req_id,
                "method": method,
                "params": params,
            }

            loop = asyncio.get_running_loop()
            fut: asyncio.Future[dict] = loop.create_future()
            self._pending[req_id] = fut

            try:
                async with self._write_lock:
                    self._writer.write((json.dumps(req) + "\n").encode("utf-8"))
                    await self._writer.drain()
            except (OSError, ConnectionResetError, BrokenPipeError) as exc:
                self._pending.pop(req_id, None)
                if not fut.done():
                    fut.cancel()
                # Try to reconnect and retry.
                logger.warning("Write failed, attempting reconnect: %s", exc)
                try:
                    await self._reconnect()
                    continue
                except ConnectionError:
                    raise ConnectionError(f"Connection lost: {exc}") from exc

            try:
                response = await asyncio.wait_for(fut, timeout=REQUEST_TIMEOUT)
            except asyncio.TimeoutError:
                self._pending.pop(req_id, None)
                raise

            if "error" in response:
                raise RuntimeError(response["error"]["message"])

            return response["result"]

        raise ConnectionError("Request failed after retry")

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Health check. Returns ``True`` if the primary responds with 'pong'."""
        result = await self._request("ping", {})
        return result == "pong"

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Forward a tool call to the primary daemon."""
        return await self._request("tool", {"name": name, "arguments": arguments})

    async def read_resource(self, uri: str) -> str:
        """Forward a resource read to the primary daemon."""
        return await self._request("resource", {"uri": uri})
