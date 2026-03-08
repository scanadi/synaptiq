"""Async Unix domain socket client for proxy synaptiq instances.

Connects to the primary daemon's Unix socket and forwards MCP
tool/resource calls using the same line-delimited JSON protocol
that :class:`~synaptiq.core.daemon.socket_server.SocketServer` speaks.

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


class SocketClient:
    """Async Unix domain socket client for inter-process communication."""

    def __init__(self, socket_path: Path) -> None:
        self._socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

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
        logger.info("Connected to primary daemon at %s", self._socket_path)

    async def close(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None
            logger.info("Disconnected from %s", self._socket_path)

    # ------------------------------------------------------------------
    # Low-level request/response
    # ------------------------------------------------------------------

    async def _request(self, method: str, params: dict) -> str:
        """Send a JSON request and wait for the response.

        Returns the ``result`` string from the server response.

        Raises :class:`ConnectionError` if not connected and
        :class:`RuntimeError` if the server returns an error payload.
        """
        if self._reader is None or self._writer is None:
            raise ConnectionError("Not connected — call connect() first")

        req = {
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        self._writer.write((json.dumps(req) + "\n").encode("utf-8"))
        await self._writer.drain()

        line = await self._reader.readline()
        if not line:
            raise ConnectionError("Connection closed by server")

        response = json.loads(line)

        if "error" in response:
            raise RuntimeError(response["error"]["message"])

        return response["result"]

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
