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
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from synaptiq.core.daemon.socket_server import DISPATCH_TIMEOUT, WRITE_DISPATCH_TIMEOUT

if TYPE_CHECKING:
    from synaptiq.core.daemon.lock import LockInfo

logger = logging.getLogger(__name__)

# Client-side timeouts derive from the server's dispatch budgets so the
# server always times out first and the client relays its error instead of
# abandoning an in-flight request.  Anyone changing the server constants
# keeps this invariant automatically.
REQUEST_TIMEOUT = DISPATCH_TIMEOUT + 10.0
REINDEX_TIMEOUT = WRITE_DISPATCH_TIMEOUT + 10.0

# Reconnection settings.
MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 0.5

# StreamReader buffer limit for responses.  asyncio's default is 64KB and
# a single line-delimited JSON response easily exceeds that (tool responses
# can run hundreds of KB) — overflowing the limit kills the connection.
SOCKET_READ_LIMIT = 16 * 1024 * 1024


class PrimaryPromotedError(ConnectionError):
    """The primary daemon is gone and this process promoted itself.

    Raised instead of a plain reconnect failure so callers holding a
    proxy reference know to dispatch locally from now on.
    """


class SocketClient:
    """Async Unix domain socket client for inter-process communication.

    Supports concurrent requests via request-ID multiplexing and
    automatic reconnection on broken connections.

    Parameters
    ----------
    socket_path:
        Unix socket of the primary daemon.
    lock_reader:
        Optional callable returning the current :class:`LockInfo` (or
        ``None``).  When reconnection to the cached socket path fails,
        the lock file is consulted: a healthy lock with a different
        socket means the primary moved (follow it); a missing or stale
        lock means the primary died.
    on_primary_lost:
        Optional async callback invoked when the primary is gone for
        good.  Should attempt takeover and return ``True`` when this
        process successfully promoted itself to primary.
    """

    def __init__(
        self,
        socket_path: Path,
        *,
        lock_reader: Callable[[], "LockInfo | None"] | None = None,
        on_primary_lost: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._lock_reader = lock_reader
        self._on_primary_lost = on_primary_lost
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # Pending responses keyed by request ID.
        self._pending: dict[str, asyncio.Future[dict]] = {}
        # Lock to serialize writes (reads are demuxed by the reader task).
        self._write_lock = asyncio.Lock()
        # Serializes reconnection: when the read loop fails every pending
        # future at once, all waiters try to reconnect — without the lock
        # each teardown would close the connection another waiter just
        # opened.
        self._reconnect_lock = asyncio.Lock()
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
                limit=SOCKET_READ_LIMIT,
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
        """Attempt to reconnect to the primary daemon (serialized).

        Concurrent waiters queue on the lock; whoever arrives after a
        successful reconnect sees a healthy connection and returns
        immediately instead of tearing it down again.
        """
        async with self._reconnect_lock:
            if (
                self._writer is not None
                and not self._writer.is_closing()
                and self._reader_task is not None
                and not self._reader_task.done()
            ):
                return  # another waiter already reconnected

            await self._teardown()

            for attempt in range(MAX_RECONNECT_ATTEMPTS):
                try:
                    self._reader, self._writer = await asyncio.open_unix_connection(
                        str(self._socket_path),
                        limit=SOCKET_READ_LIMIT,
                    )
                    self._reader_task = asyncio.create_task(self._read_loop())
                    logger.info("Reconnected to primary daemon (attempt %d)", attempt + 1)
                    return
                except (OSError, ConnectionRefusedError):
                    if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                        await asyncio.sleep(RECONNECT_DELAY * (attempt + 1))

            if await self._recover_from_lock():
                return

            raise ConnectionError(
                f"Failed to reconnect to {self._socket_path} "
                f"after {MAX_RECONNECT_ATTEMPTS} attempts"
            )

    async def _recover_from_lock(self) -> bool:
        """Recover after exhausting reconnects to the cached socket path.

        Consults the lock file: a healthy lock pointing at a *different*
        socket means the primary restarted elsewhere — follow it.  A
        missing or stale lock means the primary died; invoke the
        ``on_primary_lost`` callback so the owner can take over.

        Returns ``True`` when reconnected to a moved primary.  Raises
        :class:`PrimaryPromotedError` when the callback promoted this
        process.  Returns ``False`` when no recovery was possible.
        """
        if self._lock_reader is None:
            return False

        try:
            info = self._lock_reader()
        except Exception:
            logger.exception("Lock read failed during reconnect recovery")
            return False

        if info is not None and not info.is_stale():
            new_path = Path(info.socket)
            if new_path == self._socket_path:
                # Primary claims to be alive on the path we already tried —
                # nothing to recover; let the caller surface the failure.
                return False
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(
                    str(new_path),
                    limit=SOCKET_READ_LIMIT,
                )
            except (OSError, ConnectionRefusedError):
                return False
            self._socket_path = new_path
            self._reader_task = asyncio.create_task(self._read_loop())
            logger.info("Primary moved — followed lock file to %s", new_path)
            return True

        if self._on_primary_lost is not None:
            if await self._on_primary_lost():
                logger.info("Primary lost — this instance promoted itself")
                raise PrimaryPromotedError(
                    "Primary daemon is gone; this instance promoted itself to primary"
                )
        return False

    # ------------------------------------------------------------------
    # Low-level request/response
    # ------------------------------------------------------------------

    async def _request(
        self, method: str, params: dict, *, timeout: float | None = None
    ) -> str:
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
                except PrimaryPromotedError:
                    raise
                except ConnectionError:
                    raise ConnectionError(f"Connection lost: {exc}") from exc

            try:
                response = await asyncio.wait_for(fut, timeout=timeout or REQUEST_TIMEOUT)
            except asyncio.TimeoutError:
                self._pending.pop(req_id, None)
                raise
            except ConnectionError as exc:
                # Connection dropped between write and response.  The server
                # may already be EXECUTING the request, so re-sending could
                # double-run a non-idempotent method (reindex, forget) —
                # reconnect so future requests work, but surface the error.
                self._pending.pop(req_id, None)
                logger.warning("Connection lost mid-request: %s", exc)
                try:
                    await self._reconnect()
                except PrimaryPromotedError:
                    # The old primary died mid-request taking the in-flight
                    # work with it; the caller re-dispatches locally.
                    raise
                except ConnectionError:
                    pass
                raise ConnectionError(
                    f"Connection lost while awaiting response (the request may "
                    f"or may not have executed): {exc}"
                ) from exc

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

    async def reindex(
        self,
        *,
        full: bool = True,
        skip_embeddings: bool = False,
        embedding_model: str | None = None,
    ) -> str:
        """Request a full reindex from the primary daemon.

        *embedding_model* (a tier name, e.g. ``"quality"``/``"fast"``) is an
        explicit override for this one request — the primary's own routine
        watcher-triggered rebuilds always re-derive the tier from
        ``meta.json`` instead (see ``pipeline.build_full_index``), since
        those have no per-cycle flag to take an override from. ``None``
        (default) lets the primary do the same self-derivation here too.
        """
        return await self._request(
            "reindex",
            {
                "full": full,
                "skip_embeddings": skip_embeddings,
                "embedding_model": embedding_model,
            },
            timeout=REINDEX_TIMEOUT,
        )
