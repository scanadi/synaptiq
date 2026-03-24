"""Streamable HTTP MCP transport for Synaptiq.

Wraps the existing MCP ``Server`` instance in a Starlette ASGI app using
``StreamableHTTPSessionManager`` from the MCP SDK.  Served via uvicorn.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

if TYPE_CHECKING:
    from mcp.server import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager


def _health(request):
    """Simple health check endpoint."""
    return JSONResponse({"status": "ok", "service": "synaptiq"})


def create_starlette_app(
    mcp_server: Server,
    *,
    stateless: bool = True,
) -> tuple[Starlette, "StreamableHTTPSessionManager"]:
    """Build a Starlette ASGI app wired to the MCP server.

    Returns the app and the session manager (whose ``run()`` async context
    manager must be entered during the app lifespan).
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        stateless=stateless,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with session_manager.run():
            yield

    starlette_app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/health", _health),
            Mount("/mcp", app=session_manager.handle_request),
        ],
    )

    return starlette_app, session_manager
