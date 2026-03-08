"""MCP server for Synaptiq — exposes code intelligence tools over stdio transport.

Registers fifteen tools and three resources that give AI agents and MCP clients
access to the Synaptiq knowledge graph.  The server lazily initialises a
:class:`KuzuBackend` from the ``.synaptiq/kuzu`` directory in the current
working directory.

Concurrency
-----------
Uses an :class:`AsyncRWLock` so multiple read queries can run in parallel.
Write operations (file watcher) acquire an exclusive write lock.

Usage::

    # MCP server only
    synaptiq mcp

    # MCP server with live file watching (recommended)
    synaptiq serve --watch
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from synaptiq.core.daemon.rwlock import AsyncRWLock
from synaptiq.core.daemon.socket_server import DISPATCH_TIMEOUT
from synaptiq.core.storage.kuzu_backend import KuzuBackend
from synaptiq.mcp.resources import get_dead_code_list, get_overview, get_schema

if TYPE_CHECKING:
    from synaptiq.core.daemon.socket_client import SocketClient

from synaptiq.mcp.tools import (
    handle_call_path,
    handle_communities,
    handle_context,
    handle_coupling,
    handle_cycles,
    handle_cypher,
    handle_dead_code,
    handle_detect_changes,
    handle_explain,
    handle_file_context,
    handle_impact,
    handle_list_repos,
    handle_query,
    handle_review_risk,
    handle_test_impact,
)

logger = logging.getLogger(__name__)

server = Server("synaptiq")

_storage: KuzuBackend | None = None
_rwlock: AsyncRWLock | None = None
_proxy_client: "SocketClient | None" = None


def set_proxy_client(client: "SocketClient | None") -> None:
    """Inject a socket client for proxy mode."""
    global _proxy_client  # noqa: PLW0603
    _proxy_client = client


def set_storage(storage: KuzuBackend) -> None:
    """Inject a pre-initialised storage backend (e.g. from ``synaptiq serve --watch``)."""
    global _storage  # noqa: PLW0603
    _storage = storage


def set_rwlock(rwlock: AsyncRWLock) -> None:
    """Inject a shared RWLock for coordinating storage access with the file watcher."""
    global _rwlock  # noqa: PLW0603
    _rwlock = rwlock


async def _dispatch_under_read_lock(fn: object, *args: object) -> str:
    """Run *fn* in a thread with timeout, optionally under a shared read lock."""
    coro = asyncio.to_thread(fn, *args)
    if _rwlock is not None:
        async with _rwlock.reader():
            return await asyncio.wait_for(coro, timeout=DISPATCH_TIMEOUT)
    return await asyncio.wait_for(coro, timeout=DISPATCH_TIMEOUT)


def _get_storage() -> KuzuBackend:
    """Lazily initialise and return the KuzuDB storage backend.

    Looks for a ``.synaptiq/kuzu`` directory in the current working directory.
    If it exists, the backend is initialised from that path.  Otherwise a
    bare (uninitialised) backend is returned so that tools can still be
    called without crashing.
    """
    global _storage  # noqa: PLW0603
    if _storage is None:
        _storage = KuzuBackend()
        db_path = Path.cwd() / ".synaptiq" / "kuzu"
        if db_path.exists():
            _storage.initialize(db_path, read_only=True)
            logger.info("Initialised storage (read-only) from %s", db_path)
        else:
            logger.warning("No .synaptiq/kuzu directory found in %s", Path.cwd())
    return _storage

TOOLS: list[Tool] = [
    Tool(
        name="synaptiq_list_repos",
        description="List all indexed repositories with their stats.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="synaptiq_query",
        description=(
            "Search the knowledge graph using hybrid (keyword + vector) search. "
            "Returns ranked symbols matching the query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="synaptiq_context",
        description=(
            "Get a 360-degree view of a symbol: callers, callees, type references, "
            "heritage, and community membership."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to look up.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="synaptiq_impact",
        description=(
            "Blast radius analysis: find all symbols affected by changing a given symbol, "
            "grouped by depth with confidence scores."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to analyse.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum traversal depth (default 3, max 10).",
                    "default": 3,
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="synaptiq_dead_code",
        description="List all symbols detected as dead (unreachable) code.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="synaptiq_detect_changes",
        description=(
            "Parse a git diff and map changed files/lines to affected symbols "
            "in the knowledge graph."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Raw git diff output.",
                },
            },
            "required": ["diff"],
        },
    ),
    Tool(
        name="synaptiq_cypher",
        description="Execute a raw Cypher query against the knowledge graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher query string.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="synaptiq_coupling",
        description=(
            "Show temporal coupling for a file: which files change together in git history, "
            "and flag hidden dependencies (co-change without static import)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative file path to analyse.",
                },
                "min_strength": {
                    "type": "number",
                    "description": "Minimum coupling strength threshold (default 0.3).",
                    "default": 0.3,
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="synaptiq_call_path",
        description=(
            "Find the shortest call chain between two symbols via BFS traversal."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "from_symbol": {
                    "type": "string",
                    "description": "Name of the source symbol.",
                },
                "to_symbol": {
                    "type": "string",
                    "description": "Name of the target symbol.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum BFS depth (default 10).",
                    "default": 10,
                },
            },
            "required": ["from_symbol", "to_symbol"],
        },
    ),
    Tool(
        name="synaptiq_communities",
        description=(
            "List detected communities (Leiden clusters), or drill into a specific "
            "community to see its members and cross-community processes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "community": {
                    "type": "string",
                    "description": "Optional community name to drill into.",
                },
            },
        },
    ),
    Tool(
        name="synaptiq_explain",
        description=(
            "Get a narrative explanation of a symbol: its role, callers, callees, "
            "community, and process memberships."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to explain.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="synaptiq_review_risk",
        description=(
            "Assess PR risk from a git diff: scores risk based on entry point hits, "
            "missing co-change files, downstream dependents, and community crossings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Raw git diff output.",
                },
            },
            "required": ["diff"],
        },
    ),
    Tool(
        name="synaptiq_file_context",
        description=(
            "Get comprehensive context for a file: symbols, imports, importers, "
            "coupling, dead code, and communities."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative file path to analyse.",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="synaptiq_cycles",
        description=(
            "Detect circular dependencies using strongly connected components (igraph)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "min_size": {
                    "type": "integer",
                    "description": "Minimum cycle size to report (default 2).",
                    "default": 2,
                },
            },
        },
    ),
    Tool(
        name="synaptiq_test_impact",
        description=(
            "Find tests likely affected by code changes. Accepts either a git diff "
            "or a list of symbol names."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Raw git diff output.",
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbol names to check test impact for.",
                },
            },
        },
    ),
]

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available Synaptiq tools."""
    return TOOLS

def dispatch_tool(name: str, arguments: dict, storage: KuzuBackend) -> str:
    """Synchronous tool dispatch — called directly or via ``asyncio.to_thread``."""
    if name == "synaptiq_list_repos":
        return handle_list_repos()
    elif name == "synaptiq_query":
        return handle_query(storage, arguments.get("query", ""), limit=arguments.get("limit", 20))
    elif name == "synaptiq_context":
        return handle_context(storage, arguments.get("symbol", ""))
    elif name == "synaptiq_impact":
        return handle_impact(storage, arguments.get("symbol", ""), depth=arguments.get("depth", 3))
    elif name == "synaptiq_dead_code":
        return handle_dead_code(storage)
    elif name == "synaptiq_detect_changes":
        return handle_detect_changes(storage, arguments.get("diff", ""))
    elif name == "synaptiq_cypher":
        return handle_cypher(storage, arguments.get("query", ""))
    elif name == "synaptiq_coupling":
        return handle_coupling(
            storage,
            arguments.get("file_path", ""),
            min_strength=arguments.get("min_strength", 0.3),
        )
    elif name == "synaptiq_call_path":
        return handle_call_path(
            storage,
            arguments.get("from_symbol", ""),
            arguments.get("to_symbol", ""),
            max_depth=arguments.get("max_depth", 10),
        )
    elif name == "synaptiq_communities":
        return handle_communities(storage, community=arguments.get("community"))
    elif name == "synaptiq_explain":
        return handle_explain(storage, arguments.get("symbol", ""))
    elif name == "synaptiq_review_risk":
        return handle_review_risk(storage, arguments.get("diff", ""))
    elif name == "synaptiq_file_context":
        return handle_file_context(storage, arguments.get("file_path", ""))
    elif name == "synaptiq_cycles":
        return handle_cycles(storage, min_size=arguments.get("min_size", 2))
    elif name == "synaptiq_test_impact":
        return handle_test_impact(
            storage,
            diff=arguments.get("diff", ""),
            symbols=arguments.get("symbols"),
        )
    else:
        return f"Unknown tool: {name}"


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    if _proxy_client is not None:
        result = await _proxy_client.call_tool(name, arguments)
        return [TextContent(type="text", text=result)]

    storage = _get_storage()
    result = await _dispatch_under_read_lock(dispatch_tool, name, arguments, storage)
    return [TextContent(type="text", text=result)]

@server.list_resources()
async def list_resources() -> list[Resource]:
    """Return the list of available Synaptiq resources."""
    return [
        Resource(
            uri="synaptiq://overview",
            name="Codebase Overview",
            description="High-level statistics about the indexed codebase.",
            mimeType="text/plain",
        ),
        Resource(
            uri="synaptiq://dead-code",
            name="Dead Code Report",
            description="List of all symbols flagged as unreachable.",
            mimeType="text/plain",
        ),
        Resource(
            uri="synaptiq://schema",
            name="Graph Schema",
            description="Description of the Synaptiq knowledge graph schema.",
            mimeType="text/plain",
        ),
    ]

def dispatch_resource(uri_str: str, storage: KuzuBackend) -> str:
    """Synchronous resource dispatch."""
    if uri_str == "synaptiq://overview":
        return get_overview(storage)
    if uri_str == "synaptiq://dead-code":
        return get_dead_code_list(storage)
    if uri_str == "synaptiq://schema":
        return get_schema()
    return f"Unknown resource: {uri_str}"


@server.read_resource()
async def read_resource(uri) -> str:
    """Read the contents of an Synaptiq resource."""
    if _proxy_client is not None:
        return await _proxy_client.read_resource(str(uri))

    storage = _get_storage()
    return await _dispatch_under_read_lock(dispatch_resource, str(uri), storage)

async def main() -> None:
    """Run the Synaptiq MCP server over stdio transport."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
