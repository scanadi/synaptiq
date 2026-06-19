"""MCP server for Synaptiq — exposes code intelligence tools over stdio transport.

Registers twenty tools and three resources that give AI agents and MCP clients
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
from synaptiq.core.daemon.socket_client import PrimaryPromotedError
from synaptiq.core.daemon.socket_server import DISPATCH_TIMEOUT
from synaptiq.core.storage.kuzu_backend import KuzuBackend
from synaptiq.mcp.resources import get_dead_code_list, get_overview, get_schema

if TYPE_CHECKING:
    from synaptiq.core.daemon.socket_client import SocketClient

from synaptiq.mcp.secret_scanner import redact as _redact_secrets
from synaptiq.mcp.token_budget import truncate_response, wrap_with_metadata
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
    handle_export,
    handle_file_context,
    handle_forget,
    handle_impact,
    handle_list_repos,
    handle_query,
    handle_recall,
    handle_remember,
    handle_review_risk,
    handle_suggest,
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

    If the database is corrupted (e.g. duplicate primary key from a
    mid-write kill), the kuzu directory and meta.json are deleted so
    the next ``serve --watch`` or ``analyze`` invocation rebuilds cleanly.
    """
    from synaptiq.core.storage.kuzu_backend import open_with_recovery

    global _storage  # noqa: PLW0603
    if _storage is None:
        db_path = Path.cwd() / ".synaptiq" / "kuzu"
        if db_path.exists():
            try:
                _storage = open_with_recovery(
                    db_path,
                    Path.cwd() / ".synaptiq" / "meta.json",
                    read_only=True,
                )
                logger.info("Initialised storage (read-only) from %s", db_path)
            except RuntimeError as exc:
                if "lock on file" in str(exc).lower():
                    # Another process holds the database read-write
                    # (e.g. ``synaptiq serve --watch``). Kuzu allows
                    # read-only opens only when no writer exists.
                    raise RuntimeError(
                        "The Synaptiq database is locked by another process "
                        "(likely `synaptiq serve --watch`). Connect through "
                        "that server instead of starting a standalone "
                        "`synaptiq mcp` instance."
                    ) from exc
                raise
        else:
            _storage = KuzuBackend()
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
                "focus_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Bias results toward symbols in these files using "
                        "Personalized PageRank."
                    ),
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum response size in tokens.",
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
                "focus_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Bias results toward symbols near these files using "
                        "Personalized PageRank."
                    ),
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum response size in tokens.",
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
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum response size in tokens.",
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
            "properties": {
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum response size in tokens.",
                },
            },
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
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum response size in tokens.",
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
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum response size in tokens.",
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
    Tool(
        name="synaptiq_remember",
        description=(
            "Persist a fact about the codebase for recall in future sessions. "
            "Useful for storing architectural insights discovered during analysis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Unique key identifying the fact.",
                },
                "value": {
                    "type": "string",
                    "description": "The fact or insight to remember.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category (e.g. 'architecture', 'pattern')."
                    ),
                },
            },
            "required": ["key", "value"],
        },
    ),
    Tool(
        name="synaptiq_recall",
        description=(
            "Retrieve previously stored facts about the codebase. "
            "Searches by key (exact) or fuzzy word overlap across all facts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Key or search query to look up.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="synaptiq_forget",
        description="Remove a previously stored fact by key.",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key of the fact to remove.",
                },
            },
            "required": ["key"],
        },
    ),
    Tool(
        name="synaptiq_suggest",
        description=(
            "Get suggested tool calls for a natural language question. "
            "Returns an ordered sequence of recommended Synaptiq tools and arguments."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about the codebase.",
                },
            },
            "required": ["question"],
        },
    ),
    Tool(
        name="synaptiq_export",
        description=(
            "Graph-aware context packing: traverse from a symbol and return all "
            "structurally relevant code in a single response. Replaces multiple "
            "round-trip tool calls with one comprehensive result."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the starting symbol.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Traversal depth (default 2, max 4).",
                    "default": 2,
                },
                "include_source": {
                    "type": "boolean",
                    "description": "Include full source code of each symbol (default true).",
                    "default": True,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum response size in tokens.",
                },
            },
            "required": ["symbol"],
        },
    ),
]

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available Synaptiq tools."""
    return TOOLS

def _apply_response_pipeline(result: str, max_tokens: int | None = None) -> str:
    """Apply secret scanning, token budgeting, and metadata to a tool response."""
    result, redacted_count = _redact_secrets(result)
    if redacted_count > 0:
        result += f"\n\nWARNING: {redacted_count} potential secret(s) redacted from response."
    if max_tokens and max_tokens > 0:
        result = truncate_response(result, max_tokens)
    return wrap_with_metadata(result)


def _as_int(value: object, default: int) -> int:
    """Coerce an MCP argument to int; inputSchema types are advisory only."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    """Coerce an MCP argument to float."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    """Coerce an MCP argument to bool, accepting common string forms."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
    return default


def _as_str_list(value: object) -> list[str] | None:
    """Coerce an MCP argument to a list of strings, or None."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return None


def dispatch_tool(name: str, arguments: dict, storage: KuzuBackend) -> str:
    """Synchronous tool dispatch — called directly or via ``asyncio.to_thread``."""
    max_tokens = _as_int(arguments.get("max_tokens"), 0) or None

    if name == "synaptiq_list_repos":
        result = handle_list_repos()
    elif name == "synaptiq_query":
        result = handle_query(
            storage,
            str(arguments.get("query", "")),
            limit=_as_int(arguments.get("limit"), 20),
            focus_files=_as_str_list(arguments.get("focus_files")),
        )
    elif name == "synaptiq_context":
        result = handle_context(
            storage,
            str(arguments.get("symbol", "")),
            focus_files=_as_str_list(arguments.get("focus_files")),
        )
    elif name == "synaptiq_impact":
        result = handle_impact(
            storage, str(arguments.get("symbol", "")), depth=_as_int(arguments.get("depth"), 3)
        )
    elif name == "synaptiq_dead_code":
        result = handle_dead_code(storage)
    elif name == "synaptiq_detect_changes":
        result = handle_detect_changes(storage, str(arguments.get("diff", "")))
    elif name == "synaptiq_cypher":
        result = handle_cypher(storage, str(arguments.get("query", "")))
    elif name == "synaptiq_coupling":
        result = handle_coupling(
            storage,
            str(arguments.get("file_path", "")),
            min_strength=_as_float(arguments.get("min_strength"), 0.3),
        )
    elif name == "synaptiq_call_path":
        result = handle_call_path(
            storage,
            str(arguments.get("from_symbol", "")),
            str(arguments.get("to_symbol", "")),
            max_depth=_as_int(arguments.get("max_depth"), 10),
        )
    elif name == "synaptiq_communities":
        community = arguments.get("community")
        result = handle_communities(
            storage, community=str(community) if community is not None else None
        )
    elif name == "synaptiq_explain":
        result = handle_explain(storage, str(arguments.get("symbol", "")))
    elif name == "synaptiq_review_risk":
        result = handle_review_risk(storage, str(arguments.get("diff", "")))
    elif name == "synaptiq_file_context":
        result = handle_file_context(storage, str(arguments.get("file_path", "")))
    elif name == "synaptiq_cycles":
        result = handle_cycles(storage, min_size=_as_int(arguments.get("min_size"), 2))
    elif name == "synaptiq_test_impact":
        result = handle_test_impact(
            storage,
            diff=str(arguments.get("diff", "") or ""),
            symbols=_as_str_list(arguments.get("symbols")),
        )
    elif name == "synaptiq_remember":
        result = handle_remember(
            str(arguments.get("key", "")),
            str(arguments.get("value", "")),
            category=str(arguments.get("category", "") or ""),
        )
    elif name == "synaptiq_recall":
        result = handle_recall(str(arguments.get("query", "")))
    elif name == "synaptiq_forget":
        result = handle_forget(str(arguments.get("key", "")))
    elif name == "synaptiq_suggest":
        result = handle_suggest(storage, str(arguments.get("question", "")))
    elif name == "synaptiq_export":
        result = handle_export(
            storage,
            str(arguments.get("symbol", "")),
            depth=_as_int(arguments.get("depth"), 2),
            include_source=_as_bool(arguments.get("include_source"), True),
        )
    else:
        result = f"Unknown tool: {name}"

    return _apply_response_pipeline(result, max_tokens)


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    if _proxy_client is not None:
        try:
            result = await _proxy_client.call_tool(name, arguments)
            return [TextContent(type="text", text=result)]
        except PrimaryPromotedError:
            # The primary died and this process took over — storage and
            # rwlock are now local; fall through to local dispatch.
            pass

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
        result = get_overview(storage)
    elif uri_str == "synaptiq://dead-code":
        result = get_dead_code_list(storage)
    elif uri_str == "synaptiq://schema":
        result = get_schema()
    else:
        result = f"Unknown resource: {uri_str}"
    result, count = _redact_secrets(result)
    if count > 0:
        result += f"\n\nWARNING: {count} potential secret(s) redacted from response."
    return wrap_with_metadata(result)


@server.read_resource()
async def read_resource(uri) -> str:
    """Read the contents of an Synaptiq resource."""
    if _proxy_client is not None:
        try:
            return await _proxy_client.read_resource(str(uri))
        except PrimaryPromotedError:
            pass

    storage = _get_storage()
    return await _dispatch_under_read_lock(dispatch_resource, str(uri), storage)

async def main() -> None:
    """Run the Synaptiq MCP server over stdio transport."""
    from synaptiq.core.resources import set_profile

    # Idempotent with the CLI entry points; covers `python -m synaptiq.mcp.server`.
    set_profile("server")
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
