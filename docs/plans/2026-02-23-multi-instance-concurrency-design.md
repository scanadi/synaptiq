# Multi-Instance Concurrency for Synaptiq MCP

**Date:** 2026-02-23
**Status:** Approved

## Problem

Synaptiq is configured as an MCP server in `.mcp.json` with `synaptiq serve --watch`. Every Claude Code instance (including sub-agents in swarms) spawns this command. KuzuDB holds an exclusive file lock in write mode, so only the first instance succeeds — all subsequent ones fail to start.

KuzuDB does not support a read-write `Database` and a read-only `Database` open on the same directory simultaneously, ruling out a simple "try write, fall back to read-only" approach.

## Solution: Primary/Proxy Architecture

A single primary process owns the database and watcher. All other instances become lightweight proxies that forward queries over a Unix domain socket.

### Architecture

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Claude Code 1│  │ Claude Code 2│  │ Claude Code 3│
│   (main)     │  │  (sub-agent) │  │  (sub-agent) │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │ stdio           │ stdio           │ stdio
┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐
│ synaptiq serve   │  │ synaptiq serve   │  │ synaptiq serve   │
│ (PRIMARY)    │  │ (PROXY)      │  │ (PROXY)      │
│ DB + Watcher │  │ socket client│  │ socket client│
│ + socket srv │  └──────┬───────┘  └──────┬───────┘
└──────┬───────┘         │                 │
       │ write     ┌─────▼─────────────────▼─────┐
       ▼           │  Unix socket (.synaptiq/synaptiq.sock)│
  .synaptiq/kuzu/  ◀───└────────────────────────────────┘
```

### Lock File Protocol

Location: `.synaptiq/synaptiq.lock`

```json
{
  "pid": 12345,
  "socket": "/path/to/repo/.synaptiq/synaptiq.sock",
  "started_at": "2026-02-23T10:00:00Z",
  "version": "0.2.2"
}
```

Startup decision tree:

1. Attempt to acquire `fcntl.flock()` on `.synaptiq/synaptiq.lock`.
2. If acquired: become PRIMARY — write lock file, open DB, start watcher + MCP + socket server.
3. If not acquired: read lock file, verify PID is alive and socket responds to ping.
   - If healthy: become PROXY — start MCP stdio server, forward queries via socket.
   - If stale: remove lock file, retry from step 1.
4. On shutdown: PRIMARY releases flock, removes lock file, closes socket and DB. PROXY disconnects.

### Socket Communication Protocol

Line-delimited JSON over Unix domain socket (`.synaptiq/synaptiq.sock`).

**Request:**
```json
{"id": "uuid", "method": "tool", "params": {"name": "synaptiq_query", "arguments": {"query": "auth"}}}
```

**Response:**
```json
{"id": "uuid", "result": "## Results\n..."}
```

**Error:**
```json
{"id": "uuid", "error": {"code": -1, "message": "Storage not initialized"}}
```

Methods: `tool`, `resource`, `ping`.

The primary acquires the existing `asyncio.Lock` before dispatching queries, so watcher writes and socket queries do not conflict.

### Proxy MCP Server

From Claude Code's perspective, a proxy instance is identical to a primary — same tools, same resources, same stdio transport. The only difference is internal: `call_tool()` and `read_resource()` forward through the socket client instead of accessing KuzuDB directly.

The proxy has no `KuzuBackend` instance and no watcher.

### Branch Switching

The watcher on the primary detects file changes caused by `git checkout` and incrementally re-indexes affected files. All proxy instances get fresh results immediately since they query through the primary. No special handling is needed beyond the existing watcher logic.

## File Structure

New files:

```
src/synaptiq/core/daemon/
├── __init__.py
├── lock.py            # Lock file acquire/release/staleness check
├── socket_server.py   # asyncio Unix socket server (primary)
└── socket_client.py   # asyncio Unix socket client (proxy)
```

Modified files:

```
src/synaptiq/cli/main.py   # serve command: primary vs proxy detection
src/synaptiq/mcp/server.py # call_tool/read_resource: proxy dispatch branch
```

## Design Decisions

- **Unix domain socket over TCP:** Zero network overhead, no port conflicts, filesystem-scoped.
- **`fcntl.flock()` for lock acquisition:** Atomic, OS-level, automatically released if the process crashes.
- **No new dependencies:** `asyncio.start_unix_server()`, `fcntl`, and `json` are all stdlib.
- **Same command for all instances:** `.mcp.json` stays unchanged — the command self-detects its role.
