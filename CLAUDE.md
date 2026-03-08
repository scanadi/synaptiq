# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synaptiq is a graph-powered code intelligence engine that indexes codebases into a knowledge graph (KuzuDB) and exposes it via MCP tools for AI agents and a Typer CLI for developers. It supports Python, TypeScript, and JavaScript via tree-sitter parsing.

Package name: `synaptiq` (published to PyPI). Python 3.11+ required.

## Commands

```bash
# Setup
uv sync --all-extras

# Run from source
uv run synaptiq --help

# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/core/test_graph.py

# Run a specific test
uv run pytest tests/core/test_graph.py::test_add_node -v

# Run only fast unit tests (skip e2e)
uv run pytest tests/core/ tests/cli/ tests/mcp/

# Lint
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/
```

## Architecture

### 11-Phase Ingestion Pipeline

The core of Synaptiq is `src/synaptiq/core/ingestion/pipeline.py` which orchestrates 11 sequential phases:

1. `walker.py` — file discovery respecting `.gitignore`
2. `structure.py` — folder/file hierarchy (CONTAINS edges)
3. `parser_phase.py` — tree-sitter AST extraction → Function/Class/Method/Interface/Enum/TypeAlias nodes
4. `imports.py` — import resolution to actual files (IMPORTS edges)
5. `calls.py` — call tracing with confidence scores (CALLS edges, 1.0=exact, 0.5=fuzzy)
6. `heritage.py` — class inheritance (EXTENDS) and interface implementation (IMPLEMENTS)
7. `types.py` — type references from params/returns/variables (USES_TYPE edges)
8. `community.py` — Leiden algorithm clustering (MEMBER_OF edges)
9. `processes.py` — framework-aware entry point detection + BFS flow tracing
10. `dead_code.py` — multi-pass dead code analysis with exemptions for decorators, protocols, overrides
11. `coupling.py` — git history co-change analysis (COUPLED_WITH edges)

### Storage Layer

`src/synaptiq/core/storage/base.py` defines a `StorageBackend` Protocol. The default implementation is `kuzu_backend.py` (KuzuDB — embedded graph DB with Cypher, FTS, and vector support). Optional Neo4j backend available via `synaptiq[neo4j]`.

Data stored in `.synaptiq/` directory within each indexed repo.

### Search

`src/synaptiq/core/search/hybrid.py` implements BM25 + vector (384-dim BAAI/bge-small-en-v1.5 via fastembed) + fuzzy search fused with Reciprocal Rank Fusion.

### Multi-Instance Concurrency

`src/synaptiq/core/daemon/` implements a primary/proxy pattern for concurrent MCP sessions:
- `lock.py` — `fcntl.flock()` based lock file manager
- `socket_server.py` — async Unix domain socket server (primary)
- `socket_client.py` — async Unix domain socket client (proxy)

The `serve` command auto-detects role at startup. Design doc: `docs/plans/2026-02-23-multi-instance-concurrency-design.md`.

### MCP Server

`src/synaptiq/mcp/server.py` exposes tools (query, context, impact, dead_code, detect_changes, cypher, list_repos) and resources (overview, dead-code, schema) via FastMCP stdio transport. Proxy mode forwards calls through `SocketClient`.

### CLI

`src/synaptiq/cli/main.py` — single Typer app with commands: analyze, status, list, clean, query, context, impact, dead-code, cypher, watch, diff, setup, serve/mcp.

### Parsers

`src/synaptiq/core/parsers/` — `BaseParser` base class in `base.py`, with `python_lang.py` and `typescript.py` implementations. New language parsers extend `BaseParser` and register in `config/languages.py`.

## Code Style

- Ruff with rules `E, F, I, N, W`. Line length 100. Target Python 3.11.
- pytest with `asyncio_mode = "auto"` (no need for `@pytest.mark.asyncio` decorator in most cases, but it's used in existing tests).
- Commit messages follow conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.

## Graph Node ID Format

```
{label}:{relative_path}:{symbol_name}
# e.g., function:src/auth/validate.py:validate_user
#        method:src/models/user.py:User.save
```

## Key Dependencies

tree-sitter (parsing), kuzu (graph DB), igraph+leidenalg (community detection), fastembed (ONNX embeddings), mcp SDK (FastMCP), typer+rich (CLI), watchfiles (file watcher), pathspec (gitignore).
