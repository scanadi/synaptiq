# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synaptiq is a graph-powered code intelligence engine that indexes codebases into a knowledge graph (LadybugDB) and exposes it via MCP tools for AI agents and a Typer CLI for developers. It supports Python, TypeScript, JavaScript, Ruby, and Go via tree-sitter parsing.

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

# Release (bumps version, commits, tags, pushes → triggers PyPI publish)
./scripts/release.sh patch        # 0.5.0 → 0.5.1
./scripts/release.sh minor        # 0.5.0 → 0.6.0
./scripts/release.sh major        # 0.5.0 → 1.0.0
./scripts/release.sh 0.7.0        # explicit version
./scripts/release.sh patch --dry  # preview only
```

## Release Process

Releases are automated via `scripts/release.sh` + GitHub Actions (`.github/workflows/publish.yml`).

1. **`scripts/release.sh <bump>`** — updates version in `pyproject.toml` and `src/synaptiq/__init__.py`, commits, creates annotated tag, pushes to origin.
2. **GitHub Actions** — triggered by `v*` tag push, runs tests, then builds and publishes to PyPI via OIDC trusted publishing (no API tokens).

Guards: must be on `main`, clean working tree, synced with remote, tag must not exist.

## Architecture

### Ingestion Pipeline

The core of Synaptiq is `src/synaptiq/core/ingestion/pipeline.py` which orchestrates 12 analysis phases plus an optional embedding phase:

1. `walker.py` — file discovery respecting `.gitignore`
2. `structure.py` — folder/file hierarchy (CONTAINS edges)
3. `parser_phase.py` — tree-sitter AST extraction → Function/Class/Method/Module/Interface/Enum/TypeAlias nodes
4. `imports.py` — import resolution to actual files (IMPORTS edges)
5. `calls.py` — call tracing with confidence scores (CALLS edges, 1.0=exact, 0.8=receiver, 0.5=fuzzy)
6. `rest_linking.py` — links REST endpoints to HTTP client calls across services (Python FastAPI/Flask, TS Express/axios, Ruby Sinatra/Rails routes + HTTParty/Faraday/RestClient/Typhoeus/Net::HTTP, Go net/http HandleFunc/Handle + gin/echo/chi/gorilla router verbs + http.Get/Post/PostForm/NewRequest client calls). Per-language regex extractors live in `extract_rest_info_from_source`.
7. `heritage.py` — class inheritance (EXTENDS), interface implementation (IMPLEMENTS), Ruby module mixins (MIXES_IN, from `include`/`extend`/`prepend`), and Go embedding (struct anonymous fields + interface elements → EXTENDS). Go interface *satisfaction* is undecidable without type checking, so no IMPLEMENTS edges are emitted for Go.
8. `types.py` — type references from params/returns/variables (USES_TYPE edges)
9. `community.py` — Leiden algorithm clustering (MEMBER_OF edges), seeded for determinism
10. `processes.py` — framework-aware entry point detection + BFS flow tracing
11. `dead_code.py` — multi-pass dead code analysis with exemptions for decorators, protocols, overrides; Ruby adds `initialize` constructors, metaprogramming hooks (`method_missing`, `inherited`, ...), `attr_*`/Rails-callback macro methods, and Rails framework base classes; Go adds `main`/`init` runtime entries and `_test.go` files (Go's exported-identifier convention is surfaced as `is_exported` by the parser, and `Test*`/`Benchmark*`/`Fuzz*`/`Example*` live in `_test.go`)
12. `coupling.py` — git history co-change analysis (COUPLED_WITH edges)
13. Embeddings (optional) — fastembed BAAI/bge-small-en-v1.5 384-dim vectors for semantic search. `analyze --embeddings lazy|sync|off` (default **lazy**; `--no-embeddings` is a deprecated alias for `off`). **lazy** (W4.1): `analyze` commits the graph first and returns a queryable index in seconds, then a detached worker (`core/embeddings/lazy_worker.py`, spawned as `synaptiq _embed-worker`) encodes vectors in the background — progress lands in `.synaptiq/embeddings_state.json` (surfaced by `synaptiq status`). The worker snapshots `meta.json`'s `last_indexed_at` as a staleness anchor, encodes read-only, then stores under the single-writer lock (retry → `deferred` if a daemon/racing analyze holds it; never fights the lock, never wipes). **sync** encodes inline (pre-W4.1 behavior). Daemons (`serve`/`watch`) always embed **synchronously** in their global rebuild — lazy is a CLI-`analyze` concept. Incremental across rebuilds: vectors carry a `text_sha` of their source text, and `embed_graph(previous=...)` reuses stored vectors so only changed symbols hit ONNX — a one-file change re-encodes a handful of symbols, not all ~19k. The reused-vs-pending split is `embedder.partition_embeddings(graph, previous)` — `embed_graph` calls it internally too (one implementation, shared), but it never loads the ONNX model, so callers that only need the split get the answer at generate-text cost. Sync/daemon paths snapshot via `load_previous_embeddings` BEFORE `bulk_load` wipes the DB, same as always. **Lazy mode reuses across rebuilds too (W4.1b)**: `analyze` snapshots `load_previous_embeddings(storage)` before `run_pipeline`'s `bulk_load`, then after the graph commits it calls `partition_embeddings` and stores the reused vectors immediately (`storage.store_embeddings`, a fast COPY, no model load) — the background worker only gets spawned for the pending delta, and is skipped entirely when nothing is pending (a zero-change `analyze` prints "N vectors reused" and never spawns a worker). First-ever `analyze` (no previous index) skips the partition and falls straight to a cheap node count, so the cold path has no added overhead.

**Embedding model tiers (W4.4)**: `embedder.py` defines a `MODEL_TIERS` registry — `"quality"` (default, BAAI/bge-small-en-v1.5, 384-dim, fastembed/ONNX, ~235 texts/sec measured) and `"fast"` (minishlab/potion-base-8M, 256-dim, [model2vec](https://github.com/MinishLab/model2vec) static embeddings — token lookup + mean-pool, no transformer/ONNX at all — ~43k texts/sec measured, ~180x faster; optional dependency `synaptiq[fast-embeddings]`). Selected via `analyze --embedding-model quality|fast` (default `quality`; no persistence-inheritance — omitting the flag always means `quality`, even on a repo previously indexed with `fast`). `ensure_tier_available()` checks the requested tier's package eagerly, before the pipeline runs, so a missing `model2vec` fails fast with an install hint rather than after minutes of indexing; `_get_model()` (backend dispatch, `lru_cache`d per tier name) and `_check_model2vec_available()` raise the same message lazily for daemon/worker paths that skip the CLI's eager check. The tier is persisted in `meta.json`'s `stats.embedding_model` (`write_meta`, from `PipelineResult.embedding_model`) and re-read by every consumer that needs to match it: the query side (`mcp/tools._get_query_embedding`, via `embedder.tier_from_meta(storage.data_dir)` + `encode_query`), the lazy worker (re-derives from meta every generation — it's a detached subprocess with no CLI arg), and daemon rebuilds (`pipeline.build_full_index(..., tier=None)` self-derives from meta.json for the watcher's routine global phase and the primary's socket `reindex` handler; an explicit override flows through only from a socket-delegated `analyze --embedding-model`, via `SocketClient.reindex(embedding_model=...)`). `LadybugBackend.vector_search` guards against querying a stored index with the wrong tier's vector width — it peeks one stored row's actual length (never the schema's `EMBEDDING_DIM` placeholder, which only matters for a still-empty table) and raises a clear `RuntimeError` pointing at `synaptiq analyze` instead of letting the native FLOAT[N] cast fail cryptically or (worse) letting both the HNSW and full-scan query paths silently swallow the mismatch into an empty result. Switching tiers always forces a full re-encode: `embedder._partition_texts` salts each node's `text_sha` with the tier's model id (not just the generated text), so a "quality"-encoded vector's hash never accidentally matches what a "fast" encode of the same text would produce — `partition_embeddings` therefore returns zero reused vectors right after a tier switch instead of risking a store with mixed 256-dim/384-dim rows in the same `FLOAT[dim]` column.

Note: all edges are stored in a single LadybugDB rel table group `CodeRelation` with the
logical kind in its `rel_type` property — Cypher must filter on `r.rel_type`, not
use logical labels like `[:CALLS]`. Node `properties` dicts are persisted in the
`properties_json` column.

### Storage Layer

`src/synaptiq/core/storage/base.py` defines a `StorageBackend` Protocol. The default implementation is `ladybug_backend.py` (LadybugDB — embedded graph DB with Cypher, FTS, and vector support). LadybugDB is the actively-maintained, API-drop-in successor to the archived KuzuDB; the full replacement was an owner decision (W2.7) — see `docs/plans/2026-07-12-storage-successor-evaluation.md`. There is no legacy kuzu support: an old kuzu-format index (a directory; LadybugDB uses a single-file format) fails to open and `open_with_recovery` rebuilds it from source. Optional Neo4j backend available via `synaptiq[neo4j]`.

Data stored in `.synaptiq/` directory within each indexed repo. The on-disk index path is kept as `.synaptiq/kuzu` deliberately — reusing the path lets `analyze` detect a former-KuzuDB index on open and rebuild it in place.

### Search

`src/synaptiq/core/search/hybrid.py` implements BM25 + vector (384-dim BAAI/bge-small-en-v1.5 via fastembed) + fuzzy search fused with Reciprocal Rank Fusion. Vector search is served by a LadybugDB HNSW index (`embedding_vec_idx`, cosine metric) rebuilt by `store_embeddings`; a database whose index build failed falls back to a full `array_cosine_similarity` scan. The index pins the Embedding table — drop it before `DROP TABLE` or column updates.

### Resource Profiles

`src/synaptiq/core/resources.py` defines role-aware engine limits. Long-running daemons (`serve`, `mcp`, `watch`) call `set_profile("server")` at entry and get strict caps: LadybugDB task-scheduler threads `max(2, cores//4)`, 512 MB buffer pool, capped ONNX embedding threads, and the walk/parse worker pool also capped to `max(2, cores//4)` (vs. the interactive `min(8, cores)`). One-shot CLI commands (`analyze`, `query`, ...) keep library defaults (all cores, LadybugDB's default buffer pool). `LadybugBackend.initialize()` and the embedders read `current_limits()` at creation time — set the profile before creating either. Env overrides: `SYNAPTIQ_DB_THREADS`, `SYNAPTIQ_DB_MEMORY_MB`, `SYNAPTIQ_EMBED_THREADS` (the former `SYNAPTIQ_KUZU_THREADS` / `SYNAPTIQ_KUZU_MEMORY_MB` still work as deprecated aliases for one release, logging a one-time warning).

### Multi-Instance Concurrency

`src/synaptiq/core/daemon/` implements a primary/proxy pattern for concurrent MCP sessions:
- `lock.py` — `fcntl.flock()` based lock file manager
- `socket_server.py` — async Unix domain socket server (primary)
- `socket_client.py` — async Unix domain socket client (proxy)

The `serve` command auto-detects role at startup. Design doc: `docs/plans/2026-02-23-multi-instance-concurrency-design.md`.

### MCP Server

`src/synaptiq/mcp/server.py` exposes tools (query, context, impact, dead_code, detect_changes, cypher, list_repos) and resources (overview, dead-code, schema) via MCP. Supports both stdio and Streamable HTTP transport (`--transport http`). HTTP transport is implemented in `src/synaptiq/mcp/http_transport.py` using Starlette + uvicorn. Proxy mode forwards calls through `SocketClient`.

### CLI

`src/synaptiq/cli/main.py` — single Typer app with commands: analyze, status, list, clean, query, context, impact, dead-code, cypher, watch, diff, setup, serve/mcp. Includes a non-blocking PyPI update notifier (`cli/update_check.py`) that caches checks for 24h.

### Parsers

`src/synaptiq/core/parsers/` — `BaseParser` base class in `base.py`, with `python_lang.py`, `typescript.py`, `ruby_lang.py`, and `go_lang.py` implementations. New language parsers extend `BaseParser` and register in `config/languages.py`.

Ruby support recognizes `.rb`, `.rake`, `.gemspec`, `.ru`, `.rbi` extensions plus suffix-less special files (`Rakefile`, `Gemfile`, `Guardfile`, `Capfile`, `Vagrantfile`, `Brewfile`, `Podfile`) via `SPECIAL_FILENAMES`. The parser emits `module` symbols (→ `MODULE` nodes) and a heritage `kind="mixin"` for `include`/`extend`/`prepend` (→ `MIXES_IN` edges); `class A < B` stays `EXTENDS`. Plain Ruby has no type annotations, so `types.py`/`USES_TYPE` emission is out of scope (future Sorbet/RBS work).

Go support (`.go`) is single-pass. Symbol mapping: `func` → `function`; receiver funcs → `method` with `class_name` = the receiver type (pointer/generic stripped); `type X struct` → `class`; `type X interface` → `interface`; `type X = Y` / `type X Y` → `type_alias`; top-level `const`/`var` → `constant` (an unmapped kind — parsed, not materialised); each file's `package` clause → a `module` node. Struct embedding (anonymous fields) and interface embedding (`type_elem`) → `EXTENDS`; interface satisfaction is deliberately **not** modeled (no `IMPLEMENTS`). Go's exported convention (upper-case first letter) is surfaced via `ParseResult.exports` → `is_exported`. Import resolution (`imports.py`) maps a package import path to every `.go` file in the matching package directory (directory-suffix match — the go.mod module prefix is stripped implicitly, no manifest parse needed), one IMPORTS edge per file. Composite literals (`T{}` / `&T{}`) emit a call to the type (the Go analogue of `new`). `vendor/` is pruned by the walker; `_test.go` files are indexed (dead-code exemptions cover their `Test*` funcs).

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

tree-sitter (parsing; language grammars: tree-sitter-python/-javascript/-typescript/-ruby/-go), ladybug (LadybugDB graph DB), igraph+leidenalg (community detection), fastembed (ONNX embeddings, "quality" tier), model2vec (static embeddings, "fast" tier — optional, `synaptiq[fast-embeddings]`), mcp SDK (FastMCP), typer+rich (CLI), watchfiles (file watcher), pathspec (gitignore).
