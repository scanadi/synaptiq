# Synaptiq

**Graph-powered code intelligence engine** — indexes your codebase into a knowledge graph and exposes it via MCP tools for AI agents and a CLI for developers.

```
synaptiq analyze .

Phase 1:  Walking files...               142 files found
Phase 3:  Parsing code...                142/142
Phase 5:  Tracing calls...               847 calls resolved
Phase 7:  Analyzing types...             234 type relationships
Phase 8:  Detecting communities...       8 clusters found
Phase 9:  Detecting execution flows...   34 processes found
Phase 10: Finding dead code...           12 unreachable symbols
Phase 11: Analyzing git history...       18 coupled file pairs

Done in 4.2s — 623 symbols, 1,847 edges, 8 clusters, 34 flows
```

Most code intelligence tools treat your codebase as flat text. Synaptiq builds a **structural graph** — every function, class, import, call, type reference, and execution flow becomes a node or edge in a queryable knowledge graph. AI agents using Synaptiq don't just search for keywords; they understand how your code is connected.

---

## Why Synaptiq?

**For AI agents (Claude Code, Cursor):**
- "What breaks if I change this function?" → blast radius via call graph + type references + git coupling
- "What code is never called?" → dead code detection with framework-aware exemptions
- "Show me the login flow end-to-end" → execution flow tracing from entry points through the call graph
- "Which files always change together?" → git history change coupling analysis

**For developers:**
- Instant answers to architectural questions without grepping through files
- Find dead code, tightly coupled files, and execution flows automatically
- Raw Cypher queries against your codebase's knowledge graph
- Watch mode that re-indexes on every save

**Zero cloud dependencies.** Everything runs locally — parsing, graph storage, embeddings, search. No API keys, no data leaving your machine.

---

## Features

### 11-Phase Analysis Pipeline

Synaptiq doesn't just parse your code — it builds a deep structural understanding through 11 sequential analysis phases:

| Phase | What It Does |
|-------|-------------|
| **File Walking** | Walks repo respecting `.gitignore`, filters by supported languages |
| **Structure** | Creates File/Folder hierarchy with CONTAINS relationships |
| **Parsing** | tree-sitter AST extraction — functions, classes, methods, interfaces, enums, type aliases |
| **Import Resolution** | Resolves import statements to actual files (relative, absolute, bare specifiers) |
| **Call Tracing** | Maps function calls with confidence scores (1.0 = exact match, 0.5 = fuzzy) |
| **Heritage** | Tracks class inheritance (EXTENDS) and interface implementation (IMPLEMENTS) |
| **Type Analysis** | Extracts type references from parameters, return types, and variable annotations |
| **Community Detection** | Leiden algorithm clusters related symbols into functional communities |
| **Process Detection** | Framework-aware entry point detection + BFS flow tracing |
| **Dead Code Detection** | Multi-pass analysis with override, protocol, and decorator awareness |
| **Change Coupling** | Git history analysis — finds files that always change together |

### Hybrid Search (BM25 + Vector + RRF)

Three search strategies fused with Reciprocal Rank Fusion:

- **BM25 full-text search** — fast exact name and keyword matching via KuzuDB FTS
- **Semantic vector search** — conceptual queries via 384-dim embeddings (BAAI/bge-small-en-v1.5)
- **Fuzzy name search** — Levenshtein fallback for typos and partial matches

Test files are automatically down-ranked (0.5x), source-level functions/classes boosted (1.2x).

### Dead Code Detection

Finds unreachable symbols with intelligence — not just "zero callers" but a multi-pass analysis:

1. **Initial scan** — flags functions/methods/classes with no incoming calls
2. **Exemptions** — entry points, exports, constructors, test code, dunder methods, `__init__.py` public symbols, decorated functions, `@property` methods
3. **Override pass** — un-flags methods that override non-dead base class methods (handles dynamic dispatch)
4. **Protocol conformance** — un-flags methods on classes conforming to Protocol interfaces
5. **Protocol stubs** — un-flags all methods on Protocol classes (interface contracts)

### Impact Analysis (Blast Radius)

When you're about to change a symbol, Synaptiq traces upstream through:
- **Call graph** — every function that calls this one, recursively
- **Type references** — every function that takes, returns, or stores this type
- **Git coupling** — files that historically change alongside this one

### Community Detection

Uses the [Leiden algorithm](https://www.nature.com/articles/s41598-019-41695-z) (igraph + leidenalg) to automatically discover functional clusters in your codebase. Each community gets a cohesion score and auto-generated label based on member file paths.

### Execution Flow Tracing

Detects entry points using framework-aware patterns:
- **Python**: `@app.route`, `@router.get`, `@click.command`, `test_*` functions, `__main__` blocks
- **JavaScript/TypeScript**: Express handlers, exported functions, `handler`/`middleware` patterns

Then traces BFS execution flows from each entry point through the call graph, classifying flows as intra-community or cross-community.

### Change Coupling (Git History)

Analyzes 6 months of git history to find hidden dependencies that static analysis misses:

```
coupling(A, B) = co_changes(A, B) / max(changes(A), changes(B))
```

Files with coupling strength ≥ 0.3 and 3+ co-changes get linked. Surfaces coupled files in impact analysis.

### Watch Mode

Live re-indexing powered by a Rust-based file watcher (watchfiles):

```bash
$ synaptiq watch
Watching /Users/you/project for changes...

[10:32:15] src/auth/validate.py modified → re-indexed (0.3s)
[10:33:02] 2 files modified → re-indexed (0.5s)
```

- File-local phases (parse, imports, calls, types) run immediately on change
- Global phases (communities, processes, dead code) batch every 30 seconds

### Multi-Instance Concurrency

Synaptiq supports multiple concurrent MCP sessions (e.g., swarm sub-agents) through a primary/proxy daemon architecture:

- **Primary process**: Owns the database, watcher, and socket server
- **Proxy processes**: Forward queries over Unix socket to primary
- **Automatic role detection**: Each instance self-detects its role at startup

### Branch Comparison

Structural diff between branches using git worktrees (no stashing required):

```bash
$ synaptiq diff main..feature

Symbols added (4):
  + process_payment (Function) -- src/payments/stripe.py
  + PaymentIntent (Class) -- src/payments/models.py

Symbols modified (2):
  ~ checkout_handler (Function) -- src/routes/checkout.py

Symbols removed (1):
  - old_charge (Function) -- src/payments/legacy.py
```

---

## Supported Languages

| Language | Extensions | Parser |
|----------|-----------|--------|
| Python | `.py` | tree-sitter-python |
| TypeScript | `.ts`, `.tsx` | tree-sitter-typescript |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | tree-sitter-javascript |

---

## Installation

```bash
# With pip
pip install synaptiq

# With uv (recommended)
uv add synaptiq

# With Neo4j backend support
pip install synaptiq[neo4j]
```

Requires **Python 3.11+**.

### From Source

```bash
git clone https://github.com/scanadi/synaptiq.git
cd synaptiq
uv sync --all-extras
uv run synaptiq --help
```

---

## Quick Start

### 1. Index Your Codebase

```bash
cd your-project
synaptiq analyze .
```

### 2. Query It

```bash
# Search for symbols
synaptiq query "authentication handler"

# Get full context on a symbol
synaptiq context validate_user

# Check blast radius before changing something
synaptiq impact UserModel --depth 3

# Find dead code
synaptiq dead-code

# Run a raw Cypher query
synaptiq cypher "MATCH (n:Function) WHERE n.is_dead = true RETURN n.name, n.file_path"
```

### 3. Keep It Updated

```bash
# Watch mode — re-indexes on every save
synaptiq watch

# Or re-analyze manually
synaptiq analyze .
```

---

## CLI Reference

```
synaptiq analyze [PATH]          Index a repository (default: current directory)
    --full                       Force full rebuild (skip incremental)

synaptiq status                  Show index status for current repo
synaptiq list                    List all indexed repositories
synaptiq clean                   Delete index for current repo
    --force / -f                 Skip confirmation prompt

synaptiq query QUERY             Hybrid search the knowledge graph
    --limit / -n N               Max results (default: 20)

synaptiq context SYMBOL          360-degree view of a symbol
synaptiq impact SYMBOL           Blast radius analysis
    --depth / -d N               BFS traversal depth (default: 3)

synaptiq dead-code               List all detected dead code
synaptiq cypher QUERY            Execute a raw Cypher query (read-only)

synaptiq watch                   Watch mode — live re-indexing on file changes
synaptiq diff BASE..HEAD         Structural branch comparison

synaptiq setup                   Print MCP configuration JSON
    --claude                     For Claude Code
    --cursor                     For Cursor

synaptiq mcp                     Start the MCP server (stdio transport)
synaptiq serve                   Start the MCP server (same as synaptiq mcp)
    --watch, -w                  Enable live file watching with auto-reindex
synaptiq --version               Print version
```

---

## MCP Integration

Synaptiq exposes its full intelligence as an MCP server, giving AI agents like Claude Code and Cursor deep structural understanding of your codebase.

### Setup for Claude Code

Add to your `.claude/settings.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "synaptiq": {
      "command": "synaptiq",
      "args": ["serve", "--watch"]
    }
  }
}
```

This starts the MCP server **with live file watching** — the knowledge graph updates automatically as you edit code. To run without watching, use `"args": ["mcp"]` instead.

Or run the setup helper:

```bash
synaptiq setup --claude
```

### Claude Code Skill (Optional)

For richer Claude Code integration, copy the bundled skill into your project:

```bash
cp -r "$(python -c 'import synaptiq; import pathlib; print(pathlib.Path(synaptiq.__file__).parent.parent.parent)')"/.claude/skills/synaptiq .claude/skills/
```

Or manually copy `.claude/skills/synaptiq/SKILL.md` from the [Synaptiq repo](https://github.com/scanadi/synaptiq/tree/main/.claude/skills/synaptiq). This teaches Claude when and how to use Synaptiq's tools, the knowledge graph schema, Cypher query patterns, and investigation workflows.

### Setup for Cursor

Add to your Cursor MCP settings:

```json
{
  "synaptiq": {
    "command": "synaptiq",
    "args": ["serve", "--watch"]
  }
}
```

Or run:

```bash
synaptiq setup --cursor
```

### MCP Tools

Once connected, your AI agent gets access to these tools:

| Tool | Description |
|------|-------------|
| `synaptiq_list_repos` | List all indexed repositories with stats |
| `synaptiq_query` | Hybrid search (BM25 + vector + fuzzy) across all symbols |
| `synaptiq_context` | 360-degree view — callers, callees, type refs, community, processes |
| `synaptiq_impact` | Blast radius — all symbols affected by changing the target |
| `synaptiq_dead_code` | List all unreachable symbols grouped by file |
| `synaptiq_detect_changes` | Map a `git diff` to affected symbols in the graph |
| `synaptiq_cypher` | Execute read-only Cypher queries against the knowledge graph |

Every tool response includes a **next-step hint** guiding the agent through a natural investigation workflow:

```
query → "Next: Use context() on a specific symbol for the full picture."
context → "Next: Use impact() if planning changes to this symbol."
impact → "Tip: Review each affected symbol before making changes."
```

### MCP Resources

| Resource URI | Description |
|-------------|-------------|
| `synaptiq://overview` | Node and relationship counts by type |
| `synaptiq://dead-code` | Full dead code report |
| `synaptiq://schema` | Graph schema reference for writing Cypher queries |

---

## Knowledge Graph Model

### Nodes

| Label | Description |
|-------|-------------|
| `File` | Source file |
| `Folder` | Directory |
| `Function` | Top-level function |
| `Class` | Class definition |
| `Method` | Method within a class |
| `Interface` | Interface / Protocol definition |
| `TypeAlias` | Type alias |
| `Enum` | Enumeration |
| `Community` | Auto-detected functional cluster |
| `Process` | Detected execution flow |

### Relationships

| Type | Description | Key Properties |
|------|-------------|----------------|
| `CONTAINS` | Folder → File/Symbol hierarchy | — |
| `DEFINES` | File → Symbol it defines | — |
| `CALLS` | Symbol → Symbol it calls | `confidence` (0.0–1.0) |
| `IMPORTS` | File → File it imports from | `symbols` (names list) |
| `EXTENDS` | Class → Class it extends | — |
| `IMPLEMENTS` | Class → Interface it implements | — |
| `USES_TYPE` | Symbol → Type it references | `role` (param/return/variable) |
| `EXPORTS` | File → Symbol it exports | — |
| `MEMBER_OF` | Symbol → Community it belongs to | — |
| `STEP_IN_PROCESS` | Symbol → Process it participates in | `step_number` |
| `COUPLED_WITH` | File → File that co-changes with it | `strength`, `co_changes` |

### Node ID Format

```
{label}:{relative_path}:{symbol_name}

Examples:
  function:src/auth/validate.py:validate_user
  class:src/models/user.py:User
  method:src/models/user.py:User.save
```

---

## Architecture

```
Source Code (.py, .ts, .js, .tsx, .jsx)
    │
    ▼
┌──────────────────────────────────────────────┐
│         Ingestion Pipeline (11 phases)        │
│                                               │
│  walk → structure → parse → imports → calls   │
│  → heritage → types → communities → processes │
│  → dead_code → coupling                       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ KnowledgeGraph  │  (in-memory during build)
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐
     │ KuzuDB  │ │  FTS    │ │ Vector  │
     │ (graph) │ │ (BM25)  │ │ (HNSW)  │
     └────┬────┘ └────┬────┘ └────┬────┘
          └────────────┼────────────┘
                       │
              StorageBackend Protocol
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ┌──────────┐     ┌──────────┐
        │   MCP    │     │   CLI    │
        │  Server  │     │ (Typer)  │
        │ (stdio)  │     │          │
        └────┬─────┘     └────┬─────┘
             │                │
        Claude Code      Terminal
        / Cursor         (developer)
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Parsing | tree-sitter | Language-agnostic AST extraction |
| Graph Storage | KuzuDB | Embedded graph database with Cypher, FTS, and vector support |
| Graph Algorithms | igraph + leidenalg | Leiden community detection |
| Embeddings | fastembed | ONNX-based 384-dim vectors (~100MB, no PyTorch) |
| MCP Protocol | mcp SDK (FastMCP) | AI agent communication via stdio |
| CLI | Typer + Rich | Terminal interface with progress bars |
| File Watching | watchfiles | Rust-based file system watcher |
| Gitignore | pathspec | Full `.gitignore` pattern matching |

### Storage

Everything lives locally in your repo:

```
your-project/
└── .synaptiq/
    ├── kuzu/          # KuzuDB graph database (graph + FTS + vectors)
    └── meta.json      # Index metadata and stats
```

Add `.synaptiq/` to your `.gitignore`.

The storage layer is abstracted behind a `StorageBackend` Protocol — KuzuDB is the default, with an optional Neo4j backend available via `pip install synaptiq[neo4j]`.

---

## Example Workflows

### "I need to refactor the User class — what breaks?"

```bash
# See everything connected to User
synaptiq context User

# Check blast radius
synaptiq impact User --depth 3

# Find which files always change with user.py
synaptiq cypher "MATCH (a:File)-[r:CodeRelation]->(b:File) WHERE a.name = 'user.py' AND r.rel_type = 'coupled_with' RETURN b.name, r.strength ORDER BY r.strength DESC"
```

### "Is there dead code we should clean up?"

```bash
synaptiq dead-code
```

### "What are the main execution flows in our app?"

```bash
synaptiq cypher "MATCH (p:Process) RETURN p.name, p.properties ORDER BY p.name"
```

### "Which parts of the codebase are most tightly coupled?"

```bash
synaptiq cypher "MATCH (a:File)-[r:CodeRelation]->(b:File) WHERE r.rel_type = 'coupled_with' RETURN a.name, b.name, r.strength ORDER BY r.strength DESC LIMIT 20"
```

---

## How It Compares

| Capability | grep/ripgrep | LSP | Synaptiq |
|-----------|-------------|-----|----------|
| Text search | Yes | No | Yes (hybrid BM25 + vector) |
| Go to definition | No | Yes | Yes (graph traversal) |
| Find all callers | No | Partial | Yes (full call graph with confidence) |
| Type relationships | No | Yes | Yes (param/return/variable roles) |
| Dead code detection | No | No | Yes (multi-pass, framework-aware) |
| Execution flow tracing | No | No | Yes (entry point → flow) |
| Community detection | No | No | Yes (Leiden algorithm) |
| Change coupling (git) | No | No | Yes (6-month co-change analysis) |
| Impact analysis | No | No | Yes (calls + types + git coupling) |
| AI agent integration | No | Partial | Yes (full MCP server) |
| Structural branch diff | No | No | Yes (node/edge level) |
| Watch mode | No | Yes | Yes (Rust-based, 500ms debounce) |
| Works offline | Yes | Yes | Yes |

---

## Development

```bash
git clone https://github.com/scanadi/synaptiq.git
cd synaptiq
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check src/

# Run from source
uv run synaptiq --help
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

## Acknowledgments

Synaptiq was originally inspired by and built upon [axon](https://github.com/harshkedia177/axon) by [@harshkedia177](https://github.com/harshkedia177). This project has since evolved into an independent codebase with its own architecture, features, and direction.

---

## License

MIT — see [LICENSE](LICENSE).
