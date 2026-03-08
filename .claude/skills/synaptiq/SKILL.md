---
name: synaptiq
description: MUST consult this skill before using any Synaptiq MCP tool (synaptiq_query, synaptiq_context, synaptiq_impact, synaptiq_dead_code, synaptiq_detect_changes, synaptiq_cypher, synaptiq_list_repos) or MCP resource (synaptiq://overview, synaptiq://dead-code, synaptiq://schema). Contains the knowledge graph schema, Cypher query patterns, tool parameters, output formats, and investigation workflows needed to use Synaptiq effectively. Use whenever investigating code structure, call graphs, blast radius, dead code, file coupling, refactoring impact, cross-package dependencies, or architectural boundaries. Triggers on "what calls this", "what breaks if I change", "show dependencies", "find dead code", "blast radius", "which files change together", "trace the flow", "how is X connected", "who uses this function", "show callers", "show callees", "impact analysis", "is this code used", "structural diff", "cross-package imports", or any structural codebase question that goes beyond simple grep/glob. Also use when writing custom Cypher queries against the code graph.
---

# Synaptiq — Code Intelligence via Knowledge Graph

Synaptiq indexes the codebase into a structural knowledge graph. Every function, class, import, call, type reference, and execution flow is a node or edge you can query. The graph understands relationships that text search cannot — who calls what, what breaks if you change something, which symbols are dead code, and how the architecture clusters into functional communities.

Use Synaptiq MCP tools instead of grepping when you need **structural** understanding. Grep finds text; Synaptiq finds relationships.

## Investigation Workflow

Follow this progression — each step builds context for the next:

```
1. synaptiq_query(query="authentication handler")     → Find relevant symbols
2. synaptiq_context(symbol="handleAuthError")          → See callers, callees, types
3. synaptiq_impact(symbol="handleAuthError", depth=3)  → Blast radius before changes
```

**Always check impact before modifying a symbol that other code depends on.** This prevents surprise breakage in files you didn't know were affected.

## Tool Selection Guide

| Question | Tool |
|----------|------|
| "Find symbols related to X" | `synaptiq_query` |
| "What calls this? What does it call?" | `synaptiq_context` |
| "What breaks if I change this?" | `synaptiq_impact` |
| "What code is never called?" | `synaptiq_dead_code` |
| "Map this diff to affected symbols" | `synaptiq_detect_changes` |
| "Custom graph query" | `synaptiq_cypher` |
| "What repos are indexed?" | `synaptiq_list_repos` |

### When NOT to Use Synaptiq

- Reading file contents → use `Read`
- Simple text search → use `Grep`
- Finding files by name → use `Glob`

Synaptiq understands **structure** (call graphs, type references, coupling), not file contents.

## MCP Tools Reference

### synaptiq_query

Hybrid search (BM25 + vector + fuzzy) across all indexed symbols. Start here when you don't know exact symbol names.

**Parameters:**
- `query` (string, required) — Natural language or symbol name
- `limit` (integer, default: 20) — Max results

**Output format:**
```
1. validateUser (Function) -- packages/auth/lib/validate.ts
   Export function that validates user credentials against...
2. UserModel (Class) -- packages/db/models/user.ts
   ...

Next: Use context() on a specific symbol for the full picture.
```

Test files are auto-down-ranked, source symbols are boosted. Results include file path, symbol type, and a snippet.

### synaptiq_context

360-degree view of a single symbol: who calls it, what it calls, its type references, and which architectural community it belongs to.

**Parameters:**
- `symbol` (string, required) — The symbol name as it appears in code

**Disambiguation:** When a name is common (e.g., `handler`, `get`, `execute`), `synaptiq_context` may resolve to the wrong symbol — often a File node instead of the Function you want. To avoid this:
1. Use `synaptiq_query` first to find all symbols with that name
2. Pick the specific one you need from the results
3. Pass the **exact** symbol name to `synaptiq_context`

For unique names like `handleAuthError` or `getPortfolioOverview`, direct lookup works reliably.

**Output format:**
```
Symbol: validateUser (Function)
File: packages/auth/lib/validate.ts:15-42
Signature: async function validateUser(token: string): Promise<User>

Callers (3):
  -> authMiddleware  apps/server/middleware/auth.ts:28
  -> loginHandler    apps/server/routers/auth.ts:45
  -> sessionCheck    packages/auth/lib/session.ts:12

Callees (2):
  -> findUserById    packages/db/queries/user.ts:10
  -> verifyToken     packages/auth/lib/jwt.ts:5

Type references (1):
  -> User            packages/db/models/user.ts

Next: Use impact() if planning changes to this symbol.
```

If the symbol is flagged as dead code, the output includes `Status: DEAD CODE (unreachable)`.

### synaptiq_impact

Blast radius analysis — traces through call graph, type references, and git coupling to find all symbols affected by a change. Uses BFS traversal through upstream callers.

**Parameters:**
- `symbol` (string, required) — The symbol to analyze
- `depth` (integer, default: 3) — BFS traversal depth

**Output format:**
```
Impact analysis for: validateUser (Function)
Depth: 3
Total affected symbols: 8

  1. authMiddleware (Function) -- apps/server/middleware/auth.ts:28
  2. loginHandler (Function) -- apps/server/routers/auth.ts:45
  ...

Tip: Review each affected symbol before making changes.
```

**Depth guidance:**
- `depth=1` — Direct callers only
- `depth=2` — Callers of callers (usually sufficient)
- `depth=3` — Default, good for broadly-used utilities
- `depth=4+` — Can return very large result sets; use sparingly

Returns "No upstream callers found" if the symbol has no callers (leaf node or entry point).

### synaptiq_dead_code

Multi-pass dead code detection that goes beyond "zero callers" — accounts for entry points, exports, decorators, overrides, and protocol conformance.

**Parameters:** None

**Output format:**
```
Dead Code Report (12 symbols)
----------------------------------------

  apps/server/routers/functions/old-calc.ts:
    - calculateLegacy (line 15)
    - formatOldResult (line 42)

  packages/shared/lib/deprecated.ts:
    - oldHelper (line 8)
```

Results are grouped by file. Review before cleanup — framework entry points (route handlers, middleware) and exported API surfaces may appear as false positives because the graph doesn't track dynamic dispatch or external consumers.

**Common false positives to expect:**
- **Constructors** — called implicitly via `new`, not tracked as direct calls
- **Event handlers** — referenced in JSX (`onClick={handler}`) or framework conventions
- **React components** — invoked via JSX, not function calls
- **Route loaders/actions** — called by React Router framework, not by application code

**Filtering large reports:** Use `synaptiq_cypher` to narrow results:
```cypher
-- Dead code in a specific package only
MATCH (n) WHERE n.is_dead = true AND n.file_path STARTS WITH 'apps/server/'
RETURN n.name, n.file_path, n.start_line ORDER BY n.file_path

-- Dead code excluding constructors
MATCH (n:Function) WHERE n.is_dead = true AND n.name <> 'constructor'
RETURN n.name, n.file_path ORDER BY n.file_path
```

### synaptiq_detect_changes

Parse a git diff to identify which indexed symbols are affected by the changes. Maps changed file/line ranges to the symbols defined at those locations.

**Parameters:**
- `diff` (string, required) — Raw `git diff` output

**Output format:**
```
Changed files: 3

  packages/auth/lib/validate.ts:
    - validateUser (Function) lines 15-42
    - parseToken (Function) lines 44-60

  apps/server/routers/auth.ts:
    (no indexed symbols in changed lines)

Total affected symbols: 2

Next: Use impact() on affected symbols to see downstream effects.
```

**Usage pattern:** Pipe git diff output directly:
```
# Get the diff first, then pass to detect_changes
git diff HEAD~3..HEAD
```

Note: Changes to string literals, comments, or whitespace outside of symbol definitions will show `(no indexed symbols in changed lines)` — this is correct behavior since no code structure was affected.

### synaptiq_cypher

Execute raw Cypher queries against the knowledge graph (read-only). Write operations (DELETE, DROP, CREATE, SET, MERGE) are rejected.

**Parameters:**
- `query` (string, required) — Cypher query

**Output format:**
```
Results (15 rows):

  1. validateUser | packages/auth/lib/validate.ts | 42
  2. authMiddleware | apps/server/middleware/auth.ts | 28
  ...
```

**IMPORTANT — KùzuDB Cypher dialect:** The graph backend is KùzuDB, not Neo4j. Key differences:
- **No typed relationship patterns.** `[r:CALLS]` will fail with "Table CALLS does not exist". All relationships are stored in a single `CodeRelation` table. Filter by `r.rel_type` property instead.
- **`type(r)` does not exist.** Use `r.rel_type` to get the relationship type.
- **`labels(n)` works** for node labels but returns empty in some edge contexts.
- **`NOT EXISTS { subquery }` works** as expected.
- **`keys(n)` works** to inspect available properties.
- **`CALL show_tables() RETURN *`** lists all node/relationship tables — useful for schema discovery.

See the **Cypher Patterns** section below for tested, working queries.

### synaptiq_list_repos

List all repositories that have been indexed by Synaptiq with their stats.

**Parameters:** None

**Output format:**
```
Indexed repositories (1):

  1. lvlp-app
     Path: /Users/stevicacanadi/projects/levelup/lvlp-app
     Files: 1871  Symbols: 8038  Relationships: 37502
```

## MCP Resources

Three read-only resources are available for quick reference without querying:

| Resource URI | Description |
|-------------|-------------|
| `synaptiq://overview` | Aggregate stats — node counts by type, relationship counts |
| `synaptiq://dead-code` | Full dead code report (same as `synaptiq_dead_code` tool) |
| `synaptiq://schema` | Complete graph schema reference |

## Knowledge Graph Schema

### Node Labels

`File` | `Folder` | `Function` | `Class` | `Method` | `Interface` | `TypeAlias` | `Community` | `Process`

Note: `Enum` and `Embedding` tables exist in the schema but are typically empty in TypeScript codebases.

### Node Properties

All node types share the same property set:

`id`, `name`, `file_path`, `start_line`, `end_line`, `content`, `signature`, `language`, `class_name`, `is_dead`, `is_entry_point`, `is_exported`

The `id` property uses the format `{label}:{relative_path}:{symbol_name}` (e.g., `function:packages/auth/lib/error-handler.ts:handleAuthError`).

### Relationship Model

All relationships are stored in a **single `CodeRelation` table** with a `rel_type` property that indicates the relationship kind. This is a KùzuDB design — there are no separate tables per relationship type.

**Relationship properties:** `rel_type`, `confidence`, `role`, `step_number`, `strength`, `co_changes`, `symbols`

**Relationship types** (filter via `r.rel_type`):

| `rel_type` value | Description | Key Properties |
|---|---|---|
| `contains` | Folder → File/Symbol hierarchy | — |
| `defines` | File → Symbol it defines | — |
| `calls` | Symbol → Symbol | `confidence` (0.0–1.0) |
| `imports` | File → File | `symbols` (list) |
| `extends` | Class → Class | — |
| `implements` | Class → Interface | — |
| `uses_type` | Symbol → Type | `role` (param/return/variable) |
| `member_of` | Symbol → Community | — |
| `step_in_process` | Symbol → Process | `step_number` |
| `coupled_with` | File ↔ File (temporal) | `strength`, `co_changes` |

Note: There is no `exports` relationship. Export tracking uses the `is_exported` node property instead.

### Node ID Format

```
{label}:{relative_path}:{symbol_name}
```

Examples:
- `function:apps/server/routers/functions/portfolio.ts:getPortfolioOverview`
- `class:packages/db/models/user.ts:User`
- `method:apps/web/pages/dashboard.tsx:loader`

## Cypher Patterns

All patterns below have been tested against the actual KùzuDB backend.

### Files that always change together (coupling)
```cypher
MATCH (a:File)-[r]->(b:File)
WHERE r.rel_type = 'coupled_with'
RETURN a.name, b.name, r.strength
ORDER BY r.strength DESC LIMIT 20
```

### Functions in a specific file
```cypher
MATCH (f:File)-[r]->(fn:Function)
WHERE r.rel_type = 'defines' AND f.file_path ENDS WITH 'error-handler.ts'
RETURN fn.name, fn.start_line
```

### Cross-community calls (architectural boundary violations)
```cypher
MATCH (a)-[r1]->(c1:Community),
      (b)-[r2]->(c2:Community),
      (a)-[r3]->(b)
WHERE r1.rel_type = 'member_of'
  AND r2.rel_type = 'member_of'
  AND r3.rel_type = 'calls'
  AND c1 <> c2
RETURN a.name, c1.name, b.name, c2.name
```

### All execution flows / processes
```cypher
MATCH (p:Process)
RETURN p.name
ORDER BY p.name
```

### Cross-package dependencies (monorepo-specific)
```cypher
MATCH (f1:File)-[r]->(f2:File)
WHERE r.rel_type = 'imports'
  AND f1.file_path STARTS WITH 'apps/'
  AND f2.file_path STARTS WITH 'packages/'
RETURN f1.file_path, f2.file_path, f1.name
ORDER BY f2.file_path
```

### Exported symbols never called or referenced
```cypher
MATCH (f:File)-[r]->(s)
WHERE r.rel_type = 'defines' AND s.is_exported = true
  AND NOT EXISTS { MATCH ()-[r2]->(s) WHERE r2.rel_type = 'calls' }
  AND NOT EXISTS { MATCH ()-[r3]->(s) WHERE r3.rel_type = 'uses_type' }
  AND s.is_entry_point = false
RETURN s.name, s.file_path
```

### Most-connected symbols (refactoring risk)
```cypher
MATCH (n)<-[r]-(caller)
WHERE r.rel_type = 'calls'
RETURN n.name, n.file_path, count(r) AS caller_count
ORDER BY caller_count DESC LIMIT 20
```

### Schema discovery
```cypher
-- List all tables (node + relationship)
CALL show_tables() RETURN *

-- Count nodes by label
MATCH (n) RETURN labels(n) AS label, count(*) AS cnt ORDER BY cnt DESC

-- Count relationships by type
MATCH ()-[r]->() RETURN DISTINCT r.rel_type, count(*) AS cnt ORDER BY cnt DESC

-- Inspect properties on a node
MATCH (n:Function) RETURN keys(n) AS props LIMIT 1

-- Inspect properties on relationships
MATCH ()-[r]->() RETURN keys(r) AS props LIMIT 1
```

## Troubleshooting

**"Symbol not found"** — The tool resolves symbols by exact name match first, then falls back to full-text search. If a symbol isn't found:
1. Try the exact function/class name (case-sensitive)
2. Use `synaptiq_query` to search broadly, then use the exact name from results in `synaptiq_context`

**Wrong symbol resolved** — Common names like `handler`, `get`, `execute`, `constructor` exist across many files. `synaptiq_context` may resolve to a File node or the wrong function. Use `synaptiq_query` first to find the specific symbol, then pass the unique name.

**"No upstream callers found"** — The symbol is a leaf node (entry point, route handler, or top-level export). This is expected for things like React components, route loaders, or tRPC procedure handlers that are invoked by the framework rather than called directly by other code.

**"Table X does not exist" in Cypher** — You used a typed relationship pattern like `[r:CALLS]`. KùzuDB stores all relationships in a single `CodeRelation` table. Use `[r]` with `WHERE r.rel_type = 'calls'` instead. See the Cypher Patterns section for working examples.

**Stale results** — The MCP server runs with `--watch` and auto-reindexes on file changes. If results seem stale:
```bash
synaptiq analyze .        # Incremental update
synaptiq analyze . --full # Full rebuild
```

**Large impact results** — If `synaptiq_impact` returns too many symbols, reduce `depth` to 1 or 2 to focus on direct callers only.

## CLI Commands (non-MCP)

The `synaptiq diff` CLI command does structural branch comparison (not exposed as MCP tool):
```bash
synaptiq diff main..feature-branch
```

Other useful CLI commands:
```bash
synaptiq status           # Show index stats for current repo
synaptiq list             # List all indexed repos
synaptiq analyze .        # Incremental reindex
synaptiq analyze . --full # Full rebuild
synaptiq setup --claude   # Generate MCP config
```

## Multi-Instance Concurrency

`synaptiq serve --watch` supports multiple concurrent MCP sessions (e.g., multiple Claude Code windows). The first instance becomes the primary daemon (owns the DB), and subsequent instances automatically proxy queries over a Unix socket. No configuration needed.

## Installation

```bash
uv tool install synaptiq
# or
pip install synaptiq
```

### MCP Setup

Add to `.mcp.json`:

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

Or run `synaptiq setup --claude` to auto-generate the config.
