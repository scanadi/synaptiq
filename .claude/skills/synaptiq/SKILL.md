---
name: synaptiq
description: MUST consult this skill before using any Synaptiq MCP tool (synaptiq_query, synaptiq_context, synaptiq_impact, synaptiq_dead_code, synaptiq_detect_changes, synaptiq_cypher). Contains the knowledge graph schema, Cypher query patterns, node ID formats, and investigation workflows needed to use Synaptiq effectively. Use whenever investigating code structure, call graphs, blast radius, dead code, file coupling, refactoring impact, or architectural boundaries. Triggers on "what calls this", "what breaks if I change", "show dependencies", "find dead code", "blast radius", "which files change together", "trace the flow", "how is X connected", or any structural codebase question that goes beyond simple grep/glob. Also use when writing custom Cypher queries against the code graph.
---

# Synaptiq — Code Intelligence via Knowledge Graph

Synaptiq indexes the codebase into a structural knowledge graph. Every function, class, import, call, type reference, and execution flow is a node or edge you can query. Use Synaptiq MCP tools instead of grepping when you need **structural** understanding of the code.

## When to Use Synaptiq

| Question | Tool |
|----------|------|
| "Find symbols related to X" | `synaptiq_query` |
| "What calls this? What does it call?" | `synaptiq_context` |
| "What breaks if I change this?" | `synaptiq_impact` |
| "What code is never called?" | `synaptiq_dead_code` |
| "Map this diff to affected symbols" | `synaptiq_detect_changes` |
| "Custom graph query" | `synaptiq_cypher` |
| "What repos are indexed?" | `synaptiq_list_repos` |

## When NOT to Use Synaptiq

- Reading file contents (use `Read`)
- Simple text search / grep (use `Grep`)
- Finding files by name (use `Glob`)
- Synaptiq understands **structure**, not file contents

## Investigation Workflow

Follow this natural progression — each step builds on the previous:

```
1. synaptiq_query("authentication handler")     → Find relevant symbols
2. synaptiq_context("validateUser")             → See callers, callees, types, community
3. synaptiq_impact("validateUser", depth=3)     → Blast radius before making changes
```

**Always check impact before modifying a symbol that other code depends on.**

## MCP Tools Reference

### synaptiq_query
**Hybrid search** (BM25 + vector + fuzzy) across all symbols.

```
synaptiq_query(query="payment processing", limit=20)
```

- Use natural language or symbol names
- Test files are auto-down-ranked, source symbols boosted
- Returns ranked results with file paths and types

### synaptiq_context
**360-degree view** of a single symbol: callers, callees, type references, community membership, execution flows.

```
synaptiq_context(symbol="UserModel")
```

- Use the symbol name as it appears in code
- Shows both incoming (who calls this) and outgoing (what it calls) relationships
- Reveals which community/cluster the symbol belongs to

### synaptiq_impact
**Blast radius analysis** — all symbols affected by changing the target.

```
synaptiq_impact(symbol="UserModel", depth=3)
```

- Traces through: call graph, type references, git coupling
- `depth` controls BFS traversal depth (default: 3)
- Use before refactoring to understand the full impact

### synaptiq_dead_code
**Unreachable symbols** detected via multi-pass analysis.

```
synaptiq_dead_code()
```

- Not just "zero callers" — accounts for entry points, exports, decorators, overrides, protocol conformance
- Results grouped by file
- Review before cleanup to avoid false positives on framework entry points

### synaptiq_detect_changes
**Map a git diff to affected symbols** in the knowledge graph.

```
synaptiq_detect_changes(diff="<raw git diff output>")
```

- Pass the output of `git diff` directly
- Returns which symbols were added, modified, or removed
- Useful for understanding what a PR or commit actually touches structurally

### synaptiq_cypher
**Raw Cypher queries** against the knowledge graph (read-only).

```
synaptiq_cypher(query="MATCH (n:Function) WHERE n.is_dead = true RETURN n.name, n.file_path")
```

Use for advanced queries not covered by other tools.

## Knowledge Graph Schema

### Node Labels
`File` | `Folder` | `Function` | `Class` | `Method` | `Interface` | `TypeAlias` | `Enum` | `Community` | `Process`

### Relationship Types
| Type | Description |
|------|-------------|
| `CONTAINS` | Folder → File/Symbol hierarchy |
| `DEFINES` | File → Symbol it defines |
| `CALLS` | Symbol → Symbol (has `confidence` 0.0–1.0) |
| `IMPORTS` | File → File (has `symbols` list) |
| `EXTENDS` | Class → Class |
| `IMPLEMENTS` | Class → Interface |
| `USES_TYPE` | Symbol → Type (has `role`: param/return/variable) |
| `EXPORTS` | File → Symbol |
| `MEMBER_OF` | Symbol → Community |
| `STEP_IN_PROCESS` | Symbol → Process (has `step_number`) |
| `COUPLED_WITH` | File → File (has `strength`, `co_changes`) |

### Node ID Format
```
{label}:{relative_path}:{symbol_name}
```
Examples: `function:src/auth/validate.ts:validateUser`, `class:src/models/user.ts:User`

## Common Cypher Patterns

**Files that always change together (coupling):**
```cypher
MATCH (a:File)-[r:COUPLED_WITH]->(b:File)
RETURN a.name, b.name, r.strength
ORDER BY r.strength DESC LIMIT 20
```

**All execution flows:**
```cypher
MATCH (p:Process)
RETURN p.name, p.properties
ORDER BY p.name
```

**Functions in a specific file:**
```cypher
MATCH (f:File)-[:DEFINES]->(fn:Function)
WHERE f.name ENDS WITH 'auth.ts'
RETURN fn.name
```

**Cross-community calls (architectural boundaries):**
```cypher
MATCH (a)-[:MEMBER_OF]->(c1:Community),
      (b)-[:MEMBER_OF]->(c2:Community),
      (a)-[:CALLS]->(b)
WHERE c1 <> c2
RETURN a.name, c1.label, b.name, c2.label
```

## Multi-Instance Concurrency

`synaptiq serve --watch` supports multiple concurrent MCP sessions (e.g., multiple Claude Code windows). The first instance becomes the primary daemon (owns the DB), and subsequent instances automatically proxy queries over a Unix socket. No configuration needed — it just works.

## Re-indexing

The MCP server runs with `--watch` by default, so the graph updates automatically as you edit code. If the index seems stale, re-analyze:

```bash
synaptiq analyze .        # Incremental update
synaptiq analyze . --full # Full rebuild
```

## Installation

```bash
pip install synaptiq
# or
uv add synaptiq
```

### MCP Setup

Add to `.claude/settings.json` or `.mcp.json`:

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

Or run `synaptiq setup --claude` to generate the config.

## Tips

- **Start broad, narrow down**: `synaptiq_query` first, then `synaptiq_context` on specific results
- **Check impact before refactoring**: Always run `synaptiq_impact` before changing widely-used symbols
- **Use Cypher for custom analysis**: The schema above gives you full query power
- **Combine with git diff**: Use `synaptiq_detect_changes` to understand PR scope structurally
- **Community = cluster**: Communities are auto-detected functional groups — useful for understanding architecture boundaries
