"""MCP tool handler implementations for Synaptiq.

Each function accepts a storage backend and the tool-specific arguments,
performs the appropriate query, and returns a human-readable string suitable
for inclusion in an MCP ``TextContent`` response.
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

from synaptiq.core.cypher_guard import check_read_only
from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.ingestion.dead_code import _is_test_file
from synaptiq.core.memory import MemoryStore
from synaptiq.core.search.hybrid import hybrid_search
from synaptiq.core.storage.base import StorageBackend
from synaptiq.core.storage.ladybug_backend import deserialize_properties

logger = logging.getLogger(__name__)

MAX_TRAVERSE_DEPTH = 10


def _get_query_embedding(query: str, storage: StorageBackend) -> list[float] | None:
    """Generate a query embedding vector, returning None if unavailable.

    Encodes with whatever tier *storage*'s index was actually built with
    (:func:`~synaptiq.core.embeddings.embedder.tier_from_meta`, read from
    ``meta.json`` next to the database) rather than a hardcoded model — an
    index built with the "fast" tier must be queried with "fast" vectors,
    never "quality" ones, since the two have different widths (W4.4). The
    resolved model is cached by
    :func:`~synaptiq.core.embeddings.embedder._get_model` (keyed by tier
    name), so repeated calls in a long-running `serve`/`mcp` process reuse
    the already-loaded model instead of reloading it every query.
    """
    try:
        from synaptiq.core.embeddings.embedder import encode_query, tier_from_meta

        # duck-typed: only LadybugBackend exposes `data_dir` (see its
        # docstring) — any other backend, or a bare mock in tests, falls
        # back to the default tier via tier_from_meta(None).
        data_dir = getattr(storage, "data_dir", None)
        tier = tier_from_meta(data_dir)
        return encode_query(tier.name, query)
    except Exception:
        logger.debug("Query embedding generation failed", exc_info=True)
        return None


_DEPTH_LABELS: dict[int, str] = {
    1: "Direct callers (will break)",
    2: "Indirect (may break)",
}


def _heritage_rows(storage: StorageBackend, node_id: str) -> list:
    """EXTENDS/IMPLEMENTS parents of *node_id* as (name, file_path, rel_type) rows."""
    return (
        storage.execute_raw(
            "MATCH (n)-[r:CodeRelation]->(parent) "
            "WHERE n.id = $nid "
            "AND r.rel_type IN ['extends', 'implements'] "
            "RETURN parent.name, parent.file_path, r.rel_type",
            parameters={"nid": node_id},
        )
        or []
    )


def _community_names(storage: StorageBackend, node_id: str) -> list[str]:
    """Names of communities *node_id* belongs to."""
    rows = (
        storage.execute_raw(
            "MATCH (n)-[r:CodeRelation]->(c:Community) "
            "WHERE n.id = $nid AND r.rel_type = 'member_of' RETURN c.name",
            parameters={"nid": node_id},
        )
        or []
    )
    return [row[0] for row in rows if row and row[0]]


def _confidence_tag(confidence: float) -> str:
    """Return a visual confidence indicator for edge display."""
    if confidence >= 0.9:
        return ""
    if confidence >= 0.5:
        return " (~)"
    return " (?)"


def _resolve_symbol(storage: StorageBackend, symbol: str) -> list:
    """Resolve a symbol name to search results, preferring exact name matches."""
    if hasattr(storage, "exact_name_search"):
        results = storage.exact_name_search(symbol, limit=1)
        if results:
            return results
    return storage.fts_search(symbol, limit=1)


def handle_list_repos(registry_dir: Path | None = None) -> str:
    """List indexed repositories by scanning for .synaptiq directories.

    Scans the global registry directory (defaults to ``~/.synaptiq/repos``) for
    project metadata files and returns a formatted summary.

    Args:
        registry_dir: Directory containing repo metadata. If ``None``,
            defaults to ``~/.synaptiq/repos``.

    Returns:
        Formatted list of indexed repositories with stats, or a message
        indicating none were found.
    """
    use_cwd_fallback = registry_dir is None
    if registry_dir is None:
        registry_dir = Path.home() / ".synaptiq" / "repos"

    repos: list[dict[str, Any]] = []

    if registry_dir.exists():
        for meta_file in registry_dir.glob("*/meta.json"):
            try:
                data = json.loads(meta_file.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    if not repos and use_cwd_fallback:
        # Fall back: scan current directory for .synaptiq
        cwd_data = Path.cwd() / ".synaptiq" / "meta.json"
        if cwd_data.exists():
            try:
                data = json.loads(cwd_data.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                pass

    if not repos:
        return "No indexed repositories found. Run `synaptiq index` on a project first."

    lines = [f"Indexed repositories ({len(repos)}):"]
    lines.append("")
    for i, repo in enumerate(repos, 1):
        name = repo.get("name", "unknown")
        path = repo.get("path", "")
        stats = repo.get("stats", {})
        files = stats.get("files", "?")
        symbols = stats.get("symbols", "?")
        relationships = stats.get("relationships", "?")
        lines.append(f"  {i}. {name}")
        lines.append(f"     Path: {path}")
        lines.append(f"     Files: {files}  Symbols: {symbols}  Relationships: {relationships}")
        lines.append("")

    return "\n".join(lines)


def _group_by_process(results: list, storage: StorageBackend) -> dict[str, list]:
    """Map search results to their parent execution processes."""
    if not results:
        return {}
    node_ids = [r.node_id for r in results]
    try:
        node_to_process = storage.get_process_memberships(node_ids)
    except AttributeError:
        return {}
    groups: dict[str, list] = {}
    for r in results:
        pname = node_to_process.get(r.node_id)
        if pname:
            groups.setdefault(pname, []).append(r)
    return groups


def _format_query_results(results: list, groups: dict[str, list]) -> str:
    """Format search results with process grouping."""
    grouped_ids: set[str] = {r.node_id for group in groups.values() for r in group}
    ungrouped = [r for r in results if r.node_id not in grouped_ids]

    lines: list[str] = []
    counter = 1

    for process_name, proc_results in groups.items():
        lines.append(f"=== {process_name} ===")
        for r in proc_results:
            label = r.label.title() if r.label else "Unknown"
            lines.append(f"{counter}. {r.node_name} ({label}) -- {r.file_path}")
            if r.snippet:
                snippet = r.snippet[:200].replace("\n", " ").strip()
                lines.append(f"   {snippet}")
            counter += 1
        lines.append("")

    if ungrouped:
        if groups:
            lines.append("=== Other results ===")
        for r in ungrouped:
            label = r.label.title() if r.label else "Unknown"
            lines.append(f"{counter}. {r.node_name} ({label}) -- {r.file_path}")
            if r.snippet:
                snippet = r.snippet[:200].replace("\n", " ").strip()
                lines.append(f"   {snippet}")
            counter += 1
        lines.append("")

    lines.append("Next: Use context() on a specific symbol for the full picture.")
    return "\n".join(lines)


def handle_query(
    storage: StorageBackend,
    query: str,
    limit: int = 20,
    focus_files: list[str] | None = None,
) -> str:
    """Execute hybrid search and format results, grouped by execution process.

    Args:
        storage: The storage backend to search against.
        query: Text search query.
        limit: Maximum number of results (default 20, capped at 100).
        focus_files: Optional list of file paths to bias results toward
            via Personalized PageRank.

    Returns:
        Formatted search results grouped by process, with file, name, label,
        and snippet for each result.
    """
    limit = max(1, min(limit, 100))

    ppr_scores: dict[str, float] | None = None
    if focus_files:
        from synaptiq.core.search.pagerank import personalized_pagerank

        ppr_scores = personalized_pagerank(storage, focus_files)

    query_embedding = _get_query_embedding(query, storage)
    try:
        results = hybrid_search(
            query, storage, query_embedding=query_embedding, limit=limit, ppr_scores=ppr_scores
        )
    except RuntimeError as exc:
        # storage.vector_search's embedding-tier dimension guard (W4.4) — the
        # only exception hybrid_search's own call chain can raise (FTS/fuzzy
        # search already degrade internally rather than raising). Surface it
        # as the tool's text result, matching handle_cypher's pattern, rather
        # than letting it propagate as an MCP protocol-level error.
        return str(exc)
    if not results:
        return f"No results found for '{query}'."

    groups = _group_by_process(results, storage)
    return _format_query_results(results, groups)


def handle_context(
    storage: StorageBackend,
    symbol: str,
    focus_files: list[str] | None = None,
) -> str:
    """Provide a 360-degree view of a symbol.

    Looks up the symbol by name via full-text search, then retrieves its
    callers (with confidence), callees (with confidence), type references,
    heritage, and importers.

    Args:
        storage: The storage backend.
        symbol: The symbol name to look up.
        focus_files: Optional list of file paths to bias result ordering
            via Personalized PageRank.

    Returns:
        Formatted view including callers, callees, type refs, and guidance.
    """
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    # Compute PPR scores if focus_files provided.
    ppr_scores: dict[str, float] | None = None
    if focus_files:
        from synaptiq.core.search.pagerank import personalized_pagerank

        ppr_scores = personalized_pagerank(storage, focus_files)

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        return f"Symbol '{symbol}' not found."

    label_display = node.label.value.title() if node.label else "Unknown"
    lines = [f"Symbol: {node.name} ({label_display})"]
    lines.append(f"File: {node.file_path}:{node.start_line}-{node.end_line}")

    if node.signature:
        lines.append(f"Signature: {node.signature}")

    if node.is_dead:
        lines.append("Status: DEAD CODE (unreachable)")

    try:
        callers_raw = storage.get_callers_with_confidence(node.id)
    except (AttributeError, TypeError):
        callers_raw = [(c, 1.0) for c in storage.get_callers(node.id)]

    # Sort callers by PPR proximity if focus_files provided.
    if ppr_scores and callers_raw:
        callers_raw.sort(
            key=lambda pair: ppr_scores.get(pair[0].id, 0.0), reverse=True
        )

    if callers_raw:
        lines.append(f"\nCallers ({len(callers_raw)}):")
        for c, conf in callers_raw:
            tag = _confidence_tag(conf)
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

    try:
        callees_raw = storage.get_callees_with_confidence(node.id)
    except (AttributeError, TypeError):
        callees_raw = [(c, 1.0) for c in storage.get_callees(node.id)]

    # Sort callees by PPR proximity if focus_files provided.
    if ppr_scores and callees_raw:
        callees_raw.sort(
            key=lambda pair: ppr_scores.get(pair[0].id, 0.0), reverse=True
        )

    if callees_raw:
        lines.append(f"\nCallees ({len(callees_raw)}):")
        for c, conf in callees_raw:
            tag = _confidence_tag(conf)
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}{tag}")

    type_refs = storage.get_type_refs(node.id)
    if type_refs:
        lines.append(f"\nType references ({len(type_refs)}):")
        for t in type_refs:
            lines.append(f"  -> {t.name}  {t.file_path}")

    heritage_rows = _heritage_rows(storage, node.id)
    if heritage_rows:
        lines.append(f"\nHeritage ({len(heritage_rows)}):")
        for row in heritage_rows:
            parent_name = row[0] or "?"
            parent_file = row[1] or "?"
            rel = row[2] or "?"
            lines.append(f"  -> {rel}: {parent_name}  {parent_file}")

    if node.file_path:
        import_rows = (
            storage.execute_raw(
                "MATCH (a:File)-[r:CodeRelation]->(b:File) "
                "WHERE b.file_path = $fp "
                "AND r.rel_type = 'imports' "
                "RETURN a.file_path ORDER BY a.file_path",
                parameters={"fp": node.file_path},
            )
            or []
        )
        if import_rows:
            importers = [r[0] for r in import_rows if r[0]]
            lines.append(f"\nImported by ({len(importers)}):")
            for imp in importers:
                lines.append(f"  -> {imp}")

    lines.append("")
    lines.append("Next: Use impact() if planning changes to this symbol.")
    return "\n".join(lines)


def handle_impact(storage: StorageBackend, symbol: str, depth: int = 3) -> str:
    """Analyse the blast radius of changing a symbol, grouped by hop depth.

    Uses BFS traversal through CALLS edges to find all affected symbols
    up to the specified depth, then groups results by distance.

    Args:
        storage: The storage backend.
        symbol: The symbol name to analyse.
        depth: Maximum traversal depth (default 3).

    Returns:
        Formatted impact analysis with depth-grouped sections.
    """
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    depth = max(1, min(depth, MAX_TRAVERSE_DEPTH))

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        return f"Symbol '{symbol}' not found."

    affected_with_depth = storage.traverse_with_depth(start_node.id, depth, direction="callers")
    if not affected_with_depth:
        return f"No upstream callers found for '{symbol}'."

    by_depth: dict[int, list] = {}
    for node, d in affected_with_depth:
        by_depth.setdefault(d, []).append(node)

    total = len(affected_with_depth)
    label_display = start_node.label.value.title()
    lines = [f"Impact analysis for: {start_node.name} ({label_display})"]
    lines.append(f"Depth: {depth} | Total: {total} symbols")

    conf_lookup = {
        node.id: conf for node, conf in storage.get_callers_with_confidence(start_node.id)
    }

    counter = 1
    for d in sorted(by_depth.keys()):
        depth_label = _DEPTH_LABELS.get(d, "Transitive (review)")
        lines.append(f"\nDepth {d} — {depth_label}:")
        for node in by_depth[d]:
            label = node.label.value.title() if node.label else "Unknown"
            conf = conf_lookup.get(node.id)
            tag = f"  (confidence: {conf:.2f})" if conf is not None else ""
            lines.append(
                f"  {counter}. {node.name} ({label}) -- {node.file_path}:{node.start_line}{tag}"
            )
            counter += 1

    lines.append("")
    lines.append("Tip: Review each affected symbol before making changes.")
    return "\n".join(lines)


def handle_dead_code(storage: StorageBackend) -> str:
    """List all symbols marked as dead code.

    Delegates to :func:`~synaptiq.mcp.resources.get_dead_code_list` for the
    shared query and formatting.

    Args:
        storage: The storage backend.

    Returns:
        Formatted list of dead code symbols grouped by file.
    """
    from synaptiq.mcp.resources import get_dead_code_list

    return get_dead_code_list(storage)


_DIFF_FILE_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_DIFF_HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def _parse_diff_files(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a git diff and return {file_path: [(start, end), ...]}."""
    changed_files: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff.split("\n"):
        file_match = _DIFF_FILE_PATTERN.match(line)
        if file_match:
            current_file = file_match.group(2)
            if current_file not in changed_files:
                changed_files[current_file] = []
            continue

        hunk_match = _DIFF_HUNK_PATTERN.match(line)
        if hunk_match and current_file is not None:
            start = int(hunk_match.group(1))
            count = max(1, int(hunk_match.group(2) or "1"))
            changed_files[current_file].append((start, start + count - 1))

    return changed_files


def handle_detect_changes(storage: StorageBackend, diff: str) -> str:
    """Map git diff output to affected symbols.

    Parses the diff to find changed files and line ranges, then queries
    the storage backend to identify which symbols those lines belong to.

    Args:
        storage: The storage backend.
        diff: Raw git diff output string.

    Returns:
        Formatted list of affected symbols per changed file.
    """
    if not diff.strip():
        return "Empty diff provided."

    changed_files = _parse_diff_files(diff)

    if not changed_files:
        return "Could not parse any changed files from the diff."

    lines = [f"Changed files: {len(changed_files)}"]
    lines.append("")
    total_affected = 0

    for file_path, ranges in changed_files.items():
        affected_symbols = []
        rows = (
            storage.execute_raw(
                "MATCH (n) WHERE n.file_path = $fp "
                "AND n.start_line > 0 "
                "RETURN n.id, n.name, n.file_path, n.start_line, n.end_line",
                parameters={"fp": file_path},
            )
            or []
        )
        for row in rows:
            node_id = row[0] or ""
            name = row[1] or ""
            start_line = row[3] or 0
            end_line = row[4] or 0
            label_prefix = node_id.split(":", 1)[0] if node_id else ""
            for start, end in ranges:
                if start_line <= end and end_line >= start:
                    affected_symbols.append((name, label_prefix.title(), start_line, end_line))
                    break

        lines.append(f"  {file_path}:")
        if affected_symbols:
            for sym_name, label, s_line, e_line in affected_symbols:
                lines.append(f"    - {sym_name} ({label}) lines {s_line}-{e_line}")
                total_affected += 1
        else:
            lines.append("    (no indexed symbols in changed lines)")
        lines.append("")

    lines.append(f"Total affected symbols: {total_affected}")
    lines.append("")
    lines.append("Next: Use impact() on affected symbols to see downstream effects.")
    return "\n".join(lines)


def handle_cypher(storage: StorageBackend, query: str) -> str:
    """Execute a raw Cypher query and return formatted results.

    Only read-only queries are allowed.  Queries containing write keywords
    (DELETE, DROP, CREATE, SET, etc.) are rejected.  Comments are stripped
    before keyword scanning to prevent bypass via ``/* CREATE */``.

    Args:
        storage: The storage backend.
        query: The Cypher query string.

    Returns:
        Formatted query results, or an error message if execution fails.
    """
    rejection = check_read_only(query)
    if rejection is not None:
        return rejection

    try:
        rows = storage.execute_raw(query)
    except Exception as exc:
        return f"Cypher query failed: {exc}"

    if not rows:
        return "Query returned no results."

    lines = [f"Results ({len(rows)} rows):"]
    lines.append("")
    for i, row in enumerate(rows, 1):
        formatted_values = [str(v) for v in row]
        lines.append(f"  {i}. {' | '.join(formatted_values)}")

    return "\n".join(lines)


def handle_coupling(storage: StorageBackend, file_path: str, min_strength: float = 0.3) -> str:
    """Query temporal coupling for a file and flag hidden dependencies."""
    if not file_path or not file_path.strip():
        return "Error: 'file_path' parameter is required and cannot be empty."

    file_path = file_path.strip()

    # COUPLED_WITH edges are stored in one direction (sorted pair), so the
    # undirected pattern is required to find coupling from either side.
    rows = (
        storage.execute_raw(
            "MATCH (a:File)-[r:CodeRelation]-(b:File) "
            "WHERE a.file_path = $fp AND r.rel_type = 'coupled_with' "
            "RETURN b.file_path, r.strength, r.co_changes "
            "ORDER BY r.strength DESC",
            parameters={"fp": file_path},
        )
        or []
    )

    rows = [r for r in rows if (r[1] or 0) >= min_strength]

    if not rows:
        return f"No temporal coupling found for '{file_path}' (min strength: {min_strength})."

    import_rows = (
        storage.execute_raw(
            "MATCH (a:File)-[r:CodeRelation]->(b:File) "
            "WHERE a.file_path = $fp AND r.rel_type = 'imports' "
            "RETURN b.file_path",
            parameters={"fp": file_path},
        )
        or []
    )
    imported_files = {r[0] for r in import_rows}

    lines = [f"Temporal coupling for: {file_path}"]
    lines.append("=" * 48)
    lines.append("")

    for i, row in enumerate(rows, 1):
        coupled_path = row[0] or "?"
        strength = row[1] or 0.0
        co_changes = row[2] or 0
        has_import = coupled_path in imported_files
        import_flag = "imports: yes" if has_import else "imports: no \u26a0\ufe0f"
        lines.append(
            f"  {i}. {coupled_path}  strength: {strength:.2f}  "
            f"co_changes: {co_changes}  ({import_flag})"
        )

    lines.append("")
    hidden = [r[0] for r in rows if r[0] not in imported_files]
    if hidden:
        lines.append(
            f"\u26a0\ufe0f {len(hidden)} file(s) have hidden dependencies (no static import)."
        )
    return "\n".join(lines)


def handle_call_path(
    storage: StorageBackend, from_symbol: str, to_symbol: str, max_depth: int = 10
) -> str:
    """Find the shortest call chain between two symbols via BFS."""
    if not from_symbol or not from_symbol.strip():
        return "Error: 'from_symbol' parameter is required and cannot be empty."
    if not to_symbol or not to_symbol.strip():
        return "Error: 'to_symbol' parameter is required and cannot be empty."

    max_depth = max(1, min(max_depth, MAX_TRAVERSE_DEPTH))

    from_results = _resolve_symbol(storage, from_symbol)
    if not from_results:
        return f"Source symbol '{from_symbol}' not found."

    to_results = _resolve_symbol(storage, to_symbol)
    if not to_results:
        return f"Target symbol '{to_symbol}' not found."

    src_node = storage.get_node(from_results[0].node_id)
    tgt_node = storage.get_node(to_results[0].node_id)
    if not src_node or not tgt_node:
        return "Could not resolve one or both symbols."

    if src_node.id == tgt_node.id:
        return f"Source and target are the same symbol: {src_node.name}"

    parent: dict[str, str] = {}
    queue: deque[tuple[str, int]] = deque([(src_node.id, 0)])
    visited: set[str] = {src_node.id}

    found = False
    while queue:
        current_id, depth = queue.popleft()
        if depth >= max_depth:
            continue

        for callee in storage.get_callees(current_id):
            if callee.id in visited:
                continue
            visited.add(callee.id)
            parent[callee.id] = current_id

            if callee.id == tgt_node.id:
                found = True
                break

            queue.append((callee.id, depth + 1))

        if found:
            break

    if not found:
        return (
            f"No call path found from '{src_node.name}' to '{tgt_node.name}' "
            f"within {max_depth} hops."
        )

    path_ids: list[str] = []
    node_id = tgt_node.id
    while node_id is not None:
        path_ids.append(node_id)
        node_id = parent.get(node_id)
    path_ids.reverse()

    hop_count = len(path_ids) - 1
    path_names = []
    detail_lines = []
    for i, nid in enumerate(path_ids, 1):
        node = storage.get_node(nid)
        if node:
            label = node.label.value.title() if node.label else "Unknown"
            path_names.append(node.name)
            detail_lines.append(
                f"  {i}. {node.name} ({label}) \u2014 {node.file_path}:{node.start_line}"
            )
        else:
            path_names.append(nid)
            detail_lines.append(f"  {i}. {nid}")

    arrow = " \u2192 "
    header = f"Call path: {arrow.join(path_names)} ({hop_count} hop{'s' if hop_count != 1 else ''})"
    return header + "\n\n" + "\n".join(detail_lines)


def handle_communities(storage: StorageBackend, community: str | None = None) -> str:
    """List communities or drill into a specific one."""
    if community:
        rows = (
            storage.execute_raw(
                "MATCH (n)-[r:CodeRelation]->(c:Community) "
                "WHERE c.name = $cn AND r.rel_type = 'member_of' "
                "RETURN n.name, label(n), n.file_path, n.start_line, "
                "n.is_entry_point, n.is_exported "
                "ORDER BY n.file_path, n.start_line",
                parameters={"cn": community},
            )
            or []
        )

        if not rows:
            return f"Community '{community}' not found or has no members."

        lines = [f"Community: {community}"]
        lines.append(f"Members ({len(rows)}):")
        lines.append("")
        for row in rows:
            name = row[0] or "?"
            label = row[1] or "Unknown"
            file_path = row[2] or "?"
            start_line = row[3] or 0
            is_entry = row[4] if len(row) > 4 else False
            is_exported = row[5] if len(row) > 5 else False
            tags = []
            if is_entry:
                tags.append("entry point")
            if is_exported:
                tags.append("exported")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"  - {name} ({label}) \u2014 {file_path}:{start_line}{tag_str}")

        return "\n".join(lines)

    rows = (
        storage.execute_raw(
            "MATCH (c:Community) RETURN c.name, c.properties_json"
        )
        or []
    )

    if not rows:
        return "No communities detected. Run indexing with community detection enabled."

    # Cohesion and symbol_count live in the properties_json column;
    # parse and sort in Python.
    communities_parsed: list[tuple[str, float, Any]] = []
    for row in rows:
        name = row[0] or "?"
        props = deserialize_properties(row[1])
        cohesion = float(props.get("cohesion", 0.0) or 0.0)
        symbol_count = props.get("symbol_count", "?")
        communities_parsed.append((name, cohesion, symbol_count))

    communities_parsed.sort(key=lambda t: t[1], reverse=True)

    lines = [f"Communities ({len(communities_parsed)} detected):"]
    lines.append("")
    for i, (name, cohesion, symbol_count) in enumerate(communities_parsed, 1):
        lines.append(f"  {i}. {name}  (cohesion: {cohesion:.2f}, {symbol_count} symbols)")

    cross_procs = (
        storage.execute_raw(
            "MATCH (n)-[r1:CodeRelation]->(p:Process), "
            "(n)-[r2:CodeRelation]->(c:Community) "
            "WHERE r1.rel_type = 'step_in_process' AND r2.rel_type = 'member_of' "
            "WITH p.name AS proc, collect(DISTINCT c.name) AS comms "
            "WHERE size(comms) > 1 "
            "RETURN proc, comms"
        )
        or []
    )

    if cross_procs:
        lines.append("")
        lines.append("Cross-community processes:")
        for row in cross_procs:
            proc_name = row[0] or "?"
            comms = row[1] if len(row) > 1 else []
            comm_str = " \u2192 ".join(comms) if isinstance(comms, list) else str(comms)
            lines.append(f"  - {proc_name} ({comm_str})")

    return "\n".join(lines)


def handle_explain(storage: StorageBackend, symbol: str) -> str:
    """Produce a narrative explanation of a symbol."""
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        return f"Symbol '{symbol}' not found."

    label_display = node.label.value.title() if node.label else "Unknown"
    lines = [f"Explanation: {node.name} ({label_display})"]
    lines.append("=" * 48)
    lines.append("")

    roles = []
    if node.is_entry_point:
        roles.append("Entry point")
    if node.is_exported:
        roles.append("Exported")
    if node.is_dead:
        roles.append("Dead code (unreachable)")
    if roles:
        lines.append(f"Role: {', '.join(roles)}")

    lines.append(f"Location: {node.file_path}:{node.start_line}-{node.end_line}")

    if node.signature:
        lines.append(f"Signature: {node.signature}")

    communities = _community_names(storage, node.id)
    if communities:
        lines.append(f"Community: {communities[0]}")

    lines.append("")

    try:
        callers = storage.get_callers_with_confidence(node.id)
    except (AttributeError, TypeError):
        callers = [(c, 1.0) for c in storage.get_callers(node.id)]

    try:
        callees = storage.get_callees_with_confidence(node.id)
    except (AttributeError, TypeError):
        callees = [(c, 1.0) for c in storage.get_callees(node.id)]

    if callers:
        caller_names = ", ".join(c.name for c, _ in callers[:5])
        suffix = f" (+{len(callers) - 5} more)" if len(callers) > 5 else ""
        lines.append(f"Called by {len(callers)}: {caller_names}{suffix}")
    else:
        lines.append("Called by: nothing (root or dead)")

    if callees:
        callee_names = ", ".join(c.name for c, _ in callees[:5])
        suffix = f" (+{len(callees) - 5} more)" if len(callees) > 5 else ""
        lines.append(f"Calls {len(callees)}: {callee_names}{suffix}")
    else:
        lines.append("Calls: nothing (leaf)")

    proc_rows = (
        storage.execute_raw(
            "MATCH (n)-[r:CodeRelation]->(p:Process) "
            "WHERE n.id = $nid AND r.rel_type = 'step_in_process' RETURN p.name",
            parameters={"nid": node.id},
        )
        or []
    )
    if proc_rows:
        lines.append("")
        lines.append("Process flows through this symbol:")
        for row in proc_rows:
            proc_name = row[0] or "?"
            lines.append(f"  - {proc_name}")

    return "\n".join(lines)


def handle_review_risk(storage: StorageBackend, diff: str) -> str:
    """Assess PR risk by synthesizing multiple graph signals."""
    if not diff.strip():
        return "Empty diff provided."

    changed_files = _parse_diff_files(diff)
    if not changed_files:
        return "Could not parse any changed files from the diff."

    changed_file_set = set(changed_files.keys())
    # (node_id, name, label, file_path, dep_count) — the real node id is
    # carried through so later lookups never have to reconstruct it from
    # the bare name (method ids are ClassName-qualified, duplicates carry
    # a #L suffix; reconstruction misses both).
    all_affected_symbols: list[tuple[str, str, str, str, int]] = []
    entry_points_hit = 0
    total_dependents = 0

    for file_path, ranges in changed_files.items():
        rows = (
            storage.execute_raw(
                "MATCH (n) WHERE n.file_path = $fp "
                "AND n.start_line > 0 "
                "RETURN n.id, n.name, n.file_path, n.start_line, n.end_line",
                parameters={"fp": file_path},
            )
            or []
        )

        for row in rows:
            node_id = row[0] or ""
            name = row[1] or ""
            start_line = row[3] or 0
            end_line = row[4] or 0
            label_prefix = node_id.split(":", 1)[0].title() if node_id else ""

            hit = any(start_line <= end and end_line >= start for start, end in ranges)
            if not hit:
                continue

            node = storage.get_node(node_id)
            dep_count = 0
            if node:
                deps = storage.traverse_with_depth(node.id, 2, direction="callers")
                dep_count = len(deps)
                if node.is_entry_point:
                    entry_points_hit += 1

            total_dependents += dep_count
            all_affected_symbols.append((node_id, name, label_prefix, file_path, dep_count))

    missing_cochange: list[tuple[str, str, float]] = []
    for file_path in changed_files:
        coupling_rows = (
            storage.execute_raw(
                "MATCH (a:File)-[r:CodeRelation]-(b:File) "
                "WHERE a.file_path = $fp AND r.rel_type = 'coupled_with' "
                "AND r.strength >= 0.5 "
                "RETURN b.file_path, r.strength",
                parameters={"fp": file_path},
            )
            or []
        )
        for row in coupling_rows:
            coupled_file = row[0] or ""
            strength = row[1] or 0.0
            if coupled_file not in changed_file_set:
                missing_cochange.append((coupled_file, file_path, strength))

    communities_touched: set[str] = set()
    for node_id, _name, _label, _file_path, _deps in all_affected_symbols:
        communities_touched.update(_community_names(storage, node_id))

    score = entry_points_hit + len(missing_cochange) + total_dependents // 10
    if len(communities_touched) > 1:
        score += 2
    score = min(score, 10)

    if score <= 3:
        level = "LOW"
    elif score <= 6:
        level = "MEDIUM"
    else:
        level = "HIGH"

    lines = ["PR Risk Assessment"]
    lines.append("=" * 48)
    lines.append(f"Risk: {level} (score: {score}/10)")
    lines.append("")

    if all_affected_symbols:
        lines.append(f"Changed symbols ({len(all_affected_symbols)}):")
        for _node_id, name, label, fp, deps in all_affected_symbols:
            tags = []
            if deps > 0:
                tags.append(f"{deps} downstream dependents")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"  - {name} ({label}) \u2014 {fp}{tag_str}")
    else:
        lines.append("No indexed symbols in changed lines.")

    if missing_cochange:
        lines.append("")
        lines.append("\u26a0\ufe0f Missing co-change files (usually change together):")
        for missing, coupled_with, strength in missing_cochange:
            lines.append(f"  - {missing} (strength: {strength:.2f} with {coupled_with})")

    if len(communities_touched) > 1:
        lines.append("")
        lines.append(f"Community boundary crossings: {len(communities_touched)}")
        lines.append(f"  Spans: {', '.join(sorted(communities_touched))}")

    return "\n".join(lines)


def handle_file_context(storage: StorageBackend, file_path: str) -> str:
    """Provide comprehensive context for a single file."""
    if not file_path or not file_path.strip():
        return "Error: 'file_path' parameter is required and cannot be empty."

    file_path = file_path.strip()
    params = {"fp": file_path}

    sym_rows = (
        storage.execute_raw(
            "MATCH (n) WHERE n.file_path = $fp AND n.start_line > 0 "
            "RETURN n.name, label(n), n.start_line, n.is_dead, n.is_entry_point, n.is_exported "
            "ORDER BY n.start_line",
            parameters=params,
        )
        or []
    )

    imports_out = (
        storage.execute_raw(
            "MATCH (a:File)-[r:CodeRelation]->(b:File) "
            "WHERE a.file_path = $fp AND r.rel_type = 'imports' "
            "RETURN b.file_path ORDER BY b.file_path",
            parameters=params,
        )
        or []
    )

    imports_in = (
        storage.execute_raw(
            "MATCH (a:File)-[r:CodeRelation]->(b:File) "
            "WHERE b.file_path = $fp AND r.rel_type = 'imports' "
            "RETURN a.file_path ORDER BY a.file_path",
            parameters=params,
        )
        or []
    )

    coupling_rows = (
        storage.execute_raw(
            "MATCH (a:File)-[r:CodeRelation]-(b:File) "
            "WHERE a.file_path = $fp AND r.rel_type = 'coupled_with' "
            "RETURN b.file_path, r.strength, r.co_changes "
            "ORDER BY r.strength DESC LIMIT 5",
            parameters=params,
        )
        or []
    )

    dead_rows = (
        storage.execute_raw(
            "MATCH (n) WHERE n.is_dead = true AND n.file_path = $fp "
            "RETURN n.name, n.start_line, label(n)",
            parameters=params,
        )
        or []
    )

    comm_rows = (
        storage.execute_raw(
            "MATCH (n)-[r:CodeRelation]->(c:Community) "
            "WHERE n.file_path = $fp AND r.rel_type = 'member_of' "
            "RETURN c.name, count(n) ORDER BY count(n) DESC",
            parameters=params,
        )
        or []
    )

    if not sym_rows and not imports_out and not imports_in:
        return f"No data found for file '{file_path}'. Is it indexed?"

    lines = [f"File: {file_path}"]
    lines.append("=" * 48)

    if sym_rows:
        lines.append("")
        lines.append(f"Symbols ({len(sym_rows)}):")
        for row in sym_rows:
            name = row[0] or "?"
            label = row[1] or "Unknown"
            start_line = row[2] or 0
            is_dead = row[3] if len(row) > 3 else False
            is_entry = row[4] if len(row) > 4 else False
            is_exported = row[5] if len(row) > 5 else False
            tags = []
            if is_entry:
                tags.append("entry point")
            if is_exported:
                tags.append("exported")
            if is_dead:
                tags.append("dead")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            lines.append(f"  - {name} ({label}) line {start_line}{tag_str}")

    if imports_out:
        out_paths = [r[0] for r in imports_out if r[0]]
        lines.append("")
        lines.append(f"Imports ({len(out_paths)}): {', '.join(out_paths)}")

    if imports_in:
        in_paths = [r[0] for r in imports_in if r[0]]
        lines.append(f"Imported by ({len(in_paths)}): {', '.join(in_paths)}")

    if coupling_rows:
        lines.append("")
        lines.append(f"Coupled files ({len(coupling_rows)}):")
        for row in coupling_rows:
            coupled_path = row[0] or "?"
            strength = row[1] or 0.0
            co_changes = row[2] or 0
            lines.append(f"  - {coupled_path}  strength: {strength:.2f}  co_changes: {co_changes}")

    if dead_rows:
        lines.append("")
        lines.append(f"Dead code ({len(dead_rows)}):")
        for row in dead_rows:
            name = row[0] or "?"
            start_line = row[1] or 0
            label = row[2] or "Unknown"
            lines.append(f"  - {name} ({label}) line {start_line}")

    if comm_rows:
        lines.append("")
        comm_parts = [f"{r[0]} ({r[1]} symbols)" for r in comm_rows if r[0]]
        lines.append(f"Communities: {', '.join(comm_parts)}")

    return "\n".join(lines)


def handle_cycles(storage: StorageBackend, min_size: int = 2) -> str:
    """Detect circular dependencies using strongly connected components.

    The graph load and SCC decomposition only change when the index is
    rebuilt, so the expensive part is memoized per storage *generation*
    (see :attr:`~synaptiq.core.storage.ladybug_backend.LadybugBackend.generation`)
    \u2014 mirrors the projection cache in ``core/search/pagerank.py``.  Only the
    cheap ``min_size`` filter and text formatting run on every call.
    """
    min_size = max(2, min_size)
    generation = getattr(storage, "generation", 0)

    try:
        graph, groups = _cached_scc_groups(storage, generation)
    except Exception as exc:
        return f"Error loading graph: {exc}"

    if not groups:
        return "No symbols in the graph to analyze."

    cycles = [group for group in groups if len(group) >= min_size]

    if not cycles:
        return "No circular dependencies detected."

    cycles.sort(key=len, reverse=True)

    lines = [f"Circular Dependencies ({len(cycles)} groups)"]
    lines.append("=" * 48)

    for i, node_ids in enumerate(cycles, 1):
        nodes = [graph.get_node(nid) for nid in node_ids]
        nodes = [n for n in nodes if n is not None]

        severity = "CRITICAL" if len(nodes) >= 5 else ""
        size_label = f" \u2014 {severity}" if severity else ""
        lines.append(f"\nCycle {i} ({len(nodes)} symbols){size_label}:")
        for node in nodes:
            label = node.label.value.title() if node.label else "Unknown"
            lines.append(f"  - {node.name} ({label}) \u2014 {node.file_path}:{node.start_line}")

    return "\n".join(lines)


@lru_cache(maxsize=2)
def _cached_scc_groups(
    storage: StorageBackend, generation: int
) -> tuple[KnowledgeGraph, list[list[str]]]:
    """Load the graph and compute SCC node-id groups once per index generation.

    Independent of ``min_size``: every strongly connected component is
    returned (including singletons), so :func:`handle_cycles` can apply any
    ``min_size`` filter against the same cached decomposition without
    re-running ``load_graph`` or the SCC computation.

    Cache key is ``(storage, generation)``. ``storage`` participates via
    default (identity-based) equality \u2014 distinct backend instances (e.g. a
    freshly rebuilt database after corruption recovery) never share a stale
    entry \u2014 and ``generation`` is bumped by
    :meth:`~synaptiq.core.storage.ladybug_backend.LadybugBackend.initialize`,
    which ``bulk_load`` calls internally on every full reindex, so a reindex
    evicts the old entry on the next call. Backends without a ``generation``
    counter fall back to a constant, caching for the storage object's
    lifetime (matches the ``personalized_pagerank`` precedent).
    """
    graph = storage.load_graph()

    from synaptiq.core.ingestion.community import export_to_igraph

    ig_graph, index_to_node_id = export_to_igraph(graph)

    if ig_graph.vcount() == 0:
        return graph, []

    sccs = ig_graph.connected_components(mode="strong")
    groups = [
        [index_to_node_id[idx] for idx in component if idx in index_to_node_id]
        for component in sccs
    ]
    return graph, groups


def handle_test_impact(
    storage: StorageBackend,
    diff: str = "",
    symbols: list[str] | None = None,
) -> str:
    """Find tests likely affected by code changes."""
    changed_symbol_ids: list[tuple[str, str]] = []

    if diff and diff.strip():
        changed_files = _parse_diff_files(diff)
        for file_path, ranges in changed_files.items():
            rows = (
                storage.execute_raw(
                    "MATCH (n) WHERE n.file_path = $fp "
                    "AND n.start_line > 0 "
                    "RETURN n.id, n.name, n.start_line, n.end_line",
                    parameters={"fp": file_path},
                )
                or []
            )
            for row in rows:
                node_id = row[0] or ""
                name = row[1] or ""
                start_line = row[2] or 0
                end_line = row[3] or 0
                hit = any(start_line <= end and end_line >= start for start, end in ranges)
                if hit:
                    changed_symbol_ids.append((node_id, name))

    elif symbols:
        for sym_name in symbols:
            results = _resolve_symbol(storage, sym_name)
            if results:
                node = storage.get_node(results[0].node_id)
                if node:
                    changed_symbol_ids.append((node.id, node.name))

    else:
        return "Error: provide either 'diff' or 'symbols' parameter."

    if not changed_symbol_ids:
        return "No changed symbols found."

    test_hits: dict[str, list[tuple[str, str, int]]] = {}

    for sym_id, sym_name in changed_symbol_ids:
        for caller, depth in storage.traverse_with_depth(sym_id, 4, direction="callers"):
            if _is_test_file(caller.file_path):
                test_hits.setdefault(caller.file_path, []).append((caller.name, sym_name, depth))

    if not test_hits:
        return (
            f"No test files found in the call graph of {len(changed_symbol_ids)} "
            f"changed symbol(s). Tests may not directly call these symbols."
        )

    lines = ["Test Impact Analysis"]
    lines.append("=" * 48)
    lines.append(f"Changed symbols: {len(changed_symbol_ids)}")
    lines.append("")

    direct_files: dict[str, list[tuple[str, str, int]]] = {}
    transitive_files: dict[str, list[tuple[str, str, int]]] = {}

    for test_file, hits in sorted(test_hits.items()):
        for test_name, source_sym, depth in hits:
            if depth <= 2:
                direct_files.setdefault(test_file, []).append((test_name, source_sym, depth))
            else:
                transitive_files.setdefault(test_file, []).append((test_name, source_sym, depth))

    total_tests = sum(len(v) for v in test_hits.values())

    if direct_files:
        lines.append(f"Affected tests ({total_tests}):")
        for test_file, hits in sorted(direct_files.items()):
            lines.append(f"  {test_file}:")
            seen = set()
            for test_name, source_sym, depth in hits:
                key = (test_name, source_sym)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"    - {test_name} (calls: {source_sym})")
        lines.append("")

    if transitive_files:
        lines.append("Tests with indirect coverage (depth 3+):")
        for test_file, hits in sorted(transitive_files.items()):
            lines.append(f"  {test_file}:")
            seen = set()
            for test_name, source_sym, depth in hits:
                key = (test_name, source_sym)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"    - {test_name} (transitive via: {source_sym})")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Memory tools
# ------------------------------------------------------------------

_memory_store: MemoryStore | None = None


def _get_memory() -> MemoryStore:
    global _memory_store  # noqa: PLW0603
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def handle_remember(key: str, value: str, category: str = "") -> str:
    """Store a fact for future recall."""
    if not key or not key.strip():
        return "Error: 'key' parameter is required and cannot be empty."
    if not value or not value.strip():
        return "Error: 'value' parameter is required and cannot be empty."

    fact = _get_memory().remember(key.strip(), value.strip(), category.strip())
    cat_info = f" [{fact.category}]" if fact.category else ""
    return f"Remembered: {fact.key}{cat_info}\n  {fact.value}"


def handle_recall(query: str) -> str:
    """Retrieve stored facts matching a query."""
    if not query or not query.strip():
        return "Error: 'query' parameter is required and cannot be empty."

    facts = _get_memory().recall(query.strip())
    if not facts:
        return f"No stored facts match '{query}'."

    lines = [f"Recalled facts ({len(facts)}):"]
    lines.append("")
    for i, fact in enumerate(facts, 1):
        cat_info = f" [{fact.category}]" if fact.category else ""
        lines.append(f"  {i}. {fact.key}{cat_info}")
        lines.append(f"     {fact.value}")
    return "\n".join(lines)


def handle_forget(key: str) -> str:
    """Remove a stored fact by key."""
    if not key or not key.strip():
        return "Error: 'key' parameter is required and cannot be empty."

    removed = _get_memory().forget(key.strip())
    if removed:
        return f"Forgotten: '{key}'"
    return f"No fact found with key '{key}'."


# ------------------------------------------------------------------
# Suggest tool
# ------------------------------------------------------------------


def handle_suggest(storage: StorageBackend, question: str) -> str:
    """Return tool call suggestions for a natural language question."""
    if not question or not question.strip():
        return "Error: 'question' parameter is required and cannot be empty."

    from synaptiq.mcp.suggest import suggest_tools

    suggestions = suggest_tools(question.strip(), storage)
    if not suggestions:
        return "Could not determine appropriate tools for this question."

    lines = ["Suggested tool calls:"]
    lines.append("")
    for i, s in enumerate(suggestions, 1):
        args_str = json.dumps(s.arguments) if s.arguments else "{}"
        lines.append(f"  {i}. {s.tool_name}({args_str})")
        lines.append(f"     Reason: {s.reason}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Export tool
# ------------------------------------------------------------------


def handle_export(
    storage: StorageBackend,
    symbol: str,
    depth: int = 2,
    include_source: bool = True,
) -> str:
    """Graph-aware context packing from a starting symbol.

    Performs multi-hop BFS traversal and returns a single structurally-ordered
    context blob with callers, callees, type references, and community members.
    """
    if not symbol or not symbol.strip():
        return "Error: 'symbol' parameter is required and cannot be empty."

    depth = max(1, min(depth, 4))

    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    root_node = storage.get_node(results[0].node_id)
    if not root_node:
        return f"Symbol '{symbol}' not found."

    # Collect related nodes via BFS at each hop level.
    sections: list[str] = []

    # Root symbol.
    label_display = root_node.label.value.title() if root_node.label else "Unknown"
    root_lines = [f"=== Symbol: {root_node.name} ({label_display}) ==="]
    root_lines.append(f"File: {root_node.file_path}:{root_node.start_line}-{root_node.end_line}")
    if root_node.signature:
        root_lines.append(f"Signature: {root_node.signature}")
    if include_source and root_node.content:
        root_lines.append("")
        root_lines.append(root_node.content)
    sections.append("\n".join(root_lines))

    # Direct callees.
    try:
        callees_raw = storage.get_callees_with_confidence(root_node.id)
    except (AttributeError, TypeError):
        callees_raw = [(c, 1.0) for c in storage.get_callees(root_node.id)]

    if callees_raw:
        callee_lines = [f"\n=== Direct Callees ({len(callees_raw)}) ==="]
        for i, (node, conf) in enumerate(callees_raw, 1):
            label = node.label.value.title() if node.label else "Unknown"
            tag = _confidence_tag(conf)
            loc = f"{node.file_path}:{node.start_line}"
            callee_lines.append(f"  {i}. {node.name} ({label}) — {loc}{tag}")
            if node.signature:
                callee_lines.append(f"     Signature: {node.signature}")
            if include_source and node.content:
                callee_lines.append(f"     Source:\n{node.content}")
        sections.append("\n".join(callee_lines))

    # Direct callers.
    try:
        callers_raw = storage.get_callers_with_confidence(root_node.id)
    except (AttributeError, TypeError):
        callers_raw = [(c, 1.0) for c in storage.get_callers(root_node.id)]

    if callers_raw:
        caller_lines = [f"\n=== Direct Callers ({len(callers_raw)}) ==="]
        for i, (node, conf) in enumerate(callers_raw, 1):
            label = node.label.value.title() if node.label else "Unknown"
            tag = _confidence_tag(conf)
            loc = f"{node.file_path}:{node.start_line}"
            caller_lines.append(f"  {i}. {node.name} ({label}) — {loc}{tag}")
            if node.signature:
                caller_lines.append(f"     Signature: {node.signature}")
            if include_source and node.content:
                caller_lines.append(f"     Source:\n{node.content}")
        sections.append("\n".join(caller_lines))

    # Type references.
    type_refs = storage.get_type_refs(root_node.id)
    if type_refs:
        type_lines = [f"\n=== Type References ({len(type_refs)}) ==="]
        for i, t in enumerate(type_refs, 1):
            type_lines.append(f"  {i}. {t.name} — {t.file_path}")
        sections.append("\n".join(type_lines))

    # Heritage.
    heritage_rows = _heritage_rows(storage, root_node.id)
    if heritage_rows:
        her_lines = [f"\n=== Heritage ({len(heritage_rows)}) ==="]
        for row in heritage_rows:
            her_lines.append(f"  - {row[2]}: {row[0]}  {row[1]}")
        sections.append("\n".join(her_lines))

    # Community membership.
    communities = _community_names(storage, root_node.id)
    if communities:
        sections.append(f"\n=== Community: {communities[0]} ===")

    # Deeper hops if depth > 1.
    if depth >= 2:
        deeper_nodes = storage.traverse_with_depth(root_node.id, depth, direction="callers")
        deeper = [(n, d) for n, d in deeper_nodes if d >= 2]
        if deeper:
            deep_lines = [f"\n=== Transitive Callers (depth 2-{depth}, {len(deeper)} symbols) ==="]
            for i, (node, d) in enumerate(deeper[:20], 1):
                label = node.label.value.title() if node.label else "Unknown"
                deep_lines.append(
                    f"  {i}. {node.name} ({label}) — {node.file_path}:{node.start_line} [depth {d}]"
                )
            if len(deeper) > 20:
                deep_lines.append(f"  ... and {len(deeper) - 20} more")
            sections.append("\n".join(deep_lines))

    return "\n".join(sections)
