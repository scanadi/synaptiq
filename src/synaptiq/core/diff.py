"""Branch comparison for Synaptiq.

Compares two code graphs structurally to find added, removed, and modified
nodes and relationships.

The default (scoped) mode parses ONLY the files that changed between the
two refs — content is read via ``git show``, no worktrees, no full pipeline
— so cost scales with the diff, not the repository.  Node changes are exact;
relationship changes are scoped to edges originating in changed files and
resolved within the changed-file symbol set.

``full=True`` keeps the legacy behaviour: build the complete graph for both
sides in temporary worktrees.  Exhaustive (cross-file edges everywhere) but
cost scales with repository size — unusable on large monorepos.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptiq.core.graph.graph import KnowledgeGraph
    from synaptiq.core.ingestion.parser_phase import FileParseData

from synaptiq.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)

logger = logging.getLogger(__name__)

@dataclass
class StructuralDiff:
    """Result of comparing two code graphs."""

    added_nodes: list[GraphNode] = field(default_factory=list)
    removed_nodes: list[GraphNode] = field(default_factory=list)
    modified_nodes: list[tuple[GraphNode, GraphNode]] = field(default_factory=list)
    added_relationships: list[GraphRelationship] = field(default_factory=list)
    removed_relationships: list[GraphRelationship] = field(default_factory=list)

# Fields checked to determine if a node was "modified".
_NODE_COMPARE_FIELDS = ("content", "signature", "start_line", "end_line")

def diff_graphs(
    base_nodes: dict[str, GraphNode],
    current_nodes: dict[str, GraphNode],
    base_rels: dict[str, GraphRelationship],
    current_rels: dict[str, GraphRelationship],
) -> StructuralDiff:
    """Diff two graph snapshots by node/relationship IDs.

    Nodes present only in *current_nodes* are added; only in *base_nodes* are
    removed.  Nodes with the same ID but different content/signature/lines are
    modified.  Relationships are compared by ID only (added/removed).

    Args:
        base_nodes: ``{node_id: GraphNode}`` from the base branch.
        current_nodes: ``{node_id: GraphNode}`` from the current branch.
        base_rels: ``{rel_id: GraphRelationship}`` from the base branch.
        current_rels: ``{rel_id: GraphRelationship}`` from the current branch.

    Returns:
        A :class:`StructuralDiff` with the comparison results.
    """
    result = StructuralDiff()

    base_ids = set(base_nodes)
    current_ids = set(current_nodes)

    for nid in current_ids - base_ids:
        result.added_nodes.append(current_nodes[nid])

    for nid in base_ids - current_ids:
        result.removed_nodes.append(base_nodes[nid])

    for nid in base_ids & current_ids:
        base_node = base_nodes[nid]
        current_node = current_nodes[nid]
        if _node_changed(base_node, current_node):
            result.modified_nodes.append((base_node, current_node))

    base_rel_ids = set(base_rels)
    current_rel_ids = set(current_rels)

    for rid in current_rel_ids - base_rel_ids:
        result.added_relationships.append(current_rels[rid])

    for rid in base_rel_ids - current_rel_ids:
        result.removed_relationships.append(base_rels[rid])

    return result

def _node_changed(base: GraphNode, current: GraphNode) -> bool:
    """Return True if the two nodes differ on any comparison field."""
    for attr in _NODE_COMPARE_FIELDS:
        if getattr(base, attr) != getattr(current, attr):
            return True
    return False

def diff_branches(
    repo_path: Path,
    branch_range: str,
    *,
    full: bool = False,
) -> StructuralDiff:
    """Compare two refs structurally.

    *branch_range* should be ``"base..current"`` (e.g. ``"main..feature"``).
    If only one ref is given (no ``..``), it is treated as the base and the
    current working tree is used as the current side.

    Default (scoped) mode: only files changed between the refs are parsed
    (contents via ``git show``; working-tree side read from disk).  Symbol
    additions/removals/modifications are exact.  Relationship changes cover
    import statements and calls originating in changed files, resolved
    against the changed-file symbol set — edges into unchanged files are out
    of scope.  Untracked files are not included when diffing against the
    working tree.

    ``full=True``: legacy mode — build the complete graph for both sides in
    temporary worktrees.  Cost scales with repository size.

    Args:
        repo_path: Root of the git repository.
        branch_range: Branch range string (e.g. ``"main..feature"``).
        full: Use the exhaustive dual-graph build instead of the scoped diff.

    Returns:
        A :class:`StructuralDiff` comparing the two refs.

    Raises:
        ValueError: If the branch range format is invalid.
        RuntimeError: If git operations fail.
    """
    base_ref, current_ref = _parse_range(branch_range)

    if full:
        return _full_diff(repo_path, base_ref, current_ref)
    return _scoped_diff(repo_path, base_ref, current_ref)

def _parse_range(branch_range: str) -> tuple[str, str | None]:
    """Split ``"base..current"`` into refs; ``None`` current = working tree."""
    if ".." in branch_range:
        parts = branch_range.split("..", 1)
        base_ref = parts[0].strip()
        current_ref = parts[1].strip() if parts[1].strip() else None
    else:
        base_ref = branch_range.strip()
        current_ref = None

    if not base_ref:
        raise ValueError(f"Invalid branch range: {branch_range!r}")
    return base_ref, current_ref

# ----------------------------------------------------------------------
# Scoped diff (default): parse only changed files
# ----------------------------------------------------------------------

def _scoped_diff(
    repo_path: Path,
    base_ref: str,
    current_ref: str | None,
) -> StructuralDiff:
    """Diff only the files that changed between the refs."""
    from synaptiq.config.languages import get_language, is_supported
    from synaptiq.core.ingestion.parser_phase import parse_file

    changed = [
        (status, path)
        for status, path in _changed_files(repo_path, base_ref, current_ref)
        if is_supported(path)
    ]
    if not changed:
        return StructuralDiff()

    base_side = _SideGraph()
    current_side = _SideGraph()

    for _status, path in changed:
        language = get_language(path)
        if language is None:
            continue

        base_content = _file_content(repo_path, base_ref, path)
        if base_content is not None:
            base_side.add_file(parse_file(path, base_content, language))

        current_content = _file_content(repo_path, current_ref, path)
        if current_content is not None:
            current_side.add_file(parse_file(path, current_content, language))

    base_side.finalise()
    current_side.finalise()

    return diff_graphs(
        base_side.nodes, current_side.nodes, base_side.rels, current_side.rels
    )

def _changed_files(
    repo_path: Path,
    base_ref: str,
    current_ref: str | None,
) -> list[tuple[str, str]]:
    """Return ``(status, path)`` pairs for files changed between the refs."""
    target = f"{base_ref}..{current_ref}" if current_ref else base_ref
    proc = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", target],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git diff failed for '{target}': {proc.stderr.strip()}")

    changed: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        status, _, path = line.partition("\t")
        status = status.strip()
        path = path.strip()
        if status and path:
            changed.append((status, path))
    return changed

def _file_content(repo_path: Path, ref: str | None, path: str) -> str | None:
    """Read *path* at *ref* (``None`` = working tree); ``None`` when absent."""
    if ref is None:
        try:
            return (repo_path / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return proc.stdout if proc.returncode == 0 else None

class _SideGraph:
    """Nodes and relationships built from one side's changed files."""

    def __init__(self) -> None:
        self._files: list["FileParseData"] = []
        self.nodes: dict[str, GraphNode] = {}
        self.rels: dict[str, GraphRelationship] = {}

    def add_file(self, parse_data: "FileParseData") -> None:
        self._files.append(parse_data)

    def finalise(self) -> None:
        """Materialise nodes and changed-file-scoped relationships."""
        from synaptiq.core.ingestion.calls import _CALL_BLOCKLIST
        from synaptiq.core.ingestion.parser_phase import (
            _KIND_TO_LABEL,
            assign_symbol_ids,
        )

        name_index: dict[str, list[str]] = {}
        spans: dict[str, list[tuple[int, int, str]]] = {}

        for fpd in self._files:
            exported = set(fpd.parse_result.exports)
            symbol_ids = assign_symbol_ids(fpd.parse_result.symbols, fpd.file_path)
            for symbol, symbol_id in zip(fpd.parse_result.symbols, symbol_ids):
                if symbol_id is None:
                    continue
                self.nodes[symbol_id] = GraphNode(
                    id=symbol_id,
                    label=_KIND_TO_LABEL[symbol.kind],
                    name=symbol.name,
                    file_path=fpd.file_path,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    content=symbol.content,
                    signature=symbol.signature,
                    class_name=symbol.class_name,
                    language=fpd.language,
                    is_exported=symbol.name in exported,
                )
                name_index.setdefault(symbol.name, []).append(symbol_id)
                spans.setdefault(fpd.file_path, []).append(
                    (symbol.start_line, symbol.end_line, symbol_id)
                )

        for fpd in self._files:
            file_id = generate_id(NodeLabel.FILE, fpd.file_path)

            # Import statements as pseudo-edges keyed by module + names, so
            # both new modules and changed imported-name lists surface.
            for imp in fpd.parse_result.imports:
                names = ",".join(sorted(imp.names))
                rel_id = f"imports:{file_id}->{imp.module}:{names}"
                self.rels[rel_id] = GraphRelationship(
                    id=rel_id,
                    type=RelType.IMPORTS,
                    source=file_id,
                    target=imp.module,
                    properties={"symbols": names},
                )

            for call in fpd.parse_result.calls:
                source_id = self._containing_symbol(
                    spans.get(fpd.file_path, []), call.line
                ) or file_id

                names = [] if call.name in _CALL_BLOCKLIST else [call.name]
                names.extend(a for a in call.arguments if a not in _CALL_BLOCKLIST)
                for name in names:
                    target_id = self._resolve(name_index, name, fpd.file_path)
                    if target_id is None or target_id == source_id:
                        continue
                    rel_id = f"calls:{source_id}->{target_id}"
                    self.rels[rel_id] = GraphRelationship(
                        id=rel_id,
                        type=RelType.CALLS,
                        source=source_id,
                        target=target_id,
                    )

    @staticmethod
    def _containing_symbol(
        file_spans: list[tuple[int, int, str]], line: int
    ) -> str | None:
        """Innermost symbol whose span contains *line*."""
        best: tuple[int, str] | None = None
        for start, end, symbol_id in file_spans:
            if start <= line <= end:
                size = end - start
                if best is None or size < best[0]:
                    best = (size, symbol_id)
        return best[1] if best else None

    def _resolve(
        self,
        name_index: dict[str, list[str]],
        name: str,
        caller_file: str,
    ) -> str | None:
        """Resolve *name* within the changed-file symbol set.

        Same-file match wins; a unique cross-file match is accepted;
        ambiguous cross-file names are skipped to avoid noise.
        """
        candidates = name_index.get(name, [])
        if not candidates:
            return None
        same_file = [c for c in candidates if self.nodes[c].file_path == caller_file]
        if same_file:
            return same_file[0]
        if len(candidates) == 1:
            return candidates[0]
        return None

# ----------------------------------------------------------------------
# Full diff (legacy): build complete graphs for both sides
# ----------------------------------------------------------------------

def _full_diff(
    repo_path: Path,
    base_ref: str,
    current_ref: str | None,
) -> StructuralDiff:
    """Exhaustive comparison via two full pipeline builds (legacy)."""
    from synaptiq.core.ingestion.pipeline import build_graph

    # Build both graphs (in parallel when both need worktrees).
    if current_ref:
        with ThreadPoolExecutor(max_workers=2) as executor:
            base_future = executor.submit(_build_graph_for_ref, repo_path, base_ref)
            current_future = executor.submit(_build_graph_for_ref, repo_path, current_ref)
            base_graph = base_future.result()
            current_graph = current_future.result()
    else:
        current_graph = build_graph(repo_path)
        base_graph = _build_graph_for_ref(repo_path, base_ref)

    base_nodes = {n.id: n for n in base_graph.iter_nodes()}
    current_nodes = {n.id: n for n in current_graph.iter_nodes()}
    base_rels = {r.id: r for r in base_graph.iter_relationships()}
    current_rels = {r.id: r for r in current_graph.iter_relationships()}

    return diff_graphs(base_nodes, current_nodes, base_rels, current_rels)

def _build_graph_for_ref(repo_path: Path, ref: str) -> KnowledgeGraph:
    """Build an in-memory graph for a git ref using a temporary worktree."""
    from synaptiq.core.ingestion.pipeline import build_graph

    with tempfile.TemporaryDirectory(prefix="synaptiq_diff_") as tmp_dir:
        worktree_path = Path(tmp_dir) / "worktree"

        try:
            subprocess.run(
                ["git", "worktree", "add", str(worktree_path), ref],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to create worktree for ref '{ref}': {exc.stderr.strip()}"
            ) from exc

        try:
            graph = build_graph(worktree_path)
        finally:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree_path)],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                logger.warning("Failed to remove worktree at %s", worktree_path)

    return graph

def format_diff(diff: StructuralDiff) -> str:
    """Format a StructuralDiff as human-readable output.

    Args:
        diff: The structural diff to format.

    Returns:
        A multi-line string summarizing added, removed, and modified entities.
    """
    total_changes = (
        len(diff.added_nodes)
        + len(diff.removed_nodes)
        + len(diff.modified_nodes)
        + len(diff.added_relationships)
        + len(diff.removed_relationships)
    )

    if total_changes == 0:
        return "No structural differences found."

    lines: list[str] = []
    lines.append(f"Structural diff: {total_changes} changes")
    lines.append("")

    if diff.added_nodes:
        lines.append(f"Added nodes ({len(diff.added_nodes)}):")
        for node in sorted(diff.added_nodes, key=lambda n: n.id):
            label = node.label.value.title()
            lines.append(f"  + {node.name} ({label}) -- {node.file_path}")
        lines.append("")

    if diff.removed_nodes:
        lines.append(f"Removed nodes ({len(diff.removed_nodes)}):")
        for node in sorted(diff.removed_nodes, key=lambda n: n.id):
            label = node.label.value.title()
            lines.append(f"  - {node.name} ({label}) -- {node.file_path}")
        lines.append("")

    if diff.modified_nodes:
        lines.append(f"Modified nodes ({len(diff.modified_nodes)}):")
        for base_node, current_node in sorted(diff.modified_nodes, key=lambda p: p[0].id):
            label = current_node.label.value.title()
            lines.append(f"  ~ {current_node.name} ({label}) -- {current_node.file_path}")
        lines.append("")

    if diff.added_relationships:
        lines.append(f"Added relationships ({len(diff.added_relationships)}):")
        for rel in sorted(diff.added_relationships, key=lambda r: r.id):
            lines.append(f"  + [{rel.type.value}] {rel.source} -> {rel.target}")
        lines.append("")

    if diff.removed_relationships:
        lines.append(f"Removed relationships ({len(diff.removed_relationships)}):")
        for rel in sorted(diff.removed_relationships, key=lambda r: r.id):
            lines.append(f"  - [{rel.type.value}] {rel.source} -> {rel.target}")
        lines.append("")

    return "\n".join(lines).rstrip()
