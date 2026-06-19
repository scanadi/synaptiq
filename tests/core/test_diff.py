"""Tests for the branch diff module (diff.py)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from synaptiq.core.diff import StructuralDiff, diff_branches, diff_graphs, format_diff
from synaptiq.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(nid: str, label: NodeLabel = NodeLabel.FUNCTION, **kwargs) -> GraphNode:
    """Create a GraphNode with sensible defaults."""
    return GraphNode(
        id=nid,
        label=label,
        name=kwargs.pop("name", nid.split(":")[-1] or nid),
        file_path=kwargs.pop("file_path", "src/app.py"),
        **kwargs,
    )


def _rel(rid: str, rel_type: RelType = RelType.CALLS, **kwargs) -> GraphRelationship:
    """Create a GraphRelationship with sensible defaults."""
    return GraphRelationship(
        id=rid,
        type=rel_type,
        source=kwargs.pop("source", "a"),
        target=kwargs.pop("target", "b"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests: diff_graphs — node detection
# ---------------------------------------------------------------------------


class TestDiffGraphsAddedNodes:
    """Nodes present in current but not base are detected as added."""

    def test_added_nodes(self) -> None:
        base = {}
        current = {"n1": _node("n1")}

        result = diff_graphs(base, current, {}, {})

        assert len(result.added_nodes) == 1
        assert result.added_nodes[0].id == "n1"
        assert result.removed_nodes == []
        assert result.modified_nodes == []


class TestDiffGraphsRemovedNodes:
    """Nodes present in base but not current are detected as removed."""

    def test_removed_nodes(self) -> None:
        base = {"n1": _node("n1")}
        current = {}

        result = diff_graphs(base, current, {}, {})

        assert len(result.removed_nodes) == 1
        assert result.removed_nodes[0].id == "n1"
        assert result.added_nodes == []


class TestDiffGraphsModifiedContent:
    """Nodes with same ID but different content are detected as modified."""

    def test_modified_content(self) -> None:
        base = {"n1": _node("n1", content="old body")}
        current = {"n1": _node("n1", content="new body")}

        result = diff_graphs(base, current, {}, {})

        assert len(result.modified_nodes) == 1
        assert result.modified_nodes[0][0].content == "old body"
        assert result.modified_nodes[0][1].content == "new body"
        assert result.added_nodes == []
        assert result.removed_nodes == []


class TestDiffGraphsModifiedSignature:
    """Nodes with same ID but different signature are detected as modified."""

    def test_modified_signature(self) -> None:
        base = {"n1": _node("n1", signature="def foo()")}
        current = {"n1": _node("n1", signature="def foo(x: int)")}

        result = diff_graphs(base, current, {}, {})

        assert len(result.modified_nodes) == 1


class TestDiffGraphsModifiedLines:
    """Nodes with same ID but different line numbers are detected as modified."""

    def test_modified_start_line(self) -> None:
        base = {"n1": _node("n1", start_line=10, end_line=20)}
        current = {"n1": _node("n1", start_line=15, end_line=25)}

        result = diff_graphs(base, current, {}, {})

        assert len(result.modified_nodes) == 1


class TestDiffGraphsUnchangedNodes:
    """Identical nodes produce no diff entries."""

    def test_unchanged(self) -> None:
        n = _node("n1", content="body", signature="def f()")
        base = {"n1": n}
        current = {"n1": n}

        result = diff_graphs(base, current, {}, {})

        assert result.added_nodes == []
        assert result.removed_nodes == []
        assert result.modified_nodes == []


class TestDiffGraphsEmptyGraphs:
    """Diffing two empty graphs produces an empty diff."""

    def test_empty(self) -> None:
        result = diff_graphs({}, {}, {}, {})

        assert result == StructuralDiff()


# ---------------------------------------------------------------------------
# Tests: diff_graphs — relationship detection
# ---------------------------------------------------------------------------


class TestDiffGraphsAddedRelationships:
    """Relationships in current but not base are added."""

    def test_added_rels(self) -> None:
        base_rels: dict[str, GraphRelationship] = {}
        current_rels = {"r1": _rel("r1")}

        result = diff_graphs({}, {}, base_rels, current_rels)

        assert len(result.added_relationships) == 1
        assert result.added_relationships[0].id == "r1"


class TestDiffGraphsRemovedRelationships:
    """Relationships in base but not current are removed."""

    def test_removed_rels(self) -> None:
        base_rels = {"r1": _rel("r1")}
        current_rels: dict[str, GraphRelationship] = {}

        result = diff_graphs({}, {}, base_rels, current_rels)

        assert len(result.removed_relationships) == 1
        assert result.removed_relationships[0].id == "r1"


# ---------------------------------------------------------------------------
# Tests: diff_graphs — mixed scenarios
# ---------------------------------------------------------------------------


class TestDiffGraphsMixedChanges:
    """A realistic diff with adds, removes, and modifications."""

    def test_mixed(self) -> None:
        base_nodes = {
            "n1": _node("n1", content="old"),
            "n2": _node("n2", content="same"),
            "n3": _node("n3", content="removed"),
        }
        current_nodes = {
            "n1": _node("n1", content="new"),
            "n2": _node("n2", content="same"),
            "n4": _node("n4", content="added"),
        }
        base_rels = {
            "r1": _rel("r1"),
            "r2": _rel("r2"),
        }
        current_rels = {
            "r1": _rel("r1"),
            "r3": _rel("r3"),
        }

        result = diff_graphs(base_nodes, current_nodes, base_rels, current_rels)

        assert len(result.added_nodes) == 1
        assert result.added_nodes[0].id == "n4"

        assert len(result.removed_nodes) == 1
        assert result.removed_nodes[0].id == "n3"

        assert len(result.modified_nodes) == 1
        assert result.modified_nodes[0][0].id == "n1"

        assert len(result.added_relationships) == 1
        assert result.added_relationships[0].id == "r3"

        assert len(result.removed_relationships) == 1
        assert result.removed_relationships[0].id == "r2"


# ---------------------------------------------------------------------------
# Tests: format_diff
# ---------------------------------------------------------------------------


class TestFormatDiffEmpty:
    """Empty diff produces a 'no differences' message."""

    def test_empty(self) -> None:
        result = format_diff(StructuralDiff())
        assert "No structural differences" in result


class TestFormatDiffAddedNodes:
    """Added nodes appear with + prefix."""

    def test_added(self) -> None:
        diff = StructuralDiff(added_nodes=[_node("n1", name="my_func")])
        result = format_diff(diff)

        assert "+ my_func" in result
        assert "Added nodes (1)" in result
        assert "1 changes" in result


class TestFormatDiffRemovedNodes:
    """Removed nodes appear with - prefix."""

    def test_removed(self) -> None:
        diff = StructuralDiff(removed_nodes=[_node("n1", name="old_func")])
        result = format_diff(diff)

        assert "- old_func" in result
        assert "Removed nodes (1)" in result


class TestFormatDiffModifiedNodes:
    """Modified nodes appear with ~ prefix."""

    def test_modified(self) -> None:
        diff = StructuralDiff(
            modified_nodes=[
                (_node("n1", name="changed_func"), _node("n1", name="changed_func"))
            ]
        )
        result = format_diff(diff)

        assert "~ changed_func" in result
        assert "Modified nodes (1)" in result


class TestFormatDiffRelationships:
    """Relationship changes include type and source->target."""

    def test_rel_format(self) -> None:
        diff = StructuralDiff(
            added_relationships=[_rel("r1", source="func:a:f", target="func:b:g")],
            removed_relationships=[_rel("r2", source="func:c:h", target="func:d:i")],
        )
        result = format_diff(diff)

        assert "Added relationships (1)" in result
        assert "Removed relationships (1)" in result
        assert "[calls]" in result


class TestFormatDiffFullSummary:
    """The summary line shows total change count."""

    def test_summary(self) -> None:
        diff = StructuralDiff(
            added_nodes=[_node("n1")],
            removed_nodes=[_node("n2")],
            modified_nodes=[(_node("n3"), _node("n3"))],
        )
        result = format_diff(diff)

        assert "3 changes" in result


# ---------------------------------------------------------------------------
# Tests: scoped diff_branches against a real git repo
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture()
def ts_repo(tmp_path: Path) -> Path:
    """Two-commit repo: v1 defines helper+main, v2 reshapes both files."""
    repo = tmp_path
    _git(repo, "init", "-b", "main")

    src = repo / "src"
    src.mkdir()
    (src / "app.ts").write_text(
        "import { helper } from './util';\n"
        "export function main(): void {\n"
        "  helper();\n"
        "}\n"
        "function legacy(): void {}\n"
    )
    (src / "util.ts").write_text("export function helper(): void {}\n")
    (repo / "README.md").write_text("v1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "v1")

    (src / "app.ts").write_text(
        "import { helper, extra } from './util';\n"
        "export function main(): void {\n"
        "  helper();\n"
        "  freshOne();\n"
        "}\n"
        "export function freshOne(): void {}\n"
    )
    (repo / "README.md").write_text("v2 — not an indexed language\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "v2")
    return repo


class TestScopedDiffBranches:
    def test_two_ref_diff_detects_node_changes(self, ts_repo: Path) -> None:
        result = diff_branches(ts_repo, "HEAD~1..HEAD")

        added = {n.name for n in result.added_nodes}
        removed = {n.name for n in result.removed_nodes}
        modified = {c.name for _b, c in result.modified_nodes}

        assert added == {"freshOne"}
        assert removed == {"legacy"}
        assert "main" in modified
        # util.ts did not change between the refs — helper stays untouched.
        assert "helper" not in added | removed | modified

    def test_call_and_import_edges_within_changed_files(self, ts_repo: Path) -> None:
        result = diff_branches(ts_repo, "HEAD~1..HEAD")

        added_calls = {
            (r.source, r.target)
            for r in result.added_relationships
            if r.type is RelType.CALLS
        }
        assert any(
            "main" in src and "freshOne" in dst for src, dst in added_calls
        )

        added_imports = [
            r for r in result.added_relationships if r.type is RelType.IMPORTS
        ]
        removed_imports = [
            r for r in result.removed_relationships if r.type is RelType.IMPORTS
        ]
        # Import list changed (helper -> helper,extra): old keyed edge out,
        # new keyed edge in.
        assert any("extra" in r.properties.get("symbols", "") for r in added_imports)
        assert len(removed_imports) == 1

    def test_working_tree_diff(self, ts_repo: Path) -> None:
        (ts_repo / "src" / "app.ts").write_text(
            "export function main(): void {}\n"
        )

        result = diff_branches(ts_repo, "HEAD")

        removed = {n.name for n in result.removed_nodes}
        assert "freshOne" in removed

    def test_untracked_file_counts_as_added(self, ts_repo: Path) -> None:
        (ts_repo / "src" / "brand-new.ts").write_text(
            "export function neverCommitted(): void {}\n"
        )

        result = diff_branches(ts_repo, "HEAD")

        added = {n.name for n in result.added_nodes}
        assert "neverCommitted" in added

    def test_no_changes_returns_empty(self, ts_repo: Path) -> None:
        result = diff_branches(ts_repo, "HEAD..HEAD")
        assert format_diff(result) == "No structural differences found."

    def test_invalid_ref_raises(self, ts_repo: Path) -> None:
        with pytest.raises(RuntimeError):
            diff_branches(ts_repo, "no-such-ref..HEAD")

    def test_deleted_file_symbols_reported_removed(self, ts_repo: Path) -> None:
        _git(ts_repo, "rm", "src/util.ts")
        _git(ts_repo, "commit", "-m", "drop util")

        result = diff_branches(ts_repo, "HEAD~1..HEAD")
        removed = {n.name for n in result.removed_nodes}
        assert "helper" in removed
