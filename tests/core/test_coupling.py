"""Tests for the change coupling analysis phase (Phase 11)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, RelType, generate_id
from synaptiq.core.ingestion.coupling import (
    GitCollectionResult,
    build_cochange_matrix,
    calculate_coupling,
    collect_coupling_commits,
    parse_git_log,
    process_coupling,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a KnowledgeGraph pre-populated with File nodes.

    Layout:
    - File:src/auth.py
    - File:src/models.py
    - File:src/views.py
    - File:src/utils.py
    """
    g = KnowledgeGraph()

    for path in ("src/auth.py", "src/models.py", "src/views.py", "src/utils.py"):
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, path),
                label=NodeLabel.FILE,
                name=path.split("/")[-1],
                file_path=path,
            )
        )

    return g


# ---------------------------------------------------------------------------
# build_cochange_matrix tests
# ---------------------------------------------------------------------------


class TestBuildCochangeMatrix:
    """build_cochange_matrix produces correct pairwise counts."""

    def test_build_cochange_matrix(self) -> None:
        """Correct pair counts from commit data."""
        commits = [
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/views.py", "src/utils.py"],
        ]
        matrix = build_cochange_matrix(commits, min_cochanges=1)

        pair = ("src/auth.py", "src/models.py")
        assert pair in matrix
        assert matrix[pair] == 3

        pair_vu = ("src/utils.py", "src/views.py")
        assert pair_vu in matrix
        assert matrix[pair_vu] == 1

    def test_build_cochange_matrix_min_threshold(self) -> None:
        """Pairs below min_cochanges are filtered out."""
        commits = [
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/views.py", "src/utils.py"],
        ]
        matrix = build_cochange_matrix(commits, min_cochanges=3)

        # auth+models has 3 co-changes, should be included.
        assert ("src/auth.py", "src/models.py") in matrix

        # views+utils has only 1, should be filtered.
        assert ("src/utils.py", "src/views.py") not in matrix

    def test_build_cochange_matrix_empty(self) -> None:
        """Empty commits list returns an empty dict."""
        matrix = build_cochange_matrix([], min_cochanges=1)
        assert matrix == {}


# ---------------------------------------------------------------------------
# calculate_coupling tests
# ---------------------------------------------------------------------------


class TestCalculateCoupling:
    """calculate_coupling produces correct strength values."""

    def test_calculate_coupling(self) -> None:
        """Coupling = co_changes / max(total_a, total_b)."""
        total_changes = {"src/auth.py": 10, "src/models.py": 5}
        strength = calculate_coupling(
            "src/auth.py", "src/models.py", co_changes=5, total_changes=total_changes
        )
        # 5 / max(10, 5) = 5 / 10 = 0.5
        assert strength == pytest.approx(0.5)

    def test_calculate_coupling_equal_changes(self) -> None:
        """When both files have equal total changes, coupling = co_changes / total."""
        total_changes = {"src/auth.py": 8, "src/models.py": 8}
        strength = calculate_coupling(
            "src/auth.py", "src/models.py", co_changes=6, total_changes=total_changes
        )
        # 6 / max(8, 8) = 6 / 8 = 0.75
        assert strength == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# process_coupling tests
# ---------------------------------------------------------------------------


class TestProcessCoupling:
    """process_coupling creates COUPLED_WITH relationships in the graph."""

    def test_process_coupling_creates_relationships(
        self, graph: KnowledgeGraph
    ) -> None:
        """Mock git log via the commits parameter, verify COUPLED_WITH edges."""
        # auth.py and models.py change together 4 times out of 5 commits each.
        # views.py and utils.py change together only once.
        commits = [
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/auth.py"],
            ["src/models.py"],
            ["src/views.py", "src/utils.py"],
        ]

        count = process_coupling(
            graph,
            Path("/fake/repo"),
            min_strength=0.3,
            commits=commits,
        )

        # auth+models: coupling = 4 / max(5, 5) = 0.8 >= 0.3 -> created
        # views+utils: coupling = 1 / max(1, 1) = 1.0 >= 0.3 -> created
        assert count == 2

        coupled_rels = graph.get_relationships_by_type(RelType.COUPLED_WITH)
        assert len(coupled_rels) == 2

        # Verify properties on the auth+models relationship.
        auth_id = generate_id(NodeLabel.FILE, "src/auth.py")
        models_id = generate_id(NodeLabel.FILE, "src/models.py")

        auth_models_rel = next(
            (
                r
                for r in coupled_rels
                if r.source == auth_id and r.target == models_id
            ),
            None,
        )
        assert auth_models_rel is not None
        assert auth_models_rel.properties["strength"] == pytest.approx(0.8)
        assert auth_models_rel.properties["co_changes"] == 4

    def test_process_coupling_no_git(self, graph: KnowledgeGraph) -> None:
        """Non-git repo returns 0 gracefully (parse_git_log returns [])."""
        count = process_coupling(
            graph,
            Path("/nonexistent/repo"),
            min_strength=0.3,
            commits=[],
        )
        assert count == 0

        coupled_rels = graph.get_relationships_by_type(RelType.COUPLED_WITH)
        assert len(coupled_rels) == 0

    def test_process_coupling_filters_weak_pairs(
        self, graph: KnowledgeGraph
    ) -> None:
        """Pairs below min_strength are not added to the graph."""
        # auth changes 10 times, models 10 times, but they co-change only twice.
        # coupling = 2/10 = 0.2 which is below min_strength=0.3
        commits = [
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/auth.py"],
            ["src/auth.py"],
            ["src/auth.py"],
            ["src/auth.py"],
            ["src/auth.py"],
            ["src/auth.py"],
            ["src/models.py"],
            ["src/models.py"],
            ["src/models.py"],
            ["src/models.py"],
            ["src/models.py"],
            ["src/models.py"],
        ]

        count = process_coupling(
            graph,
            Path("/fake/repo"),
            min_strength=0.3,
            commits=commits,
        )
        assert count == 0

    def test_process_coupling_relationship_id_format(
        self, graph: KnowledgeGraph
    ) -> None:
        """Relationship IDs follow the coupled:{id_a}->{id_b} pattern."""
        commits = [
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
            ["src/auth.py", "src/models.py"],
        ]

        process_coupling(
            graph,
            Path("/fake/repo"),
            min_strength=0.3,
            commits=commits,
        )

        coupled_rels = graph.get_relationships_by_type(RelType.COUPLED_WITH)
        assert len(coupled_rels) >= 1

        for rel in coupled_rels:
            assert rel.id.startswith("coupled:")
            assert "->" in rel.id


# ---------------------------------------------------------------------------
# collect_coupling_commits / GitCollectionResult tests (W2.4)
#
# collect_coupling_commits is the thread-safe "collect" half of coupling:
# run_pipeline starts it in a background thread right after the file walk so
# the git-log subprocess wait (G11) overlaps with unrelated CPU phases
# instead of sitting serially at the end of the pipeline. It must never
# touch a KnowledgeGraph and must never raise -- both properties this class
# verifies directly, independent of run_pipeline's threading.
# ---------------------------------------------------------------------------


class TestCollectCouplingCommits:
    """collect_coupling_commits wraps parse_git_log for background-thread use."""

    def test_not_a_git_repo_degrades_gracefully(self, tmp_path: Path) -> None:
        """Mirrors parse_git_log's own "not a git repo" handling: commits=[],
        no error -- safe for a background thread to return."""
        outcome = collect_coupling_commits(tmp_path)

        assert isinstance(outcome, GitCollectionResult)
        assert outcome.commits == []
        assert outcome.error is None
        assert outcome.duration >= 0.0

    def test_success_matches_parse_git_log_directly(self, tmp_path: Path) -> None:
        """A real git repo yields the exact same commits parse_git_log
        itself would return -- collect_coupling_commits adds no filtering
        or transformation of its own."""
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        _git(tmp_path, "init", "-b", "main")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-m", "initial")

        outcome = collect_coupling_commits(tmp_path)

        assert outcome.error is None
        assert outcome.commits == [["a.py"]]
        assert outcome.commits == parse_git_log(tmp_path)

    def test_graph_files_filters_commit_output(self, tmp_path: Path) -> None:
        """graph_files restricts the returned commits to known files, the
        same filter process_coupling applies via parse_git_log today --
        this is what lets the background thread reproduce that filter
        without needing the graph itself (graph_files is derived from the
        walked file list instead; see run_pipeline)."""
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
        _git(tmp_path, "init", "-b", "main")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-m", "initial")

        outcome = collect_coupling_commits(tmp_path, graph_files={"a.py"})

        assert outcome.error is None
        assert outcome.commits == [["a.py"]]

    def test_unexpected_exception_is_captured_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exception parse_git_log does not already catch internally
        (it only catches CalledProcessError/FileNotFoundError, for the
        common "not a git repo" case) must be captured on the result
        instead of raised -- a background thread running this function must
        always return normally, never die with an unhandled exception."""

        class _BoomError(RuntimeError):
            pass

        def fake_run(*args, **kwargs):
            raise _BoomError("simulated unexpected subprocess failure")

        monkeypatch.setattr(subprocess, "run", fake_run)

        outcome = collect_coupling_commits(tmp_path)

        assert outcome.commits == []
        assert isinstance(outcome.error, _BoomError)
        assert outcome.duration >= 0.0
