"""Phase 11: Change Coupling Analysis for Synaptiq.

Analyzes git history to discover files that frequently change together.
Co-change frequency is a strong indicator of logical coupling -- files that
must be modified in tandem likely share implicit dependencies that may not
be visible in static analysis alone.

The main entry point is :func:`process_coupling`, which parses the git log,
builds a co-change matrix, computes coupling strengths, and writes
``COUPLED_WITH`` relationships into the knowledge graph.

This module's work splits into two halves so ``run_pipeline`` (W2.4) can
overlap coupling's slow part with unrelated CPU phases instead of paying
for it serially at the end of the pipeline:

* **Collect** -- :func:`collect_coupling_commits` (wrapping
  :func:`parse_git_log`) runs the single GIL-releasing ``git log``
  subprocess (G11) and returns raw commit data.  It never touches a
  :class:`KnowledgeGraph`, so it is safe to run in a background thread.
* **Apply** -- :func:`process_coupling` turns commit data into
  ``COUPLED_WITH`` edges on the graph.  This half is not thread-safe
  (``KnowledgeGraph`` has no internal locking) and must run on the same
  thread that owns the graph.
"""

from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
)

logger = logging.getLogger(__name__)


def current_git_head(repo_path: Path) -> str | None:
    """Return the current ``git rev-parse HEAD`` sha, or ``None`` if unavailable.

    The cheap coupling-staleness gate for incremental indexing (design §9 / D8):
    ``COUPLED_WITH`` depends solely on git history, so an incremental apply can
    reuse the stored coupling as long as HEAD has not moved. Stamped into
    ``IndexManifest.git_head`` on a full build and compared here on the next
    consolidation decision. ``None`` for a non-git repo, a fresh repo with no
    commits, or a missing ``git`` binary — treated by callers as "unknown", which
    conservatively lets a consolidation proceed rather than trusting stale edges.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        logger.debug("git rev-parse HEAD failed for %s — not a git repo?", repo_path)
        return None
    head = result.stdout.strip()
    return head or None


def parse_git_log(
    repo_path: Path,
    since_months: int = 6,
    *,
    graph_files: set[str] | None = None,
) -> list[list[str]]:
    """Run ``git log`` and return commits as lists of changed file paths.

    Each inner list contains the file paths that were modified in a single
    commit.  Only files present in *graph_files* (when provided) are kept,
    so the output is already filtered to source files known to the graph.

    Args:
        repo_path: Root of the git repository.
        since_months: How far back in history to look.
        graph_files: Optional set of file paths present in the graph.
            When ``None``, no filtering is applied.

    Returns:
        A list of commits, each represented as a list of changed file paths.
        Returns an empty list when the git command fails (e.g. not a repo).
    """
    cmd = [
        "git",
        "log",
        "--name-only",
        '--pretty=format:COMMIT:%H',
        f"--since={since_months} months ago",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("git log failed for %s — not a git repo?", repo_path)
        return []

    commits: list[list[str]] = []
    current_files: list[str] = []

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("COMMIT:"):
            # Start of a new commit — flush the previous one.
            if current_files:
                commits.append(current_files)
            current_files = []
        else:
            if graph_files is None or stripped in graph_files:
                current_files.append(stripped)

    # Flush the last commit.
    if current_files:
        commits.append(current_files)

    return commits

def collect_coupling_commits(
    repo_path: Path,
    graph_files: set[str] | None = None,
    since_months: int = 6,
) -> list[list[str]]:
    """Run the git-log subprocess and return raw commit data (the "collect" half).

    This wraps :func:`parse_git_log` for use from a background worker: it never
    reads or writes a :class:`KnowledgeGraph`, so it can safely run concurrently
    with the CPU-bound phases in ``run_pipeline`` while they operate on the
    graph.  ``process_coupling`` (or its ``commits=`` fast path) is the "apply"
    half -- it must stay on the thread that owns the graph.

    Graceful degradation for git-unavailable is inherited from
    :func:`parse_git_log`, which catches the common "not a git repo" case
    (``CalledProcessError`` / ``FileNotFoundError``) and returns ``[]``.  Any
    OTHER, unexpected exception propagates unchanged: ``run_pipeline`` submits
    this on a :class:`~concurrent.futures.ThreadPoolExecutor`, so the worker's
    exception is captured on the Future and re-raised -- with its original
    traceback -- by ``Future.result()``. The pipeline then crashes exactly as a
    synchronous ``parse_git_log`` call would, instead of silently swallowing a
    real bug (review F4b).
    """
    return parse_git_log(repo_path, since_months=since_months, graph_files=graph_files)

def build_cochange_matrix(
    commits: list[list[str]],
    min_cochanges: int = 3,
    max_files_per_commit: int = 50,
) -> dict[tuple[str, str], int]:
    """Build a co-change frequency matrix from commit data.

    For every pair of files that appear in the same commit, their co-change
    count is incremented.  Only pairs whose count meets or exceeds
    *min_cochanges* are retained.

    Commits touching more than *max_files_per_commit* files are skipped
    (merge commits, bulk reformats) to avoid O(n^2) pair explosion and
    coupling noise.

    Keys are *sorted* tuples ``(file_a, file_b)`` so that ``(A, B)`` and
    ``(B, A)`` map to the same entry.

    Args:
        commits: List of commits, each a list of changed file paths.
        min_cochanges: Minimum co-change count to keep a pair.
        max_files_per_commit: Skip commits with more files than this.

    Returns:
        A dict mapping ``(file_a, file_b)`` sorted tuples to their count.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for files in commits:
        unique_files = sorted(set(files))
        if len(unique_files) > max_files_per_commit:
            continue
        for a, b in combinations(unique_files, 2):
            counts[(a, b)] += 1

    return {pair: count for pair, count in counts.items() if count >= min_cochanges}

def calculate_coupling(
    file_a: str,
    file_b: str,
    co_changes: int,
    total_changes: dict[str, int],
) -> float:
    """Compute the coupling strength between two files.

    The formula is::

        coupling = co_changes / max(total_changes[file_a], total_changes[file_b])

    This yields a value in ``[0.0, 1.0]`` — higher means more tightly coupled.

    Args:
        file_a: First file path.
        file_b: Second file path.
        co_changes: Number of commits where both files changed together.
        total_changes: Mapping of file path to its total commit count.

    Returns:
        A float between 0.0 and 1.0 representing coupling strength.
    """
    max_changes = max(total_changes.get(file_a, 0), total_changes.get(file_b, 0))
    if max_changes == 0:
        return 0.0
    return co_changes / max_changes

def process_coupling(
    graph: KnowledgeGraph,
    repo_path: Path,
    min_strength: float = 0.3,
    *,
    commits: list[list[str]] | None = None,
) -> int:
    """Analyze git history and create ``COUPLED_WITH`` relationships.

    Parses the git log (or uses pre-supplied *commits* for testing),
    computes pairwise coupling strengths, and adds edges between ``File``
    nodes that exceed *min_strength*.

    Args:
        graph: The knowledge graph containing ``File`` nodes.
        repo_path: Root of the git repository.
        min_strength: Minimum coupling strength to create a relationship.
        commits: Pre-parsed commit data.  When provided, ``parse_git_log``
            is skipped — useful for deterministic testing.

    Returns:
        The number of ``COUPLED_WITH`` relationships created.
    """
    file_nodes = graph.get_nodes_by_label(NodeLabel.FILE)
    graph_files: set[str] = {n.file_path for n in file_nodes}

    if commits is None:
        commits = parse_git_log(repo_path, graph_files=graph_files)

    # Build co-change matrix (threshold of 1 — we filter by strength later).
    cochange = build_cochange_matrix(commits, min_cochanges=1)

    # Count total changes per file (across all commits).
    total_changes: dict[str, int] = defaultdict(int)
    for files in commits:
        for f in set(files):
            total_changes[f] += 1

    path_to_id: dict[str, str] = {n.file_path: n.id for n in file_nodes}

    count = 0
    for (file_a, file_b), co_changes in cochange.items():
        strength = calculate_coupling(file_a, file_b, co_changes, total_changes)
        if strength < min_strength:
            continue

        id_a = path_to_id.get(file_a)
        id_b = path_to_id.get(file_b)
        if id_a is None or id_b is None:
            continue

        rel_id = f"coupled:{id_a}->{id_b}"
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.COUPLED_WITH,
                source=id_a,
                target=id_b,
                properties={"strength": strength, "co_changes": co_changes},
            )
        )
        count += 1

    logger.info("Created %d COUPLED_WITH relationships", count)
    return count
