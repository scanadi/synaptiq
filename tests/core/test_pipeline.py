"""Tests for the pipeline orchestrator (pipeline.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synaptiq.core.ingestion.pipeline import PipelineResult, run_pipeline
from synaptiq.core.storage.kuzu_backend import KuzuBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a small Python repository under a temporary directory.

    Layout::

        tmp_repo/
        +-- src/
            +-- main.py    (imports validate from auth, calls it)
            +-- auth.py    (imports helper from utils, calls it)
            +-- utils.py   (standalone helper function)
    """
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text(
        "from .auth import validate\n"
        "\n"
        "def main():\n"
        "    validate()\n",
        encoding="utf-8",
    )

    (src / "auth.py").write_text(
        "from .utils import helper\n"
        "\n"
        "def validate():\n"
        "    helper()\n",
        encoding="utf-8",
    )

    (src / "utils.py").write_text(
        "def helper():\n"
        "    pass\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def storage(tmp_path: Path) -> KuzuBackend:
    """Provide an initialised KuzuBackend for testing."""
    db_path = tmp_path / "test_db"
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# test_run_pipeline_basic
# ---------------------------------------------------------------------------


class TestRunPipelineBasic:
    """run_pipeline completes without error and returns a PipelineResult."""

    def test_run_pipeline_basic(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(tmp_repo, storage)

        assert isinstance(result, PipelineResult)
        assert result.duration_seconds > 0.0


# ---------------------------------------------------------------------------
# test_run_pipeline_file_count
# ---------------------------------------------------------------------------


class TestRunPipelineFileCount:
    """The result reports exactly 3 files from the fixture repo."""

    def test_run_pipeline_file_count(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(tmp_repo, storage)

        assert result.files == 3


# ---------------------------------------------------------------------------
# test_run_pipeline_finds_symbols
# ---------------------------------------------------------------------------


class TestRunPipelineFindsSymbols:
    """At least 3 symbols are discovered (main, validate, helper)."""

    def test_run_pipeline_finds_symbols(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(tmp_repo, storage)

        assert result.symbols >= 3


# ---------------------------------------------------------------------------
# test_run_pipeline_finds_relationships
# ---------------------------------------------------------------------------


class TestRunPipelineFindsRelationships:
    """Relationships are created (CONTAINS, DEFINES, IMPORTS, CALLS)."""

    def test_run_pipeline_finds_relationships(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(tmp_repo, storage)

        assert result.relationships > 0


# ---------------------------------------------------------------------------
# test_run_pipeline_progress_callback
# ---------------------------------------------------------------------------


class TestRunPipelineProgressCallback:
    """The progress callback is invoked with expected phase names."""

    def test_run_pipeline_progress_callback(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        calls: list[tuple[str, float]] = []

        def callback(phase: str, pct: float) -> None:
            calls.append((phase, pct))

        run_pipeline(tmp_repo, storage, progress_callback=callback)

        # At minimum, every phase should report start (0.0) and end (1.0).
        assert len(calls) >= 2

        phase_names = {name for name, _ in calls}
        assert "Walking files" in phase_names
        assert "Processing structure" in phase_names
        assert "Parsing code" in phase_names
        assert "Resolving imports" in phase_names
        assert "Tracing calls" in phase_names
        assert "Extracting heritage" in phase_names
        assert "Loading to storage" in phase_names


# ---------------------------------------------------------------------------
# test_run_pipeline_loads_to_storage
# ---------------------------------------------------------------------------


class TestRunPipelineLoadsToStorage:
    """After the pipeline runs, nodes are retrievable from storage."""

    def test_run_pipeline_loads_to_storage(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # File nodes should be stored. The walker produces paths relative to
        # repo root, so "src/main.py" should exist as a File node.
        node = storage.get_node("file:src/main.py:")
        assert node is not None
        assert node.name == "main.py"


# ---------------------------------------------------------------------------
# Richer fixture for full-phase tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def rich_repo(tmp_path: Path) -> Path:
    """Create a repository with classes and type annotations for phases 7-11.

    Layout::

        rich_repo/
        +-- src/
            +-- models.py   (User class)
            +-- auth.py     (validate function using User type, calls check)
            +-- check.py    (check function, calls verify)
            +-- verify.py   (verify function -- standalone, no callers)
            +-- unused.py   (orphan function -- dead code candidate)
    """
    src = tmp_path / "src"
    src.mkdir()

    (src / "models.py").write_text(
        "class User:\n"
        "    def __init__(self, name: str):\n"
        "        self.name = name\n",
        encoding="utf-8",
    )

    (src / "auth.py").write_text(
        "from .models import User\n"
        "from .check import check\n"
        "\n"
        "def validate(user: User) -> bool:\n"
        "    return check(user)\n",
        encoding="utf-8",
    )

    (src / "check.py").write_text(
        "from .verify import verify\n"
        "\n"
        "def check(obj) -> bool:\n"
        "    return verify(obj)\n",
        encoding="utf-8",
    )

    (src / "verify.py").write_text(
        "def verify(obj) -> bool:\n"
        "    return obj is not None\n",
        encoding="utf-8",
    )

    (src / "unused.py").write_text(
        "def orphan_func():\n"
        "    pass\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def rich_storage(tmp_path: Path) -> KuzuBackend:
    """Provide an initialised KuzuBackend for the rich repo tests."""
    db_path = tmp_path / "rich_db"
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# test_run_pipeline_full_phases
# ---------------------------------------------------------------------------


class TestRunPipelineFullPhases:
    """Pipeline phases 7-11 populate the corresponding PipelineResult fields."""

    def test_run_pipeline_full_phases(
        self, rich_repo: Path, rich_storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(rich_repo, rich_storage)

        # Basic sanity checks.
        assert isinstance(result, PipelineResult)
        assert result.files == 5
        assert result.symbols >= 5  # User, __init__, validate, check, verify, orphan_func
        assert result.relationships > 0
        assert result.duration_seconds > 0.0

        # Phase 8 (communities) and Phase 9 (processes) return ints >= 0.
        # The exact count depends on the graph structure, but they must be
        # non-negative integers.
        assert isinstance(result.clusters, int)
        assert result.clusters >= 0

        assert isinstance(result.processes, int)
        assert result.processes >= 0

        # Phase 10 (dead code): orphan_func has no callers and is not a
        # constructor, test function, or dunder -- it should be flagged.
        assert isinstance(result.dead_code, int)
        assert result.dead_code >= 1

        # Phase 11 (coupling): no git repo, so coupling should be 0.
        assert isinstance(result.coupled_pairs, int)
        assert result.coupled_pairs == 0


# ---------------------------------------------------------------------------
# test_run_pipeline_phase_timings
# ---------------------------------------------------------------------------

# Phases that always run, regardless of storage/embeddings (see run_pipeline).
_CORE_PHASES = (
    "Walking files",
    "Processing structure",
    "Parsing code",
    "Resolving imports",
    "Tracing calls",
    "Linking REST endpoints",
    "Extracting heritage",
    "Analyzing types",
    "Detecting communities",
    "Detecting execution flows",
    "Finding dead code",
    "Analyzing git history",
)


def _assert_timings_sum_close(
    phase_timings: dict[str, float], duration_seconds: float, tolerance: float = 0.2
) -> None:
    """Assert phase_timings sums to approximately duration_seconds.

    Each phase is timed as a non-overlapping sub-interval of the pipeline's
    total wall-clock run, so the sum can never legitimately exceed the total
    (a tiny epsilon absorbs float rounding). The lower bound is generous
    because a little bit of work (e.g. the final symbol-counting pass, the
    pre-bulk_load embedding snapshot) is intentionally left untimed.
    """
    total_timed = sum(phase_timings.values())
    assert total_timed <= duration_seconds + 1e-3, (
        f"timed phases ({total_timed}) exceed duration_seconds ({duration_seconds})"
    )
    assert total_timed >= duration_seconds * (1 - tolerance), (
        f"timed phases ({total_timed}) too far below duration_seconds ({duration_seconds})"
    )


class TestRunPipelinePhaseTimingsWithStorage:
    """phase_timings is populated for every phase, including storage/embeddings."""

    def test_run_pipeline_phase_timings_with_storage(
        self, rich_repo: Path, rich_storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(rich_repo, rich_storage)

        assert isinstance(result.phase_timings, dict)

        expected_phases = {*_CORE_PHASES, "Loading to storage", "Generating embeddings"}
        assert expected_phases.issubset(result.phase_timings.keys())

        for phase, seconds in result.phase_timings.items():
            assert isinstance(seconds, float)
            assert seconds >= 0.0, f"{phase} has a negative duration"

        _assert_timings_sum_close(result.phase_timings, result.duration_seconds)


class TestRunPipelinePhaseTimingsNoStorage:
    """Without a storage backend, only the 12 core phases are timed."""

    def test_run_pipeline_phase_timings_no_storage(self, rich_repo: Path) -> None:
        _, result = run_pipeline(rich_repo)

        assert set(_CORE_PHASES).issubset(result.phase_timings.keys())
        assert "Loading to storage" not in result.phase_timings
        assert "Generating embeddings" not in result.phase_timings

        _assert_timings_sum_close(result.phase_timings, result.duration_seconds)


class TestRunPipelinePhaseTimingsSkipEmbeddings:
    """skip_embeddings=True omits the embeddings phase but keeps storage load."""

    def test_run_pipeline_phase_timings_skip_embeddings(
        self, rich_repo: Path, rich_storage: KuzuBackend
    ) -> None:
        _, result = run_pipeline(rich_repo, rich_storage, skip_embeddings=True)

        assert "Loading to storage" in result.phase_timings
        assert "Generating embeddings" not in result.phase_timings

        _assert_timings_sum_close(result.phase_timings, result.duration_seconds)


# ---------------------------------------------------------------------------
# test_run_pipeline_progress_includes_new_phases
# ---------------------------------------------------------------------------


class TestRunPipelineProgressIncludesNewPhases:
    """Progress callback includes phase names for phases 7-11."""

    def test_run_pipeline_progress_includes_new_phases(
        self, rich_repo: Path, rich_storage: KuzuBackend
    ) -> None:
        calls: list[tuple[str, float]] = []

        def callback(phase: str, pct: float) -> None:
            calls.append((phase, pct))

        run_pipeline(rich_repo, rich_storage, progress_callback=callback)

        phase_names = {name for name, _ in calls}

        # Phases 1-6 (existing).
        assert "Walking files" in phase_names
        assert "Processing structure" in phase_names
        assert "Parsing code" in phase_names
        assert "Resolving imports" in phase_names
        assert "Tracing calls" in phase_names
        assert "Extracting heritage" in phase_names

        # Phases 7-11 (new).
        assert "Analyzing types" in phase_names
        assert "Detecting communities" in phase_names
        assert "Detecting execution flows" in phase_names
        assert "Finding dead code" in phase_names
        assert "Analyzing git history" in phase_names

        # Storage loading (always present).
        assert "Loading to storage" in phase_names

        # Every phase reports both start (0.0) and end (1.0).
        for phase_name in phase_names:
            phase_pcts = {pct for name, pct in calls if name == phase_name}
            assert 0.0 in phase_pcts, f"{phase_name} missing 0.0 progress"
            assert 1.0 in phase_pcts, f"{phase_name} missing 1.0 progress"
