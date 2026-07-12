"""Tests for the pipeline orchestrator (pipeline.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synaptiq.core.ingestion.pipeline import (
    PipelineResult,
    apply_reindex,
    build_full_index,
    commit_full_index,
    parse_files,
    run_pipeline,
)
from synaptiq.core.ingestion.walker import FileEntry
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


# ---------------------------------------------------------------------------
# apply_reindex: FTS staleness contract (W1.2 -- G3)
#
# apply_reindex is the watcher's per-file-save path.  It must NOT rebuild FTS
# (BM25) indexes: `rebuild_fts_indexes()` (DROP_FTS_INDEX + CREATE_FTS_INDEX
# per table) is an O(whole corpus) operation, so paying it on every single
# save is the bug this package fixes (G3).  A guaranteed full rebuild still
# happens at the next global-phase commit (`build_full_index` +
# `commit_full_index` -> `storage.bulk_load`, which unconditionally rebuilds
# every searchable index -- see `KuzuBackend.bulk_load`).
#
# NOTE on what "stale" means here: empirically, on the exact pinned
# ``kuzu==0.11.3`` (see W1.8 -- upstream is archived, so this is permanent for
# this codebase), `QUERY_FTS_INDEX` already reflects rows inserted or deleted
# on the same live connection *without* an explicit rebuild -- this is not
# documented as a guaranteed contract by Kuzu, just observed behavior of this
# pinned version. These tests therefore do NOT assert that new content is
# hidden from FTS pre-rebuild (that would be an assertion about internal
# Kuzu behavior, not about apply_reindex). They assert the properties that
# actually matter and that the code guarantees: no per-save rebuild call, no
# errors ever, unaffected content stays correct, and a real rebuild always
# happens at the next global phase.  See `apply_reindex`'s docstring for the
# full contract.
# ---------------------------------------------------------------------------


class TestApplyReindexFtsStaleness:
    """apply_reindex defers FTS rebuilds to the next global-phase commit."""

    def test_apply_reindex_does_not_rebuild_fts(
        self, tmp_repo: Path, storage: KuzuBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-save path must never call rebuild_fts_indexes."""
        # Initial full index -- this DOES rebuild FTS once, via bulk_load.
        run_pipeline(tmp_repo, storage)

        calls: list[None] = []
        monkeypatch.setattr(storage, "rebuild_fts_indexes", lambda: calls.append(None))

        (tmp_repo / "src" / "main.py").write_text(
            "from .auth import validate\n"
            "\n"
            "def main():\n"
            "    validate()\n"
            "\n"
            "def extra():\n"
            "    pass\n",
            encoding="utf-8",
        )
        entry = FileEntry(
            path="src/main.py",
            content=(tmp_repo / "src" / "main.py").read_text(),
            language="python",
        )
        graph = parse_files([entry], tmp_repo)
        apply_reindex([entry], storage, graph)

        assert calls == []

    def test_fts_search_keeps_working_after_apply_reindex_without_rebuild(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        """FTS search must keep functioning -- never erroring -- once
        apply_reindex stops rebuilding it, and unaffected content must stay
        correctly searchable throughout.  The graph and exact-match paths
        reflect the change immediately either way (never FTS-gated)."""
        run_pipeline(tmp_repo, storage)

        # Rename auth.py's only symbol: its old node is deleted and a new
        # one inserted -- exercises both sides of the FTS index's contents.
        (tmp_repo / "src" / "auth.py").write_text(
            "from .utils import helper\n"
            "\n"
            "def brand_new_symbol():\n"
            "    helper()\n",
            encoding="utf-8",
        )
        entry = FileEntry(
            path="src/auth.py",
            content=(tmp_repo / "src" / "auth.py").read_text(),
            language="python",
        )
        graph = parse_files([entry], tmp_repo)
        apply_reindex([entry], storage, graph)

        # The graph reflects the change immediately (never FTS-gated).
        assert storage.get_node("function:src/auth.py:brand_new_symbol") is not None
        assert storage.get_node("function:src/auth.py:validate") is None

        # None of these may raise: fts_search catches per-table query
        # failures internally, so a stale or mid-mutation FTS index must
        # degrade gracefully rather than propagate an error. Covers a
        # deleted row's old name, a freshly inserted row, and untouched
        # content, in one sweep.
        for query in ("validate", "brand_new_symbol", "helper"):
            results = storage.fts_search(query, limit=10)
            assert isinstance(results, list)

        # An untouched file's content is unaffected by another file's
        # mutation and must still be found correctly.
        untouched_results = storage.fts_search("helper", limit=10)
        assert any(r.node_name == "helper" for r in untouched_results)

    def test_global_rebuild_refreshes_fts_after_apply_reindex(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        """The next global-phase rebuild guarantees FTS reflects the
        changes -- the same `build_full_index` + `commit_full_index`
        machinery used by both the watcher's `_on_build` and the socket
        `reindex` handler."""
        run_pipeline(tmp_repo, storage)

        (tmp_repo / "src" / "auth.py").write_text(
            "from .utils import helper\n"
            "\n"
            "def brand_new_symbol():\n"
            "    helper()\n",
            encoding="utf-8",
        )
        entry = FileEntry(
            path="src/auth.py",
            content=(tmp_repo / "src" / "auth.py").read_text(),
            language="python",
        )
        graph = parse_files([entry], tmp_repo)
        apply_reindex([entry], storage, graph)

        # Simulate the watcher's global phase: build_full_index (CPU work,
        # off the write lock) + commit_full_index (bulk_load, under the
        # write lock) -- exactly what watcher.py's `_on_build` and the
        # socket `reindex` handler both run. This is the guaranteed,
        # version-independent rebuild point (unlike apply_reindex alone,
        # it does not depend on Kuzu's internal FTS update behavior).
        full_graph, embeddings, _result = build_full_index(tmp_repo, skip_embeddings=True)
        commit_full_index(storage, full_graph, embeddings)

        results = storage.fts_search("brand_new_symbol", limit=10)
        assert any(r.node_name == "brand_new_symbol" for r in results)

        # The old, now-removed "validate" function node is gone from the
        # rebuilt database -- main.py's body still literally contains the
        # substring "validate()" as a call site, so it legitimately still
        # matches the query; what must NOT appear is the deleted node itself.
        assert storage.get_node("function:src/auth.py:validate") is None
        stale_node_id = "function:src/auth.py:validate"
        assert all(r.node_id != stale_node_id for r in storage.fts_search("validate", limit=10))
