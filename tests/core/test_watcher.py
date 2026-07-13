"""Tests for the watch mode module (watcher.py)."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from synaptiq.core.ingestion import watcher
from synaptiq.core.ingestion.pipeline import apply_reindex, parse_files, run_pipeline
from synaptiq.core.ingestion.walker import FileEntry, read_file
from synaptiq.core.ingestion.watcher import _apply_deletions, _prepare_entries
from synaptiq.core.storage.ladybug_backend import LadybugBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a small Python repository for watcher tests."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "app.py").write_text(
        "def hello():\n"
        "    return 'hello'\n",
        encoding="utf-8",
    )

    (src / "utils.py").write_text(
        "def helper():\n"
        "    pass\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def storage(tmp_path: Path) -> LadybugBackend:
    """Provide an initialised LadybugBackend for testing."""
    db_path = tmp_path / "test_db"
    backend = LadybugBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


def _reindex_files_compat(
    changed_paths: list[Path],
    repo_path: Path,
    storage: LadybugBackend,
    gitignore_patterns: list[str] | None = None,
) -> int:
    """Compatibility wrapper matching the old _reindex_files interface.

    Combines _prepare_entries + _apply_deletions + reindex_files.
    """
    entries, deleted = _prepare_entries(
        changed_paths, repo_path, gitignore_patterns
    )

    if deleted:
        _apply_deletions(storage, deleted)

    if entries:
        graph = parse_files(entries, repo_path)
        apply_reindex(entries, storage, graph)

    return len(entries)


# ---------------------------------------------------------------------------
# Tests: _read_file_entry
# ---------------------------------------------------------------------------


class TestReadFileEntry:
    """_read_file_entry reads a file and returns a FileEntry."""

    def test_reads_python_file(self, tmp_repo: Path) -> None:
        entry = read_file(tmp_repo, tmp_repo / "src" / "app.py")

        assert entry is not None
        assert entry.path == "src/app.py"
        assert entry.language == "python"
        assert "hello" in entry.content

    def test_returns_none_for_unsupported(self, tmp_repo: Path) -> None:
        readme = tmp_repo / "README.md"
        readme.write_text("# readme", encoding="utf-8")

        entry = read_file(tmp_repo, readme)

        assert entry is None

    def test_returns_none_for_missing(self, tmp_repo: Path) -> None:
        entry = read_file(tmp_repo, tmp_repo / "nonexistent.py")

        assert entry is None

    def test_returns_none_for_empty(self, tmp_repo: Path) -> None:
        empty = tmp_repo / "empty.py"
        empty.write_text("", encoding="utf-8")

        entry = read_file(tmp_repo, empty)

        assert entry is None


# ---------------------------------------------------------------------------
# Tests: reindex_files (pipeline function)
# ---------------------------------------------------------------------------


class TestReindexFiles:
    """reindex_files() correctly removes old nodes and adds new ones."""

    def test_reindex_updates_content(
        self, tmp_repo: Path, storage: LadybugBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)

        # Verify initial node exists.
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "hello" in node.content

        # Modify the file.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n"
            "    return 'goodbye'\n",
            encoding="utf-8",
        )

        # Re-read and reindex.
        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        graph = parse_files([entry], tmp_repo)
        apply_reindex([entry], storage, graph)

        # Verify updated node.
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "goodbye" in node.content

    def test_reindex_handles_new_symbols(
        self, tmp_repo: Path, storage: LadybugBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)

        # Add a new function to the file.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n"
            "    return 'hello'\n"
            "\n"
            "def world():\n"
            "    return 'world'\n",
            encoding="utf-8",
        )

        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        graph = parse_files([entry], tmp_repo)
        apply_reindex([entry], storage, graph)

        # Both symbols should exist.
        assert storage.get_node("function:src/app.py:hello") is not None
        assert storage.get_node("function:src/app.py:world") is not None

    def test_reindex_removes_deleted_symbols(
        self, tmp_repo: Path, storage: LadybugBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)
        assert storage.get_node("function:src/app.py:hello") is not None

        # Remove the function.
        (tmp_repo / "src" / "app.py").write_text(
            "# empty file\nX = 1\n",
            encoding="utf-8",
        )

        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        graph = parse_files([entry], tmp_repo)
        apply_reindex([entry], storage, graph)

        # Old symbol should be gone.
        assert storage.get_node("function:src/app.py:hello") is None


# ---------------------------------------------------------------------------
# Tests: watcher helpers (_prepare_entries + _apply_deletions)
# ---------------------------------------------------------------------------


class TestWatcherReindexFiles:
    """Watcher helpers filter and process changed paths."""

    def test_reindexes_changed_files(
        self, tmp_repo: Path, storage: LadybugBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Modify a file.
        app_path = tmp_repo / "src" / "app.py"
        app_path.write_text(
            "def hello():\n    return 'updated'\n",
            encoding="utf-8",
        )

        count = _reindex_files_compat([app_path], tmp_repo, storage)

        assert count == 1
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "updated" in node.content

    def test_skips_ignored_files(
        self, tmp_repo: Path, storage: LadybugBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Create a file in an ignored directory.
        cache_dir = tmp_repo / "__pycache__"
        cache_dir.mkdir()
        cached = cache_dir / "module.cpython-311.pyc"
        cached.write_bytes(b"\x00")

        count = _reindex_files_compat([cached], tmp_repo, storage)

        assert count == 0

    def test_skips_unsupported_files(
        self, tmp_repo: Path, storage: LadybugBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        readme = tmp_repo / "README.md"
        readme.write_text("# hello", encoding="utf-8")

        count = _reindex_files_compat([readme], tmp_repo, storage)

        assert count == 0

    def test_handles_deleted_files(
        self, tmp_repo: Path, storage: LadybugBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # File exists in storage but is now deleted from disk.
        deleted_path = tmp_repo / "src" / "app.py"
        assert storage.get_node("file:src/app.py:") is not None

        deleted_path.unlink()

        count = _reindex_files_compat([deleted_path], tmp_repo, storage)

        # Returns 0 because file no longer exists (was handled as deletion).
        assert count == 0

    def test_handles_multiple_files(
        self, tmp_repo: Path, storage: LadybugBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Modify both files.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n    return 'v2'\n",
            encoding="utf-8",
        )
        (tmp_repo / "src" / "utils.py").write_text(
            "def helper():\n    return 42\n",
            encoding="utf-8",
        )

        count = _reindex_files_compat(
            [tmp_repo / "src" / "app.py", tmp_repo / "src" / "utils.py"],
            tmp_repo,
            storage,
        )

        assert count == 2


# ---------------------------------------------------------------------------
# Tests: fingerprint helpers (skip-if-clean primitives)
# ---------------------------------------------------------------------------


class TestFingerprint:
    """_fingerprint / _content_hash back the skip-if-clean check."""

    def test_fingerprint_is_order_independent(self) -> None:
        a = watcher._fingerprint({"src/a.py": "h1", "src/b.py": "h2"})
        b = watcher._fingerprint({"src/b.py": "h2", "src/a.py": "h1"})
        assert a == b

    def test_fingerprint_changes_with_content(self) -> None:
        base = watcher._fingerprint({"src/a.py": "h1"})
        assert base != watcher._fingerprint({"src/a.py": "h2"})

    def test_fingerprint_distinguishes_deletion(self) -> None:
        present = watcher._fingerprint({"src/a.py": "h1"})
        deleted = watcher._fingerprint({"src/a.py": None})
        assert present != deleted

    def test_content_hash_is_stable_and_distinct(self) -> None:
        assert watcher._content_hash("x = 1\n") == watcher._content_hash("x = 1\n")
        assert watcher._content_hash("x = 1\n") != watcher._content_hash("x = 2\n")


# ---------------------------------------------------------------------------
# Tests: RebuildCoordinator + _GlobalPhaseScheduler (single-flight/debounce)
# ---------------------------------------------------------------------------


async def _cancel(task: asyncio.Task) -> None:
    """Cancel a background task and swallow the cancellation."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class TestRebuildCoordinator:
    """Single-flight guard shared by the watcher and the socket reindex."""

    async def test_serializes_concurrent_builds(self) -> None:
        """Two callers hitting the guard together never build concurrently."""
        coord = watcher.RebuildCoordinator()
        active = 0
        max_active = 0
        order: list[str] = []

        async def build(tag: str) -> str:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            order.append(tag)
            return tag

        results = await asyncio.gather(
            coord.run(lambda: build("socket")),
            coord.run(lambda: build("watcher")),
        )

        assert max_active == 1  # single-flight: never two builds at once
        assert set(results) == {"socket", "watcher"}
        assert set(order) == {"socket", "watcher"}

    async def test_run_returns_build_result(self) -> None:
        coord = watcher.RebuildCoordinator()

        async def build() -> str:
            return "stats-json"

        assert await coord.run(build) == "stats-json"


class TestGlobalPhaseScheduler:
    """Quiescence debounce, skip-if-clean, single-flight, max-staleness."""

    @pytest.fixture(autouse=True)
    def _short_intervals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Short debounce; a ceiling far above the test window unless overridden.
        monkeypatch.setattr(watcher, "GLOBAL_PHASE_INTERVAL", 0.05)
        monkeypatch.setattr(watcher, "MAX_STALENESS_SECONDS", 100.0)

    async def test_edit_burst_triggers_one_build_after_quiescence(self) -> None:
        """A burst of edits collapses into exactly one global build."""
        builds = 0

        async def on_build() -> bool:
            nonlocal builds
            builds += 1
            return True

        coord = watcher.RebuildCoordinator()
        sched = watcher._GlobalPhaseScheduler(coord, on_build, stopped=lambda: False)
        task = asyncio.create_task(sched.run())
        try:
            # Five edits spaced tighter than the debounce interval.
            for i in range(5):
                sched.notify({f"src/f{i}.py": f"hash{i}"})
                await asyncio.sleep(0.01)
            # Wait out the quiescence window plus margin.
            await asyncio.sleep(0.2)
            assert builds == 1
        finally:
            await _cancel(task)

    async def test_edit_during_build_yields_single_followup(self) -> None:
        """A change mid-build produces exactly one follow-up — no lost update."""
        builds = 0
        snapshots: list[dict[str, str | None]] = []
        build_started = asyncio.Event()
        release_build = asyncio.Event()

        async def on_build() -> bool:
            nonlocal builds
            builds += 1
            snapshots.append(dict(sched._changed))
            build_started.set()
            await release_build.wait()
            release_build.clear()
            return True

        coord = watcher.RebuildCoordinator()
        sched = watcher._GlobalPhaseScheduler(coord, on_build, stopped=lambda: False)
        task = asyncio.create_task(sched.run())
        try:
            sched.notify({"src/a.py": "h1"})
            await build_started.wait()  # first build is running
            build_started.clear()

            # An edit lands WHILE the first build is in flight.
            sched.notify({"src/b.py": "h2"})
            release_build.set()  # let the first build finish

            await build_started.wait()  # exactly one follow-up build runs
            release_build.set()  # let it finish
            await asyncio.sleep(0.2)

            assert builds == 2
            # The follow-up carried the mid-build change — nothing dropped.
            assert snapshots[0] == {"src/a.py": "h1"}
            assert snapshots[1] == {"src/a.py": "h1", "src/b.py": "h2"}
        finally:
            release_build.set()
            await _cancel(task)

    async def test_unchanged_repo_skips_pipeline(self) -> None:
        """A trigger reproducing the committed state does zero pipeline work."""
        builds = 0

        async def on_build() -> bool:
            nonlocal builds
            builds += 1
            return True

        coord = watcher.RebuildCoordinator()
        sched = watcher._GlobalPhaseScheduler(coord, on_build, stopped=lambda: False)
        task = asyncio.create_task(sched.run())
        try:
            sched.notify({"src/a.py": "h1"})
            await asyncio.sleep(0.2)
            assert builds == 1  # first build establishes the committed fingerprint

            # Re-notify identical content: fingerprint matches → skip-if-clean.
            sched.notify({"src/a.py": "h1"})
            await asyncio.sleep(0.2)
            assert builds == 1  # no additional pipeline work
        finally:
            await _cancel(task)

    async def test_socket_reindex_and_watcher_are_single_flight(self) -> None:
        """A watcher trigger during a socket reindex waits — never concurrent."""
        coord = watcher.RebuildCoordinator()
        active = 0
        max_active = 0
        socket_release = asyncio.Event()

        async def socket_build() -> str:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await socket_release.wait()
            active -= 1
            return "socket"

        async def watcher_on_build() -> bool:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return True

        sched = watcher._GlobalPhaseScheduler(coord, watcher_on_build, stopped=lambda: False)
        # A socket reindex acquires the shared guard first.
        socket_task = asyncio.create_task(coord.run(socket_build))
        sched_task = asyncio.create_task(sched.run())
        try:
            await asyncio.sleep(0.01)
            assert coord.busy

            # Fire the watcher while the socket build holds the guard.
            sched.notify({"src/a.py": "h1"})
            await asyncio.sleep(0.15)  # debounce elapses; watcher build wants in
            assert max_active == 1  # blocked on the guard — no concurrent build

            socket_release.set()  # release the socket reindex
            await socket_task
            await asyncio.sleep(0.15)  # watcher build now runs
            assert max_active == 1  # still strictly single-flight
        finally:
            socket_release.set()
            await _cancel(sched_task)

    async def test_max_staleness_forces_build_under_churn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Continuous churn that never quiesces still gets a bounded rebuild."""
        # Debounce far above the test window, so only the ceiling can fire.
        monkeypatch.setattr(watcher, "GLOBAL_PHASE_INTERVAL", 10.0)
        monkeypatch.setattr(watcher, "MAX_STALENESS_SECONDS", 0.15)
        builds = 0

        async def on_build() -> bool:
            nonlocal builds
            builds += 1
            return True

        coord = watcher.RebuildCoordinator()
        sched = watcher._GlobalPhaseScheduler(coord, on_build, stopped=lambda: False)
        task = asyncio.create_task(sched.run())

        async def churn() -> None:
            i = 0
            while True:
                sched.notify({f"src/f{i}.py": f"h{i}"})
                i += 1
                await asyncio.sleep(0.02)  # faster than the debounce

        churn_task = asyncio.create_task(churn())
        try:
            await asyncio.sleep(0.4)  # > MAX_STALENESS_SECONDS
            # Quiescence (10s) cannot have fired in 0.4s, so any build is the
            # max-staleness ceiling forcing progress despite non-stop churn.
            assert builds >= 1
        finally:
            await _cancel(churn_task)
            await _cancel(task)


# ---------------------------------------------------------------------------
# Watcher incremental global tier (W3.2e) — the two-tier interaction
# ---------------------------------------------------------------------------


class TestWatcherIncrementalTier:
    """The watcher keeps the lossy per-file immediate tier (apply_reindex) and
    layers a *scoped* incremental global tier on top. The crux this proves:
    even though apply_reindex DETACH-DELETEs a changed file (destroying inbound
    edges to it), the global tier's ``repair_inbound=True`` incremental apply
    restores them, so the result matches a full rebuild on the strict core.
    """

    _MODELS = (
        "class Account:\n"
        "    def save(self):\n"
        "        return 1\n"
        "\n\n"
        "def make_user(name):\n"
        "    acct = Account()\n"
        "    return acct\n"
    )
    _SERVICE = (
        "from pkg.models import make_user, Account\n"
        "\n\n"
        "class Service(Account):\n"
        "    def run(self):\n"
        "        return make_user('x')\n"
    )
    _REPORT = "from pkg.models import make_user\n\n\ndef report():\n    return make_user('y')\n"

    def _build_repo(self, root: Path) -> None:
        files = {
            "pkg/models.py": self._MODELS,
            "pkg/service.py": self._SERVICE,
            "pkg/report.py": self._REPORT,
        }
        for i in range(12):  # fillers keep a 1-file edit under the ratio gates
            files[f"pkg/f{i}.py"] = f"def fn{i}():\n    return {i}\n"
        for rel, content in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def test_immediate_tier_plus_incremental_global_equals_full_rebuild(self, tmp_path):
        from synaptiq.core.graph.model import NodeLabel, RelType, generate_id
        from synaptiq.core.ingestion.pipeline import (
            run_incremental,
            stamp_full_manifest,
        )

        root = tmp_path / "repo"
        self._build_repo(root)

        # 1. Full index into a backend + stamp the manifest.
        backend = LadybugBackend()
        backend.initialize(tmp_path / "watch_db")
        base_graph, _ = run_pipeline(root, None, skip_embeddings=True)
        backend.bulk_load(base_graph)
        stamp_full_manifest(backend, base_graph, tool_version="test", git_head="head0")

        make_user = generate_id(NodeLabel.FUNCTION, "pkg/models.py", "make_user")
        service_run = generate_id(NodeLabel.METHOD, "pkg/service.py", "Service.run")

        def _inbound_calls_make_user() -> set[str]:
            g = backend.load_graph()
            return {
                r.source
                for r in g.iter_relationships()
                if r.type is RelType.CALLS and r.target == make_user
            }

        assert service_run in _inbound_calls_make_user()  # baseline inbound edge

        # 2. Body-only edit on disk.
        (root / "pkg/models.py").write_text(
            self._MODELS.replace("acct = Account()", "acct = Account()  # tweak"),
            encoding="utf-8",
        )

        # 3. Immediate tier (per-file): parse + apply_reindex — the LOSSY tier
        #    that DETACH-DELETEs models.py and drops the inbound CALLS edge.
        entries = [read_file(root, root / "pkg/models.py")]
        graph = parse_files(entries, root)
        apply_reindex(entries, backend, graph)
        assert service_run not in _inbound_calls_make_user()  # damage confirmed

        # 4. Global tier: the watcher's incremental apply (repair_inbound=True).
        outcome = run_incremental(
            root, backend, tool_version="test", repair_inbound=True, git_head="head0"
        )
        assert not outcome.full_rebuild_required
        # The inbound edge the immediate tier destroyed is restored.
        assert service_run in _inbound_calls_make_user()

        incremental_graph = backend.load_graph()
        backend.close()

        # 5. Oracle: a full rebuild of the edited tree — strict-core equal.
        full_backend = LadybugBackend()
        full_backend.initialize(tmp_path / "watch_full_db")
        full_graph, _ = run_pipeline(root, None, skip_embeddings=True)
        full_backend.bulk_load(full_graph)
        full_loaded = full_backend.load_graph()
        full_backend.close()

        fringe = frozenset({NodeLabel.COMMUNITY, NodeLabel.PROCESS})
        inc_nodes = {n.id for n in incremental_graph.iter_nodes() if n.label not in fringe}
        full_nodes = {n.id for n in full_loaded.iter_nodes() if n.label not in fringe}
        assert inc_nodes == full_nodes

        def _edges(g, rt):
            return {(r.source, r.target) for r in g.iter_relationships() if r.type is rt}

        for rt in (RelType.CONTAINS, RelType.IMPORTS, RelType.EXTENDS, RelType.DEFINES):
            assert _edges(incremental_graph, rt) == _edges(full_loaded, rt), rt.value
        # Inbound + outbound high-confidence CALLS match too (no duplicates,
        # nothing lost) — the whole point of the repair.
        inc_calls = {
            (r.source, r.target)
            for r in incremental_graph.iter_relationships()
            if r.type is RelType.CALLS and (r.properties.get("confidence", 1.0) or 0) >= 0.8
        }
        full_calls = {
            (r.source, r.target)
            for r in full_loaded.iter_relationships()
            if r.type is RelType.CALLS and (r.properties.get("confidence", 1.0) or 0) >= 0.8
        }
        assert inc_calls == full_calls
