"""End-to-end test for the full Synaptiq pipeline on a Ruby project.

Copies the ``tests/fixtures/ruby_project`` sample into a temp directory,
runs the full ingestion pipeline, and verifies that Ruby symbols, imports,
calls, class inheritance, module mixins, framework entry points, and
dead-code detection all flow through to the resulting knowledge graph.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from synaptiq.core.ingestion.pipeline import PipelineResult, run_pipeline
from synaptiq.core.storage.ladybug_backend import LadybugBackend
from synaptiq.mcp.tools import handle_dead_code, handle_query

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "ruby_project"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ruby_repo(tmp_path: Path) -> Path:
    """Copy the Ruby fixture project into an isolated temp directory."""
    dest = tmp_path / "ruby_project"
    shutil.copytree(_FIXTURE_DIR, dest)
    return dest


@pytest.fixture()
def storage(tmp_path: Path) -> LadybugBackend:
    """Provide an initialised LadybugBackend."""
    db_path = tmp_path / "ruby_e2e_db"
    backend = LadybugBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


@pytest.fixture()
def pipeline_result(ruby_repo: Path, storage: LadybugBackend) -> PipelineResult:
    """Run the full pipeline once over the Ruby fixture and return the result."""
    _, result = run_pipeline(ruby_repo, storage)
    return result


def _rel_count(storage: LadybugBackend, rel_type: str) -> int:
    rows = storage.execute_raw(
        "MATCH ()-[r:CodeRelation]->() "
        f"WHERE r.rel_type = '{rel_type}' "
        "RETURN count(r)"
    )
    return rows[0][0]


# ---------------------------------------------------------------------------
# Test: File and symbol discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    """The pipeline discovers all Ruby files and their symbols."""

    def test_discovers_all_files(self, pipeline_result: PipelineResult) -> None:
        # 5 .rb files: greeter, user, user_service, application_controller,
        # users_controller.
        assert pipeline_result.files == 5

    def test_minimum_symbols(self, pipeline_result: PipelineResult) -> None:
        # 1 module (Greeter) + 4 classes + 7 methods (greet, unused_greet,
        # initialize, display, find_user, authenticate, show).
        assert pipeline_result.symbols >= 12


# ---------------------------------------------------------------------------
# Test: Node labels — Module / Class / Method
# ---------------------------------------------------------------------------


class TestNodeLabels:
    """Ruby module, class, and method nodes are materialised."""

    def test_module_node(self, storage: LadybugBackend, pipeline_result: PipelineResult) -> None:
        rows = storage.execute_raw("MATCH (n:Module) RETURN n.name")
        names = {r[0] for r in rows}
        assert "Greeter" in names

    def test_class_nodes(self, storage: LadybugBackend, pipeline_result: PipelineResult) -> None:
        rows = storage.execute_raw("MATCH (n:Class) RETURN n.name")
        names = {r[0] for r in rows}
        assert {"User", "UserService", "ApplicationController", "UsersController"} <= names

    def test_method_nodes(self, storage: LadybugBackend, pipeline_result: PipelineResult) -> None:
        rows = storage.execute_raw("MATCH (n:Method) RETURN n.name")
        names = {r[0] for r in rows}
        assert {"greet", "initialize", "display", "find_user", "show"} <= names


# ---------------------------------------------------------------------------
# Test: Relationship types
# ---------------------------------------------------------------------------


class TestRelationships:
    """The expected Ruby relationship types are present."""

    def test_contains_and_defines(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        assert _rel_count(storage, "contains") > 0
        assert _rel_count(storage, "defines") > 0

    def test_imports_resolved(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # All four require_relative statements resolve to in-project files.
        assert _rel_count(storage, "imports") >= 4

    def test_calls_exist(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        assert _rel_count(storage, "calls") > 0

    def test_extends_edge(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # UsersController < ApplicationController.
        assert _rel_count(storage, "extends") >= 1

    def test_mixes_in_edge(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # User includes Greeter.
        assert _rel_count(storage, "mixes_in") >= 1


# ---------------------------------------------------------------------------
# Test: Entry points / processes
# ---------------------------------------------------------------------------


class TestEntryPoints:
    """Rails controller actions are detected as framework entry points."""

    def test_process_detected(self, pipeline_result: PipelineResult) -> None:
        assert pipeline_result.processes >= 1

    def test_controller_action_is_entry_point(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        node = storage.get_node(
            "method:app/controllers/users_controller.rb:UsersController.show"
        )
        assert node is not None
        assert node.is_entry_point is True


# ---------------------------------------------------------------------------
# Test: Dead code detection
# ---------------------------------------------------------------------------


class TestDeadCode:
    """An uncalled module method is flagged dead; a called one is not."""

    def test_dead_code_detected(self, pipeline_result: PipelineResult) -> None:
        assert pipeline_result.dead_code >= 1

    def test_unused_method_flagged(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        node = storage.get_node("method:lib/greeter.rb:Greeter.unused_greet")
        assert node is not None
        assert node.is_dead is True

    def test_called_method_not_flagged(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        node = storage.get_node("method:lib/greeter.rb:Greeter.greet")
        assert node is not None
        assert node.is_dead is False

    def test_initialize_exempt(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # Ruby constructor must never be flagged dead.
        node = storage.get_node("method:lib/user.rb:User.initialize")
        assert node is not None
        assert node.is_dead is False


# ---------------------------------------------------------------------------
# Test: MCP tool surfaces work over Ruby data
# ---------------------------------------------------------------------------


class TestMCPTools:
    """MCP tools return Ruby symbols and dead-code results."""

    def test_query_finds_symbol(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        result = handle_query(storage, "UserService")
        assert "UserService" in result

    def test_dead_code_tool_lists_unused(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        result = handle_dead_code(storage)
        assert "unused_greet" in result
