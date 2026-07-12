"""End-to-end test for the full Synaptiq pipeline on a Go project.

Copies the ``tests/fixtures/go_project`` sample into a temp directory, runs the
full ingestion pipeline, and verifies that Go symbols, imports (cross-package,
one edge per package file), calls (incl. cross-package resolution), struct/
interface embedding, framework entry points, dead-code detection, REST linking,
and the MCP tool surfaces all flow through to the resulting knowledge graph.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from synaptiq.core.ingestion.pipeline import PipelineResult, run_pipeline
from synaptiq.core.storage.ladybug_backend import LadybugBackend
from synaptiq.mcp.tools import handle_dead_code, handle_query

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "go_project"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def go_repo(tmp_path: Path) -> Path:
    """Copy the Go fixture project into an isolated temp directory."""
    dest = tmp_path / "go_project"
    shutil.copytree(_FIXTURE_DIR, dest)
    return dest


@pytest.fixture()
def storage(tmp_path: Path) -> LadybugBackend:
    """Provide an initialised LadybugBackend."""
    db_path = tmp_path / "go_e2e_db"
    backend = LadybugBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


@pytest.fixture()
def pipeline_result(go_repo: Path, storage: LadybugBackend) -> PipelineResult:
    """Run the full pipeline once over the Go fixture and return the result."""
    _, result = run_pipeline(go_repo, storage)
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
    """The pipeline discovers all Go files and their symbols."""

    def test_discovers_all_files(self, pipeline_result: PipelineResult) -> None:
        # 7 .go files: main, models/{base,user,io,user_test}, services/
        # user_service, handlers/user_handler. (go.mod is not a source file.)
        assert pipeline_result.files == 7

    def test_minimum_symbols(self, pipeline_result: PipelineResult) -> None:
        assert pipeline_result.symbols >= 20


# ---------------------------------------------------------------------------
# Test: Node labels — Module / Class / Interface / TypeAlias / Method
# ---------------------------------------------------------------------------


class TestNodeLabels:
    """Go package, struct, interface, alias, and method nodes are materialised."""

    def test_module_nodes(self, storage: LadybugBackend, pipeline_result: PipelineResult) -> None:
        names = {r[0] for r in storage.execute_raw("MATCH (n:Module) RETURN n.name")}
        # One module node per file's package clause.
        assert {"main", "models", "services", "handlers"} <= names

    def test_struct_nodes_are_classes(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        names = {r[0] for r in storage.execute_raw("MATCH (n:Class) RETURN n.name")}
        assert {"User", "Base", "UserService", "UserHandler"} <= names

    def test_interface_nodes(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        names = {r[0] for r in storage.execute_raw("MATCH (n:Interface) RETURN n.name")}
        assert {"Reader", "Closer", "ReadCloser", "Handler"} <= names

    def test_type_alias_node(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        names = {r[0] for r in storage.execute_raw("MATCH (n:TypeAlias) RETURN n.name")}
        assert "UserID" in names

    def test_method_nodes_with_receiver_class(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        rows = storage.execute_raw("MATCH (n:Method) RETURN n.name, n.class_name")
        by_name = {r[0]: r[1] for r in rows}
        assert by_name.get("Display") == "User"
        assert by_name.get("FindUser") == "UserService"
        assert by_name.get("Describe") == "Base"


# ---------------------------------------------------------------------------
# Test: Relationship types
# ---------------------------------------------------------------------------


class TestRelationships:
    """The expected Go relationship types are present (and absent)."""

    def test_contains_and_defines(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        assert _rel_count(storage, "contains") > 0
        assert _rel_count(storage, "defines") > 0

    def test_imports_fan_out_per_package_file(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # services imports models (4 package files) → 4 edges alone.
        assert _rel_count(storage, "imports") >= 4

    def test_calls_exist(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        assert _rel_count(storage, "calls") > 0

    def test_cross_package_call_resolved(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # services/UserService.build calls models.NewUser across packages.
        rows = storage.execute_raw(
            "MATCH (a)-[r:CodeRelation]->(b) "
            "WHERE r.rel_type = 'calls' AND a.name = 'build' AND b.name = 'NewUser' "
            "RETURN count(r)"
        )
        assert rows[0][0] >= 1

    def test_struct_embedding_extends(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # User embeds Base.
        rows = storage.execute_raw(
            "MATCH (a)-[r:CodeRelation]->(b) "
            "WHERE r.rel_type = 'extends' AND a.name = 'User' AND b.name = 'Base' "
            "RETURN count(r)"
        )
        assert rows[0][0] == 1

    def test_interface_embedding_extends(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # ReadCloser embeds Reader and Closer.
        rows = storage.execute_raw(
            "MATCH (a)-[r:CodeRelation]->(b) "
            "WHERE r.rel_type = 'extends' AND a.name = 'ReadCloser' "
            "RETURN b.name"
        )
        assert {r[0] for r in rows} == {"Reader", "Closer"}

    def test_no_implements_or_mixes_in(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # Interface satisfaction is not statically derived; Go has no mixins.
        assert _rel_count(storage, "implements") == 0
        assert _rel_count(storage, "mixes_in") == 0

    def test_uses_type_edges(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        assert _rel_count(storage, "uses_type") > 0


# ---------------------------------------------------------------------------
# Test: REST linking (cross-service)
# ---------------------------------------------------------------------------


class TestRestLinking:
    """The net/http HandleFunc route links to the http.Get client call."""

    def test_rest_link_created(self, pipeline_result: PipelineResult) -> None:
        assert pipeline_result.rest_links >= 1


# ---------------------------------------------------------------------------
# Test: Entry points / processes
# ---------------------------------------------------------------------------


class TestEntryPoints:
    """main is an entry point and flows are traced."""

    def test_process_detected(self, pipeline_result: PipelineResult) -> None:
        assert pipeline_result.processes >= 1

    def test_main_is_entry_point(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        node = storage.get_node("function:main.go:main")
        assert node is not None
        assert node.is_entry_point is True


# ---------------------------------------------------------------------------
# Test: Dead code detection
# ---------------------------------------------------------------------------


class TestDeadCode:
    """An uncalled unexported function is flagged dead; exemptions hold."""

    def test_dead_code_detected(self, pipeline_result: PipelineResult) -> None:
        assert pipeline_result.dead_code >= 1

    def test_unused_function_flagged(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        node = storage.get_node("function:models/user.go:unusedHelper")
        assert node is not None
        assert node.is_dead is True

    def test_called_unexported_method_not_flagged(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # build is unexported but reached via s.build(name).
        node = storage.get_node("method:services/user_service.go:UserService.build")
        assert node is not None
        assert node.is_dead is False

    def test_exported_symbol_exempt(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # NewUser is exported (upper-case) → never dead even with no local caller.
        node = storage.get_node("function:models/user.go:NewUser")
        assert node is not None
        assert node.is_dead is False

    def test_main_and_init_exempt(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        for node_id in ("function:main.go:main", "function:main.go:init"):
            node = storage.get_node(node_id)
            assert node is not None
            assert node.is_dead is False

    def test_test_file_helper_exempt(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        # An unexported helper in a _test.go file is exempt.
        node = storage.get_node("function:models/user_test.go:helperExpected")
        assert node is not None
        assert node.is_dead is False


# ---------------------------------------------------------------------------
# Test: MCP tool surfaces work over Go data
# ---------------------------------------------------------------------------


class TestMCPTools:
    """MCP tools return Go symbols and dead-code results."""

    def test_query_finds_symbol(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        result = handle_query(storage, "UserService")
        assert "UserService" in result

    def test_dead_code_tool_lists_unused(
        self, storage: LadybugBackend, pipeline_result: PipelineResult
    ) -> None:
        result = handle_dead_code(storage)
        assert "unusedHelper" in result
