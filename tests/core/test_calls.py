"""Tests for the call tracing phase (Phase 5)."""

from __future__ import annotations

import copy
import time

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from synaptiq.core.ingestion import calls as calls_module
from synaptiq.core.ingestion.calls import (
    process_calls,
    resolve_call,
)
from synaptiq.core.ingestion.parser_phase import FileParseData, assign_symbol_ids
from synaptiq.core.ingestion.symbol_lookup import (
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)
from synaptiq.core.parsers.base import CallInfo, ParseResult, SymbolInfo, VarTypeInfo

_CALLABLE_LABELS = (NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _add_file_node(graph: KnowledgeGraph, path: str) -> str:
    """Add a File node and return its ID."""
    node_id = generate_id(NodeLabel.FILE, path)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=NodeLabel.FILE,
            name=path.rsplit("/", 1)[-1],
            file_path=path,
        )
    )
    return node_id


def _add_symbol_node(
    graph: KnowledgeGraph,
    label: NodeLabel,
    file_path: str,
    name: str,
    start_line: int,
    end_line: int,
    class_name: str = "",
) -> str:
    """Add a symbol node with a DEFINES relationship from the file node."""
    symbol_name = (
        f"{class_name}.{name}" if label == NodeLabel.METHOD and class_name else name
    )
    node_id = generate_id(label, file_path, symbol_name)
    graph.add_node(
        GraphNode(
            id=node_id,
            label=label,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            class_name=class_name,
        )
    )
    file_id = generate_id(NodeLabel.FILE, file_path)
    graph.add_relationship(
        GraphRelationship(
            id=f"defines:{file_id}->{node_id}",
            type=RelType.DEFINES,
            source=file_id,
            target=node_id,
        )
    )
    return node_id


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the test fixture specification.

    File: src/auth.py
        Function: validate (lines 1-10)
        Function: hash_password (lines 12-20)

    File: src/app.py
        Function: login (lines 1-15)

    File: src/utils.py
        Function: helper (lines 1-5)
    """
    g = KnowledgeGraph()

    # Files
    _add_file_node(g, "src/auth.py")
    _add_file_node(g, "src/app.py")
    _add_file_node(g, "src/utils.py")

    # Symbols in src/auth.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth.py", "validate", 1, 10)
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth.py", "hash_password", 12, 20)

    # Symbols in src/app.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/app.py", "login", 1, 15)

    # Symbols in src/utils.py
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/utils.py", "helper", 1, 5)

    return g


@pytest.fixture()
def parse_data() -> list[FileParseData]:
    """Parse data with calls matching the fixture specification.

    src/auth.py: hash_password() at line 5 (inside validate)
    src/app.py: validate() at line 8 (inside login)
    """
    return [
        FileParseData(
            file_path="src/auth.py",
            language="python",
            parse_result=ParseResult(
                calls=[CallInfo(name="hash_password", line=5)],
            ),
        ),
        FileParseData(
            file_path="src/app.py",
            language="python",
            parse_result=ParseResult(
                calls=[CallInfo(name="validate", line=8)],
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# build_name_index (callable labels)
# ---------------------------------------------------------------------------


class TestBuildCallIndex:
    """build_name_index creates correct mapping from graph symbol nodes."""

    def test_build_call_index(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)

        # All four functions should appear.
        assert "validate" in index
        assert "hash_password" in index
        assert "login" in index
        assert "helper" in index

        # Each name maps to exactly one node ID.
        assert len(index["validate"]) == 1
        assert len(index["hash_password"]) == 1

        # IDs match expected generate_id output.
        expected_validate = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        assert index["validate"] == [expected_validate]

    def test_build_call_index_includes_classes(self) -> None:
        """Class nodes are included (for constructor calls)."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/models.py")
        _add_symbol_node(g, NodeLabel.CLASS, "src/models.py", "User", 1, 20)

        index = build_name_index(g, _CALLABLE_LABELS)
        assert "User" in index
        assert len(index["User"]) == 1

    def test_build_call_index_multiple_same_name(self) -> None:
        """Multiple symbols with the same name produce a list with all IDs."""
        g = KnowledgeGraph()
        _add_file_node(g, "src/a.py")
        _add_file_node(g, "src/b.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/a.py", "init", 1, 5)
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/b.py", "init", 1, 5)

        index = build_name_index(g, _CALLABLE_LABELS)
        assert "init" in index
        assert len(index["init"]) == 2


# ---------------------------------------------------------------------------
# resolve_call — same-file
# ---------------------------------------------------------------------------


class TestResolveCallSameFile:
    """hash_password call in auth.py resolves locally (confidence 1.0)."""

    def test_resolve_call_same_file(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="hash_password", line=5)

        target_id, confidence = resolve_call(
            call, "src/auth.py", index, graph
        )

        expected_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "hash_password"
        )
        assert target_id == expected_id
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# resolve_call — global fuzzy
# ---------------------------------------------------------------------------


class TestResolveCallGlobal:
    """validate call in app.py resolves globally (confidence 0.5)."""

    def test_resolve_call_global(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="validate", line=8)

        target_id, confidence = resolve_call(
            call, "src/app.py", index, graph
        )

        expected_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        assert target_id == expected_id
        assert confidence == 0.5


# ---------------------------------------------------------------------------
# resolve_call — unresolved
# ---------------------------------------------------------------------------


class TestResolveCallUnresolved:
    """Call to unknown function returns None."""

    def test_resolve_call_unresolved(self, graph: KnowledgeGraph) -> None:
        index = build_name_index(graph, _CALLABLE_LABELS)
        call = CallInfo(name="nonexistent_function", line=3)

        target_id, confidence = resolve_call(
            call, "src/auth.py", index, graph
        )

        assert target_id is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# process_calls — creates relationships
# ---------------------------------------------------------------------------


class TestProcessCallsCreatesRelationships:
    """process_calls creates CALLS edges in the graph."""

    def test_process_calls_creates_relationships(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        process_calls(parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)
        assert len(calls_rels) == 2

        # Collect source->target pairs.
        pairs = {(r.source, r.target) for r in calls_rels}

        validate_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        hash_pw_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "hash_password"
        )
        login_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "login")

        # validate -> hash_password (same-file call at line 5 inside validate)
        assert (validate_id, hash_pw_id) in pairs
        # login -> validate (cross-file call at line 8 inside login)
        assert (login_id, validate_id) in pairs


# ---------------------------------------------------------------------------
# process_calls — confidence scores
# ---------------------------------------------------------------------------


class TestProcessCallsConfidence:
    """Confidence scores are set correctly on CALLS relationships."""

    def test_process_calls_confidence(
        self,
        graph: KnowledgeGraph,
        parse_data: list[FileParseData],
    ) -> None:
        process_calls(parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)

        validate_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        hash_pw_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "hash_password"
        )
        login_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "login")

        confidences = {(r.source, r.target): r.properties["confidence"] for r in calls_rels}

        # Same-file call: confidence 1.0
        assert confidences[(validate_id, hash_pw_id)] == 1.0
        # Cross-file global match: confidence 0.5
        assert confidences[(login_id, validate_id)] == 0.5


# ---------------------------------------------------------------------------
# process_calls — no duplicates
# ---------------------------------------------------------------------------


class TestProcessCallsNoDuplicates:
    """Same call twice does not create duplicate edges."""

    def test_process_calls_no_duplicates(
        self, graph: KnowledgeGraph
    ) -> None:
        # Two identical calls to hash_password inside validate.
        duplicate_parse_data = [
            FileParseData(
                file_path="src/auth.py",
                language="python",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="hash_password", line=5),
                        CallInfo(name="hash_password", line=7),
                    ],
                ),
            ),
        ]

        process_calls(duplicate_parse_data, graph)

        calls_rels = graph.get_relationships_by_type(RelType.CALLS)
        # Both calls resolve to validate -> hash_password, but only one
        # relationship should exist.
        assert len(calls_rels) == 1


class TestRubyBuiltinBlocklist:
    """Ruby Kernel/builtin names never produce CALLS edges.

    Even when a same-named symbol happens to be defined in the codebase,
    calls to blocklisted Ruby builtins (``puts``, ``new``, ``each`` …) must
    be filtered out before resolution so they don't create spurious edges.
    """

    def test_blocklisted_ruby_builtins_no_edges(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "app/widget.rb")

        # A method that would be the resolution target if not blocklisted.
        caller_id = _add_symbol_node(
            g, NodeLabel.METHOD, "app/widget.rb", "render", 1, 10, class_name="Widget"
        )
        # Decoy same-named definitions to prove blocklisting wins over a
        # would-be same-file exact match.
        for builtin in ("puts", "each", "new"):
            _add_symbol_node(
                g, NodeLabel.METHOD, "app/widget.rb", builtin, 20, 22, class_name="Widget"
            )

        parse_data = [
            FileParseData(
                file_path="app/widget.rb",
                language="ruby",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(name="puts", line=3),
                        CallInfo(name="each", line=4),
                        CallInfo(name="new", line=5, receiver="User"),
                        CallInfo(name="require", line=6),
                    ],
                ),
            ),
        ]

        process_calls(parse_data, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        targets = {r.target for r in calls_rels}
        for builtin in ("puts", "each", "new"):
            builtin_id = generate_id(NodeLabel.METHOD, "app/widget.rb", f"Widget.{builtin}")
            assert builtin_id not in targets
        # Every call in this fixture is a blocklisted builtin, so no CALLS
        # edge should originate from the caller at all.
        assert caller_id not in {r.source for r in calls_rels}
        assert len(calls_rels) == 0


# ---------------------------------------------------------------------------
# resolve_call — self.method()
# ---------------------------------------------------------------------------


class TestResolveMethodCallSelf:
    """self.method() resolves within the same class."""

    def test_resolve_method_call_self(self) -> None:
        g = KnowledgeGraph()

        _add_file_node(g, "src/service.py")
        _add_symbol_node(
            g,
            NodeLabel.CLASS,
            "src/service.py",
            "AuthService",
            1,
            30,
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.py",
            "login",
            3,
            15,
            class_name="AuthService",
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.py",
            "check_token",
            17,
            28,
            class_name="AuthService",
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="check_token", line=10, receiver="self")

        target_id, confidence = resolve_call(
            call, "src/service.py", index, g
        )

        expected_id = generate_id(
            NodeLabel.METHOD, "src/service.py", "AuthService.check_token"
        )
        assert target_id == expected_id
        assert confidence == 1.0

    def test_resolve_method_call_this(self) -> None:
        """this.method() also resolves within the same class."""
        g = KnowledgeGraph()

        _add_file_node(g, "src/service.ts")
        _add_symbol_node(
            g,
            NodeLabel.CLASS,
            "src/service.ts",
            "AuthService",
            1,
            30,
        )
        _add_symbol_node(
            g,
            NodeLabel.METHOD,
            "src/service.ts",
            "checkToken",
            17,
            28,
            class_name="AuthService",
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="checkToken", line=10, receiver="this")

        target_id, confidence = resolve_call(
            call, "src/service.ts", index, g
        )

        expected_id = generate_id(
            NodeLabel.METHOD, "src/service.ts", "AuthService.checkToken"
        )
        assert target_id == expected_id
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# resolve_call — import-resolved
# ---------------------------------------------------------------------------


class TestResolveCallImportResolved:
    """Calls to imported symbols resolve with confidence 1.0."""

    def test_resolve_call_import_resolved(self) -> None:
        g = KnowledgeGraph()

        # Two files: app.py imports validate from auth.py.
        _add_file_node(g, "src/auth.py")
        _add_file_node(g, "src/app.py")

        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/auth.py", "validate", 1, 10
        )
        _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.py", "login", 1, 15
        )

        # IMPORTS relationship: app.py -> auth.py with symbol "validate"
        app_file_id = generate_id(NodeLabel.FILE, "src/app.py")
        auth_file_id = generate_id(NodeLabel.FILE, "src/auth.py")
        g.add_relationship(
            GraphRelationship(
                id=f"imports:{app_file_id}->{auth_file_id}",
                type=RelType.IMPORTS,
                source=app_file_id,
                target=auth_file_id,
                properties={"symbols": "validate"},
            )
        )

        index = build_name_index(g, _CALLABLE_LABELS)
        call = CallInfo(name="validate", line=8)

        target_id, confidence = resolve_call(
            call, "src/app.py", index, g
        )

        expected_id = generate_id(
            NodeLabel.FUNCTION, "src/auth.py", "validate"
        )
        assert target_id == expected_id
        assert confidence == 1.0


# ---------------------------------------------------------------------------
# process_calls — orphan (module-level) calls fall back to File node
# ---------------------------------------------------------------------------


class TestOrphanCallsFallbackToFile:
    """Module-level calls outside any function should still create CALLS edges."""

    def test_module_level_call_creates_edge(self) -> None:
        """A call at the top level (not inside any function) uses the File node as source."""
        g = KnowledgeGraph()

        _add_file_node(g, "src/app.ts")
        _add_file_node(g, "src/init.ts")

        # Target function exists in another file.
        target_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/init.ts", "bootstrap", 1, 10
        )

        # No symbols in app.ts — the call at line 3 is at module level.
        parse = [
            FileParseData(
                file_path="src/app.ts",
                language="typescript",
                parse_result=ParseResult(
                    calls=[CallInfo(name="bootstrap", line=3)],
                ),
            ),
        ]

        process_calls(parse, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        # The edge should exist with the File node as source.
        file_id = generate_id(NodeLabel.FILE, "src/app.ts")
        pairs = {(r.source, r.target) for r in calls_rels}
        assert (file_id, target_id) in pairs

    def test_blocklisted_call_still_links_callback_arguments(self) -> None:
        """rows.map(formatRow) links formatRow even though 'map' is blocklisted."""
        g = KnowledgeGraph()

        _add_file_node(g, "src/core.ts")
        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/core.ts", "getSharedCompanies", 1, 20
        )
        target_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/core.ts", "formatRow", 30, 35
        )

        parse = [
            FileParseData(
                file_path="src/core.ts",
                language="typescript",
                parse_result=ParseResult(
                    calls=[
                        CallInfo(
                            name="map",
                            line=10,
                            receiver="rows",
                            arguments=["formatRow"],
                        )
                    ],
                ),
            ),
        ]

        process_calls(parse, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        pairs = {(r.source, r.target) for r in calls_rels}
        assert (caller_id, target_id) in pairs


# ---------------------------------------------------------------------------
# Type-inferred receiver method resolution
# ---------------------------------------------------------------------------


class TestTypeInferredReceiverResolution:
    """Variable type inference enables receiver method resolution."""

    def test_new_expression_type_inference_resolves_method(self) -> None:
        """pool.acquire() resolves to Pool.acquire via type inference from new Pool()."""
        g = KnowledgeGraph()

        _add_file_node(g, "src/db.ts")
        _add_symbol_node(g, NodeLabel.CLASS, "src/db.ts", "Pool", 1, 20)
        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/db.ts", "connect", 22, 30
        )
        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/db.ts", "acquire", 5, 10, class_name="Pool"
        )

        parse = [
            FileParseData(
                file_path="src/db.ts",
                language="typescript",
                parse_result=ParseResult(
                    calls=[CallInfo(name="acquire", line=25, receiver="pool")],
                    variable_types=[VarTypeInfo(var_name="pool", type_name="Pool", line=23)],
                ),
            ),
        ]

        process_calls(parse, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        pairs = {(r.source, r.target) for r in calls_rels}
        assert (caller_id, method_id) in pairs

    def test_type_annotation_inference_resolves_method(self) -> None:
        """svc.run() resolves to Service.run via type annotation const svc: Service."""
        g = KnowledgeGraph()

        _add_file_node(g, "src/app.ts")
        _add_symbol_node(g, NodeLabel.CLASS, "src/app.ts", "Service", 1, 20)
        caller_id = _add_symbol_node(
            g, NodeLabel.FUNCTION, "src/app.ts", "main", 22, 30
        )
        method_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/app.ts", "run", 5, 10, class_name="Service"
        )

        parse = [
            FileParseData(
                file_path="src/app.ts",
                language="typescript",
                parse_result=ParseResult(
                    calls=[CallInfo(name="run", line=25, receiver="svc")],
                    variable_types=[VarTypeInfo(var_name="svc", type_name="Service", line=23)],
                ),
            ),
        ]

        process_calls(parse, g)

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        pairs = {(r.source, r.target) for r in calls_rels}
        assert (caller_id, method_id) in pairs


# ---------------------------------------------------------------------------
# 11. W2.5c equivalence — call_index_by_file two-level index + symbol_ids
# carry-forward
#
# ``_resolve_self_method_reference`` / ``_resolve_call_reference`` /
# ``_process_calls_reference`` below are frozen, verbatim copies of the
# pre-W2.5c same-file resolution logic: a plain linear scan over the flat
# name->[ids] index, with no by-file grouping and no symbol_ids carry
# forward.  They are deliberately NOT the real resolve_call /
# _resolve_self_method / process_calls with the new optional arguments
# omitted -- the whole point is to pin the *old* behavior independently of
# whatever synaptiq.core.ingestion.calls does now, so a future edit to the
# real implementation can't accidentally make this test compare an
# implementation against itself (see tests/core/test_processes.py
# ``_deduplicate_flows_reference`` for the established style).  Everything
# else process_calls relies on (import resolution, fuzzy-match
# tie-breaking, receiver-method resolution, blocklists, add-edge dedup) is
# untouched by W2.5c, so the reference driver calls those real helpers
# directly.
# ---------------------------------------------------------------------------


def _resolve_self_method_reference(
    method_name: str,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    caller_class_name: str | None = None,
) -> str | None:
    """Frozen copy of the pre-W2.5c ``_resolve_self_method``: an
    unconditional linear scan over ``call_index[method_name]``."""
    fallback: str | None = None
    for nid in call_index.get(method_name, []):
        node = graph.get_node(nid)
        if (
            node is not None
            and node.label == NodeLabel.METHOD
            and node.file_path == file_path
        ):
            if caller_class_name and node.class_name == caller_class_name:
                return nid
            if fallback is None:
                fallback = nid
    return fallback


def _resolve_call_reference(
    call: CallInfo,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    caller_class_name: str | None = None,
    import_cache: dict[str, set[str]] | None = None,
) -> tuple[str | None, float]:
    """Frozen copy of the pre-W2.5c ``resolve_call``: the same-file step is
    a plain linear scan over every same-name candidate, with no by-file
    index to short-circuit it."""
    name = call.name
    receiver = call.receiver

    if receiver in ("self", "this"):
        result = _resolve_self_method_reference(
            name, file_path, call_index, graph, caller_class_name
        )
        if result is not None:
            return result, 1.0

    candidate_ids = call_index.get(name, [])
    if not candidate_ids:
        return None, 0.0

    # 1. Same-file exact match (unconditional linear scan -- this is the
    # step W2.5c replaces with an O(1) by-file dict hit).
    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path == file_path:
            return nid, 1.0

    # 2. Import-resolved match -- reuses the real (unchanged) helper.
    effective_cache = (
        import_cache
        if import_cache is not None
        else calls_module._build_import_cache(file_path, graph)
    )
    imported_target = calls_module._resolve_via_imports(
        name, candidate_ids, graph, effective_cache
    )
    if imported_target is not None:
        return imported_target, 1.0

    # 3. Global fuzzy match -- reuses the real (unchanged) helper.
    if len(candidate_ids) > 5:
        return None, 0.0
    return (
        calls_module._pick_closest(candidate_ids, graph, caller_file_path=file_path),
        0.5,
    )


def _process_calls_reference(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
) -> None:
    """Frozen copy of the pre-W2.5c ``process_calls`` driver: builds only
    the flat name index (no by-file grouping) and always recomputes
    ``assign_symbol_ids`` for the decorator loop (no ``symbol_ids`` carry
    forward, even if a caller happens to set it on the FileParseData).
    Delegates to the frozen ``_resolve_call_reference`` for resolution and
    to the real, untouched-by-W2.5c helpers for everything else.
    """
    call_index = build_name_index(graph, calls_module._CALLABLE_LABELS)
    file_sym_index = build_file_symbol_index(graph, calls_module._CALLABLE_LABELS)
    var_type_map = calls_module._build_var_type_map(parse_data)
    seen: set[str] = set()

    for fpd in parse_data:
        import_cache = calls_module._build_import_cache(fpd.file_path, graph)

        blocklist = (
            calls_module._CALL_BLOCKLIST | calls_module._RUBY_CALL_BLOCKLIST
            if fpd.language == "ruby"
            else calls_module._CALL_BLOCKLIST
        )

        for call in fpd.parse_result.calls:
            blocklisted = call.name in blocklist and call.receiver not in ("self", "this")

            source_id = find_containing_symbol(call.line, fpd.file_path, file_sym_index)
            if source_id is None:
                source_id = generate_id(NodeLabel.FILE, fpd.file_path)

            if not blocklisted:
                caller_class_name: str | None = None
                if call.receiver in ("self", "this"):
                    source_node = graph.get_node(source_id)
                    if source_node is not None:
                        caller_class_name = source_node.class_name

                target_id, confidence = _resolve_call_reference(
                    call,
                    fpd.file_path,
                    call_index,
                    graph,
                    caller_class_name=caller_class_name,
                    import_cache=import_cache,
                )
                if target_id is not None:
                    if call.is_weak_ref and confidence < 0.8:
                        confidence = calls_module._WEAK_REF_CONFIDENCE
                    calls_module._add_calls_edge(source_id, target_id, confidence, graph, seen)

            for arg_name in call.arguments:
                if arg_name in blocklist:
                    continue
                arg_call = CallInfo(name=arg_name, line=call.line)
                arg_id, arg_conf = _resolve_call_reference(
                    arg_call, fpd.file_path, call_index, graph, import_cache=import_cache
                )
                if arg_id is not None:
                    calls_module._add_calls_edge(source_id, arg_id, arg_conf * 0.8, graph, seen)

            receiver = call.receiver
            if blocklisted:
                continue
            if receiver and receiver not in ("self", "this"):
                receiver_call = CallInfo(name=receiver, line=call.line)
                recv_id, recv_conf = _resolve_call_reference(
                    receiver_call, fpd.file_path, call_index, graph, import_cache=import_cache
                )
                if recv_id is not None:
                    calls_module._add_calls_edge(source_id, recv_id, recv_conf, graph, seen)

                resolved_receiver = var_type_map.get(fpd.file_path, {}).get(receiver, receiver)

                calls_module._resolve_receiver_method(
                    resolved_receiver,
                    call.name,
                    source_id,
                    fpd.file_path,
                    call_index,
                    graph,
                    seen,
                )

        symbol_ids = assign_symbol_ids(fpd.parse_result.symbols, fpd.file_path)
        for symbol, source_id in zip(fpd.parse_result.symbols, symbol_ids):
            if not symbol.decorators or source_id is None:
                continue

            for dec_name in symbol.decorators:
                base_name = dec_name.rsplit(".", 1)[-1] if "." in dec_name else dec_name
                call_obj = CallInfo(name=base_name, line=symbol.start_line)
                target_id, confidence = _resolve_call_reference(
                    call_obj, fpd.file_path, call_index, graph, import_cache=import_cache
                )
                if target_id is None and "." in dec_name:
                    call_obj = CallInfo(name=dec_name, line=symbol.start_line)
                    target_id, confidence = _resolve_call_reference(
                        call_obj, fpd.file_path, call_index, graph, import_cache=import_cache
                    )
                if target_id is None:
                    continue

                rel_id = f"calls:{source_id}->{target_id}"
                if rel_id in seen:
                    continue
                seen.add(rel_id)

                graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=RelType.CALLS,
                        source=source_id,
                        target=target_id,
                        properties={"confidence": confidence},
                    )
                )


def _canonical_calls(graph: KnowledgeGraph) -> list[tuple]:
    """Sorted, comparable snapshot of every CALLS edge (id, source, target,
    properties) -- used to diff two independent process_calls runs."""
    return sorted(
        (r.id, r.source, r.target, tuple(sorted(r.properties.items())))
        for r in graph.get_relationships_by_type(RelType.CALLS)
    )


def _build_multi_file_equivalence_fixture() -> (
    tuple[KnowledgeGraph, list[FileParseData], dict[str, str]]
):
    """Multi-file fixture with same-named functions/methods spread across
    several files, self/this calls, plain + dotted decorators, an
    import-resolved call, a global-fuzzy tie-break, and Ruby-blocklist
    decoys -- exercises every resolve_call/_resolve_self_method path with
    deliberate name collisions across files to stress the by-file index.

    ``symbol_ids`` is pre-populated on every returned FileParseData (as
    process_parsing's Phase 2 would do) so the same fixture also exercises
    the symbol_ids carry-forward path in the decorator loop.
    """
    g = KnowledgeGraph()

    for path in (
        "src/auth/service.py",
        "src/utils/helpers.py",
        "src/jobs/worker.py",
        "src/web/app.ts",
        "src/web/init.ts",
        "app/widget.rb",
    ):
        _add_file_node(g, path)

    # -- src/auth/service.py -------------------------------------------
    _add_symbol_node(g, NodeLabel.CLASS, "src/auth/service.py", "AuthService", 1, 60)
    login_id = _add_symbol_node(
        g, NodeLabel.METHOD, "src/auth/service.py", "login", 3, 15, class_name="AuthService"
    )
    check_token_id = _add_symbol_node(
        g,
        NodeLabel.METHOD,
        "src/auth/service.py",
        "check_token",
        17,
        25,
        class_name="AuthService",
    )
    auth_validate_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/auth/service.py", "validate", 27, 33
    )
    auth_run_id = _add_symbol_node(g, NodeLabel.FUNCTION, "src/auth/service.py", "run", 35, 45)

    # -- src/utils/helpers.py -------------------------------------------
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/utils/helpers.py", "validate", 1, 6)
    _add_symbol_node(g, NodeLabel.FUNCTION, "src/utils/helpers.py", "run", 8, 14)
    route_id = _add_symbol_node(g, NodeLabel.FUNCTION, "src/utils/helpers.py", "route", 16, 20)
    log_call_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/utils/helpers.py", "log_call", 22, 26
    )

    # -- src/jobs/worker.py -----------------------------------------------
    worker_run_id = _add_symbol_node(g, NodeLabel.FUNCTION, "src/jobs/worker.py", "run", 1, 6)
    process_job_id = _add_symbol_node(
        g, NodeLabel.FUNCTION, "src/jobs/worker.py", "process_job", 8, 20
    )

    # -- src/web/app.ts ---------------------------------------------------
    _add_symbol_node(g, NodeLabel.CLASS, "src/web/app.ts", "Service", 1, 40)
    service_run_id = _add_symbol_node(
        g, NodeLabel.METHOD, "src/web/app.ts", "run", 3, 9, class_name="Service"
    )
    start_id = _add_symbol_node(
        g, NodeLabel.METHOD, "src/web/app.ts", "start", 11, 18, class_name="Service"
    )
    bootstrap_id = _add_symbol_node(g, NodeLabel.FUNCTION, "src/web/app.ts", "bootstrap", 20, 28)

    # -- src/web/init.ts --------------------------------------------------
    helper_id = _add_symbol_node(g, NodeLabel.FUNCTION, "src/web/init.ts", "helper", 1, 6)

    # app.ts imports `helper` from init.ts.
    app_file_id = generate_id(NodeLabel.FILE, "src/web/app.ts")
    init_file_id = generate_id(NodeLabel.FILE, "src/web/init.ts")
    g.add_relationship(
        GraphRelationship(
            id=f"imports:{app_file_id}->{init_file_id}",
            type=RelType.IMPORTS,
            source=app_file_id,
            target=init_file_id,
            properties={"symbols": "helper"},
        )
    )

    # -- app/widget.rb (Ruby blocklist decoys) ----------------------------
    render_id = _add_symbol_node(
        g, NodeLabel.METHOD, "app/widget.rb", "render", 1, 10, class_name="Widget"
    )
    helper_method_id = _add_symbol_node(
        g, NodeLabel.METHOD, "app/widget.rb", "helper_method", 12, 16, class_name="Widget"
    )
    for builtin, ln in (("puts", 20), ("each", 24), ("new", 28)):
        _add_symbol_node(
            g, NodeLabel.METHOD, "app/widget.rb", builtin, ln, ln + 2, class_name="Widget"
        )

    parse_data = [
        FileParseData(
            file_path="src/auth/service.py",
            language="python",
            parse_result=ParseResult(
                symbols=[
                    SymbolInfo(
                        name="login",
                        kind="method",
                        start_line=3,
                        end_line=15,
                        content="",
                        class_name="AuthService",
                        decorators=["log_call"],
                    ),
                    SymbolInfo(
                        name="check_token",
                        kind="method",
                        start_line=17,
                        end_line=25,
                        content="",
                        class_name="AuthService",
                    ),
                    SymbolInfo(
                        name="validate", kind="function", start_line=27, end_line=33, content=""
                    ),
                    SymbolInfo(
                        name="run",
                        kind="function",
                        start_line=35,
                        end_line=45,
                        content="",
                        decorators=["app.route"],
                    ),
                ],
                calls=[
                    CallInfo(name="check_token", line=6, receiver="self"),
                    CallInfo(name="validate", line=8),
                ],
            ),
        ),
        FileParseData(
            file_path="src/utils/helpers.py",
            language="python",
            parse_result=ParseResult(),
        ),
        FileParseData(
            file_path="src/jobs/worker.py",
            language="python",
            parse_result=ParseResult(
                calls=[
                    CallInfo(name="run", line=10),
                    CallInfo(name="validate", line=11),
                ],
            ),
        ),
        FileParseData(
            file_path="src/web/app.ts",
            language="typescript",
            parse_result=ParseResult(
                calls=[
                    CallInfo(name="run", line=13, receiver="this"),
                    CallInfo(name="helper", line=22),
                ],
            ),
        ),
        FileParseData(
            file_path="src/web/init.ts",
            language="typescript",
            parse_result=ParseResult(),
        ),
        FileParseData(
            file_path="app/widget.rb",
            language="ruby",
            parse_result=ParseResult(
                calls=[
                    CallInfo(name="puts", line=3),
                    CallInfo(name="each", line=4),
                    CallInfo(name="new", line=5, receiver="User"),
                    CallInfo(name="helper_method", line=6),
                ],
            ),
        ),
    ]

    # Simulate process_parsing's Phase 2: carry the computed symbol IDs
    # forward so this fixture also exercises the symbol_ids carry-forward
    # path (the frozen reference above ignores this field and always
    # recomputes, so both sides must still agree).
    for fpd in parse_data:
        fpd.symbol_ids = assign_symbol_ids(fpd.parse_result.symbols, fpd.file_path)

    return g, parse_data, {
        "login_id": login_id,
        "check_token_id": check_token_id,
        "auth_validate_id": auth_validate_id,
        "auth_run_id": auth_run_id,
        "route_id": route_id,
        "log_call_id": log_call_id,
        "worker_run_id": worker_run_id,
        "process_job_id": process_job_id,
        "service_run_id": service_run_id,
        "start_id": start_id,
        "bootstrap_id": bootstrap_id,
        "helper_id": helper_id,
        "render_id": render_id,
        "helper_method_id": helper_method_id,
    }


class TestProcessCallsCallIndexByFileEquivalence:
    """process_calls's output (source, target, confidence, properties for
    every CALLS edge) must be byte-identical to the frozen pre-W2.5c
    reference on a multi-file fixture built specifically to stress
    same-name-different-file resolution.
    """

    def test_matches_reference_on_multi_file_fixture(self) -> None:
        g, parse_data, ids = _build_multi_file_equivalence_fixture()

        graph_old = copy.deepcopy(g)
        graph_new = copy.deepcopy(g)

        _process_calls_reference(parse_data, graph_old)
        process_calls(parse_data, graph_new)

        assert _canonical_calls(graph_old) == _canonical_calls(graph_new)

        # Sanity: the fixture actually produced a meaningful, non-trivial
        # edge set (guards against a vacuous "both empty" pass).
        new_pairs = {
            (r.source, r.target, r.properties.get("confidence"))
            for r in graph_new.get_relationships_by_type(RelType.CALLS)
        }
        assert new_pairs == {
            (ids["login_id"], ids["check_token_id"], 1.0),
            (ids["login_id"], ids["auth_validate_id"], 1.0),
            (ids["process_job_id"], ids["worker_run_id"], 1.0),
            (ids["process_job_id"], ids["auth_validate_id"], 0.5),
            (ids["start_id"], ids["service_run_id"], 1.0),
            (ids["bootstrap_id"], ids["helper_id"], 1.0),
            (ids["render_id"], ids["helper_method_id"], 1.0),
            (ids["login_id"], ids["log_call_id"], 0.5),
            (ids["auth_run_id"], ids["route_id"], 0.5),
        }
        # `run` has 4 same-named candidates repo-wide (auth/service.py,
        # helpers.py, worker.py, and app.ts's Service.run METHOD); worker.py
        # must resolve its own local call to worker.py's `run` specifically
        # -- the exact scenario the by-file index exists for.
        assert (ids["process_job_id"], ids["worker_run_id"]) in {
            (r.source, r.target) for r in graph_new.get_relationships_by_type(RelType.CALLS)
        }


class TestCallIndexBySameFileMultipleCandidatesEquivalence:
    """Same file defining two same-named METHOD candidates whose class
    doesn't match the caller (forcing ``_resolve_self_method``'s *fallback*
    branch, not its immediate exact-class-match return) -- mirrors
    assign_symbol_ids's ``#L{line}`` collision suffix.  The by-file bucket
    must preserve the same relative order as the original flat scan so
    "first candidate encountered" resolves identically either way.
    """

    def test_matches_reference_with_duplicate_same_file_method(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/legacy/handlers.py")

        first_id = generate_id(NodeLabel.METHOD, "src/legacy/handlers.py", "Other.handle")
        g.add_node(
            GraphNode(
                id=first_id,
                label=NodeLabel.METHOD,
                name="handle",
                file_path="src/legacy/handlers.py",
                start_line=5,
                end_line=8,
                class_name="Other",
            )
        )
        second_id = generate_id(NodeLabel.METHOD, "src/legacy/handlers.py", "Other.handle#L20")
        g.add_node(
            GraphNode(
                id=second_id,
                label=NodeLabel.METHOD,
                name="handle",
                file_path="src/legacy/handlers.py",
                start_line=20,
                end_line=24,
                class_name="Other",
            )
        )
        caller_id = _add_symbol_node(
            g, NodeLabel.METHOD, "src/legacy/handlers.py", "run", 1, 3, class_name="Job"
        )

        parse_data = [
            FileParseData(
                file_path="src/legacy/handlers.py",
                language="python",
                parse_result=ParseResult(
                    calls=[CallInfo(name="handle", line=2, receiver="self")],
                ),
            ),
        ]

        graph_old = copy.deepcopy(g)
        graph_new = copy.deepcopy(g)

        _process_calls_reference(parse_data, graph_old)
        process_calls(parse_data, graph_new)

        assert _canonical_calls(graph_old) == _canonical_calls(graph_new)

        # Pin the concrete outcome: the *first*-added candidate wins, both
        # in the old scan and in the new by-file bucket.
        new_pairs = {
            (r.source, r.target) for r in graph_new.get_relationships_by_type(RelType.CALLS)
        }
        assert (caller_id, first_id) in new_pairs
        assert (caller_id, second_id) not in new_pairs


class TestSymbolIdsCarryForwardEquivalence:
    """process_calls's decorator-edge resolution must produce identical
    output whether FileParseData.symbol_ids is pre-populated (as
    process_parsing's Phase 2 does) or left None (forcing recomputation via
    assign_symbol_ids) -- exercises both branches of the carry-forward
    fallback directly.
    """

    def test_decorator_edges_identical_with_and_without_carried_ids(self) -> None:
        g = KnowledgeGraph()
        _add_file_node(g, "src/api/routes.py")
        _add_symbol_node(g, NodeLabel.FUNCTION, "src/api/routes.py", "route", 1, 5)
        handler_symbol = SymbolInfo(
            name="handler",
            kind="function",
            start_line=10,
            end_line=14,
            content="",
            decorators=["route"],
        )
        parse_result = ParseResult(symbols=[handler_symbol])

        fpd_recompute = FileParseData(
            file_path="src/api/routes.py",
            language="python",
            parse_result=parse_result,
        )
        graph_recompute = copy.deepcopy(g)
        process_calls([fpd_recompute], graph_recompute)

        fpd_carried = FileParseData(
            file_path="src/api/routes.py",
            language="python",
            parse_result=parse_result,
            symbol_ids=assign_symbol_ids(parse_result.symbols, "src/api/routes.py"),
        )
        graph_carried = copy.deepcopy(g)
        process_calls([fpd_carried], graph_carried)

        assert fpd_recompute.symbol_ids is None
        assert fpd_carried.symbol_ids is not None
        assert _canonical_calls(graph_recompute) == _canonical_calls(graph_carried)
        assert len(_canonical_calls(graph_carried)) == 1


class TestCallIndexByFileScale:
    """Demonstrates the O(1) same-file lookup fix (plan §7 W2.5 acceptance:
    "complexity fix demonstrated with a synthetic large fixture"): many
    files share a symbol name, and each file's local call must resolve to
    ITS OWN same-named symbol -- not scan hundreds of repo-wide candidates
    to find it.  Also guards against a regression back to an O(files) scan
    per call site.
    """

    def test_many_files_same_name_resolves_locally(self) -> None:
        g = KnowledgeGraph()
        parse_data: list[FileParseData] = []
        n_files = 500

        expected_pairs: set[tuple[str, str]] = set()
        for i in range(n_files):
            path = f"src/mod{i}/handler.py"
            _add_file_node(g, path)
            # "process" is itself in _CALL_BLOCKLIST (Node.js global), so
            # the shared name under test must avoid it -- "compute_result"
            # is a plain, unreserved user function name.
            target_id = _add_symbol_node(g, NodeLabel.FUNCTION, path, "compute_result", 1, 5)
            caller_id = _add_symbol_node(g, NodeLabel.FUNCTION, path, "entrypoint", 7, 12)
            parse_data.append(
                FileParseData(
                    file_path=path,
                    language="python",
                    parse_result=ParseResult(
                        calls=[CallInfo(name="compute_result", line=9)]
                    ),
                )
            )
            expected_pairs.add((caller_id, target_id))

        start = time.perf_counter()
        process_calls(parse_data, g)
        elapsed = time.perf_counter() - start

        calls_rels = g.get_relationships_by_type(RelType.CALLS)
        actual_pairs = {(r.source, r.target) for r in calls_rels}

        # Every file's entrypoint must resolve to ITS OWN local
        # `compute_result`, not one of the 499 other same-named candidates.
        assert actual_pairs == expected_pairs
        assert len(calls_rels) == n_files

        # Regression guard: O(1) same-file lookup keeps this comfortably
        # fast even at 500 files sharing a name; a regression back to the
        # old O(files) linear scan per call site would be quadratic here.
        assert elapsed < 5.0
