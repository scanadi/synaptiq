"""Tests for the process / execution flow detection phase (Phase 9)."""

from __future__ import annotations

import random
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
from synaptiq.core.ingestion.processes import (
    deduplicate_flows,
    find_entry_points,
    generate_process_label,
    process_processes,
    trace_flow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_function(
    graph: KnowledgeGraph,
    name: str,
    file_path: str = "src/app.py",
    *,
    content: str = "",
    language: str = "python",
    is_exported: bool = False,
) -> GraphNode:
    """Add a FUNCTION node and return it."""
    node_id = generate_id(NodeLabel.FUNCTION, file_path, name)
    node = GraphNode(
        id=node_id,
        label=NodeLabel.FUNCTION,
        name=name,
        file_path=file_path,
        content=content,
        language=language,
        is_exported=is_exported,
    )
    graph.add_node(node)
    return node


def _add_method(
    graph: KnowledgeGraph,
    name: str,
    *,
    class_name: str = "",
    file_path: str = "app/models/x.rb",
    content: str = "",
    language: str = "ruby",
    is_exported: bool = False,
) -> GraphNode:
    """Add a METHOD node and return it."""
    symbol = f"{class_name}.{name}" if class_name else name
    node_id = generate_id(NodeLabel.METHOD, file_path, symbol)
    node = GraphNode(
        id=node_id,
        label=NodeLabel.METHOD,
        name=name,
        file_path=file_path,
        content=content,
        language=language,
        class_name=class_name,
        is_exported=is_exported,
    )
    graph.add_node(node)
    return node


def _add_call(
    graph: KnowledgeGraph,
    source: GraphNode,
    target: GraphNode,
    confidence: float = 1.0,
) -> None:
    """Add a CALLS relationship between two nodes."""
    rel_id = f"calls:{source.id}->{target.id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.CALLS,
            source=source.id,
            target=target.id,
            properties={"confidence": confidence},
        )
    )


def _add_member_of(
    graph: KnowledgeGraph,
    node: GraphNode,
    community_id: str,
) -> None:
    """Add a MEMBER_OF relationship from *node* to a community."""
    rel_id = f"member_of:{node.id}->{community_id}"
    graph.add_relationship(
        GraphRelationship(
            id=rel_id,
            type=RelType.MEMBER_OF,
            source=node.id,
            target=community_id,
        )
    )


# ---------------------------------------------------------------------------
# Fixture: call graph
#
#   main() --> validate() --> hash_password()
#                         \-> query_db() --> format_result()
#
#   orphan_func() <-- (has incoming call from some_caller)
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Build a graph matching the specification.

    - main() calls validate()
    - validate() calls hash_password() and query_db()
    - query_db() calls format_result()
    - orphan_func() has an incoming call (so it is NOT an entry point)
    """
    g = KnowledgeGraph()

    main = _add_function(g, "main")
    validate = _add_function(g, "validate")
    hash_password = _add_function(g, "hash_password")
    query_db = _add_function(g, "query_db")
    format_result = _add_function(g, "format_result")
    orphan_func = _add_function(g, "orphan_func")

    # Also add a caller for orphan_func so it has an incoming CALLS edge.
    some_caller = _add_function(g, "some_caller")

    _add_call(g, main, validate)
    _add_call(g, validate, hash_password)
    _add_call(g, validate, query_db)
    _add_call(g, query_db, format_result)
    _add_call(g, some_caller, orphan_func)

    return g


# ---------------------------------------------------------------------------
# 1. test_find_entry_points
# ---------------------------------------------------------------------------


class TestFindEntryPoints:
    """Entry points are functions with no incoming CALLS edges."""

    def test_find_entry_points(self, graph: KnowledgeGraph) -> None:
        """main is identified as entry point; orphan_func is NOT."""
        entry_points = find_entry_points(graph)
        ep_names = {n.name for n in entry_points}

        # main has no incoming CALLS -> entry point.
        assert "main" in ep_names
        # orphan_func HAS an incoming CALLS edge -> not an entry point
        # (unless matched by framework pattern, which it does not).
        assert "orphan_func" not in ep_names

    def test_entry_point_flag_set(self, graph: KnowledgeGraph) -> None:
        """is_entry_point is set to True on detected entry points."""
        entry_points = find_entry_points(graph)
        for ep in entry_points:
            assert ep.is_entry_point is True


# ---------------------------------------------------------------------------
# 2. test_find_entry_points_framework
# ---------------------------------------------------------------------------


class TestFindEntryPointsFramework:
    """Framework patterns are recognised as entry points."""

    def test_test_function_is_entry_point(self) -> None:
        """A function named test_something is detected as entry point."""
        g = KnowledgeGraph()
        test_fn = _add_function(g, "test_something", language="python")

        # Give it an incoming call so *only* the framework pattern triggers.
        caller = _add_function(g, "runner")
        _add_call(g, caller, test_fn)

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "test_something" in ep_names

    def test_decorator_pattern_entry_point(self) -> None:
        """A function with @app.route in content is an entry point."""
        g = KnowledgeGraph()
        _add_function(
            g,
            "index",
            content='@app.route("/")\ndef index():\n    pass',
            language="python",
        )

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "index" in ep_names

    def test_ts_handler_is_entry_point(self) -> None:
        """A TypeScript function named handler is an entry point."""
        g = KnowledgeGraph()
        _add_function(
            g,
            "handler",
            file_path="src/api.ts",
            language="typescript",
        )

        entry_points = find_entry_points(g)
        ep_names = {n.name for n in entry_points}
        assert "handler" in ep_names


# ---------------------------------------------------------------------------
# 2b. Ruby framework entry points
# ---------------------------------------------------------------------------


class TestFindEntryPointsRuby:
    """Ruby/Rails/Sinatra entry-point conventions are recognised."""

    def test_rails_controller_action_by_class_name(self) -> None:
        """A method on a *Controller class is an entry point despite callers."""
        g = KnowledgeGraph()
        action = _add_method(
            g,
            "show",
            class_name="UsersController",
            file_path="app/controllers/users_controller.rb",
        )
        # Give it an incoming call so only the framework pattern can trigger.
        caller = _add_function(g, "dispatch", language="ruby")
        _add_call(g, caller, action)

        ep_names = {n.name for n in find_entry_points(g)}
        assert "show" in ep_names

    def test_rails_controller_action_by_filename(self) -> None:
        """A controller action is detected via the *_controller.rb filename."""
        g = KnowledgeGraph()
        action = _add_method(
            g,
            "index",
            class_name="",
            file_path="app/controllers/posts_controller.rb",
        )
        caller = _add_function(g, "dispatch", language="ruby")
        _add_call(g, caller, action)

        ep_names = {n.name for n in find_entry_points(g)}
        assert "index" in ep_names

    def test_job_perform_is_entry_point(self) -> None:
        """ActiveJob #perform on a *Job class is a framework entry point."""
        g = KnowledgeGraph()
        m = _add_method(
            g,
            "perform",
            class_name="EmailJob",
            file_path="app/jobs/email_job.rb",
        )
        caller = _add_function(g, "enqueue", language="ruby")
        _add_call(g, caller, m)

        ep_names = {n.name for n in find_entry_points(g)}
        assert "perform" in ep_names

    def test_sinatra_route_block_is_entry_point(self) -> None:
        """A handler whose content is an inline route DSL is an entry point."""
        g = KnowledgeGraph()
        route = _add_function(
            g,
            "get_user",
            file_path="app.rb",
            language="ruby",
            content='get "/users/:id" do\n  User.find(params[:id])\nend',
        )
        caller = _add_function(g, "boot", language="ruby")
        _add_call(g, caller, route)

        ep_names = {n.name for n in find_entry_points(g)}
        assert "get_user" in ep_names

    def test_rake_task_file_is_entry_point(self) -> None:
        """A definition in a .rake file is an entry point via heuristic."""
        g = KnowledgeGraph()
        _add_function(g, "build", file_path="lib/tasks/build.rake", language="ruby")

        ep_names = {n.name for n in find_entry_points(g)}
        assert "build" in ep_names

    def test_spec_file_definition_is_entry_point(self) -> None:
        """A definition in a *_spec.rb file is an entry point via heuristic."""
        g = KnowledgeGraph()
        _add_function(
            g, "setup_data", file_path="spec/models/user_spec.rb", language="ruby"
        )

        ep_names = {n.name for n in find_entry_points(g)}
        assert "setup_data" in ep_names

    def test_config_ru_definition_is_entry_point(self) -> None:
        """A definition in config.ru is an entry point via heuristic."""
        g = KnowledgeGraph()
        _add_function(g, "run_app", file_path="config.ru", language="ruby")

        ep_names = {n.name for n in find_entry_points(g)}
        assert "run_app" in ep_names

    def test_rakefile_definition_is_entry_point(self) -> None:
        """A definition in a Rakefile is an entry point via heuristic."""
        g = KnowledgeGraph()
        _add_function(g, "default_task", file_path="Rakefile", language="ruby")

        ep_names = {n.name for n in find_entry_points(g)}
        assert "default_task" in ep_names

    def test_plain_private_method_with_caller_not_entry_point(self) -> None:
        """A plain method with an incoming call is NOT an entry point."""
        g = KnowledgeGraph()
        helper = _add_method(
            g,
            "calculate_total",
            class_name="Invoice",
            file_path="app/models/invoice.rb",
        )
        caller = _add_method(
            g,
            "total",
            class_name="Invoice",
            file_path="app/models/invoice.rb",
        )
        _add_call(g, caller, helper)

        ep_names = {n.name for n in find_entry_points(g)}
        assert "calculate_total" not in ep_names

    def test_non_route_dsl_method_not_misdetected(self) -> None:
        """A method whose body merely ends in a verb is not a route handler."""
        g = KnowledgeGraph()
        m = _add_method(
            g,
            "widget",
            class_name="Builder",
            file_path="app/models/builder.rb",
            content='def widget\n  target "x"\nend',
        )
        caller = _add_function(g, "render", language="ruby")
        _add_call(g, caller, m)

        ep_names = {n.name for n in find_entry_points(g)}
        assert "widget" not in ep_names

    def test_plain_rb_function_without_callers_not_entry_via_heuristic(
        self,
    ) -> None:
        """A regular .rb function with no callers is not an entry point.

        Only special Ruby files (specs, rake, config.ru) qualify via the
        file heuristic — ordinary library files must not.
        """
        g = KnowledgeGraph()
        _add_function(g, "private_helper", file_path="lib/util.rb", language="ruby")

        ep_names = {n.name for n in find_entry_points(g)}
        assert "private_helper" not in ep_names


# ---------------------------------------------------------------------------
# 3. test_trace_flow
# ---------------------------------------------------------------------------


class TestTraceFlow:
    """BFS traces the correct path from an entry point."""

    def test_trace_flow(self, graph: KnowledgeGraph) -> None:
        """Tracing from main covers the full call chain."""
        main_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "main")
        main_node = graph.get_node(main_id)
        assert main_node is not None

        flow = trace_flow(main_node, graph)
        flow_names = [n.name for n in flow]

        # BFS from main: main -> validate -> {hash_password, query_db} -> format_result
        assert flow_names[0] == "main"
        assert "validate" in flow_names
        assert "hash_password" in flow_names
        assert "query_db" in flow_names
        assert "format_result" in flow_names
        assert len(flow) == 5

    def test_trace_flow_no_cycles(self, graph: KnowledgeGraph) -> None:
        """Visited tracking prevents infinite loops in cyclic graphs."""
        g = KnowledgeGraph()
        a = _add_function(g, "a")
        b = _add_function(g, "b")
        _add_call(g, a, b)
        _add_call(g, b, a)  # cycle

        flow = trace_flow(a, g)
        assert len(flow) == 2  # a, b -- no revisit


# ---------------------------------------------------------------------------
# 4. test_trace_flow_max_depth
# ---------------------------------------------------------------------------


class TestTraceFlowMaxDepth:
    """Depth limit is respected."""

    def test_trace_flow_max_depth(self, graph: KnowledgeGraph) -> None:
        """With max_depth=1, only the direct callees are included."""
        main_id = generate_id(NodeLabel.FUNCTION, "src/app.py", "main")
        main_node = graph.get_node(main_id)
        assert main_node is not None

        flow = trace_flow(main_node, graph, max_depth=1)
        flow_names = [n.name for n in flow]

        # main -> validate (depth 1), but hash_password/query_db at depth 2 are cut off.
        assert "main" in flow_names
        assert "validate" in flow_names
        # Depth-2 nodes should NOT appear.
        assert "hash_password" not in flow_names
        assert "query_db" not in flow_names


# ---------------------------------------------------------------------------
# 4b. W2.5a: trace_flow's heap-based branch selection (processes.py:238-240)
#
# These pin two subtleties of the pre-existing full-sort-then-skip loop
# that a naive ``heapq.nlargest(max_branching, ...)`` truncation would get
# wrong:
#   1. An edge to an already-visited node must not consume the branch
#      budget -- the search has to keep looking past it.
#   2. Equal-confidence edges keep their original (insertion) order among
#      the selected branches (Python's stable sort + reverse=True keeps
#      ties in original order, not reversed order).
# ---------------------------------------------------------------------------


class TestTraceFlowBranchingSkipsVisitedWithinBudget:
    """A high-confidence edge to an already-visited node must not consume
    the branch budget -- the search must continue past it to find enough
    NEW nodes, exactly like the pre-heap full-sort-then-skip loop did.
    """

    def test_looks_past_visited_top_ranked_edge(self) -> None:
        g = KnowledgeGraph()
        e = _add_function(g, "e")
        d = _add_function(g, "d")
        p = _add_function(g, "p")
        q = _add_function(g, "q")
        r = _add_function(g, "r")

        # E's two outgoing edges both get taken (budget 2): d first
        # (highest confidence), then p.
        _add_call(g, e, d, confidence=1.0)
        _add_call(g, e, p, confidence=0.9)

        # d's outgoing edges, ranked by confidence: p (already visited via
        # e), then q, then r. With a branch budget of 2, the algorithm must
        # skip p (visited, doesn't consume budget) and then take BOTH q
        # and r -- a naive top-2-by-confidence truncation would only ever
        # consider {p, q}, skip p, and stop after q, silently dropping r.
        _add_call(g, d, p, confidence=1.0)
        _add_call(g, d, q, confidence=0.9)
        _add_call(g, d, r, confidence=0.5)

        flow = trace_flow(e, g, max_branching=2)
        flow_names = {n.name for n in flow}

        assert flow_names == {"e", "d", "p", "q", "r"}


class TestTraceFlowBranchingTieBreak:
    """Equal-confidence outgoing edges keep their original (insertion)
    order among the selected branches, matching the old
    ``list.sort(key=..., reverse=True)`` stable-sort tie-break.
    """

    def test_equal_confidence_ties_keep_insertion_order(self) -> None:
        g = KnowledgeGraph()
        e = _add_function(g, "e")
        x = _add_function(g, "x")
        y = _add_function(g, "y")
        z = _add_function(g, "z")

        # All three tie on confidence; only the budget's worth (2) should
        # be taken, in insertion order: x, then y (z is dropped).
        _add_call(g, e, x, confidence=0.5)
        _add_call(g, e, y, confidence=0.5)
        _add_call(g, e, z, confidence=0.5)

        flow = trace_flow(e, g, max_branching=2)
        flow_names = [n.name for n in flow]

        assert flow_names == ["e", "x", "y"]


# ---------------------------------------------------------------------------
# 5. test_generate_process_label
# ---------------------------------------------------------------------------


class TestGenerateProcessLabel:
    """Process labels are formatted correctly."""

    def test_generate_process_label(self) -> None:
        """Multi-step label uses arrow notation with max 4 steps."""
        nodes = [
            GraphNode(id=f"n{i}", label=NodeLabel.FUNCTION, name=name)
            for i, name in enumerate(
                ["main", "validate", "hash_password", "query_db", "format_result"]
            )
        ]
        label = generate_process_label(nodes)
        # Max 4 steps in the label.
        assert label == "main \u2192 validate \u2192 hash_password \u2192 query_db"

    def test_generate_process_label_single(self) -> None:
        """Single-step label is just the function name."""
        nodes = [GraphNode(id="n0", label=NodeLabel.FUNCTION, name="main")]
        label = generate_process_label(nodes)
        assert label == "main"

    def test_generate_process_label_empty(self) -> None:
        """Empty input gives empty string."""
        assert generate_process_label([]) == ""


# ---------------------------------------------------------------------------
# 6. test_deduplicate_flows
# ---------------------------------------------------------------------------


class TestDeduplicateFlows:
    """Similar flows are merged by keeping the longer one."""

    def test_deduplicate_flows(self) -> None:
        """A short flow that overlaps >70% with a longer flow is discarded."""
        # Create nodes.
        a = GraphNode(id="a", label=NodeLabel.FUNCTION, name="a")
        b = GraphNode(id="b", label=NodeLabel.FUNCTION, name="b")
        c = GraphNode(id="c", label=NodeLabel.FUNCTION, name="c")
        d = GraphNode(id="d", label=NodeLabel.FUNCTION, name="d")

        long_flow = [a, b, c, d]
        short_flow = [a, b, c]  # 100% overlap with long_flow (3/3)

        result = deduplicate_flows([short_flow, long_flow])
        assert len(result) == 1
        assert len(result[0]) == 4  # Kept the longer flow.

    def test_deduplicate_keeps_distinct(self) -> None:
        """Flows with low overlap are both kept."""
        a = GraphNode(id="a", label=NodeLabel.FUNCTION, name="a")
        b = GraphNode(id="b", label=NodeLabel.FUNCTION, name="b")
        c = GraphNode(id="c", label=NodeLabel.FUNCTION, name="c")
        d = GraphNode(id="d", label=NodeLabel.FUNCTION, name="d")

        flow1 = [a, b]
        flow2 = [c, d]

        result = deduplicate_flows([flow1, flow2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 7. test_process_processes_creates_nodes
# ---------------------------------------------------------------------------


class TestProcessProcessesCreatesNodes:
    """process_processes creates Process nodes in the graph."""

    def test_process_processes_creates_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        process_processes(graph)

        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        assert len(process_nodes) > 0

        # Each Process node has a name and step_count property.
        for pn in process_nodes:
            assert pn.name != ""
            assert pn.properties["step_count"] > 1


# ---------------------------------------------------------------------------
# 8. test_process_processes_creates_steps
# ---------------------------------------------------------------------------


class TestProcessProcessesCreatesSteps:
    """STEP_IN_PROCESS relationships are created with step numbers."""

    def test_process_processes_creates_steps(
        self, graph: KnowledgeGraph
    ) -> None:
        process_processes(graph)

        step_rels = graph.get_relationships_by_type(RelType.STEP_IN_PROCESS)
        assert len(step_rels) > 0

        # All step relationships should have a step_number property.
        for rel in step_rels:
            assert "step_number" in rel.properties
            assert isinstance(rel.properties["step_number"], int)

        # Verify step numbers start at 0 for each process.
        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        for pn in process_nodes:
            incoming = graph.get_incoming(pn.id, RelType.STEP_IN_PROCESS)
            step_numbers = sorted(
                r.properties["step_number"] for r in incoming
            )
            assert step_numbers[0] == 0
            assert step_numbers == list(range(len(step_numbers)))


# ---------------------------------------------------------------------------
# 9. test_process_processes_returns_count
# ---------------------------------------------------------------------------


class TestProcessProcessesReturnsCount:
    """process_processes returns the correct count of processes created."""

    def test_process_processes_returns_count(
        self, graph: KnowledgeGraph
    ) -> None:
        count = process_processes(graph)

        process_nodes = graph.get_nodes_by_label(NodeLabel.PROCESS)
        assert count == len(process_nodes)
        assert count > 0


# ---------------------------------------------------------------------------
# 10. W2.5a equivalence + scale — deduplicate_flows inverted-index rewrite
#
# ``_deduplicate_flows_reference`` below is a frozen, verbatim copy of the
# pre-W2.5a all-pairs O(n^2) algorithm (the one that used to live in
# ``deduplicate_flows`` itself). It is deliberately NOT imported from
# source -- the whole point is to pin the *old* behaviour independently of
# whatever ``synaptiq.core.ingestion.processes.deduplicate_flows`` does now,
# so a future edit to the real implementation can't accidentally make this
# test compare an implementation against itself.
# ---------------------------------------------------------------------------


def _deduplicate_flows_reference(
    flows: list[list[GraphNode]],
) -> list[list[GraphNode]]:
    """Frozen copy of the pre-W2.5a O(n^2) all-pairs implementation."""
    flows_sorted = sorted(flows, key=len, reverse=True)

    kept: list[list[GraphNode]] = []
    kept_sets: list[set[str]] = []

    for flow in flows_sorted:
        flow_ids = {n.id for n in flow}
        is_duplicate = False

        for kept_set in kept_sets:
            if not flow_ids or not kept_set:
                continue
            intersection = flow_ids & kept_set
            smaller_size = min(len(flow_ids), len(kept_set))
            overlap = len(intersection) / smaller_size
            if overlap > 0.5:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(flow)
            kept_sets.append(flow_ids)

    return kept


def _make_flow(node_ids: list[str]) -> list[GraphNode]:
    """Build a flow (list of GraphNode) from a list of node ids."""
    return [GraphNode(id=nid, label=NodeLabel.FUNCTION, name=nid) for nid in node_ids]


class TestDeduplicateFlowsEquivalence:
    """The inverted-index rewrite must match the frozen O(n^2) reference."""

    def test_matches_reference_on_many_overlapping_flows(self) -> None:
        """500 flows across 50 disjoint clusters, with heavy intra-cluster
        overlap (some near-duplicates should collapse) and zero
        inter-cluster overlap (most flows should survive) -- this exercises
        both the "shares a node -> must compare" and "shares nothing -> safe
        to skip" branches of the inverted-index pruning.
        """
        rng = random.Random(20260712)
        flows: list[list[GraphNode]] = []

        for cluster in range(50):
            base = [f"c{cluster}n{i}" for i in range(20)]
            for _variant in range(10):
                length = rng.randint(3, 20)
                node_ids = rng.sample(base, length)
                flows.append(_make_flow(node_ids))

        assert len(flows) == 500

        expected = _deduplicate_flows_reference(list(flows))
        actual = deduplicate_flows(list(flows))

        expected_ids = [[n.id for n in f] for f in expected]
        actual_ids = [[n.id for n in f] for f in actual]
        assert actual_ids == expected_ids
        # Sanity: both some collapsing (intra-cluster) and some survival
        # (inter-cluster) actually happened, so the comparison is meaningful.
        assert 50 <= len(actual) < 500

    def test_matches_reference_with_equal_length_ties(self) -> None:
        """Many same-length flows exercise the stable-sort tie-break: ties
        must keep their original relative order, exactly like Python's
        ``sorted(..., reverse=True)``.
        """
        rng = random.Random(9)
        flows: list[list[GraphNode]] = []
        for i in range(120):
            # All flows the same length (10) so `sorted(key=len, reverse=True)`
            # is an all-ties sort -- original order must be preserved.
            node_ids = [f"t{i}n{j}" for j in range(10)]
            # Deliberately overlap a couple of neighbours' ids so some
            # duplicate decisions are actually exercised, not just no-ops.
            if i % 7 == 0 and i > 0:
                node_ids[:3] = [f"t{i - 1}n{j}" for j in range(3)]
            rng.shuffle(node_ids)
            flows.append(_make_flow(node_ids))

        expected = _deduplicate_flows_reference(list(flows))
        actual = deduplicate_flows(list(flows))

        expected_ids = [[n.id for n in f] for f in expected]
        actual_ids = [[n.id for n in f] for f in actual]
        assert actual_ids == expected_ids

    def test_empty_flow_is_always_kept(self) -> None:
        """A flow with zero nodes never registers as a duplicate (matches
        the original guard: ``if not flow_ids or not kept_set: continue``).
        """
        empty_flow: list[GraphNode] = []
        other = _make_flow(["a", "b", "c"])

        expected = _deduplicate_flows_reference([empty_flow, other, empty_flow])
        actual = deduplicate_flows([empty_flow, other, empty_flow])

        assert len(actual) == len(expected) == 3


class TestDeduplicateFlowsScale:
    """The inverted-index rewrite must stay fast on a large, mostly-disjoint
    input -- the scenario that made the old all-pairs algorithm O(n^2).
    """

    def test_500_disjoint_flows_completes_quickly(self) -> None:
        flows = [_make_flow([f"f{i}n{j}" for j in range(20)]) for i in range(500)]

        start = time.perf_counter()
        result = deduplicate_flows(flows)
        elapsed = time.perf_counter() - start

        # Fully disjoint flows can never overlap -> all 500 survive.
        assert len(result) == 500
        # Generous bound (the new implementation should take low
        # milliseconds here); this only needs to catch a regression back
        # to O(n^2) all-pairs comparison, not chase a tight benchmark.
        assert elapsed < 2.0
