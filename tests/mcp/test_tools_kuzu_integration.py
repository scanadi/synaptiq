"""Integration tests: every MCP tool handler against a real KuzuBackend.

The unit tests in ``test_tools.py`` use mocked storage, which means the
Cypher strings inside the handlers are never validated against the real
schema.  This module seeds a small but complete graph (calls, imports,
coupling, community, process, heritage, types) into an actual KuzuDB and
exercises each handler end-to-end.
"""

from __future__ import annotations

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
)
from synaptiq.core.storage.kuzu_backend import KuzuBackend
from synaptiq.mcp.resources import get_dead_code_list, get_overview
from synaptiq.mcp.tools import (
    handle_call_path,
    handle_communities,
    handle_context,
    handle_coupling,
    handle_cycles,
    handle_cypher,
    handle_detect_changes,
    handle_explain,
    handle_export,
    handle_file_context,
    handle_impact,
    handle_query,
    handle_review_risk,
    handle_suggest,
    handle_test_impact,
)

SAMPLE_DIFF = """diff --git a/src/auth.py b/src/auth.py
index 0000000..1111111 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,4 +2,5 @@
 changed lines
"""


def _seed_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()

    g.add_node(GraphNode(
        id="file:src/auth.py:", label=NodeLabel.FILE, name="auth.py",
        file_path="src/auth.py", content="def validate_user(user: User): ...",
    ))
    g.add_node(GraphNode(
        id="file:src/models.py:", label=NodeLabel.FILE, name="models.py",
        file_path="src/models.py", content="class User: ...",
    ))
    g.add_node(GraphNode(
        id="file:tests/test_auth.py:", label=NodeLabel.FILE, name="test_auth.py",
        file_path="tests/test_auth.py", content="def test_validate(): ...",
    ))

    g.add_node(GraphNode(
        id="function:src/auth.py:validate_user", label=NodeLabel.FUNCTION,
        name="validate_user", file_path="src/auth.py", start_line=1, end_line=8,
        content="def validate_user(user: User):\n    user.save()",
        signature="def validate_user(user: User)",
    ))
    g.add_node(GraphNode(
        id="class:src/models.py:User", label=NodeLabel.CLASS, name="User",
        file_path="src/models.py", start_line=1, end_line=20,
        content="class User: ...",
    ))
    g.add_node(GraphNode(
        id="method:src/models.py:User.save", label=NodeLabel.METHOD, name="save",
        file_path="src/models.py", start_line=5, end_line=10,
        content="def save(self): ...", class_name="User",
    ))
    g.add_node(GraphNode(
        id="class:src/models.py:Admin", label=NodeLabel.CLASS, name="Admin",
        file_path="src/models.py", start_line=22, end_line=30,
        content="class Admin(User): ...",
    ))
    g.add_node(GraphNode(
        id="function:tests/test_auth.py:test_validate", label=NodeLabel.FUNCTION,
        name="test_validate", file_path="tests/test_auth.py",
        start_line=1, end_line=4, content="def test_validate(): ...",
    ))
    g.add_node(GraphNode(
        id="function:src/auth.py:unused_helper", label=NodeLabel.FUNCTION,
        name="unused_helper", file_path="src/auth.py", start_line=10,
        end_line=12, content="def unused_helper(): ...", is_dead=True,
    ))

    g.add_node(GraphNode(
        id="community:community_0:", label=NodeLabel.COMMUNITY, name="Auth",
        properties={"cohesion": 0.8, "symbol_count": 2},
    ))
    g.add_node(GraphNode(
        id="process:process_0:", label=NodeLabel.PROCESS,
        name="validate_user → save",
        properties={"step_count": 2, "kind": "intra_community"},
    ))

    def rel(rid, rtype, src, tgt, props=None):
        g.add_relationship(GraphRelationship(
            id=rid, type=rtype, source=src, target=tgt, properties=props or {},
        ))

    rel("defines:1", RelType.DEFINES, "file:src/auth.py:",
        "function:src/auth.py:validate_user")
    rel("defines:2", RelType.DEFINES, "file:src/models.py:", "class:src/models.py:User")
    rel("imports:1", RelType.IMPORTS, "file:src/auth.py:", "file:src/models.py:",
        {"symbols": "User"})
    rel("calls:1", RelType.CALLS, "function:src/auth.py:validate_user",
        "method:src/models.py:User.save", {"confidence": 1.0})
    rel("calls:2", RelType.CALLS, "function:tests/test_auth.py:test_validate",
        "function:src/auth.py:validate_user", {"confidence": 1.0})
    rel("uses_type:1", RelType.USES_TYPE, "function:src/auth.py:validate_user",
        "class:src/models.py:User", {"role": "param"})
    rel("extends:1", RelType.EXTENDS, "class:src/models.py:Admin",
        "class:src/models.py:User")
    rel("member_of:1", RelType.MEMBER_OF, "function:src/auth.py:validate_user",
        "community:community_0:")
    rel("member_of:2", RelType.MEMBER_OF, "method:src/models.py:User.save",
        "community:community_0:")
    rel("step:1", RelType.STEP_IN_PROCESS, "function:src/auth.py:validate_user",
        "process:process_0:", {"step_number": 0})
    rel("step:2", RelType.STEP_IN_PROCESS, "method:src/models.py:User.save",
        "process:process_0:", {"step_number": 1})
    rel("coupled:1", RelType.COUPLED_WITH, "file:src/auth.py:",
        "file:src/models.py:", {"strength": 0.75, "co_changes": 3})

    return g


@pytest.fixture(scope="module")
def storage(tmp_path_factory: pytest.TempPathFactory) -> KuzuBackend:
    db_path = tmp_path_factory.mktemp("integration") / "kuzu"
    backend = KuzuBackend()
    backend.initialize(db_path)
    backend.bulk_load(_seed_graph())
    yield backend
    backend.close()


@pytest.fixture(autouse=True)
def _no_query_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep handle_query hermetic — no embedding model download."""
    monkeypatch.setattr(
        "synaptiq.mcp.tools._get_query_embedding", lambda _q: None
    )


class TestEveryToolAgainstRealKuzu:
    def test_query(self, storage: KuzuBackend) -> None:
        result = handle_query(storage, "validate_user")
        assert "validate_user" in result

    def test_context(self, storage: KuzuBackend) -> None:
        result = handle_context(storage, "validate_user")
        assert "validate_user" in result
        assert "Callers" in result
        assert "test_validate" in result
        assert "Imported by" not in result or "src" in result

    def test_impact(self, storage: KuzuBackend) -> None:
        result = handle_impact(storage, "save")
        assert "validate_user" in result

    def test_coupling_from_either_side(self, storage: KuzuBackend) -> None:
        # COUPLED_WITH is stored in one direction — both sides must work.
        for fp in ("src/auth.py", "src/models.py"):
            result = handle_coupling(storage, fp)
            assert "strength: 0.75" in result, result
            assert "rejected" not in result.lower()

    def test_communities_list(self, storage: KuzuBackend) -> None:
        result = handle_communities(storage)
        assert "Auth" in result
        assert "cohesion: 0.80" in result
        assert "2 symbols" in result

    def test_communities_drill(self, storage: KuzuBackend) -> None:
        result = handle_communities(storage, community="Auth")
        assert "validate_user" in result
        assert "save" in result

    def test_explain(self, storage: KuzuBackend) -> None:
        result = handle_explain(storage, "validate_user")
        assert "Community: Auth" in result
        assert "Process flows" in result

    def test_file_context(self, storage: KuzuBackend) -> None:
        result = handle_file_context(storage, "src/auth.py")
        assert "validate_user" in result
        assert "src/models.py" in result  # imports + coupling
        assert "Coupled files" in result

    def test_review_risk(self, storage: KuzuBackend) -> None:
        result = handle_review_risk(storage, SAMPLE_DIFF)
        assert "Risk:" in result
        assert "validate_user" in result

    def test_detect_changes(self, storage: KuzuBackend) -> None:
        result = handle_detect_changes(storage, SAMPLE_DIFF)
        assert "validate_user" in result

    def test_test_impact(self, storage: KuzuBackend) -> None:
        result = handle_test_impact(storage, symbols=["save"])
        assert "tests/test_auth.py" in result

    def test_call_path(self, storage: KuzuBackend) -> None:
        result = handle_call_path(storage, "test_validate", "save")
        assert "Call path" in result
        assert "validate_user" in result

    def test_cycles(self, storage: KuzuBackend) -> None:
        result = handle_cycles(storage)
        assert "Error" not in result

    def test_export(self, storage: KuzuBackend) -> None:
        result = handle_export(storage, "validate_user")
        assert "=== Symbol: validate_user" in result
        assert "Community: Auth" in result

    def test_cypher(self, storage: KuzuBackend) -> None:
        result = handle_cypher(
            storage, "MATCH (n:Function) RETURN n.name ORDER BY n.name"
        )
        assert "validate_user" in result

    def test_suggest(self, storage: KuzuBackend) -> None:
        result = handle_suggest(storage, "what calls validate_user?")
        assert "Suggested tool calls" in result

    def test_overview_resource(self, storage: KuzuBackend) -> None:
        result = get_overview(storage)
        assert "Node counts by type" in result
        assert "Relationship counts by type" in result
        assert "Could not retrieve" not in result

    def test_dead_code_resource(self, storage: KuzuBackend) -> None:
        result = get_dead_code_list(storage)
        assert "unused_helper" in result
