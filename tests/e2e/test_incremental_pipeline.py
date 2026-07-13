"""End-to-end incremental indexing (W3.2e).

Full analyze → edit → incremental analyze → MCP queries return the updated graph
→ ``--full`` forces a rebuild → the incremental result matches a full rebuild on
the strict structural core. Exercises the whole seam at the storage +
orchestration level (mirroring ``test_full_pipeline.py``'s style), not the CLI
subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synaptiq.core.graph.model import NodeLabel, RelType, generate_id
from synaptiq.core.ingestion.pipeline import (
    REASON_FORCED_FULL,
    run_incremental,
    run_pipeline,
    stamp_full_manifest,
)
from synaptiq.core.storage.ladybug_backend import LadybugBackend
from synaptiq.mcp.tools import handle_impact, handle_query

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


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


@pytest.fixture()
def indexed_repo(tmp_path: Path):
    """A full-indexed repo + backend (manifest stamped), ready for incremental."""
    root = tmp_path / "repo"
    _write(root, "pkg/models.py", _MODELS)
    _write(root, "pkg/service.py", _SERVICE)
    for i in range(12):  # fillers keep a one-file edit under the ratio gates
        _write(root, f"pkg/f{i}.py", f"def fn{i}():\n    return {i}\n")

    backend = LadybugBackend()
    backend.initialize(tmp_path / "db")
    graph, _ = run_pipeline(root, None, skip_embeddings=True)
    backend.bulk_load(graph)
    stamp_full_manifest(backend, graph, tool_version="test", git_head="head0")
    yield root, backend, tmp_path
    backend.close()


def test_incremental_indexes_new_symbol_and_edges(indexed_repo):
    """Add a function → incremental apply → the new symbol + its CALLS edge are
    queryable, and existing inbound edges survive."""
    root, backend, _ = indexed_repo
    make_user = generate_id(NodeLabel.FUNCTION, "pkg/models.py", "make_user")
    helper = generate_id(NodeLabel.FUNCTION, "pkg/models.py", "helper")
    service_run = generate_id(NodeLabel.METHOD, "pkg/service.py", "Service.run")

    # Edit: add a new function that calls make_user (identity change to models.py).
    _write(root, "pkg/models.py", _MODELS + "\n\ndef helper():\n    return make_user('h')\n")
    outcome = run_incremental(root, backend, tool_version="test")

    assert not outcome.full_rebuild_required
    assert outcome.reason == "incremental"

    # New symbol is indexed.
    assert backend.get_node(helper) is not None

    # helper -> make_user CALLS edge exists, and the pre-existing service.run ->
    # make_user inbound edge was NOT lost by the surgical apply.
    caller_ids = {n.id for n in backend.get_callers(make_user)}
    assert helper in caller_ids
    assert service_run in caller_ids

    # MCP query surface reflects the update: impact(make_user) lists both callers.
    impact = handle_impact(backend, "make_user", depth=2)
    assert "helper" in impact and "run" in impact

    # And the new symbol is findable via hybrid query.
    result = handle_query(backend, "helper", limit=10)
    assert "helper" in result


def test_full_flag_forces_rebuild_and_matches_incremental(indexed_repo):
    """``--full`` (force_full) yields a full verdict; the incremental-built graph
    matches a from-scratch full rebuild on the strict core (equivalence oracle)."""
    root, backend, tmp_path = indexed_repo

    # Incremental edit first.
    _write(root, "pkg/models.py", _MODELS + "\n\ndef helper():\n    return make_user('h')\n")
    assert not run_incremental(root, backend, tool_version="test").full_rebuild_required
    incremental_graph = backend.load_graph()

    # --full forces the full path (orchestration verdict).
    forced = run_incremental(root, backend, tool_version="test", force_full=True)
    assert forced.full_rebuild_required and forced.reason == REASON_FORCED_FULL

    # Oracle: a from-scratch full rebuild of the edited tree.
    full_backend = LadybugBackend()
    full_backend.initialize(tmp_path / "full_db")
    full_graph, _ = run_pipeline(root, None, skip_embeddings=True)
    full_backend.bulk_load(full_graph)
    full_loaded = full_backend.load_graph()
    full_backend.close()

    fringe = frozenset({NodeLabel.COMMUNITY, NodeLabel.PROCESS})
    inc_nodes = {n.id for n in incremental_graph.iter_nodes() if n.label not in fringe}
    full_nodes = {n.id for n in full_loaded.iter_nodes() if n.label not in fringe}
    assert inc_nodes == full_nodes

    def _edges(g, rt, min_conf=None):
        out = set()
        for r in g.iter_relationships():
            if r.type is not rt:
                continue
            if min_conf is not None and (r.properties.get("confidence", 1.0) or 0) < min_conf:
                continue
            out.add((r.source, r.target))
        return out

    for rt in (RelType.CONTAINS, RelType.DEFINES, RelType.IMPORTS, RelType.EXTENDS):
        assert _edges(incremental_graph, rt) == _edges(full_loaded, rt), rt.value
    assert _edges(incremental_graph, RelType.CALLS, 0.8) == _edges(full_loaded, RelType.CALLS, 0.8)
