"""Tests for scoped incremental resolution → GraphDelta assembly (W3.2c).

Exercises :func:`~synaptiq.core.ingestion.incremental_build.build_incremental_delta`
end-to-end over a multi-language (Python + TypeScript + Go) fixture with rich
cross-file edges (import-resolved CALLS, EXTENDS, IMPLEMENTS, USES_TYPE, a REST
link):

* **Targeted delta shape** — body-only edit (zero dependents; edges from the
  edited file replaced; cross-file *inbound* edges left untouched), identity
  change / rename (dependents reparsed, old id removed, callers re-resolved),
  file deletion, file addition (outbound resolved against the global index), and
  the D7 rest_linking-over-the-scoped-set re-run.
* **Equivalence (the heart)** — apply the delta with a real ``LadybugBackend``
  (:meth:`apply_graph_delta`) onto the previous full-index DB and assert the
  result is **byte-identical to a from-scratch full pipeline of the edited tree**
  on the strict structural core (all nodes + their parse-derived properties;
  CONTAINS / DEFINES / IMPORTS / EXTENDS / IMPLEMENTS / MIXES_IN / USES_TYPE;
  CALLS ≥ 0.8), with the D9 bounded-stale fringe (communities, processes,
  ``is_dead`` / ``is_entry_point``, low-confidence fuzzy CALLS, coupling)
  documented and deliberately excluded — those converge only at consolidation
  (W3.2e), which this layer never runs.
"""

from __future__ import annotations

import json

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, RelType, generate_id
from synaptiq.core.ingestion.incremental import build_current_manifest, plan_incremental
from synaptiq.core.ingestion.incremental_build import (
    _reconstruct_stub,
    build_incremental_delta,
)
from synaptiq.core.ingestion.manifest import Manifest, build_manifest, content_sha
from synaptiq.core.ingestion.parser_phase import process_parsing
from synaptiq.core.ingestion.pipeline import run_pipeline
from synaptiq.core.ingestion.structure import process_structure
from synaptiq.core.ingestion.walker import FileEntry, walk_repo
from synaptiq.core.storage.ladybug_backend import LadybugBackend

# ---------------------------------------------------------------------------
# Multi-language fixture (Python + TypeScript + Go), rich in cross-file edges
# ---------------------------------------------------------------------------

_MODELS_PY = (
    "class Account:\n"
    "    def save(self):\n"
    "        return 1\n"
    "\n"
    "\n"
    "def make_user(name):\n"
    "    acct = Account()\n"
    "    return acct\n"
)

# Body-only edit: a comment inside make_user — content moves, identity does not.
_MODELS_PY_BODY_EDIT = (
    "class Account:\n"
    "    def save(self):\n"
    "        return 1\n"
    "\n"
    "\n"
    "def make_user(name):\n"
    "    # tweaked body, same signature\n"
    "    acct = Account()\n"
    "    return acct\n"
)

# Identity change: make_user -> create_user (renamed symbol id).
_MODELS_PY_RENAMED = (
    "class Account:\n"
    "    def save(self):\n"
    "        return 1\n"
    "\n"
    "\n"
    "def create_user(name):\n"
    "    acct = Account()\n"
    "    return acct\n"
)

# Add a symbol to an existing file (identity change: added id only).
_MODELS_PY_ADDED_SYMBOL = (
    "class Account:\n"
    "    def save(self):\n"
    "        return 1\n"
    "\n"
    "\n"
    "def make_user(name):\n"
    "    acct = Account()\n"
    "    return acct\n"
    "\n"
    "\n"
    "def helper():\n"
    "    return make_user('h')\n"
)

_SERVICE_PY = (
    "from py.models import make_user, Account\n"
    "\n"
    "\n"
    "class Service(Account):\n"
    "    def run(self):\n"
    "        return make_user('x')\n"
)

# Caller updated to the renamed symbol (used with _MODELS_PY_RENAMED).
_SERVICE_PY_RENAMED = (
    "from py.models import create_user, Account\n"
    "\n"
    "\n"
    "class Service(Account):\n"
    "    def run(self):\n"
    "        return create_user('x')\n"
)

_HANDLER_PY = (
    "from py.service import Service\n\n\ndef handle():\n    s = Service()\n    return s.run()\n"
)

# Unchanged caller of make_user — becomes a depth-1 *dependent* on a rename.
_REPORT_PY = "from py.models import make_user\n\n\ndef report():\n    return make_user('y')\n"

# Self-contained REST endpoint + client (Python) for the D7 scoped-rest test.
_REST_PY = (
    "import requests\n"
    "\n"
    "\n"
    "@app.get('/ping')\n"
    "def ping():\n"
    "    return 'pong'\n"
    "\n"
    "\n"
    "def call_ping():\n"
    "    return requests.get('/ping')\n"
)

_TYPES_TS = "export interface Repo {\n  find(): string;\n}\n\nexport class Store {}\n"

_APP_TS = (
    "import { Repo, Store } from './types';\n"
    "\n"
    "export class App implements Repo {\n"
    "  find(): string { return 'x'; }\n"
    "  make(): Store { return new Store(); }\n"
    "}\n"
)

_GO_MOD = "module example.com/proj\n\ngo 1.21\n"

_GO_USER = (
    "package models\n"
    "\n"
    "type User struct {\n"
    "\tName string\n"
    "}\n"
    "\n"
    "func NewUser() *User {\n"
    "\treturn &User{}\n"
    "}\n"
)

_GO_MAIN = (
    "package main\n"
    "\n"
    'import "example.com/proj/go/models"\n'
    "\n"
    "func main() {\n"
    "\tu := models.NewUser()\n"
    "\t_ = u\n"
    "}\n"
)

BASE_FILES: dict[str, str] = {
    "py/models.py": _MODELS_PY,
    "py/service.py": _SERVICE_PY,
    "py/handler.py": _HANDLER_PY,
    "py/report.py": _REPORT_PY,
    "py/rest_self.py": _REST_PY,
    "ts/types.ts": _TYPES_TS,
    "ts/app.ts": _APP_TS,
    "go/go.mod": _GO_MOD,
    "go/models/user.go": _GO_USER,
    "go/main.go": _GO_MAIN,
}

# --- handy node ids ---------------------------------------------------------
MAKE_USER = generate_id(NodeLabel.FUNCTION, "py/models.py", "make_user")
CREATE_USER = generate_id(NodeLabel.FUNCTION, "py/models.py", "create_user")
ACCOUNT = generate_id(NodeLabel.CLASS, "py/models.py", "Account")
ACCOUNT_SAVE = generate_id(NodeLabel.METHOD, "py/models.py", "Account.save")
SERVICE_RUN = generate_id(NodeLabel.METHOD, "py/service.py", "Service.run")
REPORT_FN = generate_id(NodeLabel.FUNCTION, "py/report.py", "report")
HANDLE_FN = generate_id(NodeLabel.FUNCTION, "py/handler.py", "handle")
FILE_HANDLER = generate_id(NodeLabel.FILE, "py/handler.py", "")
PING = generate_id(NodeLabel.FUNCTION, "py/rest_self.py", "ping")
CALL_PING = generate_id(NodeLabel.FUNCTION, "py/rest_self.py", "call_ping")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_repo(root, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _apply_edit(base: dict[str, str], *, set_: dict | None = None, delete=()) -> dict[str, str]:
    files = dict(base)
    for path in delete:
        files.pop(path, None)
    if set_:
        files.update(set_)
    return files


def _reparse_changed(walk: list[FileEntry], previous: Manifest) -> KnowledgeGraph:
    """Parse (structure + symbols only) every content-changed / added file.

    Mirrors what the W3.2e caller feeds :func:`build_current_manifest`: the
    planner reads only ``symbol_sigs`` off the current manifest, so resolved
    edges are unnecessary here — a structure+parse pass is enough and keeps the
    helper fast.
    """
    prev_sha = {p: fm.content_sha for p, fm in previous.files.items()}
    changed = [e for e in walk if content_sha(e.content) != prev_sha.get(e.path)]
    graph = KnowledgeGraph()
    process_structure(changed, graph)
    process_parsing(changed, graph)
    return graph


def _plan_and_delta(edited_root, previous: Manifest):
    """Wire the full W3.2b→W3.2c path: walk → current manifest → plan → delta."""
    walk = walk_repo(edited_root)
    current = build_current_manifest(walk, _reparse_changed(walk, previous), tool_version="test")
    plan = plan_incremental(previous, current)
    delta, new_manifest = build_incremental_delta(plan, previous, walk, tool_version="test")
    return walk, plan, delta, new_manifest


@pytest.fixture(scope="module")
def base_index(tmp_path_factory):
    """Full-index the base fixture once: (root, in-memory graph, manifest)."""
    root = tmp_path_factory.mktemp("incbuild_base")
    _write_repo(root, BASE_FILES)
    graph, _ = run_pipeline(root, None, skip_embeddings=True)
    previous = build_manifest(graph, tool_version="test")
    return root, graph, previous


def _edited_repo(tmp_path, files: dict[str, str]):
    root = tmp_path / "edited"
    root.mkdir()
    _write_repo(root, files)
    return root


# --- comparison utilities ---------------------------------------------------

# Genuinely-global / deferred artifacts — the D9 bounded-stale fringe. Excluded
# from the strict equivalence assertions (they converge only at consolidation).
_FRINGE_NODE_LABELS = frozenset({NodeLabel.COMMUNITY, NodeLabel.PROCESS})
_STRICT_EDGE_TYPES = (
    RelType.CONTAINS,
    RelType.DEFINES,
    RelType.IMPORTS,
    RelType.EXTENDS,
    RelType.IMPLEMENTS,
    RelType.MIXES_IN,
    RelType.USES_TYPE,
)


def _strict_node_props(node: GraphNode):
    """Parse/heritage-derived, delta-strict node fields.

    Excludes ``is_dead`` (recount is local-in-degree only; the global
    false-positive passes are deferred) and ``is_entry_point`` (processes phase
    deferred) — both D9 fringe.
    """
    return (
        node.label,
        node.name,
        node.file_path,
        node.class_name,
        node.signature,
        node.start_line,
        node.end_line,
        node.content,
        node.language,
        node.is_exported,
        json.dumps(node.properties, sort_keys=True),
    )


def _edge_set(graph: KnowledgeGraph, rel_type: RelType, min_conf: float | None = None):
    out: set[tuple[str, str]] = set()
    for rel in graph.iter_relationships():
        if rel.type is not rel_type:
            continue
        if min_conf is not None:
            conf = rel.properties.get("confidence", 1.0)
            if conf is None or conf < min_conf:
                continue
        out.add((rel.source, rel.target))
    return out


def _assert_strict_equivalence(incremental: KnowledgeGraph, full: KnowledgeGraph) -> None:
    """Assert the delta-applied graph == a full rebuild on the strict core."""
    inc_nodes = {n.id: n for n in incremental.iter_nodes() if n.label not in _FRINGE_NODE_LABELS}
    full_nodes = {n.id: n for n in full.iter_nodes() if n.label not in _FRINGE_NODE_LABELS}

    assert set(inc_nodes) == set(full_nodes), (
        f"node id set diverged; "
        f"only-incremental={set(inc_nodes) - set(full_nodes)}, "
        f"only-full={set(full_nodes) - set(inc_nodes)}"
    )
    for nid, full_node in full_nodes.items():
        assert _strict_node_props(inc_nodes[nid]) == _strict_node_props(full_node), (
            f"node properties diverged for {nid}"
        )

    for rel_type in _STRICT_EDGE_TYPES:
        inc_edges = _edge_set(incremental, rel_type)
        full_edges = _edge_set(full, rel_type)
        assert inc_edges == full_edges, (
            f"{rel_type.value} edges diverged; "
            f"only-incremental={inc_edges - full_edges}, only-full={full_edges - inc_edges}"
        )

    inc_calls = _edge_set(incremental, RelType.CALLS, min_conf=0.8)
    full_calls = _edge_set(full, RelType.CALLS, min_conf=0.8)
    assert inc_calls == full_calls, (
        f"CALLS>=0.8 diverged; "
        f"only-incremental={inc_calls - full_calls}, only-full={full_calls - inc_calls}"
    )


# ---------------------------------------------------------------------------
# _reconstruct_stub — id → target-only node
# ---------------------------------------------------------------------------


def test_reconstruct_stub_function():
    stub = _reconstruct_stub("function:py/models.py:make_user", "py/models.py")
    assert stub is not None
    assert stub.label is NodeLabel.FUNCTION
    assert stub.name == "make_user"
    assert stub.class_name == ""
    assert stub.file_path == "py/models.py"
    # Target-only: no content/line data (never read for a target).
    assert stub.content == "" and stub.start_line == 0


def test_reconstruct_stub_method_splits_class():
    stub = _reconstruct_stub("method:py/models.py:Account.save", "py/models.py")
    assert stub is not None
    assert stub.label is NodeLabel.METHOD
    assert stub.name == "save"
    assert stub.class_name == "Account"


def test_reconstruct_stub_method_collision_suffix_stripped():
    # assign_symbol_ids appends #L{line} to a duplicate; the id keeps it, the
    # reconstructed name/class do not.
    stub = _reconstruct_stub("method:a/b.py:User.save#L42", "a/b.py")
    assert stub is not None
    assert stub.id == "method:a/b.py:User.save#L42"
    assert stub.name == "save"
    assert stub.class_name == "User"


def test_reconstruct_stub_file_uses_basename():
    stub = _reconstruct_stub("file:py/models.py:", "py/models.py")
    assert stub is not None
    assert stub.label is NodeLabel.FILE
    assert stub.name == "models.py"


def test_reconstruct_stub_path_with_colon():
    # A path containing a colon still splits — the known prefix is stripped exactly.
    stub = _reconstruct_stub("function:weird:dir/x.py:foo", "weird:dir/x.py")
    assert stub is not None
    assert stub.file_path == "weird:dir/x.py"
    assert stub.name == "foo"


def test_reconstruct_stub_unknown_label_rejected():
    assert _reconstruct_stub("bogus:py/x.py:foo", "py/x.py") is None
    assert _reconstruct_stub("function:other/path.py:foo", "py/x.py") is None


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


def test_full_rebuild_plan_is_rejected(base_index):
    _, _, previous = base_index
    empty = Manifest(index=previous.index, files={})
    # previous=None => the planner returns a full-rebuild verdict.
    plan = plan_incremental(None, empty)
    assert plan.full_rebuild_required
    with pytest.raises(ValueError, match="requires an incremental plan"):
        build_incremental_delta(plan, previous, [])


# ---------------------------------------------------------------------------
# Targeted delta shape
# ---------------------------------------------------------------------------


def test_body_only_edit_zero_dependents_and_scoped_upsert(base_index, tmp_path):
    _, _, previous = base_index
    edited = _edited_repo(
        tmp_path, _apply_edit(BASE_FILES, set_={"py/models.py": _MODELS_PY_BODY_EDIT})
    )

    _, plan, delta, _ = _plan_and_delta(edited, previous)

    assert not plan.full_rebuild_required
    # Body-only edit ⇒ NO dependent closure (design's headline win, §5.2 Q2).
    assert plan.dependents == frozenset()
    assert plan.files_to_reparse == frozenset({"py/models.py"})

    upsert_ids = {n.id for n in delta.nodes_upsert}
    # Only the edited file's own symbols are upserted…
    assert {MAKE_USER, ACCOUNT, ACCOUNT_SAVE} <= upsert_ids
    # …and no symbol/File node from any other file (the parent Folder node, whose
    # file_path is the directory, is re-emitted idempotently and is expected).
    non_folder_paths = {n.file_path for n in delta.nodes_upsert if n.label is not NodeLabel.FOLDER}
    assert non_folder_paths == {"py/models.py"}
    assert not delta.nodes_remove


def test_body_only_edit_replaces_own_edges_but_not_inbound(base_index, tmp_path):
    _, _, previous = base_index
    edited = _edited_repo(
        tmp_path, _apply_edit(BASE_FILES, set_={"py/models.py": _MODELS_PY_BODY_EDIT})
    )

    _, _, delta, _ = _plan_and_delta(edited, previous)

    remove_pairs = {(e.rel_type, e.source, e.target) for e in delta.edges_remove}
    add_pairs = {(r.type.value, r.source, r.target) for r in delta.edges_add}

    # The edited file's OWN outbound CALLS edge is deleted-then-re-added (replaced).
    own = (RelType.CALLS.value, MAKE_USER, ACCOUNT)
    assert own in remove_pairs
    assert own in add_pairs

    # A cross-file INBOUND edge (Service.run -> make_user, source in an unchanged
    # file) is NEVER in edges_remove — surviving inbound edges are left in place.
    assert all(e.source != SERVICE_RUN for e in delta.edges_remove)
    assert (RelType.CALLS.value, SERVICE_RUN, MAKE_USER) not in remove_pairs


def test_identity_rename_reparses_dependents_and_reresolves(base_index, tmp_path):
    _, _, previous = base_index
    # Rename make_user -> create_user AND update one caller (service); leave the
    # other caller (report) unchanged so it enters as a depth-1 dependent.
    edited = _edited_repo(
        tmp_path,
        _apply_edit(
            BASE_FILES,
            set_={"py/models.py": _MODELS_PY_RENAMED, "py/service.py": _SERVICE_PY_RENAMED},
        ),
    )

    _, plan, delta, _ = _plan_and_delta(edited, previous)

    assert not plan.full_rebuild_required
    # The unchanged caller of the renamed symbol is pulled in (depth-1 closure).
    assert plan.dependents == frozenset({"py/report.py"})

    # Old id removed, new id upserted.
    assert MAKE_USER in delta.nodes_remove
    assert CREATE_USER in {n.id for n in delta.nodes_upsert}

    add_pairs = {(r.type.value, r.source, r.target) for r in delta.edges_add}
    # The updated caller's edge is re-resolved to the NEW id …
    assert (RelType.CALLS.value, SERVICE_RUN, CREATE_USER) in add_pairs
    # … and nothing points at the removed old id any more.
    assert all(r.target != MAKE_USER for r in delta.edges_add)


def test_file_deletion_removes_symbols_and_owned_edges(base_index, tmp_path):
    _, _, previous = base_index
    edited = _edited_repo(tmp_path, _apply_edit(BASE_FILES, delete=["py/handler.py"]))

    _, plan, delta, _ = _plan_and_delta(edited, previous)

    assert not plan.full_rebuild_required
    assert plan.diff is not None and "py/handler.py" in plan.diff.deleted_files
    # Deleted file's symbols + File node are removed…
    assert {HANDLE_FN, FILE_HANDLER} <= set(delta.nodes_remove)
    # …nothing from the deleted file is upserted…
    assert all(n.file_path != "py/handler.py" for n in delta.nodes_upsert)
    # …and its owned outbound edges (handle -> Service) are in edges_remove.
    remove_pairs = {(e.source, e.target) for e in delta.edges_remove}
    assert (HANDLE_FN, generate_id(NodeLabel.CLASS, "py/service.py", "Service")) in remove_pairs


def test_file_addition_resolves_outbound_against_global_index(base_index, tmp_path):
    _, _, previous = base_index
    # New leaf module that imports+calls an EXISTING (unchanged) symbol.
    extra = "from py.models import make_user\n\n\ndef spawn():\n    return make_user('z')\n"
    edited = _edited_repo(tmp_path, _apply_edit(BASE_FILES, set_={"py/extra.py": extra}))

    _, plan, delta, _ = _plan_and_delta(edited, previous)

    assert not plan.full_rebuild_required
    assert plan.files_to_reparse == frozenset({"py/extra.py"})
    assert not delta.nodes_remove

    spawn_id = generate_id(NodeLabel.FUNCTION, "py/extra.py", "spawn")
    file_extra = generate_id(NodeLabel.FILE, "py/extra.py", "")
    add_pairs = {(r.type.value, r.source, r.target) for r in delta.edges_add}
    # Outbound edges of the new file resolve against the GLOBAL (stub-seeded) index:
    assert (RelType.CALLS.value, spawn_id, MAKE_USER) in add_pairs  # import-resolved call
    assert (
        RelType.IMPORTS.value,
        file_extra,
        generate_id(NodeLabel.FILE, "py/models.py", ""),
    ) in add_pairs
    # make_user's in-degree changed ⇒ it must be recounted for deadness.
    assert MAKE_USER in delta.dead_recount


def test_rest_linking_runs_over_scoped_set(base_index, tmp_path):
    """D7: rest_linking is re-run over the re-parsed set (both sides scoped)."""
    _, _, previous = base_index
    # Body edit inside call_ping so rest_self.py re-parses; its endpoint+client
    # are both in the re-parse set, so the scoped rest link is reproduced.
    edited_rest = _REST_PY.replace(
        "return requests.get('/ping')", "x = 1\n    return requests.get('/ping')"
    )
    edited = _edited_repo(tmp_path, _apply_edit(BASE_FILES, set_={"py/rest_self.py": edited_rest}))

    _, plan, delta, _ = _plan_and_delta(edited, previous)

    assert plan.files_to_reparse == frozenset({"py/rest_self.py"})
    add_pairs = {(r.source, r.target) for r in delta.edges_add}
    # rest_linking produced the client -> endpoint CALLS edge within the scope.
    assert (CALL_PING, PING) in add_pairs


# ---------------------------------------------------------------------------
# Equivalence — the heart: delta-applied DB == full rebuild (strict core)
# ---------------------------------------------------------------------------


def _edited_files(scenario: str) -> dict[str, str]:
    if scenario == "body_only":
        return _apply_edit(BASE_FILES, set_={"py/models.py": _MODELS_PY_BODY_EDIT})
    if scenario == "add_symbol":
        return _apply_edit(BASE_FILES, set_={"py/models.py": _MODELS_PY_ADDED_SYMBOL})
    if scenario == "rename":
        return _apply_edit(
            BASE_FILES,
            set_={"py/models.py": _MODELS_PY_RENAMED, "py/service.py": _SERVICE_PY_RENAMED},
        )
    if scenario == "delete":
        return _apply_edit(BASE_FILES, delete=["py/handler.py"])
    if scenario == "add_same_folder":
        extra = "from py.models import make_user\n\n\ndef spawn():\n    return make_user('z')\n"
        return _apply_edit(BASE_FILES, set_={"py/extra.py": extra})
    if scenario == "add_new_folder":
        util = "from py.models import make_user\n\n\ndef util_fn():\n    return make_user('u')\n"
        return _apply_edit(BASE_FILES, set_={"lib/util.py": util})
    raise AssertionError(scenario)


@pytest.mark.parametrize(
    "scenario",
    ["body_only", "add_symbol", "rename", "delete", "add_same_folder", "add_new_folder"],
)
def test_delta_equivalent_to_full_rebuild(base_index, tmp_path, scenario):
    _, base_graph, previous = base_index
    edited_root = _edited_repo(tmp_path, _edited_files(scenario))

    # --- incremental side: previous full DB + apply_graph_delta(delta) --------
    inc_backend = LadybugBackend()
    inc_backend.initialize(tmp_path / "inc_db")
    inc_backend.bulk_load(base_graph)
    _, plan, delta, _ = _plan_and_delta(edited_root, previous)
    assert not plan.full_rebuild_required, f"{scenario}: unexpected full-rebuild verdict"
    inc_backend.apply_graph_delta(delta)
    incremental_graph = inc_backend.load_graph()
    inc_backend.close()

    # --- oracle side: from-scratch full pipeline of the edited tree -----------
    full_backend = LadybugBackend()
    full_backend.initialize(tmp_path / "full_db")
    full_mem, _ = run_pipeline(edited_root, None, skip_embeddings=True)
    full_backend.bulk_load(full_mem)
    full_graph = full_backend.load_graph()
    full_backend.close()

    _assert_strict_equivalence(incremental_graph, full_graph)


def test_equivalence_preserves_cross_file_inbound_and_fringe_is_documented(base_index, tmp_path):
    """A body-only edit leaves every cross-file inbound edge intact, and the
    bounded-stale fringe (communities, is_dead) is present but NOT asserted equal.
    """
    _, base_graph, previous = base_index
    edited_root = _edited_repo(
        tmp_path, _apply_edit(BASE_FILES, set_={"py/models.py": _MODELS_PY_BODY_EDIT})
    )

    inc_backend = LadybugBackend()
    inc_backend.initialize(tmp_path / "inc_db")
    inc_backend.bulk_load(base_graph)
    _, _, delta, _ = _plan_and_delta(edited_root, previous)
    inc_backend.apply_graph_delta(delta)
    incremental_graph = inc_backend.load_graph()
    inc_backend.close()

    # The inbound cross-file caller edge (Service.run -> make_user) survived the
    # delta untouched — the whole point of not DETACH-DELETEing the changed file.
    calls = _edge_set(incremental_graph, RelType.CALLS)
    assert (SERVICE_RUN, MAKE_USER) in calls

    # Fringe still lives in the DB (carried from the initial full index); it is
    # deliberately excluded from the strict assertions (D9) and reconciled at
    # consolidation — assert only that it EXISTS, not that it matches a rebuild.
    member_of = _edge_set(incremental_graph, RelType.MEMBER_OF)
    assert member_of, "communities (MEMBER_OF) are carried as bounded-stale fringe"
