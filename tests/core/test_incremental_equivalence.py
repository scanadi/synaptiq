"""Property-based incremental-equivalence harness (W3.2f).

The spine of the incremental design's acceptance (design §11 / plan §8
"randomized edit scripts"): a seeded generator emits deterministic edit scripts
over a polyglot fixture (Python + TypeScript + Ruby + Go, a ``go.mod``, a REST
pair — see :mod:`tests.core.incremental_equivalence_fixture`); after **every**
edit step the harness runs the real incremental path (``run_incremental`` against
a real :class:`LadybugBackend`, dispatching a full rebuild on a full-rebuild
verdict exactly as ``analyze`` does) *and* an independent from-scratch full
pipeline of the same tree, then asserts:

* **strict-core equivalence** — the delta-applied index equals the full rebuild
  on the D9 strict core (the shared :mod:`~tests.core.incremental_equivalence_utils`
  comparator: every non-fringe node + parse props, CONTAINS / DEFINES / IMPORTS /
  EXTENDS / IMPLEMENTS / MIXES_IN / USES_TYPE, and ``CALLS >= 0.8``);
* **fringe only drifts in the documented direction** — no dangling edge (a
  survivor never loses an inbound edge; a stale community never points at a
  removed symbol), and communities carried by the delta never vanish while the
  full build still has them;
* **provenance never rots** — the persisted manifest's per-file provenance equals
  a fresh ``build_manifest`` of the current tree (the W3.2e carry-forward class).

At least two scripts cross a **consolidation** mid-run (forced via the
``SYNAPTIQ_CONSOLIDATION_APPLIES`` knob): equivalence must hold across the
incremental → full → incremental transition, and the full build must reset the
pending/staleness stamp.

Scale: the default suite is 10 seeded scripts (~2-4 min). A 100-script soak lives
behind the ``equivalence_soak`` marker (excluded by default; run with
``-m equivalence_soak``). On any mismatch the assertion dumps the minimal diff
plus the seed, step index, and that step's ops, so a CI failure is reproducible
from the seed alone (``EditScriptGenerator(seed)``).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import pytest

import synaptiq.core.ingestion.pipeline as _pipeline
from synaptiq.core.graph.model import NodeLabel, RelType, generate_id
from synaptiq.core.ingestion import incremental as _incremental
from synaptiq.core.ingestion.incremental import REASON_CONSOLIDATE_APPLY_LIMIT
from synaptiq.core.ingestion.manifest import build_manifest
from synaptiq.core.ingestion.pipeline import run_incremental, run_pipeline, stamp_full_manifest
from synaptiq.core.storage.ladybug_backend import LadybugBackend
from tests.core import incremental_equivalence_utils as equiv
from tests.core.incremental_equivalence_fixture import (
    GHOST_SYMBOL,
    GHOST_TARGET,
    EditScriptGenerator,
    RepoState,
    Struct,
    sync_disk,
    write_tree,
)

_TOOL = "test"

#: Default CI-friendly suite: 10 deterministic seeds (~2-4 min total).
DEFAULT_SEEDS = list(range(10))
#: Scripts (>= 4 steps each) driven across a forced consolidation mid-run.
CONSOLIDATION_SEEDS = [0, 1, 3]
#: 100-script soak (opt-in via the ``equivalence_soak`` marker). Disjoint seeds.
SOAK_SEEDS = list(range(10_000, 10_100))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class StepOutcome:
    """What one edit step did — for the interleaving / determinism assertions."""

    index: int
    reason: str
    consolidated: bool
    ops: list[str]


def _full_index(root: Path, backend: LadybugBackend):
    """Full-build *root* into *backend* and stamp a fresh manifest (like analyze)."""
    graph, _ = run_pipeline(root, None, skip_embeddings=True)
    backend.bulk_load(graph)
    stamp_full_manifest(backend, graph, tool_version=_TOOL, git_head=None)
    return graph


def run_equivalence_script(
    seed: int,
    tmp_path: Path,
    *,
    apply_limit: int | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> list[StepOutcome]:
    """Drive one seeded edit script, asserting equivalence after every step.

    ``apply_limit`` forces the consolidation cadence via the
    ``SYNAPTIQ_CONSOLIDATION_APPLIES`` knob (its production env var) — bound
    through the ``should_consolidate`` seam ``run_incremental`` calls, since the
    module constant is frozen into that function's default at import time.
    """
    if apply_limit is not None:
        assert monkeypatch is not None, "apply_limit requires a monkeypatch fixture"
        monkeypatch.setenv("SYNAPTIQ_CONSOLIDATION_APPLIES", str(apply_limit))
        monkeypatch.setattr(
            _pipeline,
            "should_consolidate",
            functools.partial(_incremental.should_consolidate, apply_limit=apply_limit),
        )

    script = EditScriptGenerator(seed).build()
    root = tmp_path / "repo"
    write_tree(root, script.base_tree)

    inc = LadybugBackend()
    inc.initialize(tmp_path / "inc_db")
    oracle = LadybugBackend()
    oracle.initialize(tmp_path / "oracle_db")
    outcomes: list[StepOutcome] = []
    try:
        _full_index(root, inc)
        prev_tree = script.base_tree
        for i, step in enumerate(script.steps):
            sync_disk(root, prev_tree, step.tree)
            prev_tree = step.tree

            # --- incremental path (dispatch full on a full-rebuild verdict) ----
            outcome = run_incremental(
                root, inc, tool_version=_TOOL, git_head=None, now=None, apply=True
            )
            consolidated = outcome.full_rebuild_required
            if consolidated:
                _full_index(root, inc)
            inc_graph = inc.load_graph()

            # --- oracle: from-scratch full pipeline, round-tripped for symmetry -
            full_mem, _ = run_pipeline(root, None, skip_embeddings=True)
            oracle.bulk_load(full_mem)
            full_graph = oracle.load_graph()

            ctx = f"seed={seed} step={i} reason={outcome.reason!r} ops={step.ops}"

            # (1) strict-core equivalence (dumps the minimal diff on mismatch).
            equiv.assert_strict_equivalence(inc_graph, full_graph, context=ctx)

            # (2) fringe only in the documented direction:
            #     no dangling edge (survivors keep inbound edges; stale fringe
            #     never points at a removed node) …
            dangling = equiv.dangling_edges(inc_graph)
            assert not dangling, f"{ctx}: dangling edges {dangling[:8]}"
            #     … and carried communities never vanish while the full build has
            #     them (they lag stale, they do not disappear).
            if equiv.member_of_present(full_graph):
                assert equiv.member_of_present(inc_graph), (
                    f"{ctx}: MEMBER_OF (communities) vanished from the incremental index"
                )

            # (3) provenance never rots: persisted manifest == fresh full build's.
            persisted = inc.read_manifest()
            equiv.assert_manifest_provenance(
                persisted, build_manifest(full_mem, tool_version=_TOOL), context=ctx
            )

            # (4) a consolidation resets the pending/staleness stamp (transition).
            if consolidated:
                assert persisted.index.pending.applies_since_consolidation == 0, (
                    f"{ctx}: consolidation did not reset the pending stamp"
                )

            outcomes.append(StepOutcome(i, outcome.reason, consolidated, step.ops))
    finally:
        inc.close()
        oracle.close()
    return outcomes


# ---------------------------------------------------------------------------
# The property: incremental == full after every step, across 10 seeded scripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", DEFAULT_SEEDS)
def test_incremental_equivalent_to_full_after_each_step(seed, tmp_path):
    outcomes = run_equivalence_script(seed, tmp_path)
    assert outcomes, f"seed={seed}: the script produced no checked steps"


# ---------------------------------------------------------------------------
# Consolidation interleaving: equivalence across incremental → full → incremental
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", CONSOLIDATION_SEEDS)
def test_equivalence_holds_across_forced_consolidation(seed, tmp_path, monkeypatch):
    outcomes = run_equivalence_script(seed, tmp_path, apply_limit=2, monkeypatch=monkeypatch)
    reasons = [o.reason for o in outcomes]

    # A consolidation fired mid-script, and via the apply-limit knob specifically.
    assert REASON_CONSOLIDATE_APPLY_LIMIT in reasons, (
        f"seed={seed}: expected a knob-forced consolidation; reasons={reasons}"
    )
    # The transition is genuinely incremental → full → incremental: some
    # consolidated step has an incremental step both before and after it.
    consolidated_idx = [o.index for o in outcomes if o.consolidated]
    incremental_idx = [o.index for o in outcomes if not o.consolidated]
    assert any(
        any(j < k for j in incremental_idx) and any(m > k for m in incremental_idx)
        for k in consolidated_idx
    ), f"seed={seed}: no incremental→full→incremental interleaving; reasons={reasons}"


# ---------------------------------------------------------------------------
# Targeted: the imported-later closure fix (add a file an existing import wants)
# ---------------------------------------------------------------------------


def test_imported_later_closure_relinks_dangling_import(tmp_path):
    """Adding a module an existing file already imports (but that did not exist
    at index time) re-resolves that importer incrementally — the added-file
    closure over ``unresolved_imports`` (design §5.2), equal to a full rebuild.
    """
    state = RepoState()
    base_tree = state.render()
    root = tmp_path / "repo"
    write_tree(root, base_tree)

    backend = LadybugBackend()
    backend.initialize(tmp_path / "db")
    try:
        _full_index(root, backend)

        # The ghost client's import of py.ghost is unresolved at base time.
        ghost_module = GHOST_TARGET[: -len(".py")].replace("/", ".")
        base_manifest = backend.read_manifest()
        client = base_manifest.files["py/ghost_client.py"]
        assert ghost_module in client.unresolved_imports, (
            f"expected {ghost_module!r} unresolved at base; got {client.unresolved_imports}"
        )

        # Add the module the ghost client has been waiting for.
        state.structural[GHOST_TARGET] = Struct("py", f"def {GHOST_SYMBOL}():\n    return 42\n")
        sync_disk(root, base_tree, state.render())

        outcome = run_incremental(root, backend, tool_version=_TOOL, git_head=None, now=None)
        assert not outcome.full_rebuild_required, f"unexpected full: {outcome.reason}"
        # The added-file closure pulled the previously-dangling importer in.
        assert "py/ghost_client.py" in outcome.plan.dependents

        inc_graph = backend.load_graph()
        # The IMPORTS edge now resolves to the freshly-added file.
        imports = equiv.edge_set(inc_graph, RelType.IMPORTS)
        assert (
            generate_id(NodeLabel.FILE, "py/ghost_client.py"),
            generate_id(NodeLabel.FILE, GHOST_TARGET),
        ) in imports
    finally:
        backend.close()

    # …and it equals a full rebuild on the strict core.
    full_backend = LadybugBackend()
    full_backend.initialize(tmp_path / "full_db")
    try:
        full_graph_mem, _ = run_pipeline(root, None, skip_embeddings=True)
        full_backend.bulk_load(full_graph_mem)
        equiv.assert_strict_equivalence(
            inc_graph, full_backend.load_graph(), context="imported-later closure"
        )
    finally:
        full_backend.close()


# ---------------------------------------------------------------------------
# The generator itself is deterministic (so a seed reproduces a failure exactly)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 5, 9])
def test_script_generation_is_deterministic(seed):
    a = EditScriptGenerator(seed).build()
    b = EditScriptGenerator(seed).build()
    assert a.base_tree == b.base_tree
    assert [s.ops for s in a.steps] == [s.ops for s in b.steps]
    assert [s.tree for s in a.steps] == [s.tree for s in b.steps]


# ---------------------------------------------------------------------------
# Soak — 100 scripts, opt-in (excluded by default; see tests/conftest.py hook)
# ---------------------------------------------------------------------------


@pytest.mark.equivalence_soak
@pytest.mark.parametrize("seed", SOAK_SEEDS)
def test_equivalence_soak(seed, tmp_path):
    """100-script soak, one test per seed (run with ``pytest -m equivalence_soak``).

    Per-seed parametrization keeps a failure addressable by its test id
    (``test_equivalence_soak[10042]``) and lets a bounded local run select a
    seed subset by node id.
    """
    run_equivalence_script(seed, tmp_path)
