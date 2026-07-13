"""Tests for the incremental scope planner (W3.2b).

Drives :func:`~synaptiq.core.ingestion.incremental.plan_incremental` — the pure
planner — through every design §5.3 change category, the depth-1 dependent
closure (sourced strictly from the previous manifest's ``out_edges``), the
symbol-removal sets, the §6.3 ratio thresholds (incl. their boundaries), the §8
fallback triggers, and a property-style invariant sweep over randomized edit
scenarios. Also covers :func:`build_current_manifest`, the walk+reparse → current
manifest bridge.

Synthetic manifests are built directly (no storage, no parser) so the planner's
purity is exercised in isolation.
"""

from __future__ import annotations

import random

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, RelType, generate_id
from synaptiq.core.ingestion.incremental import (
    CONSOLIDATION_APPLY_LIMIT,
    CONSOLIDATION_MAX_SECONDS,
    CONSOLIDATION_SYMBOL_RATIO,
    FILE_RATIO_THRESHOLD,
    REASON_CONSOLIDATE_APPLY_LIMIT,
    REASON_CONSOLIDATE_GIT_HEAD,
    REASON_CONSOLIDATE_STALENESS,
    REASON_CONSOLIDATE_SYMBOL_RATIO,
    REASON_CORRUPT_MANIFEST,
    REASON_FILE_RATIO,
    REASON_INCREMENTAL,
    REASON_NO_MANIFEST,
    REASON_SYMBOL_RATIO,
    REASON_VERSION_MISMATCH,
    SYMBOL_RATIO_THRESHOLD,
    IncrementalPlan,
    build_current_manifest,
    plan_incremental,
    should_consolidate,
)
from synaptiq.core.ingestion.manifest import (
    CURRENT_MANIFEST_VERSION,
    EdgeRef,
    FileManifest,
    IndexManifest,
    IndexPending,
    Manifest,
    build_manifest,
    compute_fingerprint,
    content_sha,
)
from synaptiq.core.ingestion.walker import FileEntry

# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------

FILE_A = generate_id(NodeLabel.FILE, "src/a.py")
FUNC_FOO = generate_id(NodeLabel.FUNCTION, "src/a.py", "foo")
FILE_B = generate_id(NodeLabel.FILE, "src/b.py")
FUNC_BAR = generate_id(NodeLabel.FUNCTION, "src/b.py", "bar")
FILE_C = generate_id(NodeLabel.FILE, "src/c.py")
FUNC_BAZ = generate_id(NodeLabel.FUNCTION, "src/c.py", "baz")

IMPORTS = RelType.IMPORTS.value
CALLS = RelType.CALLS.value
EXTENDS = RelType.EXTENDS.value

# ---------------------------------------------------------------------------
# Synthetic-manifest builders
# ---------------------------------------------------------------------------


def _fm(
    path: str,
    csha: str,
    sigs: dict[str, str],
    *,
    out_edges: list[EdgeRef] | None = None,
    unresolved_imports: list[str] | None = None,
) -> FileManifest:
    """FileManifest with ``symbol_ids`` derived from ``sigs`` (build_manifest
    keeps them in lockstep, so synthetic rows must too)."""
    return FileManifest(
        path=path,
        content_sha=csha,
        symbol_ids=sorted(sigs),
        symbol_sigs=dict(sigs),
        out_edges=list(out_edges or []),
        unresolved_imports=list(unresolved_imports or []),
    )


def _manifest(
    files: list[FileManifest],
    *,
    version: int = CURRENT_MANIFEST_VERSION,
    fingerprint: str | None = None,
) -> Manifest:
    """Assemble a Manifest, auto-computing a *consistent* ``full_fingerprint``
    unless one is supplied (so the corruption gate stays quiet by default)."""
    fmap = {f.path: f for f in files}
    if fingerprint is None:
        fingerprint = compute_fingerprint({p: f.content_sha for p, f in fmap.items()})
    return Manifest(
        index=IndexManifest(manifest_version=version, full_fingerprint=fingerprint),
        files=fmap,
    )


# ===========================================================================
# §5.3 change categories through the planner
# ===========================================================================


def test_no_change_carries_everything_untouched():
    """Identical manifests → nothing to reparse, everything carried."""
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f"})])
    curr = _manifest([_fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f"})])

    plan = plan_incremental(prev, curr)

    assert not plan.full_rebuild_required
    assert plan.reason == REASON_INCREMENTAL
    assert plan.files_to_reparse == frozenset()
    assert plan.files_unchanged_to_carry == frozenset({"src/a.py"})
    assert plan.symbols_to_remove == frozenset()
    assert plan.dependents == frozenset()


def test_body_only_change_reparses_self_only():
    """Body-only edit (same sigs, new content_sha) → reparse self, ZERO closure."""
    prev = _manifest(
        [
            _fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"}),
            _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"}),
        ]
    )
    # a.py content changed but every identity fingerprint is unchanged.
    curr = _manifest(
        [
            _fm("src/a.py", "A2", {FILE_A: "s", FUNC_FOO: "f1"}),
            _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"}),
        ]
    )

    plan = plan_incremental(prev, curr)

    assert plan.files_to_reparse == frozenset({"src/a.py"})
    assert plan.dependents == frozenset()  # the design's headline economy
    assert plan.files_unchanged_to_carry == frozenset({"src/b.py"})
    assert plan.symbols_to_remove == frozenset()
    assert plan.affected_symbol_count == 0  # body-only excluded from affected


def test_added_symbol_is_identity_change():
    """Adding a symbol makes the file identity-set-changed and reparsed."""
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"})])
    new_func = generate_id(NodeLabel.FUNCTION, "src/a.py", "new")
    curr = _manifest([_fm("src/a.py", "A2", {FILE_A: "s", FUNC_FOO: "f1", new_func: "n1"})])

    plan = plan_incremental(prev, curr)

    assert plan.files_to_reparse == frozenset({"src/a.py"})
    assert plan.diff is not None
    assert plan.diff.changed["src/a.py"].added == frozenset({new_func})
    assert plan.diff.changed["src/a.py"].identity_set_changed
    assert plan.symbols_to_remove == frozenset()  # nothing removed, only added


def test_removed_symbol_goes_to_removal_set():
    """Removing a symbol lands its id in symbols_to_remove."""
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"})])
    curr = _manifest([_fm("src/a.py", "A2", {FILE_A: "s"})])  # foo gone

    plan = plan_incremental(prev, curr)

    assert plan.files_to_reparse == frozenset({"src/a.py"})
    assert plan.symbols_to_remove == frozenset({FUNC_FOO})


def test_rename_is_removed_old_plus_added_new():
    """A rename changes the id → old id removed, new id added (node-id format)."""
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"})])
    renamed = generate_id(NodeLabel.FUNCTION, "src/a.py", "foo2")
    curr = _manifest([_fm("src/a.py", "A2", {FILE_A: "s", renamed: "f1"})])

    plan = plan_incremental(prev, curr)

    fd = plan.diff.changed["src/a.py"]
    assert fd.removed == frozenset({FUNC_FOO})
    assert fd.added == frozenset({renamed})
    assert plan.symbols_to_remove == frozenset({FUNC_FOO})  # only the stale old id


def test_identity_change_keeps_id_out_of_removal_set():
    """A signature change (same id, different sig) is upserted, never removed."""
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"})])
    curr = _manifest([_fm("src/a.py", "A2", {FILE_A: "s", FUNC_FOO: "f2"})])

    plan = plan_incremental(prev, curr)

    fd = plan.diff.changed["src/a.py"]
    assert fd.identity_changed == frozenset({FUNC_FOO})
    assert plan.symbols_to_remove == frozenset()  # same id → SET/MERGE, not remove


def test_added_file_is_reparsed_not_removed():
    """A brand-new file joins files_to_reparse; its symbols are never removed."""
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s"})])
    curr = _manifest(
        [
            _fm("src/a.py", "A1", {FILE_A: "s"}),
            _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"}),
        ]
    )

    plan = plan_incremental(prev, curr)

    assert "src/b.py" in plan.files_to_reparse
    assert plan.diff.added_files == frozenset({"src/b.py"})
    assert plan.symbols_to_remove == frozenset()


def test_deleted_file_removes_all_its_symbols():
    """Deleting a file marks every one of its symbols for removal."""
    prev = _manifest(
        [
            _fm("src/a.py", "A1", {FILE_A: "s"}),
            _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"}),
        ]
    )
    curr = _manifest([_fm("src/a.py", "A1", {FILE_A: "s"})])  # b.py gone

    plan = plan_incremental(prev, curr)

    assert plan.diff.deleted_files == frozenset({"src/b.py"})
    assert plan.symbols_to_remove == frozenset({FILE_B, FUNC_BAR})
    assert "src/b.py" not in plan.files_to_reparse  # deleted, not re-resolved


# ===========================================================================
# Depth-1 dependent closure — sourced from previous.out_edges (§5.2, §5.4-5.5)
# ===========================================================================


def test_importer_is_dependent_on_identity_change():
    """A imports B; B's identity changes → A is pulled into re-resolution."""
    a = _fm(
        "src/a.py",
        "A1",
        {FILE_A: "s", FUNC_FOO: "f1"},
        out_edges=[EdgeRef(IMPORTS, FILE_A, FILE_B), EdgeRef(CALLS, FUNC_FOO, FUNC_BAR)],
    )
    b_v1 = _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"})
    b_v2 = _fm("src/b.py", "B2", {FILE_B: "s", FUNC_BAR: "b2"})  # bar sig changed

    prev = _manifest([a, b_v1])
    curr = _manifest([_fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"}), b_v2])

    plan = plan_incremental(prev, curr)

    assert plan.dependents == frozenset({"src/a.py"})
    assert plan.files_to_reparse == frozenset({"src/a.py", "src/b.py"})
    assert "src/a.py" not in plan.files_unchanged_to_carry


def test_body_only_change_pulls_in_no_dependents():
    """A imports B; B changes body-only → nobody else is reparsed."""
    a = _fm(
        "src/a.py",
        "A1",
        {FILE_A: "s", FUNC_FOO: "f1"},
        out_edges=[EdgeRef(IMPORTS, FILE_A, FILE_B)],
    )
    prev = _manifest([a, _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"})])
    # b.py content changes but bar keeps its identity → body-only.
    curr = _manifest(
        [
            _fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"}),
            _fm("src/b.py", "B2", {FILE_B: "s", FUNC_BAR: "b1"}),
        ]
    )

    plan = plan_incremental(prev, curr)

    assert plan.dependents == frozenset()
    assert plan.files_to_reparse == frozenset({"src/b.py"})
    assert plan.files_unchanged_to_carry == frozenset({"src/a.py"})


def test_caller_is_dependent_via_calls_edge():
    """C calls A.foo; foo's identity changes → C is a dependent (reverse CALLS)."""
    c = _fm(
        "src/c.py",
        "C1",
        {FILE_C: "s", FUNC_BAZ: "z1"},
        out_edges=[EdgeRef(CALLS, FUNC_BAZ, FUNC_FOO)],
    )
    prev = _manifest([c, _fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"})])
    curr = _manifest(
        [
            _fm("src/c.py", "C1", {FILE_C: "s", FUNC_BAZ: "z1"}),
            _fm("src/a.py", "A2", {FILE_A: "s", FUNC_FOO: "f2"}),  # foo sig changed
        ]
    )

    plan = plan_incremental(prev, curr)

    assert plan.dependents == frozenset({"src/c.py"})


def test_deleted_file_pulls_in_its_importers():
    """Deleting B forces every file whose edge targeted B's symbols to re-resolve."""
    a = _fm(
        "src/a.py",
        "A1",
        {FILE_A: "s", FUNC_FOO: "f1"},
        out_edges=[EdgeRef(IMPORTS, FILE_A, FILE_B)],
    )
    prev = _manifest([a, _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"})])
    curr = _manifest([_fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"})])  # b gone

    plan = plan_incremental(prev, curr)

    assert plan.dependents == frozenset({"src/a.py"})
    assert "src/a.py" in plan.files_to_reparse
    assert plan.symbols_to_remove == frozenset({FILE_B, FUNC_BAR})


def test_dependents_come_only_from_recorded_out_edges():
    """No recorded out_edge → no dependent, even if the file 'conceptually' imports.

    Proves the closure is a reverse lookup over the *stored* provenance, not a
    live graph query or a name scan.
    """
    # a.py has NO out_edges recorded, though it references bar.
    a = _fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"}, out_edges=[])
    prev = _manifest([a, _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"})])
    curr = _manifest(
        [
            _fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"}),
            _fm("src/b.py", "B2", {FILE_B: "s", FUNC_BAR: "b2"}),  # identity change
        ]
    )

    plan = plan_incremental(prev, curr)

    assert plan.dependents == frozenset()  # nothing recorded → nothing pulled in


def test_non_resolver_edge_does_not_create_dependent():
    """Only RESOLVER_REL_TYPES edges drive the closure."""
    # A COUPLED_WITH edge (not a resolver edge) must be ignored.
    a = _fm(
        "src/a.py",
        "A1",
        {FILE_A: "s"},
        out_edges=[EdgeRef(RelType.COUPLED_WITH.value, FILE_A, FILE_B)],
    )
    prev = _manifest([a, _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"})])
    curr = _manifest(
        [
            _fm("src/a.py", "A1", {FILE_A: "s"}),
            _fm("src/b.py", "B2", {FILE_B: "s", FUNC_BAR: "b2"}),
        ]
    )

    plan = plan_incremental(prev, curr)

    assert plan.dependents == frozenset()


def test_heritage_edge_creates_dependent():
    """EXTENDS is a resolver edge → a subclass depends on its changed base."""
    a = _fm(
        "src/a.py",
        "A1",
        {FILE_A: "s", FUNC_FOO: "f1"},
        out_edges=[EdgeRef(EXTENDS, FUNC_FOO, FUNC_BAR)],
    )
    prev = _manifest([a, _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"})])
    curr = _manifest(
        [
            _fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"}),
            _fm("src/b.py", "B2", {FILE_B: "s", FUNC_BAR: "b2"}),
        ]
    )

    plan = plan_incremental(prev, curr)

    assert plan.dependents == frozenset({"src/a.py"})


def test_changed_file_is_not_double_counted_as_dependent():
    """A file changed for its own edit is reparsed, never listed as a dependent."""
    # a.py imports b.py AND a.py itself changes identity; b.py also changes.
    a = _fm(
        "src/a.py",
        "A1",
        {FILE_A: "s", FUNC_FOO: "f1"},
        out_edges=[EdgeRef(IMPORTS, FILE_A, FILE_B)],
    )
    prev = _manifest([a, _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"})])
    curr = _manifest(
        [
            _fm("src/a.py", "A2", {FILE_A: "s", FUNC_FOO: "f2"}),  # a identity-changed
            _fm("src/b.py", "B2", {FILE_B: "s", FUNC_BAR: "b2"}),  # b identity-changed
        ]
    )

    plan = plan_incremental(prev, curr)

    # a.py is changed, so it is NOT in the (unchanged-only) dependents set...
    assert plan.dependents == frozenset()
    # ...but it is still reparsed (for its own change).
    assert plan.files_to_reparse == frozenset({"src/a.py", "src/b.py"})


# ===========================================================================
# Ratio thresholds incl. boundaries (§6.3 / D4)
# ===========================================================================


def _many_files(n: int, changed_count: int, *, body_only: bool) -> tuple[Manifest, Manifest]:
    """Build previous/current with ``n`` single-symbol files, ``changed_count``
    of which change. ``body_only`` keeps identities stable (0 affected symbols),
    isolating the FILE ratio."""
    prev_files, curr_files = [], []
    for i in range(n):
        path = f"src/f{i}.py"
        sid = generate_id(NodeLabel.FUNCTION, path, "fn")
        prev_files.append(_fm(path, f"{i}_v1", {sid: "sig1"}))
        if i < changed_count:
            new_sig = "sig1" if body_only else "sig2"
            curr_files.append(_fm(path, f"{i}_v2", {sid: new_sig}))
        else:
            curr_files.append(_fm(path, f"{i}_v1", {sid: "sig1"}))
    return _manifest(prev_files), _manifest(curr_files)


def test_file_ratio_at_boundary_stays_incremental():
    """3/10 == 0.30 is not > 0.30 → incremental (strict comparison)."""
    prev, curr = _many_files(10, 3, body_only=True)
    plan = plan_incremental(prev, curr)

    assert plan.file_change_ratio == pytest.approx(0.30)
    assert not plan.full_rebuild_required
    assert plan.reason == REASON_INCREMENTAL


def test_file_ratio_over_boundary_forces_full():
    """4/10 == 0.40 > 0.30 → full rebuild, reason file ratio."""
    prev, curr = _many_files(10, 4, body_only=True)
    plan = plan_incremental(prev, curr)

    assert plan.file_change_ratio == pytest.approx(0.40)
    assert plan.full_rebuild_required
    assert plan.reason == REASON_FILE_RATIO


def _one_big_file(total_symbols: int, identity_changed: int) -> tuple[Manifest, Manifest]:
    """One changed file with ``total_symbols`` symbols, ``identity_changed`` of
    which change identity; plus 9 empty unchanged files so the FILE ratio stays
    low (0.10) and the SYMBOL ratio is isolated."""
    big = "src/big.py"
    prev_sigs = {generate_id(NodeLabel.FUNCTION, big, f"s{i}"): "v1" for i in range(total_symbols)}
    curr_sigs = {}
    for i, sid in enumerate(prev_sigs):
        curr_sigs[sid] = "v2" if i < identity_changed else "v1"

    prev_files = [_fm(big, "big_v1", prev_sigs)]
    curr_files = [_fm(big, "big_v2", curr_sigs)]
    for i in range(9):
        path = f"src/pad{i}.py"
        prev_files.append(_fm(path, f"pad{i}", {}))
        curr_files.append(_fm(path, f"pad{i}", {}))
    return _manifest(prev_files), _manifest(curr_files)


def test_symbol_ratio_at_boundary_stays_incremental():
    """4/10 == 0.40 is not > 0.40 → incremental (file ratio kept low)."""
    prev, curr = _one_big_file(total_symbols=10, identity_changed=4)
    plan = plan_incremental(prev, curr)

    assert plan.file_change_ratio == pytest.approx(0.10)  # 1 of 10 files
    assert plan.symbol_change_ratio == pytest.approx(0.40)
    assert not plan.full_rebuild_required
    assert plan.reason == REASON_INCREMENTAL


def test_symbol_ratio_over_boundary_forces_full():
    """5/10 == 0.50 > 0.40 → full rebuild, reason symbol ratio."""
    prev, curr = _one_big_file(total_symbols=10, identity_changed=5)
    plan = plan_incremental(prev, curr)

    assert plan.symbol_change_ratio == pytest.approx(0.50)
    assert plan.full_rebuild_required
    assert plan.reason == REASON_SYMBOL_RATIO


def test_custom_thresholds_are_honored():
    """Thresholds are parameters (module constants are just the defaults)."""
    prev, curr = _many_files(10, 3, body_only=True)  # 0.30 file ratio

    # Tighten the file threshold below 0.30 → now over.
    plan = plan_incremental(prev, curr, file_ratio_threshold=0.25)
    assert plan.full_rebuild_required
    assert plan.reason == REASON_FILE_RATIO

    # Loosen it above 0.30 → stays incremental.
    plan = plan_incremental(prev, curr, file_ratio_threshold=0.50)
    assert not plan.full_rebuild_required


def test_threshold_defaults_match_design_d4():
    assert FILE_RATIO_THRESHOLD == 0.30
    assert SYMBOL_RATIO_THRESHOLD == 0.40


# ===========================================================================
# Fallback triggers (§8)
# ===========================================================================


def test_no_manifest_forces_full():
    """previous is None (first index / pre-manifest install) → full rebuild."""
    curr = _manifest([_fm("src/a.py", "A1", {FILE_A: "s"})])
    plan = plan_incremental(None, curr)

    assert plan.full_rebuild_required
    assert plan.reason == REASON_NO_MANIFEST
    assert plan.diff is None
    assert plan.files_to_reparse == frozenset()
    assert plan.files_unchanged_to_carry == frozenset()
    assert plan.symbols_to_remove == frozenset()


def test_version_mismatch_forces_full():
    """A stale manifest_version (schema/scoping bump) → full rebuild."""
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s"})], version=CURRENT_MANIFEST_VERSION + 1)
    curr = _manifest([_fm("src/a.py", "A2", {FILE_A: "s"})])
    plan = plan_incremental(prev, curr)

    assert plan.full_rebuild_required
    assert plan.reason == REASON_VERSION_MISMATCH


def test_corrupt_fingerprint_forces_full():
    """A previous manifest whose fingerprint disagrees with its files → full."""
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s"})], fingerprint="deadbeef-not-consistent")
    curr = _manifest([_fm("src/a.py", "A2", {FILE_A: "s"})])
    plan = plan_incremental(prev, curr)

    assert plan.full_rebuild_required
    assert plan.reason == REASON_CORRUPT_MANIFEST


def test_empty_fingerprint_is_treated_as_consistent():
    """An unstamped (empty) fingerprint is not 'corrupt' — nothing to check.

    Padded to 10 files (only 1 changed) so the ratio gate does not fire and the
    fingerprint gate is what's actually under test.
    """
    pad = [_fm(f"src/pad{i}.py", f"p{i}", {}) for i in range(9)]
    prev = _manifest([_fm("src/a.py", "A1", {FILE_A: "s"}), *pad], fingerprint="")
    curr = _manifest([_fm("src/a.py", "A2", {FILE_A: "s"}), *pad])
    plan = plan_incremental(prev, curr)

    assert not plan.full_rebuild_required
    assert plan.reason == REASON_INCREMENTAL


def test_version_gate_precedes_ratio_gate():
    """The version trigger wins even when the ratio would also fire."""
    prev, curr = _many_files(10, 9, body_only=True)  # ratio far over
    prev.index.manifest_version = CURRENT_MANIFEST_VERSION + 5
    prev.index.full_fingerprint = compute_fingerprint(
        {p: f.content_sha for p, f in prev.files.items()}
    )
    plan = plan_incremental(prev, curr)

    assert plan.full_rebuild_required
    assert plan.reason == REASON_VERSION_MISMATCH


# ===========================================================================
# Property-style invariant sweep over randomized edit scenarios
# ===========================================================================


def _random_scenario(rng: random.Random) -> tuple[Manifest, Manifest]:
    """Generate a previous manifest with inter-file resolver edges, then a
    current manifest via random per-file mutations (unchanged / body-only /
    identity-change / delete) plus optional new files."""
    n = rng.randint(2, 8)
    paths = [f"src/m{i}.py" for i in range(n)]
    file_ids = {p: generate_id(NodeLabel.FILE, p) for p in paths}
    # Each file gets a file: symbol + a few functions.
    func_ids: dict[str, list[str]] = {}
    prev_files: list[FileManifest] = []
    for p in paths:
        k = rng.randint(1, 3)
        funcs = [generate_id(NodeLabel.FUNCTION, p, f"fn{j}") for j in range(k)]
        func_ids[p] = funcs
        sigs = {file_ids[p]: "fsig"}
        for f in funcs:
            sigs[f] = f"sig-{rng.randint(0, 5)}"
        # Random resolver out_edges to other files' symbols.
        out: list[EdgeRef] = []
        for other in paths:
            if other == p:
                continue
            if rng.random() < 0.35:
                out.append(EdgeRef(IMPORTS, file_ids[p], file_ids[other]))
            if func_ids.get(other) and rng.random() < 0.35:
                out.append(EdgeRef(CALLS, funcs[0], rng.choice(func_ids[other])))
        prev_files.append(_fm(p, f"{p}-v1", sigs, out_edges=out))
    prev = _manifest(prev_files)

    curr_files: list[FileManifest] = []
    for fm in prev_files:
        action = rng.choice(["unchanged", "body", "identity", "delete"])
        if action == "delete":
            continue
        if action == "unchanged":
            curr_files.append(_fm(fm.path, fm.content_sha, dict(fm.symbol_sigs)))
            continue
        # body-only or identity: content_sha changes either way.
        sigs = dict(fm.symbol_sigs)
        if action == "identity":
            # perturb / add / drop a symbol
            mode = rng.choice(["perturb", "add", "drop"])
            funcs = [s for s in sigs if s.startswith("function:")]
            if mode == "perturb" and funcs:
                sigs[rng.choice(funcs)] = f"sig-{rng.randint(6, 12)}"
            elif mode == "add":
                sigs[generate_id(NodeLabel.FUNCTION, fm.path, f"added{rng.randint(0, 99)}")] = "new"
            elif mode == "drop" and funcs:
                del sigs[rng.choice(funcs)]
        curr_files.append(_fm(fm.path, f"{fm.path}-v2", sigs))

    # Occasionally add a brand-new file.
    if rng.random() < 0.4:
        p = f"src/added{rng.randint(0, 99)}.py"
        sid = generate_id(NodeLabel.FUNCTION, p, "brand_new")
        curr_files.append(_fm(p, f"{p}-v1", {generate_id(NodeLabel.FILE, p): "fsig", sid: "s"}))

    return prev, _manifest(curr_files)


def test_property_output_set_invariants():
    """Across many random edit scripts, the planner's output sets always obey
    the structural contract (task's property requirement)."""
    rng = random.Random(1337)
    for _ in range(200):
        prev, curr = _random_scenario(rng)
        plan = plan_incremental(prev, curr)
        diff = plan.diff
        assert diff is not None

        changed = frozenset(diff.changed)
        added = diff.added_files
        unchanged = diff.unchanged_files

        # 1. reparse set is exactly changed ∪ added ∪ dependents.
        assert plan.files_to_reparse == (changed | added | plan.dependents)
        # 2. carry set is exactly unchanged minus dependents.
        assert plan.files_unchanged_to_carry == (unchanged - plan.dependents)
        # 3. dependents are drawn only from unchanged files (pure closure).
        assert plan.dependents <= unchanged
        # 4. reparse ⊆ (changed ∪ dependents ∪ added) — the task's subset rule.
        assert plan.files_to_reparse <= (changed | plan.dependents | added)
        # 5. reparse never includes an unchanged non-dependent.
        unchanged_non_dependents = unchanged - plan.dependents
        assert not (plan.files_to_reparse & unchanged_non_dependents)
        # 6. reparse and carry are disjoint.
        assert not (plan.files_to_reparse & plan.files_unchanged_to_carry)

        # 7. every removed symbol is a genuinely-removed or deleted-file symbol.
        prev_symbols = {s for fm in prev.files.values() for s in fm.symbol_ids}
        assert plan.symbols_to_remove <= prev_symbols
        justified: set[str] = set()
        for fd in diff.changed.values():
            justified |= fd.removed
        for p in diff.deleted_files:
            justified |= set(prev.files[p].symbol_ids)
        assert plan.symbols_to_remove == frozenset(justified)

        # 8. no identity-set change anywhere ⇒ no dependents (body-only economy).
        if not diff.identity_changed_files and not diff.deleted_files:
            assert plan.dependents == frozenset()


def test_property_body_only_repo_never_pulls_dependents():
    """A repo where every changed file is body-only produces zero closure,
    regardless of how densely the files import/call each other."""
    rng = random.Random(99)
    paths = [f"src/b{i}.py" for i in range(6)]
    file_ids = {p: generate_id(NodeLabel.FILE, p) for p in paths}
    prev_files, curr_files = [], []
    for p in paths:
        sid = generate_id(NodeLabel.FUNCTION, p, "fn")
        sigs = {file_ids[p]: "fsig", sid: "sig"}
        out = [
            EdgeRef(IMPORTS, file_ids[p], file_ids[o])
            for o in paths
            if o != p and rng.random() < 0.6
        ]
        prev_files.append(_fm(p, f"{p}-v1", sigs, out_edges=out))
        # Every file changes content but keeps identical sigs → body-only.
        curr_files.append(_fm(p, f"{p}-v2", dict(sigs)))

    plan = plan_incremental(_manifest(prev_files), _manifest(curr_files))

    assert plan.dependents == frozenset()
    assert plan.files_to_reparse == frozenset(paths)  # each reparses itself only
    assert plan.affected_symbol_count == 0


# ===========================================================================
# build_current_manifest — the walk + reparse → current-manifest bridge
# ===========================================================================


def _graph_with_file(path: str, content: str, funcs: dict[str, str]) -> KnowledgeGraph:
    """A minimal reparse graph: one File node (with content) + function nodes."""
    g = KnowledgeGraph()
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.FILE, path),
            label=NodeLabel.FILE,
            name=path.rsplit("/", 1)[-1],
            file_path=path,
            content=content,
            language="python",
        )
    )
    for name, sig in funcs.items():
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FUNCTION, path, name),
                label=NodeLabel.FUNCTION,
                name=name,
                file_path=path,
                signature=sig,
                language="python",
            )
        )
    return g


def test_build_current_manifest_content_sha_from_walk():
    """content_sha for every current file comes from the walk (authoritative)."""
    walk = [
        FileEntry(path="src/a.py", content="def foo(): pass\n", language="python"),
        FileEntry(path="src/b.py", content="def bar(): pass\n", language="python"),
    ]
    reparsed = _graph_with_file("src/a.py", "def foo(): pass\n", {"foo": "foo()"})

    curr = build_current_manifest(walk, reparsed)

    assert curr.files["src/a.py"].content_sha == content_sha("def foo(): pass\n")
    assert curr.files["src/b.py"].content_sha == content_sha("def bar(): pass\n")
    # Fingerprint spans every walked file.
    assert curr.index.full_fingerprint == compute_fingerprint(
        {p: f.content_sha for p, f in curr.files.items()}
    )


def test_build_current_manifest_symbols_from_reparse():
    """Reparsed (changed/added) files carry fresh symbol_sigs; unchanged files
    carry content_sha only."""
    walk = [
        FileEntry(path="src/a.py", content="def foo(): pass\n", language="python"),
        FileEntry(path="src/b.py", content="unchanged\n", language="python"),
    ]
    reparsed = _graph_with_file("src/a.py", "def foo(): pass\n", {"foo": "foo()"})

    curr = build_current_manifest(walk, reparsed)

    a = curr.files["src/a.py"]
    assert generate_id(NodeLabel.FUNCTION, "src/a.py", "foo") in a.symbol_sigs
    assert generate_id(NodeLabel.FILE, "src/a.py") in a.symbol_sigs
    # b.py wasn't reparsed → content hash only, no symbols.
    assert curr.files["src/b.py"].symbol_sigs == {}


def test_build_current_manifest_drives_planner_end_to_end():
    """A realistic flow: build_manifest(previous graph) → previous; a body-only
    edit → build_current_manifest → planner sees exactly one body-only file.

    Uses 5 files (only a.py edited) so the 30% file-ratio gate stays quiet and
    the scoped decision is what's exercised.
    """
    files = {
        "src/a.py": ("foo", "def foo():\n    return 1\n"),
        "src/b.py": ("bar", "def bar():\n    return 2\n"),
        "src/c.py": ("baz", "def baz():\n    return 3\n"),
        "src/d.py": ("qux", "def qux():\n    return 4\n"),
        "src/e.py": ("quux", "def quux():\n    return 5\n"),
    }
    prev_graph = KnowledgeGraph()
    for path, (name, content) in files.items():
        prev_graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, path),
                label=NodeLabel.FILE,
                name=path.rsplit("/", 1)[-1],
                file_path=path,
                content=content,
                language="python",
            )
        )
        prev_graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FUNCTION, path, name),
                label=NodeLabel.FUNCTION,
                name=name,
                file_path=path,
                signature=f"{name}()",
                language="python",
            )
        )
    previous = build_manifest(prev_graph)

    # a.py edited body-only (same signature, new body); the rest untouched.
    new_a = "def foo():\n    return 999\n"
    walk = [FileEntry(path="src/a.py", content=new_a, language="python")]
    for path, (_, content) in files.items():
        if path != "src/a.py":
            walk.append(FileEntry(path=path, content=content, language="python"))
    reparsed = _graph_with_file("src/a.py", new_a, {"foo": "foo()"})
    current = build_current_manifest(walk, reparsed)

    plan = plan_incremental(previous, current)

    assert not plan.full_rebuild_required
    assert plan.files_to_reparse == frozenset({"src/a.py"})
    assert plan.files_unchanged_to_carry == frozenset(
        {"src/b.py", "src/c.py", "src/d.py", "src/e.py"}
    )
    assert plan.dependents == frozenset()
    assert plan.symbols_to_remove == frozenset()


def test_build_current_manifest_detects_deletion_via_planner():
    """A file dropped from the walk is seen as deleted by the planner."""
    prev_graph = _graph_with_file("src/a.py", "a\n", {"foo": "foo()"})
    prev_graph_b = _graph_with_file("src/b.py", "b\n", {"bar": "bar()"})
    for node in prev_graph_b.iter_nodes():
        prev_graph.add_node(node)
    previous = build_manifest(prev_graph)

    # b.py absent from the walk → deleted.
    walk = [FileEntry(path="src/a.py", content="a\n", language="python")]
    reparsed = KnowledgeGraph()  # nothing changed content-wise
    current = build_current_manifest(walk, reparsed)

    plan = plan_incremental(previous, current)

    assert plan.diff.deleted_files == frozenset({"src/b.py"})
    assert generate_id(NodeLabel.FUNCTION, "src/b.py", "bar") in plan.symbols_to_remove


def test_plan_is_incrementalplan_instance():
    """Guard the public return type."""
    curr = _manifest([_fm("src/a.py", "A1", {FILE_A: "s"})])
    assert isinstance(plan_incremental(None, curr), IncrementalPlan)


# ===========================================================================
# Added-file dependent closure (design's added-file-closure fix, W3.2e #5)
# ===========================================================================


def _filler(i: int, prefix: str = "src") -> FileManifest:
    """A symbol-bearing filler file, so a lone added-file symbol never dominates
    the symbol ratio and the closure logic (not the ratio gate) is what's tested."""
    path = f"{prefix}/p{i}.py"
    return _fm(path, f"p{i}v1", {generate_id(NodeLabel.FUNCTION, path, "fn"): "v"})


def test_added_file_pulls_in_unchanged_importer_via_unresolved_import():
    """An unchanged file whose recorded unresolved import matches a newly-ADDED
    file's importable identity is pulled into re-resolution."""
    imp_sigs = {generate_id(NodeLabel.FILE, "src/importer.py"): "s"}
    prev = _manifest(
        [_fm("src/importer.py", "I1", imp_sigs, unresolved_imports=["src.late"]),
         *[_filler(i) for i in range(9)]]
    )
    # late.py is added; importer.py is otherwise unchanged.
    late = _fm("src/late.py", "L1", {generate_id(NodeLabel.FILE, "src/late.py"): "s"})
    curr = _manifest(
        [_fm("src/importer.py", "I1", imp_sigs, unresolved_imports=["src.late"]),
         late, *[_filler(i) for i in range(9)]]
    )

    plan = plan_incremental(prev, curr)

    assert not plan.full_rebuild_required
    assert "src/late.py" in plan.diff.added_files
    assert plan.dependents == frozenset({"src/importer.py"})
    assert "src/importer.py" in plan.files_to_reparse


def test_added_file_does_not_pull_unrelated_unresolved_import():
    """A non-matching unresolved import (external gem/stdlib) is never pulled in."""
    imp_sigs = {generate_id(NodeLabel.FILE, "src/importer.py"): "s"}
    prev = _manifest(
        [_fm("src/importer.py", "I1", imp_sigs, unresolved_imports=["requests"]),
         *[_filler(i) for i in range(9)]]
    )
    late = _fm("src/late.py", "L1", {generate_id(NodeLabel.FILE, "src/late.py"): "s"})
    curr = _manifest(
        [_fm("src/importer.py", "I1", imp_sigs, unresolved_imports=["requests"]),
         late, *[_filler(i) for i in range(9)]]
    )

    plan = plan_incremental(prev, curr)

    assert plan.dependents == frozenset()  # "requests" != any form of "src/late"


def test_added_file_closure_matches_source_root_layout():
    """src/pkg/late.py added matches an unchanged file that imported `pkg.late`."""
    imp_sigs = {generate_id(NodeLabel.FILE, "app.py"): "s"}
    prev = _manifest(
        [_fm("app.py", "A1", imp_sigs, unresolved_imports=["pkg.late"]),
         *[_filler(i) for i in range(9)]]
    )
    late = _fm("src/pkg/late.py", "L1", {generate_id(NodeLabel.FILE, "src/pkg/late.py"): "s"})
    curr = _manifest(
        [_fm("app.py", "A1", imp_sigs, unresolved_imports=["pkg.late"]),
         late, *[_filler(i) for i in range(9)]]
    )

    plan = plan_incremental(prev, curr)
    # importable_identities("src/pkg/late.py") includes "pkg/late"/"pkg.late".
    assert plan.dependents == frozenset({"app.py"})


# ===========================================================================
# repair_inbound — the watcher re-resolves dependents of body-only files too
# ===========================================================================


def test_repair_inbound_pulls_dependents_of_body_only_files():
    """With repair_inbound, an unchanged importer of a *body-only*-edited file is
    re-resolved (to restore inbound edges the immediate tier DETACH-DELETEs)."""
    a = _fm(
        "src/a.py",
        "A1",
        {FILE_A: "s", FUNC_FOO: "f1"},
        out_edges=[EdgeRef(CALLS, FUNC_FOO, FUNC_BAR)],
    )
    prev = _manifest([a, _fm("src/b.py", "B1", {FILE_B: "s", FUNC_BAR: "b1"})])
    # b.py body-only (bar keeps its identity).
    curr = _manifest(
        [
            _fm("src/a.py", "A1", {FILE_A: "s", FUNC_FOO: "f1"},
                out_edges=[EdgeRef(CALLS, FUNC_FOO, FUNC_BAR)]),
            _fm("src/b.py", "B2", {FILE_B: "s", FUNC_BAR: "b1"}),
        ]
    )

    # Default (analyze): body-only ⇒ no closure.
    assert plan_incremental(prev, curr).dependents == frozenset()
    # repair_inbound (watcher): a.py is pulled in to repair its inbound edge.
    repaired = plan_incremental(prev, curr, repair_inbound=True)
    assert repaired.dependents == frozenset({"src/a.py"})
    assert "src/a.py" in repaired.files_to_reparse


# ===========================================================================
# should_consolidate — the §7/§9 consolidation gates (D5, D8, N, staleness)
# ===========================================================================


def _consol_manifest(
    *,
    total_symbols: int = 100,
    affected: int = 0,
    applies: int = 0,
    git_head: str | None = "head0",
    consolidated_at: str = "",
) -> Manifest:
    """A previous manifest carrying pending counters for the consolidation gate.

    ``total_symbols`` synthetic single-symbol files supply the symbol-ratio
    denominator (via the plan's total_symbol_count)."""
    files = [
        _fm(f"src/s{i}.py", f"s{i}", {generate_id(NodeLabel.FUNCTION, f"src/s{i}.py", "fn"): "v"})
        for i in range(total_symbols)
    ]
    idx = IndexManifest(
        manifest_version=CURRENT_MANIFEST_VERSION,
        git_head=git_head,
        consolidated_at=consolidated_at,
        pending=IndexPending(affected_symbols=affected, applies_since_consolidation=applies),
        full_fingerprint=compute_fingerprint({f.path: f.content_sha for f in files}),
    )
    return Manifest(index=idx, files={f.path: f for f in files})


def _incremental_plan(previous: Manifest, *, affected_this_burst: int = 0) -> IncrementalPlan:
    """An incremental plan against ``previous`` whose this-burst affected count is
    controllable (one identity-changed file with N single symbols)."""
    files = list(previous.files.values())
    # Perturb the first `affected_this_burst` files' single symbol identities.
    curr_files = []
    for i, fm in enumerate(files):
        sigs = dict(fm.symbol_sigs)
        if i < affected_this_burst:
            (sid,) = sigs
            curr_files.append(_fm(fm.path, f"{fm.content_sha}-v2", {sid: "v2"}))
        else:
            curr_files.append(_fm(fm.path, fm.content_sha, sigs))
    plan = plan_incremental(previous, _manifest(curr_files))
    assert not plan.full_rebuild_required
    return plan


def test_consolidate_when_git_head_moved():
    """D8: git HEAD moved ⇒ consolidate (coupling stale), highest precedence."""
    prev = _consol_manifest(git_head="head0")
    plan = _incremental_plan(prev, affected_this_burst=1)
    do, why = should_consolidate(prev, plan, git_head_moved=True)
    assert do and why == REASON_CONSOLIDATE_GIT_HEAD


def test_consolidate_when_symbol_ratio_crossed():
    """D5: cumulative affected-symbol ratio over the threshold ⇒ consolidate."""
    # 100 symbols; pending already 10 affected; this burst adds 5 → 15/100 = 0.15 > 0.12.
    prev = _consol_manifest(total_symbols=100, affected=10)
    plan = _incremental_plan(prev, affected_this_burst=5)
    do, why = should_consolidate(prev, plan, git_head_moved=False)
    assert do and why == REASON_CONSOLIDATE_SYMBOL_RATIO


def test_no_consolidate_under_symbol_ratio():
    """Just under the D5 threshold stays incremental."""
    # pending 5 + this burst 2 = 7/100 = 0.07 < 0.12.
    prev = _consol_manifest(total_symbols=100, affected=5)
    plan = _incremental_plan(prev, affected_this_burst=2)
    do, why = should_consolidate(prev, plan, git_head_moved=False)
    assert not do and why == ""


def test_consolidate_after_n_applies():
    """N applies since last consolidation ⇒ consolidate (design's N gate)."""
    prev = _consol_manifest(total_symbols=1000, applies=CONSOLIDATION_APPLY_LIMIT - 1)
    plan = _incremental_plan(prev, affected_this_burst=1)  # tiny — ratio gate quiet
    do, why = should_consolidate(prev, plan, git_head_moved=False)
    assert do and why == REASON_CONSOLIDATE_APPLY_LIMIT


def test_consolidate_after_staleness_ceiling():
    """Unconsolidated past the wall-clock ceiling ⇒ consolidate."""
    from datetime import datetime, timedelta, timezone

    stale = datetime.now(timezone.utc) - timedelta(seconds=CONSOLIDATION_MAX_SECONDS + 60)
    prev = _consol_manifest(total_symbols=1000, consolidated_at=stale.isoformat())
    plan = _incremental_plan(prev, affected_this_burst=1)
    now = datetime.now(timezone.utc).timestamp()
    do, why = should_consolidate(prev, plan, git_head_moved=False, now=now)
    assert do and why == REASON_CONSOLIDATE_STALENESS


def test_no_consolidate_when_all_gates_quiet():
    """Small change, HEAD unchanged, few applies, recently consolidated ⇒ delta."""
    from datetime import datetime, timezone

    fresh = datetime.now(timezone.utc).isoformat()
    prev = _consol_manifest(total_symbols=1000, affected=1, applies=1, consolidated_at=fresh)
    plan = _incremental_plan(prev, affected_this_burst=1)
    now = datetime.now(timezone.utc).timestamp()
    do, why = should_consolidate(prev, plan, git_head_moved=False, now=now)
    assert not do and why == ""


def test_consolidation_thresholds_match_design():
    assert CONSOLIDATION_SYMBOL_RATIO == 0.12
    assert CONSOLIDATION_APPLY_LIMIT == 50
    assert CONSOLIDATION_MAX_SECONDS == 600
