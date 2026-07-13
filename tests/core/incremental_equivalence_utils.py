"""Shared strict-core equivalence comparator for incremental-indexing tests.

Extracted from ``test_incremental_build`` (W3.2c) so the property-based
equivalence harness (W3.2f) and the delta-shape tests share **one** comparator —
the design's D9 strict-core exact-match set — instead of drifting copies. This
module is intentionally *not* ``test_``-prefixed, so pytest never collects it;
it is imported as ``tests.core.incremental_equivalence_utils``.

The strict core (must match a ``--full`` rebuild exactly, per design §11):

* every non-fringe node id and its parse/heritage-derived properties;
* ``CONTAINS`` / ``DEFINES`` / ``IMPORTS`` / ``EXTENDS`` / ``IMPLEMENTS`` /
  ``MIXES_IN`` / ``USES_TYPE`` edges;
* ``CALLS`` at confidence ``>= 0.8`` (same-file / import-resolved / receiver).

The D9 **bounded-stale fringe** (deliberately excluded from the strict set —
it converges only at a consolidation): ``COMMUNITY`` / ``PROCESS`` nodes and
their edges, ``is_dead`` / ``is_entry_point`` flags, ``CALLS`` at confidence
``<= 0.5`` (global-fuzzy) and ``0.3`` (weak-ref), and ``COUPLED_WITH``.

One fringe edge the *graph* comparator cannot see and so does not try to: a
**cross-file REST-link CALLS** (``rest_link=True``, confidence up to ``1.0``,
re-linked over the re-parsed set only → deferred for an unchanged counterpart,
D7). ``LadybugBackend.load_graph`` reconstructs a fixed relationship-property
schema (``confidence`` / coupling / process fields) and drops ``rest_link``, so a
persisted graph cannot be told apart from an import-resolved call. The harness
therefore keeps its REST pair *self-contained in one file* (always re-linked
together, never deferred), so no cross-file REST link ever enters the strict set;
the deferral itself is a documented convenience-edge fringe reconciled at
consolidation, out of scope for this strict-core comparator.

The comparator's provenance twin — the per-file manifest strict-provenance
diff — lives here too, since the harness asserts *both* the graph and the
persisted manifest against a fresh full build after every edit step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, RelType
from synaptiq.core.ingestion.manifest import Manifest

# ---------------------------------------------------------------------------
# The strict-core partition (design §11 / D9)
# ---------------------------------------------------------------------------

#: Genuinely-global / deferred node artifacts — the D9 bounded-stale fringe.
FRINGE_NODE_LABELS = frozenset({NodeLabel.COMMUNITY, NodeLabel.PROCESS})

#: High-confidence structural edges that must match a full rebuild exactly.
STRICT_EDGE_TYPES = (
    RelType.CONTAINS,
    RelType.DEFINES,
    RelType.IMPORTS,
    RelType.EXTENDS,
    RelType.IMPLEMENTS,
    RelType.MIXES_IN,
    RelType.USES_TYPE,
)

#: CALLS at or above this confidence are strict; below it they are fringe.
STRICT_CALLS_MIN_CONF = 0.8

#: Structural resolver edge kinds — the always-reproducible core of a file's
#: outbound provenance. CALLS are deliberately excluded from the manifest
#: comparison and left to the graph comparator, which owns call correctness
#: (strict ``CALLS >= 0.8``): a carried file's *fuzzy* (< 0.8) CALLS out-edges are
#: bounded-stale (not chased across dependents, §5.2) and can legitimately differ
#: from a fresh build's, and the manifest's ``EdgeRef`` — ``(rel_type, src, tgt,
#: confidence)`` — carries no marker to tell a deferred REST-link call from an
#: import-resolved one. The manifest check therefore owns structural provenance
#: (imports/heritage/types) + symbol identity + content hashes.
_STRUCTURAL_RESOLVER_RELS = frozenset(
    {
        RelType.IMPORTS.value,
        RelType.EXTENDS.value,
        RelType.IMPLEMENTS.value,
        RelType.MIXES_IN.value,
        RelType.USES_TYPE.value,
    }
)


def strict_node_props(node: GraphNode) -> tuple:
    """Parse/heritage-derived, delta-strict node fields (design §11).

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


def edge_set(
    graph: KnowledgeGraph, rel_type: RelType, min_conf: float | None = None
) -> set[tuple[str, str]]:
    """The ``(source, target)`` pairs of *rel_type* edges (optionally conf-gated)."""
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


# ---------------------------------------------------------------------------
# Structured strict-core diff (failure ergonomics — design §11 harness)
# ---------------------------------------------------------------------------

# Cap emitted lists so a CI failure dump stays legible even for a big diff.
_MAX_ITEMS = 12


def _cap(items) -> list:
    ordered = sorted(items, key=repr)
    if len(ordered) <= _MAX_ITEMS:
        return ordered
    return ordered[:_MAX_ITEMS] + [f"... (+{len(ordered) - _MAX_ITEMS} more)"]


@dataclass
class StrictCoreDiff:
    """The minimal, human-readable difference between two graphs' strict cores.

    Empty (:meth:`is_empty`) iff the two graphs are strict-core-equivalent.
    """

    only_incremental_nodes: list = field(default_factory=list)
    only_full_nodes: list = field(default_factory=list)
    node_prop_mismatches: list = field(default_factory=list)
    edge_diffs: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (
            self.only_incremental_nodes
            or self.only_full_nodes
            or self.node_prop_mismatches
            or self.edge_diffs
        )

    def render(self) -> str:
        if self.is_empty():
            return "strict cores are equivalent"
        parts: list[str] = []
        if self.only_incremental_nodes:
            parts.append(f"only-incremental nodes: {_cap(self.only_incremental_nodes)}")
        if self.only_full_nodes:
            parts.append(f"only-full nodes: {_cap(self.only_full_nodes)}")
        for mm in self.node_prop_mismatches[:_MAX_ITEMS]:
            parts.append(
                f"node {mm['id']} field {mm['field']!r}: "
                f"incremental={mm['incremental']!r} full={mm['full']!r}"
            )
        for rel_type, diff in self.edge_diffs.items():
            if diff.get("only_incremental"):
                parts.append(f"{rel_type} only-incremental: {_cap(diff['only_incremental'])}")
            if diff.get("only_full"):
                parts.append(f"{rel_type} only-full: {_cap(diff['only_full'])}")
        return "\n  ".join(parts)


# Node property field names, aligned position-for-position with strict_node_props,
# so a mismatch can be reported by the field that actually diverged.
_NODE_PROP_FIELDS = (
    "label",
    "name",
    "file_path",
    "class_name",
    "signature",
    "start_line",
    "end_line",
    "content",
    "language",
    "is_exported",
    "properties",
)


def strict_core_diff(incremental: KnowledgeGraph, full: KnowledgeGraph) -> StrictCoreDiff:
    """Compute the strict-core difference (empty ⇒ equivalent)."""
    inc_nodes = {n.id: n for n in incremental.iter_nodes() if n.label not in FRINGE_NODE_LABELS}
    full_nodes = {n.id: n for n in full.iter_nodes() if n.label not in FRINGE_NODE_LABELS}

    diff = StrictCoreDiff(
        only_incremental_nodes=list(set(inc_nodes) - set(full_nodes)),
        only_full_nodes=list(set(full_nodes) - set(inc_nodes)),
    )
    for nid in set(inc_nodes) & set(full_nodes):
        inc_props = strict_node_props(inc_nodes[nid])
        full_props = strict_node_props(full_nodes[nid])
        if inc_props != full_props:
            for field_name, iv, fv in zip(_NODE_PROP_FIELDS, inc_props, full_props):
                if iv != fv:
                    diff.node_prop_mismatches.append(
                        {"id": nid, "field": field_name, "incremental": iv, "full": fv}
                    )

    for rel_type in STRICT_EDGE_TYPES:
        inc_edges = edge_set(incremental, rel_type)
        full_edges = edge_set(full, rel_type)
        if inc_edges != full_edges:
            diff.edge_diffs[rel_type.value] = {
                "only_incremental": list(inc_edges - full_edges),
                "only_full": list(full_edges - inc_edges),
            }

    inc_calls = edge_set(incremental, RelType.CALLS, min_conf=STRICT_CALLS_MIN_CONF)
    full_calls = edge_set(full, RelType.CALLS, min_conf=STRICT_CALLS_MIN_CONF)
    if inc_calls != full_calls:
        diff.edge_diffs[f"{RelType.CALLS.value}>={STRICT_CALLS_MIN_CONF}"] = {
            "only_incremental": list(inc_calls - full_calls),
            "only_full": list(full_calls - inc_calls),
        }

    return diff


def assert_strict_equivalence(
    incremental: KnowledgeGraph, full: KnowledgeGraph, *, context: str = ""
) -> None:
    """Assert the delta-applied graph == a full rebuild on the strict core."""
    diff = strict_core_diff(incremental, full)
    if not diff.is_empty():
        prefix = f"{context}\n  " if context else ""
        raise AssertionError(f"{prefix}strict-core equivalence FAILED:\n  {diff.render()}")


# ---------------------------------------------------------------------------
# Fringe / consistency invariants (design §1, §6 — "documented directions")
# ---------------------------------------------------------------------------


def dangling_edges(graph: KnowledgeGraph) -> list[tuple[str, str, str]]:
    """Every edge whose source or target node is absent (a real corruption).

    A phantom/dangling edge is the exact bug class the surgical delta guards
    against (inbound-edge loss on a survivor, a stale community pointing at a
    removed symbol). Holds for *all* edge kinds — strict core and fringe alike.
    """
    ids = {n.id for n in graph.iter_nodes()}
    dangling: list[tuple[str, str, str]] = []
    for rel in graph.iter_relationships():
        if rel.source not in ids or rel.target not in ids:
            dangling.append((rel.type.value, rel.source, rel.target))
    return dangling


def member_of_present(graph: KnowledgeGraph) -> bool:
    """Whether the graph carries any ``MEMBER_OF`` (community) edge."""
    return any(rel.type is RelType.MEMBER_OF for rel in graph.iter_relationships())


# ---------------------------------------------------------------------------
# Manifest strict-provenance comparison (the W3.2e carry-forward regression)
# ---------------------------------------------------------------------------


def _strict_out_edges(fm) -> frozenset[tuple[str, str, str]]:
    """A file's outbound *structural* resolver edges (imports/heritage/types).

    The depth-1 dependent closure keeps these exact for carried files, so they
    must match a fresh full build's provenance. CALLS are excluded (see
    :data:`_STRUCTURAL_RESOLVER_RELS`): the fuzzy sub-bucket is bounded-stale and
    a deferred REST-link CALLS cannot be told apart from an import-resolved one in
    the manifest's ``EdgeRef``, so call correctness is left to the graph comparator.
    """
    return frozenset(
        (e.rel_type, e.src, e.tgt) for e in fm.out_edges if e.rel_type in _STRUCTURAL_RESOLVER_RELS
    )


def _file_provenance_key(fm) -> tuple:
    """The strict, always-reproducible provenance of one file manifest row."""
    return (
        fm.content_sha,
        fm.language,
        tuple(sorted(fm.symbol_ids)),
        tuple(sorted(fm.symbol_sigs.items())),
        tuple(sorted(fm.unresolved_imports)),
        _strict_out_edges(fm),
    )


@dataclass
class ManifestDiff:
    """The strict per-file provenance difference between two manifests."""

    only_persisted_files: list = field(default_factory=list)
    only_fresh_files: list = field(default_factory=list)
    file_mismatches: list = field(default_factory=list)
    fingerprint_mismatch: tuple | None = None

    def is_empty(self) -> bool:
        return not (
            self.only_persisted_files
            or self.only_fresh_files
            or self.file_mismatches
            or self.fingerprint_mismatch
        )

    def render(self) -> str:
        parts: list[str] = []
        if self.fingerprint_mismatch:
            parts.append(
                f"full_fingerprint: persisted={self.fingerprint_mismatch[0]} "
                f"fresh={self.fingerprint_mismatch[1]}"
            )
        if self.only_persisted_files:
            parts.append(f"only-persisted files: {_cap(self.only_persisted_files)}")
        if self.only_fresh_files:
            parts.append(f"only-fresh files: {_cap(self.only_fresh_files)}")
        for mm in self.file_mismatches[:_MAX_ITEMS]:
            parts.append(f"file {mm['path']} provenance diverged: {mm['detail']}")
        return "\n  ".join(parts)


def _explain_file_mismatch(path: str, persisted, fresh) -> dict:
    """A compact reason a single file's provenance diverged (for the dump)."""
    if set(persisted.symbol_ids) != set(fresh.symbol_ids):
        p, f = set(persisted.symbol_ids), set(fresh.symbol_ids)
        return {
            "path": path,
            "detail": f"symbol_ids only-persisted={_cap(p - f)} only-fresh={_cap(f - p)}",
        }
    if persisted.symbol_sigs != fresh.symbol_sigs:
        return {"path": path, "detail": "symbol_sigs differ (identity fingerprint mismatch)"}
    if _strict_out_edges(persisted) != _strict_out_edges(fresh):
        p, f = _strict_out_edges(persisted), _strict_out_edges(fresh)
        return {
            "path": path,
            "detail": f"strict out_edges only-persisted={_cap(p - f)} only-fresh={_cap(f - p)}",
        }
    if persisted.content_sha != fresh.content_sha:
        return {"path": path, "detail": "content_sha differ"}
    if sorted(persisted.unresolved_imports) != sorted(fresh.unresolved_imports):
        return {"path": path, "detail": "unresolved_imports differ"}
    return {"path": path, "detail": "language differ"}


def manifest_provenance_diff(persisted: Manifest, fresh: Manifest) -> ManifestDiff:
    """Diff a persisted manifest's per-file provenance vs a fresh full build.

    "Provenance never rots" (the W3.2e carry-forward regression class): after any
    incremental apply the stored manifest's per-file symbol ids / identity
    fingerprints / content hashes / unresolved-imports / strict out-edges must
    equal a from-scratch :func:`build_manifest` of the same tree. Index-level
    accumulators (``pending`` / ``consolidated_at`` / ``git_head`` /
    ``tool_version``) legitimately differ between an incremental carry-forward and
    a fresh build and are *not* compared here; the ``full_fingerprint`` (a pure
    function of content hashes) is.
    """
    diff = ManifestDiff(
        only_persisted_files=list(set(persisted.files) - set(fresh.files)),
        only_fresh_files=list(set(fresh.files) - set(persisted.files)),
    )
    for path in set(persisted.files) & set(fresh.files):
        pf, ff = persisted.files[path], fresh.files[path]
        if _file_provenance_key(pf) != _file_provenance_key(ff):
            diff.file_mismatches.append(_explain_file_mismatch(path, pf, ff))
    if persisted.index.full_fingerprint != fresh.index.full_fingerprint:
        diff.fingerprint_mismatch = (
            persisted.index.full_fingerprint,
            fresh.index.full_fingerprint,
        )
    return diff


def assert_manifest_provenance(persisted: Manifest, fresh: Manifest, *, context: str = "") -> None:
    """Assert the persisted manifest's provenance == a fresh full build's."""
    diff = manifest_provenance_diff(persisted, fresh)
    if not diff.is_empty():
        prefix = f"{context}\n  " if context else ""
        raise AssertionError(
            f"{prefix}manifest provenance ROTTED (carry-forward regression):\n  {diff.render()}"
        )
