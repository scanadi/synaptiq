"""Incremental scope planner (W3.2b).

The scope-planning layer between a manifest diff (W3.2a) and the scoped
resolution (W3.2c). Given the previous
:class:`~synaptiq.core.ingestion.manifest.Manifest` and a freshly-built
*current* manifest, it decides — **purely**, with no storage access and no
resolution — exactly which files must be re-parsed / re-resolved, which
unchanged files can be carried untouched, which stale symbol ids must be
removed, and whether the change is large enough that a full rebuild is cheaper
and simpler.

Design: ``docs/plans/2026-07-12-incremental-indexing-design.md`` §5 (scoping
rules / the two-question closure), §6.3 (ratio fallback, D4), §8 (fallback
triggers). This module owns *the intellectual core* of the leaner incremental
design: **the depth-1 dependent closure**.

Key economies, straight from the design:

* **Body-only edits get ZERO closure** (§5.2 Q2). If a file's *identity set*
  (added ∪ removed ∪ identity-changed symbols) is empty, every inbound edge
  still points at a same-identity symbol, so nothing that depends on it needs
  re-resolution. This is the design's headline win.
* **Depth-1, not transitive** (§5.2). A dependent's *outbound* edge set may
  change but its *symbol identities* do not, so the closure never propagates a
  second hop for structural edges.
* **Dependents come from the previous manifest's stored ``out_edges``**
  (§5.4-5.5) — a reverse resolver-edge lookup, never a live graph query and
  never a ``load_graph``. That is *why* the manifest records ``out_edges``.

The planner never touches storage or the parser: :func:`plan_incremental` is a
pure function of two manifests. W3.2e wires ``storage.read_manifest()`` →
``previous`` and the walk+reparse → ``current`` (via
:func:`build_current_manifest`) and dispatches on
:attr:`IncrementalPlan.full_rebuild_required`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.ingestion.manifest import (
    CURRENT_MANIFEST_VERSION,
    RESOLVER_REL_TYPES,
    FileManifest,
    IndexManifest,
    IndexPending,
    Manifest,
    ManifestDiff,
    build_manifest,
    compute_fingerprint,
    content_sha,
    diff_manifests,
)
from synaptiq.core.ingestion.walker import FileEntry

__all__ = [
    "FILE_RATIO_THRESHOLD",
    "SYMBOL_RATIO_THRESHOLD",
    "REASON_NO_MANIFEST",
    "REASON_VERSION_MISMATCH",
    "REASON_CORRUPT_MANIFEST",
    "REASON_FILE_RATIO",
    "REASON_SYMBOL_RATIO",
    "REASON_INCREMENTAL",
    "IncrementalPlan",
    "plan_incremental",
    "build_current_manifest",
]

# ---------------------------------------------------------------------------
# Full-rebuild ratio thresholds (design §6.3 / decision D4)
# ---------------------------------------------------------------------------
#
# When too much of the repo changes at once (branch switch, ``git pull``,
# formatter run) the scoped delta loses to the well-tested full ``bulk_load``,
# which is a bulk COPY + one FTS pass. Past the knee, fall back to full. The
# figures mirror the plan's suggestion and MUST be measured in W3.2; until then
# they are conservative defaults that keep large change-sets on the full path.
# Comparison is strict ``>`` (exactly at the threshold stays incremental).

#: Fraction of files that may change before a full rebuild is preferred.
FILE_RATIO_THRESHOLD = 0.30

#: Fraction of symbols that may be *structurally* affected before full rebuild.
SYMBOL_RATIO_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Full-rebuild reason codes (surfaced for diagnostics / status / tests)
# ---------------------------------------------------------------------------

#: No trustworthy previous manifest — first index or a pre-manifest install
#: (§8 trigger 1). ``storage.read_manifest()`` returns ``None`` here.
REASON_NO_MANIFEST = "no_manifest"
#: Stored ``manifest_version`` differs from the running code (§8 trigger 2).
REASON_VERSION_MISMATCH = "version_mismatch"
#: Previous manifest's ``full_fingerprint`` disagrees with its own file rows —
#: a corrupt / partial manifest (§8 triggers 3/5, planner-checkable subset).
REASON_CORRUPT_MANIFEST = "corrupt_manifest"
#: Changed-file ratio crossed :data:`FILE_RATIO_THRESHOLD` (§6.3).
REASON_FILE_RATIO = "file_ratio_exceeded"
#: Affected-symbol ratio crossed :data:`SYMBOL_RATIO_THRESHOLD` (§6.3).
REASON_SYMBOL_RATIO = "symbol_ratio_exceeded"
#: Change is small enough to apply as a scoped delta — the happy path.
REASON_INCREMENTAL = "incremental"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class IncrementalPlan:
    """The scoped work plan produced from a manifest diff (design §5, §6.3, §8).

    Consumed by W3.2c (scoped resolution → ``GraphDelta``) and W3.2e (dispatch /
    consolidation). When :attr:`full_rebuild_required` is ``True`` the scoped
    fields are advisory only — the caller runs the full ``bulk_load`` path — but
    they are still populated (except when there is no previous manifest) so
    ``synaptiq status`` / diagnostics can explain the decision.

    Set relationships (always hold, and are asserted by the property tests):

    * ``files_to_reparse == (changed ∪ added_files ∪ dependents)``
    * ``files_unchanged_to_carry == (unchanged_files − dependents)``
    * ``dependents ⊆ unchanged_files`` (pure closure — a *changed* file is
      already reparsed for its own edit, so it is never also a "dependent")
    * ``files_to_reparse ∩ files_unchanged_to_carry == ∅`` — the reparse set
      never contains an unchanged non-dependent.
    """

    #: Hand off to the full ``bulk_load`` path instead of applying a delta.
    full_rebuild_required: bool
    #: One of the ``REASON_*`` codes — why full (or ``REASON_INCREMENTAL``).
    reason: str

    #: Files whose outbound edges must be recomputed: every changed file
    #: (body-only *and* identity-changed — Q1), every added file, and every
    #: depth-1 dependent (Q2). These are the files a re-parse must cover.
    files_to_reparse: frozenset[str] = field(default_factory=frozenset)
    #: Unchanged files with no dependency on a changed symbol — carried as-is,
    #: their nodes and inbound edges left untouched.
    files_unchanged_to_carry: frozenset[str] = field(default_factory=frozenset)
    #: The depth-1 dependent closure: *unchanged* files pulled into re-resolution
    #: because a resolver edge of theirs points at a symbol of an
    #: identity-changed or deleted file (reverse lookup over the previous
    #: manifest's ``out_edges``). Empty whenever no file's identity set changed.
    dependents: frozenset[str] = field(default_factory=frozenset)
    #: Stale symbol ids to delete: every *removed* symbol of a changed file
    #: (renames leave the old id here) plus every symbol of a deleted file.
    #: Identity-changed (same-id) and body-only symbols are re-upserted, not
    #: removed, so they are deliberately absent.
    symbols_to_remove: frozenset[str] = field(default_factory=frozenset)

    #: Files touched (added ∪ deleted ∪ content-changed) — the ratio numerator.
    changed_file_count: int = 0
    #: Distinct files across previous ∪ current — the file-ratio denominator.
    total_file_count: int = 0
    #: Distinct *structurally* affected symbols (added ∪ removed ∪
    #: identity-changed across changed files, plus all symbols of added/deleted
    #: files). Body-only symbols are excluded — they are the cheap case the
    #: whole incremental path optimizes for, and the file ratio already catches
    #: widespread reformats.
    affected_symbol_count: int = 0
    #: Distinct symbols in the previous manifest — the symbol-ratio denominator.
    total_symbol_count: int = 0

    #: The underlying whole-manifest diff (per-file symbol classifications), so
    #: W3.2c need not recompute it. ``None`` only when there is no previous
    #: manifest.
    diff: ManifestDiff | None = None

    @property
    def file_change_ratio(self) -> float:
        """``changed_file_count / total_file_count`` (0.0 if the repo is empty)."""
        if self.total_file_count == 0:
            return 0.0
        return self.changed_file_count / self.total_file_count

    @property
    def symbol_change_ratio(self) -> float:
        """``affected_symbol_count / total_symbol_count`` (0.0 if none)."""
        if self.total_symbol_count == 0:
            return 0.0
        return self.affected_symbol_count / self.total_symbol_count


# ---------------------------------------------------------------------------
# Planner (pure)
# ---------------------------------------------------------------------------


def _previous_manifest_is_consistent(previous: Manifest) -> bool:
    """Does the previous manifest's stored fingerprint match its own file rows?

    Defense-in-depth for §8 (corrupt / partial manifest): a manifest whose
    ``full_fingerprint`` disagrees with the fingerprint recomputed over its own
    ``content_sha`` set is untrustworthy, so we fall back to a full rebuild
    rather than diff against it. An empty fingerprint means "not stamped"
    (e.g. a hand-built manifest) and is treated as consistent — there is nothing
    to disagree with. ``read_manifest`` handles the read/parse-error case
    upstream (returns ``None``); this catches a manifest that parsed cleanly but
    is internally inconsistent.
    """
    fingerprint = previous.index.full_fingerprint
    if not fingerprint:
        return True
    recomputed = compute_fingerprint({p: f.content_sha for p, f in previous.files.items()})
    return recomputed == fingerprint


def plan_incremental(
    previous: Manifest | None,
    current: Manifest,
    *,
    file_ratio_threshold: float = FILE_RATIO_THRESHOLD,
    symbol_ratio_threshold: float = SYMBOL_RATIO_THRESHOLD,
) -> IncrementalPlan:
    """Plan the scoped incremental work from ``previous`` → ``current`` manifests.

    Pure and deterministic — no I/O, no storage, no parsing beyond the two
    manifests handed in. ``current`` must describe *every* current file (so
    deletions are detected) with an up-to-date ``content_sha``, and carry fresh
    ``symbol_sigs`` for every file whose content changed (so body-only vs
    identity-changed classification is exact); :func:`build_current_manifest`
    produces exactly such a manifest from a walk + a reparse.

    Returns an :class:`IncrementalPlan`. When ``previous`` is ``None`` the plan
    is an empty full-rebuild verdict (:data:`REASON_NO_MANIFEST`); otherwise the
    full diff and scoped sets are always computed, and
    :attr:`~IncrementalPlan.full_rebuild_required` reflects the version /
    corruption / ratio gates (design §8, §6.3).
    """
    if previous is None:
        return IncrementalPlan(
            full_rebuild_required=True,
            reason=REASON_NO_MANIFEST,
        )

    diff = diff_manifests(previous, current)

    # --- ratio numerators / denominators -----------------------------------
    total_file_count = (
        len(diff.added_files)
        + len(diff.deleted_files)
        + len(diff.changed)
        + len(diff.unchanged_files)
    )
    changed_file_count = len(diff.added_files) + len(diff.deleted_files) + len(diff.changed)

    total_symbol_ids: set[str] = set()
    for fm in previous.files.values():
        total_symbol_ids.update(fm.symbol_ids)
    total_symbol_count = len(total_symbol_ids)

    affected_ids: set[str] = set()
    for fd in diff.changed.values():
        affected_ids |= fd.added | fd.removed | fd.identity_changed
    for path in diff.deleted_files:
        affected_ids.update(previous.files[path].symbol_ids)
    for path in diff.added_files:
        fm = current.files.get(path)
        if fm is not None:
            affected_ids.update(fm.symbol_ids)
    affected_symbol_count = len(affected_ids)

    # --- depth-1 dependent closure (the design's core; §5.2, §5.4-5.5) ------
    # Only identity-changed and deleted files project a "changed symbol" set;
    # body-only edits contribute nothing, so they pull in no dependents.
    affected_targets: set[str] = set()
    for path in diff.identity_changed_files:
        affected_targets.update(previous.files[path].symbol_ids)
    for path in diff.deleted_files:
        affected_targets.update(previous.files[path].symbol_ids)

    dependents: set[str] = set()
    if affected_targets:
        # Reverse resolver-edge lookup over the PREVIOUS manifest's out_edges:
        # an unchanged file whose imports/calls/heritage/types edge targets one
        # of the affected symbols must be re-resolved. Restricting to unchanged
        # files keeps `dependents` the *pure* closure — changed/added files are
        # already reparsed for their own edits, and deleted files are gone.
        for path, fm in previous.files.items():
            if path not in diff.unchanged_files:
                continue
            for edge in fm.out_edges:
                if edge.rel_type in RESOLVER_REL_TYPES and edge.tgt in affected_targets:
                    dependents.add(path)
                    break

    files_to_reparse = set(diff.changed) | set(diff.added_files) | dependents
    files_unchanged_to_carry = set(diff.unchanged_files) - dependents

    # --- stale symbol ids to remove ----------------------------------------
    symbols_to_remove: set[str] = set()
    for fd in diff.changed.values():
        symbols_to_remove |= fd.removed
    for path in diff.deleted_files:
        symbols_to_remove.update(previous.files[path].symbol_ids)

    # --- full-rebuild verdict (§8 triggers 1/2/5 + §6.3 ratios) -------------
    file_ratio = changed_file_count / total_file_count if total_file_count else 0.0
    symbol_ratio = affected_symbol_count / total_symbol_count if total_symbol_count else 0.0

    full_rebuild = False
    reason = REASON_INCREMENTAL
    if previous.index.manifest_version != CURRENT_MANIFEST_VERSION:
        full_rebuild, reason = True, REASON_VERSION_MISMATCH
    elif not _previous_manifest_is_consistent(previous):
        full_rebuild, reason = True, REASON_CORRUPT_MANIFEST
    elif file_ratio > file_ratio_threshold:
        full_rebuild, reason = True, REASON_FILE_RATIO
    elif symbol_ratio > symbol_ratio_threshold:
        full_rebuild, reason = True, REASON_SYMBOL_RATIO

    return IncrementalPlan(
        full_rebuild_required=full_rebuild,
        reason=reason,
        files_to_reparse=frozenset(files_to_reparse),
        files_unchanged_to_carry=frozenset(files_unchanged_to_carry),
        dependents=frozenset(dependents),
        symbols_to_remove=frozenset(symbols_to_remove),
        changed_file_count=changed_file_count,
        total_file_count=total_file_count,
        affected_symbol_count=affected_symbol_count,
        total_symbol_count=total_symbol_count,
        diff=diff,
    )


# ---------------------------------------------------------------------------
# Current-manifest assembly from a walk + reparse (the walk-results consumer)
# ---------------------------------------------------------------------------


def build_current_manifest(
    walk: list[FileEntry],
    reparsed: KnowledgeGraph,
    *,
    tool_version: str = "",
    git_head: str | None = None,
) -> Manifest:
    """Assemble the ``current`` :class:`Manifest` from a walk + a partial reparse.

    The bridge from the pipeline's raw inputs to the planner: ``walk`` supplies
    the authoritative ``content_sha`` for *every* current file (so
    :func:`plan_incremental` can detect added / deleted / unchanged files), and
    ``reparsed`` — the phase 2-7 graph of only the files whose content changed
    (and any added files) — supplies fresh ``symbol_ids`` / ``symbol_sigs`` /
    ``out_edges`` for those files.

    Caller contract (honored by W3.2e): ``reparsed`` MUST cover every file whose
    content changed and every added file. A content-changed file missing from
    ``reparsed`` would carry empty ``symbol_sigs`` and its symbols would look
    spuriously removed. Unchanged files are intentionally *not* in ``reparsed``;
    they carry ``content_sha`` only, which is all the diff's content-hash
    short-circuit reads for them.

    Pure — reuses :func:`~synaptiq.core.ingestion.manifest.build_manifest` for
    the reparsed graph's symbol/edge provenance and
    :func:`~synaptiq.core.ingestion.manifest.content_sha` for the per-file hash.
    """
    reparsed_manifest = build_manifest(reparsed, tool_version=tool_version, git_head=git_head)

    files: dict[str, FileManifest] = {}
    for entry in walk:
        sha = content_sha(entry.content)
        reparsed_fm = reparsed_manifest.files.get(entry.path)
        if reparsed_fm is not None:
            # Changed / added file: keep the reparse's symbol + edge provenance,
            # but pin content_sha to the walk (the exact bytes just read).
            reparsed_fm.content_sha = sha
            if not reparsed_fm.language:
                reparsed_fm.language = entry.language
            files[entry.path] = reparsed_fm
        else:
            # Unchanged file: content hash only — symbol_sigs are never read for
            # it (the diff short-circuits on the matching content_sha).
            files[entry.path] = FileManifest(
                path=entry.path, content_sha=sha, language=entry.language
            )

    index = IndexManifest(
        manifest_version=CURRENT_MANIFEST_VERSION,
        tool_version=tool_version,
        git_head=git_head,
        full_fingerprint=compute_fingerprint({p: f.content_sha for p, f in files.items()}),
        consolidated_at="",
        pending=IndexPending(),
    )
    return Manifest(index=index, files=files)
