"""Scoped incremental resolution → GraphDelta assembly (W3.2c).

The bridge between the pure scope *planner* (W3.2b,
:func:`~synaptiq.core.ingestion.incremental.plan_incremental`) and the storage
*apply* (W3.2d, :meth:`StorageBackend.apply_graph_delta`).
:func:`build_incremental_delta` takes an :class:`IncrementalPlan` (that did *not*
demand a full rebuild), the previous :class:`Manifest`, and the fresh walk, and
returns ``(GraphDelta, new Manifest)`` — a surgical change set the caller applies
in one transaction plus the manifest to persist alongside it. It writes nothing;
W3.2e wires it to storage.

Design: ``docs/plans/2026-07-12-incremental-indexing-design.md`` §5.5 (producing
the ``GraphDelta``), §6 (delta assembly / apply order), §9 (global-phase policy),
D7 (rest_linking re-run over the scoped set), D9 (strict core + bounded-stale
fringe).

The mechanism (design §5.5) — how scoped resolution stays *global-correct*
-----------------------------------------------------------------------------
The resolvers (``process_imports`` / ``process_calls`` / ``process_heritage`` /
``process_types``) resolve a call/import/heritage/type reference against a name
and file index built from *the whole graph*. Re-running them over a mini-graph
that held only the re-parsed files would silently drop every edge whose target
lives in an unchanged file. So the mini-graph is **seeded with the full repo's
symbol set** before resolution runs:

* Re-parsed files (``plan.files_to_reparse``) contribute **real** nodes with
  full content/line data — produced by the same ``process_structure`` +
  ``process_parsing`` machinery a full build uses, so their symbols, DEFINES and
  CONTAINS edges are byte-for-byte what a full rebuild would emit.
* Every other file (``plan.files_unchanged_to_carry``) contributes **lightweight
  stub** nodes reconstructed from the previous manifest — one per recorded
  symbol id, carrying only the four fields resolution ever reads off a *target*
  (``name``, ``file_path``, ``label``, ``class_name``). Stubs carry no
  content/signature/line data (never read for a target) and never enter the
  delta; they exist only so the resolvers' global indexes see the whole repo.
  ``name``/``class_name`` are derived from the contractual node-id format
  ``{label}:{path}:{name}`` (CLAUDE.md) — the manifest stores ids + identity
  *hashes*, not the bare name, so the id is the only faithful source (see
  :func:`_reconstruct_stub`).

This is the design's "resolve against a GLOBAL name/file index rebuilt from the
manifest" (§5.4-5.5) made concrete: the index is *the seeded mini-graph itself*,
so the resolvers run **unmodified** — the least-divergent, highest-fidelity way
to keep the re-resolved edges identical to a full rebuild. Only whole-repo
``content`` reads are avoided (§5.4 "no ``load_graph``"): a stub is the id string
plus three cheap fields, so seeding is the O(symbols) in-memory index build the
design explicitly accepts as the one repo-size term on the hot path (§5.3).

Because ``parse_data`` covers only the re-parsed files, every edge the resolvers
emit *originates from a re-parsed file* (its target may be a stub). So the fresh
edge set is exactly "the outbound edges of the re-parsed ∪ dependent files" — Q1
+ Q2 of the scoping rule (§5.2) — and the previous manifest's ``out_edges`` for
those same files is exactly what must be deleted first (``edges_remove``).

Determinism & the bounded-stale fringe (design D9, §9, §11)
-----------------------------------------------------------
The delta is **strict-equal to a full rebuild on the high-confidence structural
core** and **bounded-stale on the uncertain / genuinely-global fringe**, the
staleness closed by the watcher's debounced consolidation (W3.2e owns that — this
function never touches it):

* **Strict-equal (this function produces exactly a full rebuild's):** all
  re-parsed nodes and their parse-derived properties; CONTAINS; DEFINES; IMPORTS;
  EXTENDS / IMPLEMENTS / MIXES_IN; USES_TYPE; CALLS at confidence ≥ 0.8
  (same-file / import-resolved / receiver) — for the outbound edges of the
  re-parsed ∪ dependent files. Inbound edges from *unchanged* files onto
  surviving symbols are left in place (never DETACH-DELETEd), so they stay
  correct too.
* **Bounded-stale (may lag between consolidations):**
  - **communities** (``MEMBER_OF`` / Community nodes) and **processes**
    (``is_entry_point`` / ``STEP_IN_PROCESS`` / Process nodes) — genuinely
    global and *discontinuous*; deferred (§9, D5). Re-parsed nodes therefore
    carry ``is_entry_point = False`` and gain no ``MEMBER_OF`` here.
  - **FTS / HNSW vectors** — no incremental API on the engine; deferred (§7).
  - **low-confidence fuzzy CALLS** (0.5 global-fuzzy, 0.3 weak-ref) *into
    unchanged files* — best-effort in the delta, reconciled at consolidation
    (§5.2, §11). Fuzzy edges *from* a re-parsed file are still refreshed.
  - **rest_linking** — re-run over the re-parsed set only (D7): a re-parsed
    caller links to a re-parsed endpoint, but a re-parsed caller → *unchanged*
    endpoint (or vice-versa) is deferred. REST edges are a convenience edge, not
    structural (§9).
  - **coupling** (``COUPLED_WITH``) — depends only on git history, untouched
    here; recomputed at consolidation iff ``git HEAD`` moved (§9, D8).
  - **``is_dead``** — recomputed *by the storage apply* (W3.2d
    ``apply_graph_delta``) over :attr:`GraphDelta.dead_recount` using the
    *local in-degree* predicate (zero incoming CALLS + not exempt). That is
    exact for the common case but does **not** re-run the global false-positive
    passes (override / protocol / alive-class / inner-function / Ruby-macro) that
    ``process_dead_code`` layers on top, nor does it see the deferred
    ``is_entry_point``; deadness that hinges on those converges at consolidation.

An added file that *unchanged* files import is out of this function's hands: the
plan decides the re-resolve set, and the current planner scopes dependents to the
symbols of identity-changed / deleted files (not added ones), so an unchanged
importer of a brand-new module is reconciled at consolidation. This function
faithfully resolves whatever ``plan.files_to_reparse`` names and no more.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphNode,
    NodeLabel,
    RelType,
    generate_id,
)
from synaptiq.core.ingestion.calls import process_calls
from synaptiq.core.ingestion.heritage import process_heritage
from synaptiq.core.ingestion.imports import process_imports
from synaptiq.core.ingestion.incremental import IncrementalPlan, build_current_manifest
from synaptiq.core.ingestion.manifest import Manifest
from synaptiq.core.ingestion.parser_phase import process_parsing
from synaptiq.core.ingestion.rest_linking import process_rest_linking
from synaptiq.core.ingestion.structure import process_structure
from synaptiq.core.ingestion.types import process_types
from synaptiq.core.ingestion.walker import FileEntry
from synaptiq.core.storage.base import EdgeRef, GraphDelta

__all__ = ["build_incremental_delta"]

#: Node labels that carry a meaningful ``is_dead`` flag — mirrors
#: ``dead_code._SYMBOL_LABELS``. Only these re-parsed upserts seed the
#: ``dead_recount`` set (the storage recount skips every other label anyway).
_DEAD_RECOUNT_LABELS: frozenset[NodeLabel] = frozenset(
    {NodeLabel.FUNCTION, NodeLabel.METHOD, NodeLabel.CLASS}
)

#: Trailing collision suffix ``assign_symbol_ids`` appends to a duplicate-named
#: symbol's id (``…#L{start_line}``). Stripped when recovering the bare name for
#: a stub — ``#`` is not a valid identifier char in any supported language, so
#: the match is unambiguous. The *id* keeps the suffix (it is the real node id);
#: only the reconstructed ``name`` drops it.
_COLLISION_SUFFIX = re.compile(r"#L\d+$")


def _reconstruct_stub(symbol_id: str, file_path: str) -> GraphNode | None:
    """Rebuild a lightweight target-only :class:`GraphNode` from a manifest id.

    The manifest records symbol *ids* and identity *hashes*, not the bare name a
    resolver's name index keys on, so the name is recovered from the contractual
    id format ``{label.value}:{file_path}:{symbol_name}`` (CLAUDE.md). *file_path*
    is the FileManifest key that owns this id — passing it in (rather than parsing
    it out) keeps the split robust even for the rare path containing a colon: the
    known ``{label}:{file_path}:`` prefix is stripped exactly.

    Only the fields a resolver reads off a *target* are populated — ``name``
    (name-index key), ``file_path`` (same-file / import / prefix scoring),
    ``label`` (method/type filtering) and ``class_name`` (``self``/receiver
    method + heritage scoping). Content/signature/line data is deliberately left
    empty: it is never read for a target, and its absence keeps the stub out of
    ``build_file_symbol_index`` (guarded on ``start_line > 0``), which only ever
    locates *source* symbols in re-parsed files.

    Returns ``None`` for an id whose label prefix is unknown or that does not
    match the expected ``{label}:{file_path}:`` shape — such an id is skipped
    rather than trusted.
    """
    first = symbol_id.find(":")
    if first <= 0:
        return None
    label_value = symbol_id[:first]
    try:
        label = NodeLabel(label_value)
    except ValueError:
        return None

    prefix = f"{label_value}:{file_path}:"
    if not symbol_id.startswith(prefix):
        return None
    symbol_name = symbol_id[len(prefix) :]

    # ``method`` ids fold the owning class in: ``Class.method`` (assign_symbol_ids).
    # Strip a collision suffix first so ``User.save#L42`` recovers ``save``/``User``.
    bare = _COLLISION_SUFFIX.sub("", symbol_name)
    name = bare
    class_name = ""
    if label is NodeLabel.METHOD and "." in bare:
        class_name, name = bare.rsplit(".", 1)
    elif label is NodeLabel.FILE:
        # A File node's id folds no name (``file:{path}:`` — empty symbol part),
        # so its basename is taken from the path. File stubs are import-resolution
        # targets keyed by path, never by name; this only keeps the stub faithful.
        name = file_path.rsplit("/", 1)[-1]

    return GraphNode(
        id=symbol_id, label=label, name=name, file_path=file_path, class_name=class_name
    )


def _seed_global_stubs(
    graph: KnowledgeGraph,
    previous: Manifest,
    carry_paths: frozenset[str],
    real_ids: frozenset[str],
) -> None:
    """Seed stub target nodes for every carried (unchanged, non-dependent) file.

    Turns the re-parse-only mini-graph into a *global* one so the resolvers'
    name/file indexes see the whole repo (design §5.4-5.5). Only ``carry_paths``
    are seeded — re-parsed and dependent files already hold real nodes, deleted
    files must NOT be seeded (their symbols are gone), and ``carry_paths`` is
    exactly ``plan.files_unchanged_to_carry`` (which excludes all three). A stub
    whose id already names a real node is skipped, so a real node always wins.
    """
    for path in carry_paths:
        fm = previous.files.get(path)
        if fm is None:
            continue
        for symbol_id in fm.symbol_ids:
            if symbol_id in real_ids:
                continue
            stub = _reconstruct_stub(symbol_id, path)
            if stub is not None:
                graph.add_node(stub)


def build_incremental_delta(
    plan: IncrementalPlan,
    previous: Manifest,
    walk: list[FileEntry],
    *,
    tool_version: str = "",
    git_head: str | None = None,
) -> tuple[GraphDelta, Manifest]:
    """Assemble a scoped :class:`GraphDelta` + the ``new`` :class:`Manifest`.

    Parses only ``plan.files_to_reparse``, resolves their outbound edges against a
    global name/file index seeded from ``previous``, and packages the result as a
    delta the storage layer applies in one transaction (W3.2d). Pure with respect
    to storage — reads nothing, writes nothing; returns the delta and the manifest
    to persist with it (W3.2e wires both).

    Args:
        plan: The scope plan from :func:`plan_incremental`. Must have
            ``full_rebuild_required is False`` — the caller routes full rebuilds
            to ``bulk_load`` instead; this function raises ``ValueError`` for a
            full-rebuild plan rather than silently mis-scoping.
        previous: The manifest the plan diffed against — the source of the global
            symbol set (stub seeding) and of the ``edges_remove`` provenance.
        walk: The current walk (every present file with fresh content), from which
            the re-parse subset is selected and the ``new`` manifest's per-file
            ``content_sha`` is taken.
        tool_version: synaptiq version stamped into the ``new`` manifest.
        git_head: ``HEAD`` sha stamped into the ``new`` manifest (coupling gate);
            not otherwise used here.

    Returns:
        ``(delta, new_manifest)``. ``delta`` is empty-safe (all fields empty when
        nothing changed). ``new_manifest`` describes the post-edit repo (re-parsed
        files' fresh provenance + carried files' content hashes) and is what the
        caller stores next to the applied delta.
    """
    if plan.full_rebuild_required:
        raise ValueError(
            "build_incremental_delta requires an incremental plan; "
            f"got full_rebuild_required=True (reason={plan.reason!r})"
        )

    reparse_paths = plan.files_to_reparse
    deleted_paths = plan.diff.deleted_files if plan.diff is not None else frozenset()

    # --- 1. Parse the re-parse set into a fresh mini-graph (real nodes) --------
    # walk is sorted by path (walker guarantee); filtering preserves that order,
    # so re-parsed nodes enter the graph in the same order a full build would —
    # keeps candidate ordering (hence resolution ties) as close to a full rebuild
    # as the scoped set allows.
    reparse_entries = [entry for entry in walk if entry.path in reparse_paths]

    graph = KnowledgeGraph()
    process_structure(reparse_entries, graph)
    parse_data = process_parsing(reparse_entries, graph)

    # Every node present now is a REAL re-parsed node (File / Folder / symbol).
    # Snapshot before stubs so the two are cleanly separable: only reals become
    # ``nodes_upsert`` / manifest provenance; stubs are resolution scaffolding.
    real_ids = frozenset(node.id for node in graph.iter_nodes())

    # --- 2. Seed the global symbol set so cross-file resolution is complete ----
    _seed_global_stubs(graph, previous, plan.files_unchanged_to_carry, real_ids)

    # --- 3. Resolve outbound edges of the re-parse set (global index) ----------
    # Same phase order as run_pipeline: imports before calls (calls reads the
    # freshly-created IMPORTS edges), rest_linking after calls (phase 5b, D7 —
    # scoped both sides), heritage + types last.
    process_imports(parse_data, graph)
    process_calls(parse_data, graph)
    process_rest_linking(parse_data, graph)
    process_heritage(parse_data, graph)
    process_types(parse_data, graph)

    # --- 4. Split the resolved mini-graph into real nodes vs. all edges --------
    # Stubs carry no edges (only nodes were seeded) and every resolver edge
    # originates from a re-parsed file, so ALL edges are fresh delta content and
    # only real nodes are upserted. Build a clean graph (reals + edges, no stubs)
    # for the manifest so stub sigs never pollute carried files' provenance.
    nodes_upsert = [node for node in graph.iter_nodes() if node.id in real_ids]
    edges_add = list(graph.iter_relationships())

    clean_graph = KnowledgeGraph()
    for node in nodes_upsert:
        clean_graph.add_node(node)
    for rel in edges_add:
        clean_graph.add_relationship(rel)

    # --- 5. edges_remove: exactly what the re-resolved / deleted files owned ----
    # Sourced from the PREVIOUS manifest's out_edges (resolver edges only), so the
    # apply deletes precisely the stale contribution before re-inserting the fresh
    # one — idempotent without a global DETACH DELETE, and inbound edges from
    # untouched files are never removed. Added files have no previous provenance.
    edges_remove: list[EdgeRef] = []
    for path in reparse_paths | deleted_paths:
        fm = previous.files.get(path)
        if fm is None:
            continue
        for edge in fm.out_edges:
            edges_remove.append(EdgeRef(edge.rel_type, edge.src, edge.tgt))

    # --- 6. nodes_remove: genuinely-removed symbols + deleted files' nodes ------
    # ``symbols_to_remove`` already includes every deleted file's symbol ids
    # (incl. its File node id) and every changed file's removed symbols; the
    # explicit File-node union is belt-and-suspenders (a set dedups it).
    nodes_remove_set = set(plan.symbols_to_remove)
    for path in deleted_paths:
        nodes_remove_set.add(generate_id(NodeLabel.FILE, path))
    # Emptied-directory cleanup: deleting the last file in a directory leaves its
    # Folder node (and the CONTAINS edge into it) with nothing under it, but a full
    # rebuild's ``process_structure`` only emits a folder for a dir that still holds
    # a file — so those orphans must be removed or CONTAINS/Folder drift off the
    # strict core. Folders are not manifest-tracked (§4.3 excludes them), so the
    # symbol diff cannot surface this; derive it from the walk instead: an ancestor
    # dir of a deleted file that no longer contains any surviving file is emptied.
    # Adding the Folder id to ``nodes_remove`` lets the by-id DETACH DELETE cascade
    # its CONTAINS edges (both the parent→folder and folder→child edges).
    if deleted_paths:
        surviving_dirs = {
            str(parent)
            for entry in walk
            for parent in PurePosixPath(entry.path).parents
            if str(parent) != "."
        }
        for path in deleted_paths:
            for parent in PurePosixPath(path).parents:
                dir_str = str(parent)
                if dir_str != "." and dir_str not in surviving_dirs:
                    nodes_remove_set.add(generate_id(NodeLabel.FOLDER, dir_str))
    nodes_remove = sorted(nodes_remove_set)

    # --- 7. dead_recount: symbols whose incoming-CALLS in-degree may have moved -
    # Targets of removed/added CALLS (their in-degree changed) ∪ re-parsed symbols
    # (their own edges were rebuilt) ∪ removed ids (harmlessly skipped once gone).
    # The storage layer recomputes is_dead locally over this set (design §5.5).
    dead_recount: set[str] = set()
    for ref in edges_remove:
        if ref.rel_type == RelType.CALLS.value:
            dead_recount.add(ref.target)
    for rel in edges_add:
        if rel.type is RelType.CALLS:
            dead_recount.add(rel.target)
    dead_recount.update(node.id for node in nodes_upsert if node.label in _DEAD_RECOUNT_LABELS)
    dead_recount.update(nodes_remove_set)

    delta = GraphDelta(
        nodes_upsert=nodes_upsert,
        nodes_remove=nodes_remove,
        edges_add=edges_add,
        edges_remove=edges_remove,
        dead_recount=dead_recount,
    )

    # --- 8. new manifest: re-parsed provenance + carried-forward provenance ----
    # carry_from=previous so every UNCHANGED file inherits its full prior
    # provenance (symbol_ids / out_edges / unresolved_imports), not just a content
    # hash. The storage write is a full replace, so the persisted manifest must be
    # complete or the *next* cycle's dependent closure and removal sets break.
    new_manifest = build_current_manifest(
        walk, clean_graph, tool_version=tool_version, git_head=git_head, carry_from=previous
    )

    return delta, new_manifest
