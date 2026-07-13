"""Incremental-indexing manifest core (W3.2a).

The manifest is the provenance record that lets the incremental path answer,
without re-reading the whole repo:

1. **What changed?** ``content_sha`` per file → the changed-file set.
2. **What did each file contribute?** the exact symbol ids, their *identity*
   fingerprints, and the outbound resolver edges a file produced — so a
   symbol-level diff (not just "file changed") drives surgical updates.
3. **Is the manifest still trustworthy?** a ``manifest_version`` + a whole-index
   ``full_fingerprint``, so a schema/tool bump or a "world moved under us"
   change falls back to a full rebuild.

This module owns the *logical* schema (:class:`FileManifest`,
:class:`IndexManifest`, :class:`Manifest`), the fingerprints, the symbol-level
:func:`diff_manifests`, and the (de)serialization + version/corruption gating
that :class:`~synaptiq.core.storage.ladybug_backend.LadybugBackend` calls to
read/write the in-DB manifest tables. Storage I/O (the Cypher) lives in the
backend; everything provenance/gating-shaped lives here so it is pure and
unit-testable.

Design: ``docs/plans/2026-07-12-incremental-indexing-design.md`` §4, §5.1, §8.
Storage sink is the **in-DB table** option (design D3), confirmed viable by the
W3.2a spike (manifest tables survive ``bulk_load``'s ``.rebuild`` swap; see the
backend's ``_create_manifest_tables``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, RelType

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

#: Whole-manifest schema/scoping version stamp (design §4.3 ``manifest_version``).
#:
#: **Bump this on ANY change that could make a stored manifest disagree with a
#: freshly built one**: the manifest table columns, the ``symbol_sigs`` /
#: fingerprint formulas, the set of resolver edge kinds recorded in
#: ``out_edges``, the scoping rules that consume the manifest, OR a storage
#: engine upgrade that changes on-disk/query semantics. A mismatch forces a full
#: rebuild (§8 trigger 2) — we never migrate a manifest; the source of truth is
#: always the code (cf. ``open_with_recovery``'s "rebuildable derived artifact"
#: stance). This is the "engine + synaptiq schema version" gate.
CURRENT_MANIFEST_VERSION = 1

#: Edge kinds recorded in ``FileManifest.out_edges`` — the outbound edges the
#: *resolvers* produce (imports / calls / heritage / types, design §4.3).
#: Structural (CONTAINS/DEFINES/EXPORTS), community (MEMBER_OF), process
#: (STEP_IN_PROCESS) and coupling (COUPLED_WITH) edges are owned by other
#: mechanisms and are deliberately NOT provenance-tracked here.
RESOLVER_REL_TYPES: frozenset[str] = frozenset(
    {
        RelType.IMPORTS.value,
        RelType.CALLS.value,
        RelType.EXTENDS.value,
        RelType.IMPLEMENTS.value,
        RelType.MIXES_IN.value,
        RelType.USES_TYPE.value,
    }
)

#: Node labels that are NOT per-file symbols (never recorded in a FileManifest).
#: Folders are structural; Community/Process are global-phase artifacts.
_NON_FILE_LABELS: frozenset[NodeLabel] = frozenset(
    {NodeLabel.FOLDER, NodeLabel.COMMUNITY, NodeLabel.PROCESS}
)

# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

# All hashing uses utf-8/surrogatepass so a File node whose content contains
# lone surrogates still hashes, and so ``content_sha`` matches the watcher's
# ``_content_hash`` byte-for-byte.
_ENC = ("utf-8", "surrogatepass")


def content_sha(content: str) -> str:
    """Return ``sha256(content)`` — identical to ``watcher._content_hash``.

    The manifest's per-file ``content_sha`` is computed from the stored File
    node ``content`` at build time; the watcher computes the same hash from disk
    at detect time. They must agree, so this mirrors the watcher exactly.
    """
    return hashlib.sha256(content.encode(*_ENC)).hexdigest()


def symbol_signature(node: GraphNode) -> str:
    """Identity fingerprint: ``sha256(name|kind|class_name|signature|is_exported)``.

    Changes iff the symbol's *identity* (not its body) changed. A body-only edit
    leaves this stable, which lets the planner (W3.2b) prove no dependent needs
    re-resolution. ``kind`` is the node label value (function/class/method/...).
    A NUL join keeps the fields unambiguous (identifiers/signatures never contain
    NUL).
    """
    parts = (
        node.name,
        node.label.value,
        node.class_name,
        node.signature,
        "1" if node.is_exported else "0",
    )
    return hashlib.sha256("\x00".join(parts).encode(*_ENC)).hexdigest()


def compute_fingerprint(path_to_sha: dict[str, str]) -> str:
    """Whole-index fingerprint: ``sha256`` over ``sorted(path → content_sha)``.

    The "am I consistent" check (design §8 trigger 3): compare a manifest's
    stored ``full_fingerprint`` to this recomputed over the *current* walk
    before trusting any delta. Sorted so it is order-independent.
    """
    h = hashlib.sha256()
    for path in sorted(path_to_sha):
        h.update(path.encode(*_ENC))
        h.update(b"\x00")
        h.update(path_to_sha[path].encode(*_ENC))
        h.update(b"\x00")
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeRef:
    """A provenance reference to one outbound resolver edge a file produced.

    ``confidence`` is retained in full (not just a bucket) so W3.2c can rebuild
    the edge and W3.2d can distinguish the high-confidence structural core
    (``>= 0.8``) from the fuzzy fringe (``<= 0.5``); the delete key used by
    ``edges_remove`` is only ``(rel_type, src, tgt)``.
    """

    rel_type: str
    src: str
    tgt: str
    confidence: float = 1.0


@dataclass
class FileManifest:
    """Provenance for one file, keyed by repo-relative posix path (design §4.3)."""

    path: str
    content_sha: str
    language: str = ""
    #: Every node id this file defined (function:/class:/method:/module:/file:/...).
    symbol_ids: list[str] = field(default_factory=list)
    #: id → identity fingerprint (see :func:`symbol_signature`). Authoritative
    #: id set for the symbol-level diff.
    symbol_sigs: dict[str, str] = field(default_factory=dict)
    #: Outbound resolver edges this file's resolution produced.
    out_edges: list[EdgeRef] = field(default_factory=list)
    #: Import module strings this file referenced that did NOT resolve to any
    #: project file at build time (external gems/stdlib — or the case this field
    #: exists for: a module that does not exist *yet*). Captured from
    #: ``process_imports`` for the added-file dependent closure: when a brand-new
    #: file is later ADDED whose importable identity matches one of these, this
    #: file is pulled into re-resolution so the import links up incrementally
    #: instead of waiting for a consolidation (see ``plan_incremental`` and
    #: ``imports.importable_identities``).
    unresolved_imports: list[str] = field(default_factory=list)


@dataclass
class IndexPending:
    """Accumulated-since-consolidation counters + staleness flags (design §4.3).

    A full build resets everything to fresh (all-zero / all-clean).
    """

    affected_symbols: int = 0
    changed_files: int = 0
    #: Incremental delta applies since the last consolidation (full rebuild).
    #: One of the consolidation gates (design §9 "every N applies") — reset to 0
    #: whenever a full build stamps a fresh manifest.
    applies_since_consolidation: int = 0
    fts_dirty: bool = False
    hnsw_dirty: bool = False
    community_dirty: bool = False
    process_dirty: bool = False


@dataclass
class IndexManifest:
    """Index-level version + fingerprint stamp (one row; design §4.3)."""

    manifest_version: int = CURRENT_MANIFEST_VERSION
    #: synaptiq ``__version__`` at build time (provenance / status reporting).
    tool_version: str = ""
    #: HEAD sha coupling was computed against, or ``None`` if not a git repo /
    #: not yet known. A full build via ``bulk_load`` cannot see git from the
    #: graph alone and leaves this ``None``; the pipeline/consolidation owner
    #: (W3.2e) populates it when it computes coupling.
    git_head: str | None = None
    #: sha256 over sorted(path → content_sha) — the "am I consistent" check.
    full_fingerprint: str = ""
    #: Last time FTS/HNSW/community/process were made fresh.
    consolidated_at: str = ""
    pending: IndexPending = field(default_factory=IndexPending)


@dataclass
class Manifest:
    """The whole manifest: index-level stamp + per-file records keyed by path."""

    index: IndexManifest
    files: dict[str, FileManifest] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build (from an in-memory KnowledgeGraph)
# ---------------------------------------------------------------------------


def build_manifest(
    graph: KnowledgeGraph,
    *,
    tool_version: str = "",
    git_head: str | None = None,
    consolidated_at: str | None = None,
    pending: IndexPending | None = None,
) -> Manifest:
    """Build a complete :class:`Manifest` from an in-memory graph.

    A cheap by-product of a full build: one pass over nodes (group symbol ids +
    identity sigs + ``content_sha`` per file) and one over relationships (bucket
    resolver edges by their source file). Deterministic — ``symbol_ids`` and
    ``out_edges`` are sorted so the serialized manifest is stable across builds.

    Everything comes from the graph except ``git_head`` (not derivable from the
    graph; defaults ``None`` — see :attr:`IndexManifest.git_head`). A full build
    just made every global phase fresh, so ``pending`` defaults to all-clean and
    ``consolidated_at`` to now.
    """
    files: dict[str, FileManifest] = {}

    for node in graph.iter_nodes():
        if node.label in _NON_FILE_LABELS:
            continue
        path = node.file_path
        if not path:
            continue
        fm = files.get(path)
        if fm is None:
            fm = FileManifest(path=path, content_sha="", language=node.language or "")
            files[path] = fm
        fm.symbol_ids.append(node.id)
        fm.symbol_sigs[node.id] = symbol_signature(node)
        if node.label is NodeLabel.FILE:
            fm.content_sha = content_sha(node.content)
            if node.language:
                fm.language = node.language
            # Unresolved imports are stashed on the File node by process_imports
            # (the only place with both the parsed imports and the resolution
            # verdict); build_manifest is the by-product hook that lifts them
            # into provenance. Absent/malformed → empty (never trust a stray).
            raw_unresolved = node.properties.get("unresolved_imports")
            if isinstance(raw_unresolved, (list, tuple)):
                fm.unresolved_imports = [str(m) for m in raw_unresolved]

    for rel in graph.iter_relationships():
        if rel.type.value not in RESOLVER_REL_TYPES:
            continue
        src = graph.get_node(rel.source)
        if src is None or not src.file_path:
            continue
        fm = files.get(src.file_path)
        if fm is None:
            continue
        confidence = 1.0
        if rel.properties:
            try:
                confidence = float(rel.properties.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
        fm.out_edges.append(EdgeRef(rel.type.value, rel.source, rel.target, confidence))

    # Canonicalize for deterministic serialization / stable fingerprints.
    for fm in files.values():
        fm.symbol_ids.sort()
        fm.out_edges.sort(key=lambda e: (e.rel_type, e.src, e.tgt, e.confidence))
        fm.unresolved_imports = sorted(set(fm.unresolved_imports))

    index = IndexManifest(
        manifest_version=CURRENT_MANIFEST_VERSION,
        tool_version=tool_version,
        git_head=git_head,
        full_fingerprint=compute_fingerprint({p: f.content_sha for p, f in files.items()}),
        consolidated_at=consolidated_at if consolidated_at is not None else _now_iso(),
        pending=pending if pending is not None else IndexPending(),
    )
    return Manifest(index=index, files=files)


# ---------------------------------------------------------------------------
# Symbol-level diff (design §5.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileDiff:
    """Per-symbol classification for one changed file (design §5.1)."""

    path: str
    #: symbol ids present now, absent before.
    added: frozenset[str]
    #: present before, absent now.
    removed: frozenset[str]
    #: id present in both but identity fingerprint differs.
    identity_changed: frozenset[str]
    #: id present in both, identity fingerprint equal (body/line-range moved).
    body_only: frozenset[str]

    @property
    def identity_set_changed(self) -> bool:
        """``added ∪ removed ∪ identity_changed`` non-empty (design §5.1).

        When ``False`` the edit is body-only: every inbound edge still points at
        a same-identity symbol, so no dependent needs re-resolution.
        """
        return bool(self.added or self.removed or self.identity_changed)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.identity_changed or self.body_only)


def diff_file(old: FileManifest, new: FileManifest) -> FileDiff:
    """Classify the symbol-set change of a single file (both versions present)."""
    old_ids = set(old.symbol_sigs)
    new_ids = set(new.symbol_sigs)
    added = new_ids - old_ids
    removed = old_ids - new_ids
    common = old_ids & new_ids
    identity_changed = {i for i in common if old.symbol_sigs.get(i) != new.symbol_sigs.get(i)}
    body_only = common - identity_changed
    return FileDiff(
        path=new.path,
        added=frozenset(added),
        removed=frozenset(removed),
        identity_changed=frozenset(identity_changed),
        body_only=frozenset(body_only),
    )


@dataclass(frozen=True)
class ManifestDiff:
    """Whole-manifest diff across the design §5.3 categories.

    ``changed`` holds only files whose ``content_sha`` moved; unchanged files
    (equal ``content_sha``) are reported in ``unchanged_files`` and carry no
    per-symbol work.
    """

    added_files: frozenset[str]
    deleted_files: frozenset[str]
    changed: dict[str, FileDiff]
    unchanged_files: frozenset[str]

    @property
    def identity_changed_files(self) -> frozenset[str]:
        """Changed files whose *identity set* moved (drive dependent closure)."""
        return frozenset(p for p, d in self.changed.items() if d.identity_set_changed)

    @property
    def body_only_files(self) -> frozenset[str]:
        """Changed files whose edit was body-only (no dependent closure)."""
        return frozenset(p for p, d in self.changed.items() if not d.identity_set_changed)


def diff_manifests(old: Manifest, new: Manifest) -> ManifestDiff:
    """Diff two manifests into added/deleted/changed(+per-symbol)/unchanged files.

    The foundational change-detection primitive the W3.2b scope planner builds
    on. ``new`` need only contain the files being compared (e.g. re-parsed +
    surviving); files absent from ``new`` are treated as deleted.
    """
    old_paths = set(old.files)
    new_paths = set(new.files)
    added_files = new_paths - old_paths
    deleted_files = old_paths - new_paths

    changed: dict[str, FileDiff] = {}
    unchanged: set[str] = set()
    for path in old_paths & new_paths:
        of = old.files[path]
        nf = new.files[path]
        if of.content_sha == nf.content_sha:
            unchanged.add(path)
        else:
            changed[path] = diff_file(of, nf)

    return ManifestDiff(
        added_files=frozenset(added_files),
        deleted_files=frozenset(deleted_files),
        changed=changed,
        unchanged_files=frozenset(unchanged),
    )


# ---------------------------------------------------------------------------
# (De)serialization + version/corruption gating
# ---------------------------------------------------------------------------
#
# The backend stores the manifest in two node tables (see the backend's
# ``_create_manifest_tables``):
#
#   FileManifest(path PK, content_sha, language, symbol_ids, symbol_sigs, out_edges)
#   IndexManifest(id PK, manifest_version, data)
#
# ``symbol_ids`` / ``symbol_sigs`` / ``out_edges`` and the whole IndexManifest
# ``data`` are JSON STRING columns. The row column order below is contractual
# with the backend's DDL and read queries — keep them in lockstep.

#: FileManifest column order (DDL, serialize, and read query must all match).
FILE_MANIFEST_COLUMNS: tuple[str, ...] = (
    "path",
    "content_sha",
    "language",
    "symbol_ids",
    "symbol_sigs",
    "out_edges",
    "unresolved_imports",
)


def serialize_file_manifest(fm: FileManifest) -> list[str]:
    """FileManifest → row values in :data:`FILE_MANIFEST_COLUMNS` order.

    JSON columns are never ``""`` (always ``"[]"``/``"{}"``) so the CSV COPY
    path — which stores ``""`` as NULL — round-trips them intact.
    """
    return [
        fm.path,
        fm.content_sha,
        fm.language,
        json.dumps(fm.symbol_ids),
        json.dumps(fm.symbol_sigs, sort_keys=True),
        json.dumps([[e.rel_type, e.src, e.tgt, e.confidence] for e in fm.out_edges]),
        json.dumps(fm.unresolved_imports),
    ]


def serialize_index_manifest(im: IndexManifest) -> tuple[int, str]:
    """IndexManifest → ``(manifest_version, data_json)`` for the singleton row.

    ``manifest_version`` is surfaced as a typed column for a cheap version gate
    that does not depend on parsing ``data``; ``data`` carries the full record.
    """
    data = {
        "manifest_version": im.manifest_version,
        "tool_version": im.tool_version,
        "git_head": im.git_head,
        "full_fingerprint": im.full_fingerprint,
        "consolidated_at": im.consolidated_at,
        "pending": {
            "affected_symbols": im.pending.affected_symbols,
            "changed_files": im.pending.changed_files,
            "applies_since_consolidation": im.pending.applies_since_consolidation,
            "fts_dirty": im.pending.fts_dirty,
            "hnsw_dirty": im.pending.hnsw_dirty,
            "community_dirty": im.pending.community_dirty,
            "process_dirty": im.pending.process_dirty,
        },
    }
    return im.manifest_version, json.dumps(data)


def _parse_file_manifest_row(row: list[Any]) -> FileManifest | None:
    """One FileManifest DB row → :class:`FileManifest`, or ``None`` if corrupt.

    A corrupt row makes the whole manifest untrustworthy (caller returns
    ``None`` → full rebuild); we never partially trust a manifest.
    """
    try:
        path = row[0]
        content = row[1] or ""
        language = row[2] or ""
        symbol_ids = json.loads(row[3]) if row[3] else []
        symbol_sigs = json.loads(row[4]) if row[4] else {}
        raw_edges = json.loads(row[5]) if row[5] else []
        # ``unresolved_imports`` was added after the first manifest schema; a row
        # written before it (or a NULL from CSV COPY) has fewer than 7 cols —
        # tolerate that with an empty list rather than treating it as corrupt.
        raw_unresolved = json.loads(row[6]) if len(row) > 6 and row[6] else []
    except (IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(path, str) or not path:
        return None
    if not isinstance(symbol_ids, list) or not isinstance(symbol_sigs, dict):
        return None
    if not isinstance(raw_edges, list) or not isinstance(raw_unresolved, list):
        return None
    out_edges: list[EdgeRef] = []
    for e in raw_edges:
        if not isinstance(e, (list, tuple)) or len(e) < 3:
            return None
        conf = e[3] if len(e) > 3 else 1.0
        try:
            out_edges.append(EdgeRef(str(e[0]), str(e[1]), str(e[2]), float(conf)))
        except (TypeError, ValueError):
            return None
    return FileManifest(
        path=path,
        content_sha=content,
        language=language,
        symbol_ids=[str(s) for s in symbol_ids],
        symbol_sigs={str(k): str(v) for k, v in symbol_sigs.items()},
        out_edges=out_edges,
        unresolved_imports=[str(m) for m in raw_unresolved],
    )


def _parse_index_manifest(version_col: Any, data_col: Any) -> IndexManifest | None:
    """Index singleton (typed version + JSON blob) → :class:`IndexManifest`.

    Returns ``None`` on a version mismatch (§8 trigger 2) or an unparseable blob
    (§8 trigger 5). The typed column is the authoritative gate: it is checked
    before — and independently of — parsing ``data``.
    """
    try:
        if int(version_col) != CURRENT_MANIFEST_VERSION:
            return None
    except (TypeError, ValueError):
        return None
    try:
        data = json.loads(data_col) if data_col else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Defense in depth: the blob's own version must also agree with the gate.
    if data.get("manifest_version") != CURRENT_MANIFEST_VERSION:
        return None
    raw_pending = data.get("pending") or {}
    if not isinstance(raw_pending, dict):
        raw_pending = {}
    pending = IndexPending(
        affected_symbols=int(raw_pending.get("affected_symbols", 0) or 0),
        changed_files=int(raw_pending.get("changed_files", 0) or 0),
        applies_since_consolidation=int(raw_pending.get("applies_since_consolidation", 0) or 0),
        fts_dirty=bool(raw_pending.get("fts_dirty", False)),
        hnsw_dirty=bool(raw_pending.get("hnsw_dirty", False)),
        community_dirty=bool(raw_pending.get("community_dirty", False)),
        process_dirty=bool(raw_pending.get("process_dirty", False)),
    )
    git_head = data.get("git_head")
    return IndexManifest(
        manifest_version=CURRENT_MANIFEST_VERSION,
        tool_version=str(data.get("tool_version", "")),
        git_head=str(git_head) if git_head else None,
        full_fingerprint=str(data.get("full_fingerprint", "")),
        consolidated_at=str(data.get("consolidated_at", "")),
        pending=pending,
    )


def load_manifest_from_rows(
    index_row: list[Any] | None, file_rows: list[list[Any]]
) -> Manifest | None:
    """Assemble a :class:`Manifest` from raw DB rows, or ``None`` to force full.

    This is the single gating point for §8 triggers 1/2/5 — the caller (and the
    backend's ``read_manifest``) treat ``None`` as "no trustworthy manifest, use
    the full path". Never crashes, never partially trusts:

    * ``index_row is None``  → no manifest (trigger 1).
    * version mismatch       → ``None`` (trigger 2).
    * any unparseable row    → ``None`` (trigger 5).

    The fingerprint check (§8 trigger 3) is the caller's — it needs the current
    walk — via :func:`compute_fingerprint` vs :attr:`IndexManifest.full_fingerprint`.
    """
    if index_row is None:
        return None
    if len(index_row) < 2:
        return None
    index = _parse_index_manifest(index_row[0], index_row[1])
    if index is None:
        return None
    files: dict[str, FileManifest] = {}
    for row in file_rows:
        fm = _parse_file_manifest_row(row)
        if fm is None:
            return None
        files[fm.path] = fm
    return Manifest(index=index, files=files)
