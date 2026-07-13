"""Tests for the incremental orchestration `run_incremental` (W3.2e).

Drives the decide-and-apply seam behind ``analyze`` (default incremental) and the
watcher's global phase over a real :class:`LadybugBackend`: the D2 default
selection and every §8 / §6.3 / §9 fallback trigger, the scoped delta apply's
equivalence to a full rebuild on the strict core, the manifest carry-forward
across repeated applies, and the ``apply=False`` (compute-only) split the watcher
uses to keep the write lock hold to the surgical apply.
"""

from __future__ import annotations

from synaptiq.core.graph.model import NodeLabel, RelType
from synaptiq.core.ingestion.incremental import (
    REASON_CONSOLIDATE_SYMBOL_RATIO,
    REASON_INCREMENTAL,
)
from synaptiq.core.ingestion.pipeline import (
    REASON_FORCED_FULL,
    REASON_NO_MANIFEST,
    _count_manifest_symbols,
    run_incremental,
    run_pipeline,
    stamp_full_manifest,
)
from synaptiq.core.storage.ladybug_backend import LadybugBackend

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
_REPORT = "from pkg.models import make_user\n\n\ndef report():\n    return make_user('y')\n"


def _fillers(n: int) -> dict[str, str]:
    """Enough single-symbol files that a 1-file edit stays under the ratio gates."""
    return {f"pkg/f{i}.py": f"def fn{i}():\n    return {i}\n" for i in range(n)}


def _repo(tmp_path, files: dict[str, str]):
    root = tmp_path / "repo"
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


def _base_files() -> dict[str, str]:
    return {
        "pkg/models.py": _MODELS,
        "pkg/service.py": _SERVICE,
        "pkg/report.py": _REPORT,
        **_fillers(12),
    }


def _write(root, rel, content):
    (root / rel).write_text(content, encoding="utf-8")


def _full_index(root, tmp_path, name: str) -> LadybugBackend:
    """Full-index *root* into a fresh backend (bulk_load auto-stamps a manifest)."""
    backend = LadybugBackend()
    backend.initialize(tmp_path / name)
    graph, _ = run_pipeline(root, None, skip_embeddings=True)
    backend.bulk_load(graph)
    stamp_full_manifest(backend, graph, tool_version="test", git_head="head0")
    return backend


# --- strict comparator (mirrors test_incremental_build) ----------------------

_FRINGE = frozenset({NodeLabel.COMMUNITY, NodeLabel.PROCESS})
_STRICT_EDGES = (
    RelType.CONTAINS,
    RelType.DEFINES,
    RelType.IMPORTS,
    RelType.EXTENDS,
    RelType.IMPLEMENTS,
    RelType.MIXES_IN,
    RelType.USES_TYPE,
)


def _edges(graph, rel_type, min_conf=None):
    out = set()
    for rel in graph.iter_relationships():
        if rel.type is not rel_type:
            continue
        if min_conf is not None and (rel.properties.get("confidence", 1.0) or 0) < min_conf:
            continue
        out.add((rel.source, rel.target))
    return out


def _assert_strict_equivalent(inc, full):
    inc_nodes = {n.id for n in inc.iter_nodes() if n.label not in _FRINGE}
    full_nodes = {n.id for n in full.iter_nodes() if n.label not in _FRINGE}
    assert inc_nodes == full_nodes, (
        f"nodes diverged: only-inc={inc_nodes - full_nodes}, only-full={full_nodes - inc_nodes}"
    )
    for rt in _STRICT_EDGES:
        assert _edges(inc, rt) == _edges(full, rt), f"{rt.value} edges diverged"
    assert _edges(inc, RelType.CALLS, 0.8) == _edges(full, RelType.CALLS, 0.8)


# ---------------------------------------------------------------------------
# D2 default selection + §8 fallback triggers
# ---------------------------------------------------------------------------


def test_no_manifest_forces_full(tmp_path):
    """A fresh index (no manifest) ⇒ full-rebuild verdict, no storage writes."""
    root = _repo(tmp_path, _base_files())
    backend = LadybugBackend()
    backend.initialize(tmp_path / "empty_db")
    try:
        outcome = run_incremental(root, backend, tool_version="test")
        assert outcome.full_rebuild_required
        assert outcome.reason == REASON_NO_MANIFEST
    finally:
        backend.close()


def test_force_full_returns_full_verdict_without_writing(tmp_path):
    """--full (force_full) ⇒ full verdict even with a valid manifest, no writes."""
    root = _repo(tmp_path, _base_files())
    backend = _full_index(root, tmp_path, "db")
    try:
        before = backend.read_manifest()
        outcome = run_incremental(root, backend, tool_version="test", force_full=True)
        assert outcome.full_rebuild_required and outcome.reason == REASON_FORCED_FULL
        after = backend.read_manifest()
        assert after.index.full_fingerprint == before.index.full_fingerprint  # untouched
    finally:
        backend.close()


def test_incremental_applies_and_equals_full_rebuild(tmp_path):
    """Body-only edit ⇒ incremental apply, and the DB matches a full rebuild."""
    root = _repo(tmp_path, _base_files())
    backend = _full_index(root, tmp_path, "inc_db")
    try:
        _write(root, "pkg/models.py", _MODELS.replace("acct = Account()", "acct = Account()  # x"))
        outcome = run_incremental(root, backend, tool_version="test", now=None)
        assert not outcome.full_rebuild_required
        assert outcome.reason == REASON_INCREMENTAL
        assert outcome.changed_files == 1
        assert outcome.symbols_updated >= 1
        incremental_graph = backend.load_graph()
    finally:
        backend.close()

    full_backend = LadybugBackend()
    full_backend.initialize(tmp_path / "full_db")
    try:
        full_graph, _ = run_pipeline(root, None, skip_embeddings=True)
        full_backend.bulk_load(full_graph)
        full_loaded = full_backend.load_graph()
    finally:
        full_backend.close()

    _assert_strict_equivalent(incremental_graph, full_loaded)


def test_ratio_blowout_forces_full(tmp_path):
    """Rewriting most files crosses the file-ratio gate ⇒ full verdict."""
    root = _repo(tmp_path, _base_files())
    backend = _full_index(root, tmp_path, "ratio_db")
    try:
        # Rewrite every filler → well over 30% of files changed.
        for i in range(12):
            _write(root, f"pkg/f{i}.py", f"def fn{i}():\n    return {i} + 1\n")
        outcome = run_incremental(root, backend, tool_version="test")
        assert outcome.full_rebuild_required
        assert outcome.reason in ("file_ratio_exceeded", "symbol_ratio_exceeded")
    finally:
        backend.close()


def test_consolidation_gate_forces_full(tmp_path):
    """A stored manifest with high accumulated pending ⇒ consolidation verdict."""
    root = _repo(tmp_path, _base_files())
    backend = _full_index(root, tmp_path, "consol_db")
    try:
        man = backend.read_manifest()
        total = sum(len(fm.symbol_ids) for fm in man.files.values())
        man.index.pending.affected_symbols = total  # 100% cumulative → over D5
        backend.write_manifest(man)
        _write(root, "pkg/models.py", _MODELS + "\n\ndef extra():\n    return 1\n")
        outcome = run_incremental(root, backend, tool_version="test")
        assert outcome.full_rebuild_required
        assert outcome.reason == REASON_CONSOLIDATE_SYMBOL_RATIO
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Manifest carry-forward across repeated incremental applies
# ---------------------------------------------------------------------------


def test_repeated_incremental_keeps_manifest_complete(tmp_path):
    """Two incremental applies in a row: the persisted manifest keeps full
    provenance for unchanged files (else the next dependent closure breaks)."""
    root = _repo(tmp_path, _base_files())
    backend = _full_index(root, tmp_path, "carry_db")
    try:
        _write(root, "pkg/f0.py", "def fn0():\n    return 0  # a\n")
        run_incremental(root, backend, tool_version="test")
        _write(root, "pkg/f1.py", "def fn1():\n    return 1  # b\n")
        run_incremental(root, backend, tool_version="test")

        man = backend.read_manifest()
        # service.py was never re-parsed across the two applies, yet its full
        # provenance (its IMPORTS/CALLS out_edges) survived the carry-forward.
        service = man.files["pkg/service.py"]
        assert service.symbol_ids  # not erased to a bare content-hash row
        assert any(e.rel_type == RelType.IMPORTS.value for e in service.out_edges)
        # pending accumulated across BOTH applies.
        assert man.index.pending.applies_since_consolidation == 2
        assert man.index.pending.changed_files == 2
        assert man.index.pending.fts_dirty and man.index.pending.hnsw_dirty
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# meta.json symbol count: full == incremental == graph truth (Bug 3 regression)
# ---------------------------------------------------------------------------


def test_meta_symbols_full_equals_incremental_with_monster_file(tmp_path):
    """meta.symbols must be identical across a full build, an incremental apply,
    and the in-memory graph truth — even with a 200+-symbol file whose manifest
    row is exactly the one the pre-2.0.3 storage round trip used to truncate.

    ``meta.symbols`` comes from ``result.symbols``: a label count over the graph
    on the full path, and ``_count_manifest_symbols(new_manifest)`` on the
    incremental path. If the persisted manifest lost a big file's provenance
    (Bug 1), the incremental count would silently drop below the truth — and the
    carry-forward rewrite would compound it. This pins them together."""
    monster = "".join(f"def sym_{i:04d}(a, b, c):\n    return a + b + c\n\n\n" for i in range(240))
    files = {"pkg/monster.py": monster, "pkg/models.py": _MODELS, **_fillers(6)}
    root = _repo(tmp_path, files)

    # Graph truth = exactly what the full path stamps into meta.symbols.
    _full_graph, full_result = run_pipeline(root, None, skip_embeddings=True)
    truth = full_result.symbols
    assert truth > 200, "monster file alone contributes 240 symbols"

    backend = _full_index(root, tmp_path, "meta_monster_db")
    try:
        # Full build's persisted manifest → the incremental counter must agree
        # with the graph truth (proves the 240-symbol row survived write→read).
        assert _count_manifest_symbols(backend.read_manifest()) == truth

        # Incremental apply touching only a SMALL file: the monster row is
        # carried forward untouched. A body-only edit changes no symbol count.
        _write(root, "pkg/f0.py", "def fn0():\n    return 0  # edit\n")
        outcome = run_incremental(root, backend, tool_version="test")
        assert not outcome.full_rebuild_required
        assert outcome.new_manifest is not None
        assert _count_manifest_symbols(outcome.new_manifest) == truth
        assert _count_manifest_symbols(backend.read_manifest()) == truth

        # A second (no-change) apply keeps the count stable — no compounding loss.
        run_incremental(root, backend, tool_version="test")
        assert _count_manifest_symbols(backend.read_manifest()) == truth
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# apply=False — the watcher's compute-only split
# ---------------------------------------------------------------------------


def test_apply_false_computes_delta_without_writing(tmp_path):
    """apply=False returns the delta + manifest but performs no storage write."""
    root = _repo(tmp_path, _base_files())
    backend = _full_index(root, tmp_path, "noapply_db")
    try:
        before = backend.read_manifest().index.full_fingerprint
        _write(root, "pkg/f2.py", "def fn2():\n    return 2  # z\n")
        outcome = run_incremental(root, backend, tool_version="test", apply=False)
        assert not outcome.full_rebuild_required
        assert outcome.delta is not None and outcome.new_manifest is not None
        # Storage untouched — the caller (watcher) applies under its write lock.
        assert backend.read_manifest().index.full_fingerprint == before
        # Applying the returned delta by hand lands the change.
        backend.apply_graph_delta(outcome.delta)
        backend.write_manifest(outcome.new_manifest)
        assert backend.read_manifest().index.full_fingerprint != before
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Incremental embeddings — only the changed file's symbols are (re)encoded
# ---------------------------------------------------------------------------


def test_upsert_graph_scopes_embeddings_to_changed_file(tmp_path):
    """The embedding partition input (`upsert_graph`) holds only the edited file's
    symbols, so a lazy/sync re-encode touches only changed texts — every unchanged
    symbol keeps its stored vector (apply_graph_delta never wiped the table)."""
    from synaptiq.core.embeddings.embedder import EMBEDDABLE_LABELS, partition_embeddings

    root = _repo(tmp_path, _base_files())
    backend = _full_index(root, tmp_path, "emb_db")
    try:
        _write(root, "pkg/models.py", _MODELS.replace("return acct", "return acct  # e"))
        outcome = run_incremental(root, backend, tool_version="test")
        assert not outcome.full_rebuild_required

        upsert = outcome.upsert_graph()
        embeddable = [n for n in upsert.iter_nodes() if n.label in EMBEDDABLE_LABELS]
        assert embeddable, "the edited file has embeddable symbols to re-encode"
        # No unchanged file's symbol leaks into the re-encode set.
        assert all(n.file_path == "pkg/models.py" for n in embeddable)
        # With no prior vectors for these, every one is pending (would be encoded);
        # the text_sha reuse that skips *unchanged* symbols is proven in
        # test_embedder.py — here we prove the SCOPE is the changed file only.
        _reused, pending = partition_embeddings(upsert, None, tier="quality")
        assert pending == len(embeddable)
    finally:
        backend.close()
