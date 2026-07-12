"""Tests for the incremental-indexing manifest core (W3.2a).

Covers the pure logical layer (build / fingerprint / symbol-level diff /
serialize / version + corruption gating) and the in-DB storage integration —
most importantly a manifest round-trip through a REAL ``bulk_load`` + ``.rebuild``
swap, the D3 mechanism the design hinges on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synaptiq import __version__
from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
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
    diff_manifests,
    load_manifest_from_rows,
    serialize_file_manifest,
    serialize_index_manifest,
    symbol_signature,
)
from synaptiq.core.storage.ladybug_backend import LadybugBackend

# ---------------------------------------------------------------------------
# Ids + sample graph
# ---------------------------------------------------------------------------

FOLDER_SRC = generate_id(NodeLabel.FOLDER, "src", "")
FILE_A = generate_id(NodeLabel.FILE, "src/a.py", "")
FUNC_FOO = generate_id(NodeLabel.FUNCTION, "src/a.py", "foo")
FILE_B = generate_id(NodeLabel.FILE, "src/b.py", "")
FUNC_BAR = generate_id(NodeLabel.FUNCTION, "src/b.py", "bar")

A_CONTENT = "import b\n\n\ndef foo():\n    return bar()\n"
B_CONTENT = "def bar():\n    return 1\n"


def _sample_graph() -> KnowledgeGraph:
    """A 2-file graph with a folder, resolver edges (CALLS/IMPORTS), and a
    structural CONTAINS edge (which must be excluded from the manifest)."""
    g = KnowledgeGraph()
    g.add_node(GraphNode(id=FOLDER_SRC, label=NodeLabel.FOLDER, name="src", file_path="src"))
    g.add_node(
        GraphNode(
            id=FILE_A,
            label=NodeLabel.FILE,
            name="a.py",
            file_path="src/a.py",
            content=A_CONTENT,
            language="python",
        )
    )
    g.add_node(
        GraphNode(
            id=FUNC_FOO,
            label=NodeLabel.FUNCTION,
            name="foo",
            file_path="src/a.py",
            signature="foo()",
            language="python",
            content="def foo():\n    return bar()",
            start_line=4,
            end_line=5,
        )
    )
    g.add_node(
        GraphNode(
            id=FILE_B,
            label=NodeLabel.FILE,
            name="b.py",
            file_path="src/b.py",
            content=B_CONTENT,
            language="python",
        )
    )
    g.add_node(
        GraphNode(
            id=FUNC_BAR,
            label=NodeLabel.FUNCTION,
            name="bar",
            file_path="src/b.py",
            signature="bar()",
            language="python",
        )
    )
    g.add_relationship(
        GraphRelationship(id="contains1", type=RelType.CONTAINS, source=FOLDER_SRC, target=FILE_A)
    )
    g.add_relationship(
        GraphRelationship(
            id="call1",
            type=RelType.CALLS,
            source=FUNC_FOO,
            target=FUNC_BAR,
            properties={"confidence": 1.0},
        )
    )
    g.add_relationship(
        GraphRelationship(
            id="imp1",
            type=RelType.IMPORTS,
            source=FILE_A,
            target=FILE_B,
            properties={"confidence": 1.0},
        )
    )
    return g


@pytest.fixture()
def backend(tmp_path: Path) -> LadybugBackend:
    b = LadybugBackend()
    b.initialize(tmp_path / "kuzu")
    yield b
    b.close()


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


def test_build_manifest_records_per_file():
    m = build_manifest(_sample_graph(), tool_version="9.9.9")
    # Folders are not files; only the two .py files appear.
    assert set(m.files) == {"src/a.py", "src/b.py"}

    a = m.files["src/a.py"]
    assert a.content_sha == content_sha(A_CONTENT)
    assert a.language == "python"
    assert FILE_A in a.symbol_ids and FUNC_FOO in a.symbol_ids
    assert FUNC_FOO in a.symbol_sigs

    edge_keys = {(e.rel_type, e.src, e.tgt) for e in a.out_edges}
    assert (RelType.CALLS.value, FUNC_FOO, FUNC_BAR) in edge_keys
    assert (RelType.IMPORTS.value, FILE_A, FILE_B) in edge_keys
    # Structural CONTAINS is NOT provenance-tracked.
    assert all(e.rel_type != RelType.CONTAINS.value for e in a.out_edges)

    assert m.index.manifest_version == CURRENT_MANIFEST_VERSION
    assert m.index.tool_version == "9.9.9"
    assert m.index.git_head is None
    assert m.index.pending == IndexPending()


def test_build_manifest_excludes_community_and_process():
    g = _sample_graph()
    g.add_node(GraphNode(id="community:0", label=NodeLabel.COMMUNITY, name="c0"))
    g.add_node(GraphNode(id="process:main", label=NodeLabel.PROCESS, name="main"))
    m = build_manifest(g)
    assert set(m.files) == {"src/a.py", "src/b.py"}


def test_build_manifest_is_deterministic():
    a = build_manifest(_sample_graph(), tool_version="1", consolidated_at="t")
    b = build_manifest(_sample_graph(), tool_version="1", consolidated_at="t")
    assert a.files == b.files
    assert a.index == b.index


# ---------------------------------------------------------------------------
# symbol_signature — identity vs body
# ---------------------------------------------------------------------------


def test_symbol_signature_stable_on_body_change():
    n1 = GraphNode(
        id=FUNC_FOO,
        label=NodeLabel.FUNCTION,
        name="foo",
        signature="foo()",
        content="return 1",
        start_line=1,
        end_line=1,
    )
    n2 = GraphNode(
        id=FUNC_FOO,
        label=NodeLabel.FUNCTION,
        name="foo",
        signature="foo()",
        content="return 999  # rewritten body",
        start_line=1,
        end_line=9,
    )
    assert symbol_signature(n1) == symbol_signature(n2)


@pytest.mark.parametrize(
    "mutation",
    [
        {"name": "foo2"},
        {"signature": "foo(x)"},
        {"class_name": "C"},
        {"is_exported": True},
        {"label": NodeLabel.METHOD},
    ],
)
def test_symbol_signature_changes_on_identity(mutation):
    base = GraphNode(id=FUNC_FOO, label=NodeLabel.FUNCTION, name="foo", signature="foo()")
    other = GraphNode(id=FUNC_FOO, label=NodeLabel.FUNCTION, name="foo", signature="foo()")
    for k, v in mutation.items():
        setattr(other, k, v)
    assert symbol_signature(base) != symbol_signature(other)


# ---------------------------------------------------------------------------
# Fingerprint (design §8 trigger 3 primitive)
# ---------------------------------------------------------------------------


def test_fingerprint_order_independent():
    assert compute_fingerprint({"a": "1", "b": "2"}) == compute_fingerprint({"b": "2", "a": "1"})


def test_fingerprint_changes_with_content():
    assert compute_fingerprint({"a": "1"}) != compute_fingerprint({"a": "2"})
    assert compute_fingerprint({"a": "1"}) != compute_fingerprint({"a": "1", "b": "1"})


def test_build_manifest_fingerprint_matches_helper():
    m = build_manifest(_sample_graph())
    expected = compute_fingerprint({p: f.content_sha for p, f in m.files.items()})
    assert m.index.full_fingerprint == expected


# ---------------------------------------------------------------------------
# Symbol-level diff (design §5.1 / §5.3 categories)
# ---------------------------------------------------------------------------


def _fm(path: str, sha: str, sigs: dict[str, str]) -> FileManifest:
    return FileManifest(path=path, content_sha=sha, symbol_ids=sorted(sigs), symbol_sigs=dict(sigs))


def _manifest(*files: FileManifest) -> Manifest:
    return Manifest(index=IndexManifest(), files={f.path: f for f in files})


def test_diff_unchanged_file():
    old = _manifest(_fm("a.py", "sha1", {FUNC_FOO: "sigF"}))
    new = _manifest(_fm("a.py", "sha1", {FUNC_FOO: "sigF"}))
    d = diff_manifests(old, new)
    assert d.unchanged_files == frozenset({"a.py"})
    assert d.changed == {}
    assert not d.added_files and not d.deleted_files


def test_diff_body_only():
    old = _manifest(_fm("a.py", "sha1", {FUNC_FOO: "sigF"}))
    new = _manifest(_fm("a.py", "sha2", {FUNC_FOO: "sigF"}))  # content moved, identity same
    d = diff_manifests(old, new)
    fd = d.changed["a.py"]
    assert fd.body_only == frozenset({FUNC_FOO})
    assert not fd.identity_set_changed
    assert d.body_only_files == frozenset({"a.py"})
    assert d.identity_changed_files == frozenset()


def test_diff_identity_changed():
    old = _manifest(_fm("a.py", "sha1", {FUNC_FOO: "sigOLD"}))
    new = _manifest(_fm("a.py", "sha2", {FUNC_FOO: "sigNEW"}))
    d = diff_manifests(old, new)
    fd = d.changed["a.py"]
    assert fd.identity_changed == frozenset({FUNC_FOO})
    assert fd.identity_set_changed
    assert d.identity_changed_files == frozenset({"a.py"})


def test_diff_added_and_removed_symbol():
    old = _manifest(_fm("a.py", "sha1", {FUNC_FOO: "sigF"}))
    new = _manifest(_fm("a.py", "sha2", {FUNC_FOO: "sigF", "function:src/a.py:baz": "sigB"}))
    d = diff_manifests(old, new)
    fd = d.changed["a.py"]
    assert fd.added == frozenset({"function:src/a.py:baz"})
    assert fd.body_only == frozenset({FUNC_FOO})
    assert fd.identity_set_changed
    # Reverse direction → the same symbol reads as removed.
    back = diff_manifests(new, old)
    assert back.changed["a.py"].removed == frozenset({"function:src/a.py:baz"})


def test_diff_added_and_deleted_file():
    a = _fm("a.py", "sha1", {FUNC_FOO: "sigF"})
    b = _fm("b.py", "shaB", {FUNC_BAR: "sigBar"})
    d = diff_manifests(_manifest(a), _manifest(a, b))
    assert d.added_files == frozenset({"b.py"})
    assert d.deleted_files == frozenset()
    back = diff_manifests(_manifest(a, b), _manifest(a))
    assert back.deleted_files == frozenset({"b.py"})


# ---------------------------------------------------------------------------
# Serialization + gating (pure)
# ---------------------------------------------------------------------------


def _index_row(im: IndexManifest) -> list:
    version, data = serialize_index_manifest(im)
    return [version, data]


def test_serialize_roundtrip_pure():
    m = build_manifest(_sample_graph(), tool_version="1.2.3")
    file_rows = [serialize_file_manifest(f) for f in m.files.values()]
    loaded = load_manifest_from_rows(_index_row(m.index), file_rows)
    assert loaded is not None
    assert loaded.files == m.files
    assert loaded.index.tool_version == "1.2.3"
    assert loaded.index.full_fingerprint == m.index.full_fingerprint
    assert loaded.index.manifest_version == CURRENT_MANIFEST_VERSION
    assert loaded.index.git_head is None
    assert loaded.index.pending == m.index.pending


def test_serialize_preserves_edge_confidence_and_git_head():
    im = IndexManifest(tool_version="x", git_head="abc123", full_fingerprint="fp")
    fm = FileManifest(
        path="a.py",
        content_sha="s",
        language="python",
        symbol_ids=["function:a.py:foo"],
        symbol_sigs={"function:a.py:foo": "sig"},
        out_edges=[EdgeRef("calls", "function:a.py:foo", "function:b.py:bar", 0.5)],
    )
    loaded = load_manifest_from_rows(_index_row(im), [serialize_file_manifest(fm)])
    assert loaded is not None
    assert loaded.index.git_head == "abc123"
    assert loaded.files["a.py"].out_edges == [
        EdgeRef("calls", "function:a.py:foo", "function:b.py:bar", 0.5)
    ]


def test_load_missing_index_returns_none():
    assert load_manifest_from_rows(None, []) is None


def test_load_version_mismatch_returns_none():
    _, data = serialize_index_manifest(IndexManifest())
    # Typed column disagrees.
    assert load_manifest_from_rows([CURRENT_MANIFEST_VERSION + 1, data], []) is None
    # Typed column agrees but the blob's own version was tampered.
    bad = json.loads(data)
    bad["manifest_version"] = 999
    assert load_manifest_from_rows([CURRENT_MANIFEST_VERSION, json.dumps(bad)], []) is None


def test_load_corrupt_blob_returns_none():
    assert load_manifest_from_rows([CURRENT_MANIFEST_VERSION, "{not json"], []) is None
    assert load_manifest_from_rows([CURRENT_MANIFEST_VERSION, "null"], []) is None
    assert load_manifest_from_rows([CURRENT_MANIFEST_VERSION, ""], []) is None


def test_load_corrupt_file_row_poisons_whole_manifest():
    """One unreadable file row ⇒ the whole manifest is untrusted (never partial)."""
    m = build_manifest(_sample_graph())
    good = serialize_file_manifest(next(iter(m.files.values())))
    corrupt = ["bad.py", "sha", "python", "{not json", "{}", "[]"]
    assert load_manifest_from_rows(_index_row(m.index), [good, corrupt]) is None


# ---------------------------------------------------------------------------
# In-DB storage: write/read + REAL bulk_load + swap
# ---------------------------------------------------------------------------


def test_read_manifest_none_before_any_write(backend: LadybugBackend):
    # Tables exist (created in _create_schema) but hold no singleton ⇒ None.
    assert backend.read_manifest() is None


def test_write_then_read_manifest_direct(backend: LadybugBackend):
    m = build_manifest(_sample_graph(), tool_version="7.7.7")
    backend.write_manifest(m)
    loaded = backend.read_manifest()
    assert loaded is not None
    assert loaded.files == m.files
    assert loaded.index.tool_version == "7.7.7"
    assert loaded.index.full_fingerprint == m.index.full_fingerprint


def test_write_manifest_is_full_replace(backend: LadybugBackend):
    backend.write_manifest(build_manifest(_sample_graph(), tool_version="1"))
    # Rewrite with a single-file manifest — the other file's row must be gone.
    only = Manifest(
        index=IndexManifest(tool_version="2", full_fingerprint="fp2"),
        files={"only.py": FileManifest(path="only.py", content_sha="z", language="python")},
    )
    backend.write_manifest(only)
    loaded = backend.read_manifest()
    assert loaded is not None
    assert set(loaded.files) == {"only.py"}
    assert loaded.index.tool_version == "2"


def test_bulk_load_writes_manifest_that_survives_swap(backend: LadybugBackend):
    """The D3 contract: a full build stamps a manifest into the .rebuild DB and
    it survives the atomic swap into the live index."""
    g = _sample_graph()
    backend.bulk_load(g)  # builds into .rebuild, writes manifest, swaps

    loaded = backend.read_manifest()
    assert loaded is not None, "manifest did not survive the bulk_load swap"

    expected = build_manifest(g, tool_version=__version__)
    assert loaded.files == expected.files
    assert loaded.index.full_fingerprint == expected.index.full_fingerprint
    assert loaded.index.manifest_version == CURRENT_MANIFEST_VERSION
    assert loaded.index.tool_version == __version__
    assert loaded.index.git_head is None
    assert loaded.index.pending == IndexPending()

    # The graph itself is intact alongside the manifest (manifest tables do not
    # disturb node/edge storage or load_graph).
    reloaded = backend.load_graph()
    assert reloaded.node_count == g.node_count
    assert reloaded.relationship_count == g.relationship_count


def test_bulk_load_twice_refreshes_manifest(backend: LadybugBackend):
    backend.bulk_load(_sample_graph())

    g2 = KnowledgeGraph()
    g2.add_node(
        GraphNode(
            id=FILE_A,
            label=NodeLabel.FILE,
            name="a.py",
            file_path="src/a.py",
            content="print('changed')\n",
            language="python",
        )
    )
    backend.bulk_load(g2)

    loaded = backend.read_manifest()
    assert loaded is not None
    assert set(loaded.files) == {"src/a.py"}
    assert loaded.files["src/a.py"].content_sha == content_sha("print('changed')\n")
