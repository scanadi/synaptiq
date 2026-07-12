"""Tests for the parsing processor (Phase 3)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, RelType, generate_id
from synaptiq.core.ingestion import parser_phase as parser_phase_module
from synaptiq.core.ingestion.parser_phase import (
    FileParseData,
    get_parser,
    parse_file,
    process_parsing,
)
from synaptiq.core.ingestion.walker import FileEntry
from synaptiq.core.parsers.python_lang import PythonParser
from synaptiq.core.parsers.typescript import TypeScriptParser
from synaptiq.core.resources import set_jobs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a KnowledgeGraph pre-populated with File nodes for test files."""
    g = KnowledgeGraph()

    # Python file node
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.FILE, "src/utils.py"),
            label=NodeLabel.FILE,
            name="utils.py",
            file_path="src/utils.py",
            language="python",
        )
    )

    # TypeScript file node
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.FILE, "src/app.ts"),
            label=NodeLabel.FILE,
            name="app.ts",
            file_path="src/app.ts",
            language="typescript",
        )
    )

    return g


PYTHON_CODE = """\
class UserService:
    def get_user(self, user_id: int) -> str:
        return str(user_id)

    def delete_user(self, user_id: int) -> None:
        pass

def helper(x: int) -> int:
    return x + 1
"""

TYPESCRIPT_CODE = """\
interface Config {
    host: string;
    port: number;
}

class App {
    start(): void {}
}

function run(config: Config): void {
    const app = new App();
    app.start();
}
"""

JAVASCRIPT_CODE = """\
function add(a, b) {
    return a + b;
}
"""


def _make_file_entry(
    path: str, content: str, language: str
) -> FileEntry:
    return FileEntry(path=path, content=content, language=language)


# ---------------------------------------------------------------------------
# get_parser tests
# ---------------------------------------------------------------------------


class TestGetParserPython:
    """get_parser returns PythonParser for 'python'."""

    def test_get_parser_python(self) -> None:
        parser = get_parser("python")
        assert isinstance(parser, PythonParser)


class TestGetParserTypeScript:
    """get_parser returns TypeScriptParser for 'typescript'."""

    def test_get_parser_typescript(self) -> None:
        parser = get_parser("typescript")
        assert isinstance(parser, TypeScriptParser)
        assert parser.dialect == "typescript"


class TestGetParserJavaScript:
    """get_parser returns TypeScriptParser with 'javascript' dialect."""

    def test_get_parser_javascript(self) -> None:
        parser = get_parser("javascript")
        assert isinstance(parser, TypeScriptParser)
        assert parser.dialect == "javascript"


class TestGetParserUnsupported:
    """get_parser raises ValueError for unknown languages."""

    def test_get_parser_unsupported(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            get_parser("rust")


# ---------------------------------------------------------------------------
# parse_file tests
# ---------------------------------------------------------------------------


class TestParseFilePython:
    """parse_file parses Python source and returns correct symbols."""

    def test_parse_file_python(self) -> None:
        data = parse_file("src/utils.py", PYTHON_CODE, "python")

        assert isinstance(data, FileParseData)
        assert data.file_path == "src/utils.py"
        assert data.language == "python"

        symbol_names = [s.name for s in data.parse_result.symbols]
        assert "UserService" in symbol_names
        assert "get_user" in symbol_names
        assert "delete_user" in symbol_names
        assert "helper" in symbol_names

    def test_method_has_class_name(self) -> None:
        data = parse_file("src/utils.py", PYTHON_CODE, "python")
        methods = [s for s in data.parse_result.symbols if s.kind == "method"]
        for m in methods:
            assert m.class_name == "UserService"


class TestParseFileTypeScript:
    """parse_file parses TypeScript source and returns correct symbols."""

    def test_parse_file_typescript(self) -> None:
        data = parse_file("src/app.ts", TYPESCRIPT_CODE, "typescript")

        assert isinstance(data, FileParseData)
        assert data.file_path == "src/app.ts"
        assert data.language == "typescript"

        symbol_names = [s.name for s in data.parse_result.symbols]
        assert "Config" in symbol_names
        assert "App" in symbol_names
        assert "run" in symbol_names


# ---------------------------------------------------------------------------
# process_parsing tests
# ---------------------------------------------------------------------------


class TestProcessParsingCreatesFunctionNodes:
    """process_parsing creates Function nodes in the graph."""

    def test_process_parsing_creates_function_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        func_nodes = graph.get_nodes_by_label(NodeLabel.FUNCTION)
        func_names = {n.name for n in func_nodes}
        assert "helper" in func_names

    def test_function_node_properties(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        func_id = generate_id(NodeLabel.FUNCTION, "src/utils.py", "helper")
        node = graph.get_node(func_id)
        assert node is not None
        assert node.name == "helper"
        assert node.file_path == "src/utils.py"
        assert node.start_line > 0
        assert node.end_line >= node.start_line
        assert "def helper" in node.content
        assert node.signature != ""


class TestProcessParsingCreatesClassNodes:
    """process_parsing creates Class nodes in the graph."""

    def test_process_parsing_creates_class_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        class_nodes = graph.get_nodes_by_label(NodeLabel.CLASS)
        class_names = {n.name for n in class_nodes}
        assert "UserService" in class_names

    def test_class_node_has_content(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        class_id = generate_id(NodeLabel.CLASS, "src/utils.py", "UserService")
        node = graph.get_node(class_id)
        assert node is not None
        assert "class UserService" in node.content


class TestProcessParsingCreatesMethodNodes:
    """process_parsing creates Method nodes with class_name set."""

    def test_process_parsing_creates_method_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        method_names = {n.name for n in method_nodes}
        assert "get_user" in method_names
        assert "delete_user" in method_names

    def test_method_nodes_have_class_name(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        for method in method_nodes:
            assert method.class_name == "UserService"

    def test_method_node_id_uses_class_dot_method(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_id = generate_id(
            NodeLabel.METHOD, "src/utils.py", "UserService.get_user"
        )
        node = graph.get_node(method_id)
        assert node is not None
        assert node.name == "get_user"


class TestProcessParsingCreatesDefinesRelationships:
    """process_parsing creates DEFINES relationships from File to Symbol."""

    def test_process_parsing_creates_defines_relationships(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        assert len(defines_rels) > 0

        file_id = generate_id(NodeLabel.FILE, "src/utils.py")
        # All DEFINES relationships should originate from the file node.
        for rel in defines_rels:
            assert rel.source == file_id

    def test_defines_relationship_targets_symbol(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        target_ids = {rel.target for rel in defines_rels}

        # The function node should be a target.
        func_id = generate_id(NodeLabel.FUNCTION, "src/utils.py", "helper")
        assert func_id in target_ids

        # The class node should be a target.
        class_id = generate_id(NodeLabel.CLASS, "src/utils.py", "UserService")
        assert class_id in target_ids

    def test_defines_relationship_id_format(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        for rel in defines_rels:
            assert rel.id.startswith("defines:")
            assert "->" in rel.id


class TestProcessParsingReturnsParseData:
    """process_parsing returns FileParseData for use by later phases."""

    def test_process_parsing_returns_parse_data(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [
            _make_file_entry("src/utils.py", PYTHON_CODE, "python"),
            _make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript"),
        ]
        result = process_parsing(files, graph)

        assert len(result) == 2
        assert all(isinstance(d, FileParseData) for d in result)

    def test_parse_data_carries_symbol_ids(
        self, graph: KnowledgeGraph
    ) -> None:
        """Phase 2 carries the computed node IDs forward on symbol_ids (W2.5c)
        so later phases (calls.py's decorator-edge resolution) can reuse
        them instead of recomputing via assign_symbol_ids a second time."""
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        result = process_parsing(files, graph)

        fpd = result[0]
        assert fpd.symbol_ids is not None
        assert len(fpd.symbol_ids) == len(fpd.parse_result.symbols)
        # No unresolved collisions in this fixture -- every entry is a
        # real ID, and it matches what assign_symbol_ids computes directly.
        assert all(sid is not None for sid in fpd.symbol_ids)
        assert fpd.symbol_ids == parser_phase_module.assign_symbol_ids(
            fpd.parse_result.symbols, "src/utils.py"
        )

        # And the carried IDs are exactly the IDs the nodes were stored
        # under in the graph.
        for sid in fpd.symbol_ids:
            assert graph.get_node(sid) is not None

    def test_parse_data_carries_imports(
        self, graph: KnowledgeGraph
    ) -> None:
        code_with_import = "import os\n\ndef main():\n    pass\n"
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/main.py"),
                label=NodeLabel.FILE,
                name="main.py",
                file_path="src/main.py",
                language="python",
            )
        )
        files = [_make_file_entry("src/main.py", code_with_import, "python")]
        result = process_parsing(files, graph)

        assert len(result[0].parse_result.imports) > 0

    def test_parse_data_carries_calls(
        self, graph: KnowledgeGraph
    ) -> None:
        code_with_call = "def foo():\n    bar()\n"
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/caller.py"),
                label=NodeLabel.FILE,
                name="caller.py",
                file_path="src/caller.py",
                language="python",
            )
        )
        files = [_make_file_entry("src/caller.py", code_with_call, "python")]
        result = process_parsing(files, graph)

        call_names = [c.name for c in result[0].parse_result.calls]
        assert "bar" in call_names


class TestProcessParsingHandlesError:
    """process_parsing handles bad content gracefully without crashing."""

    def test_process_parsing_handles_error(
        self, graph: KnowledgeGraph
    ) -> None:
        # Provide an unsupported language to trigger the error path.
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/bad.rs"),
                label=NodeLabel.FILE,
                name="bad.rs",
                file_path="src/bad.rs",
                language="rust",
            )
        )
        files = [_make_file_entry("src/bad.rs", "fn main() {}", "rust")]
        result = process_parsing(files, graph)

        # Should still return a FileParseData with empty result.
        assert len(result) == 1
        assert result[0].parse_result.symbols == []
        assert result[0].parse_result.imports == []

    def test_error_does_not_affect_other_files(
        self, graph: KnowledgeGraph
    ) -> None:
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/bad.rs"),
                label=NodeLabel.FILE,
                name="bad.rs",
                file_path="src/bad.rs",
                language="rust",
            )
        )
        files = [
            _make_file_entry("src/bad.rs", "fn main() {}", "rust"),
            _make_file_entry("src/utils.py", PYTHON_CODE, "python"),
        ]
        result = process_parsing(files, graph)

        assert len(result) == 2
        # The Rust file should have empty symbols.
        assert result[0].parse_result.symbols == []
        # The Python file should parse successfully.
        assert len(result[1].parse_result.symbols) > 0


class TestProcessParsingTypeScript:
    """process_parsing handles TypeScript interface and class nodes."""

    def test_creates_interface_nodes(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript")]
        process_parsing(files, graph)

        iface_nodes = graph.get_nodes_by_label(NodeLabel.INTERFACE)
        iface_names = {n.name for n in iface_nodes}
        assert "Config" in iface_names

    def test_creates_ts_class_and_method_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript")]
        process_parsing(files, graph)

        class_nodes = graph.get_nodes_by_label(NodeLabel.CLASS)
        class_names = {n.name for n in class_nodes}
        assert "App" in class_names

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        method_names = {n.name for n in method_nodes}
        assert "start" in method_names


class TestProcessParsingWorkerCount:
    """max_workers resolution: explicit value wins; default derives from
    current_limits().pool_workers (W1.4 — analyze --jobs)."""

    @staticmethod
    def _spy_executor(captured: dict) -> type[ThreadPoolExecutor]:
        class SpyExecutor(ThreadPoolExecutor):
            def __init__(self, max_workers=None, *args, **kwargs):
                captured["max_workers"] = max_workers
                super().__init__(max_workers=max_workers, *args, **kwargs)

        return SpyExecutor

    def test_explicit_max_workers_is_honored(
        self, graph: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            parser_phase_module, "ThreadPoolExecutor", self._spy_executor(captured)
        )
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]

        process_parsing(files, graph, max_workers=2)

        assert captured["max_workers"] == 2

    def test_default_worker_count_derives_from_jobs_override(
        self, graph: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}
        monkeypatch.setattr(
            parser_phase_module, "ThreadPoolExecutor", self._spy_executor(captured)
        )
        set_jobs(3)
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]

        process_parsing(files, graph)

        assert captured["max_workers"] == 3

    def test_default_worker_count_falls_back_to_pool_default(
        self, graph: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No --jobs, no explicit max_workers: min(8, cpu_count) applies."""
        captured: dict = {}
        monkeypatch.setattr(
            parser_phase_module, "ThreadPoolExecutor", self._spy_executor(captured)
        )
        monkeypatch.setattr("synaptiq.core.resources.os.cpu_count", lambda: 4)
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]

        process_parsing(files, graph)

        assert captured["max_workers"] == 4


# ---------------------------------------------------------------------------
# W2.1: process-parallel parsing — fan-out selection, fallback, equivalence
# ---------------------------------------------------------------------------


# Curated multi-language snippets to exercise the ParseResult fields that a
# plain Python corpus under-covers: TS/JS symbols, type_refs, variable_types,
# heritage "extends" *and* Ruby "mixin", exports.  (endpoints/http_calls are
# populated by the later rest_linking phase, not by parse_file, so they stay
# empty here — the equivalence check still compares them, i.e. empty==empty.)
_FASTAPI_PY = '''\
import requests
from fastapi import FastAPI

app = FastAPI()

__all__ = ["get_item", "Client"]


class Base:
    pass


class Client(Base):
    def fetch(self, item_id: int) -> dict:
        resp = requests.get(f"https://api.test/items/{item_id}")
        return resp.json()


@app.get("/items/{item_id}")
def get_item(item_id: int) -> dict:
    client = Client()
    return client.fetch(item_id)
'''

_AXIOS_TS = '''\
import axios from "axios";

export interface Item {
    id: number;
    name: string;
}

class Widget extends BaseWidget {
    render(item: Item): void {
        renderThing();
    }
}

export function loadItem(id: number): void {
    const w = new Widget();
    axios.get(`/items/${id}`);
    w.render({ id, name: "x" });
}
'''

_RUBY = '''\
require "httparty"

module Greetable
  def greet
    "hi"
  end
end

class Animal
end

class Dog < Animal
  include Greetable

  def fetch_remote(id)
    HTTParty.get("https://api.test/dogs/#{id}")
  end
end
'''

_JS = '''\
function add(a, b) {
    return a + b;
}

class Calc {
    sum(xs) {
        return xs.reduce(add, 0);
    }
}
'''


def _curated_multi_language_corpus() -> list[FileEntry]:
    return [
        _make_file_entry("svc/api.py", _FASTAPI_PY, "python"),
        _make_file_entry("web/item.ts", _AXIOS_TS, "typescript"),
        _make_file_entry("lib/dog.rb", _RUBY, "ruby"),
        _make_file_entry("web/calc.js", _JS, "javascript"),
    ]


def _repo_python_corpus(limit: int | None = None) -> list[FileEntry]:
    """Real in-repo Python source files as FileEntry objects (sorted by path)."""
    pkg_dir = Path(parser_phase_module.__file__).resolve().parents[2]
    paths = sorted(pkg_dir.rglob("*.py"))
    if limit is not None:
        paths = paths[:limit]
    entries: list[FileEntry] = []
    for p in paths:
        try:
            content = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if content:
            entries.append(_make_file_entry(str(p), content, "python"))
    return entries


def _many_trivial_files(n: int) -> list[FileEntry]:
    return [
        _make_file_entry(f"src/mod_{i}.py", f"def f_{i}():\n    return {i}\n", "python")
        for i in range(n)
    ]


def _assert_parse_data_equal(
    threads: list[FileParseData], processes: list[FileParseData]
) -> None:
    """Assert two FileParseData lists are byte-identical, field by field.

    Comparison is order-sensitive within each file (list ``==``), which is
    the W2.1 equivalence contract.
    """
    assert len(threads) == len(processes)
    for a, b in zip(threads, processes):
        assert a.file_path == b.file_path
        assert a.language == b.language
        assert a.content == b.content
        ra, rb = a.parse_result, b.parse_result
        assert ra.symbols == rb.symbols, a.file_path
        assert ra.imports == rb.imports, a.file_path
        assert ra.calls == rb.calls, a.file_path
        assert ra.heritage == rb.heritage, a.file_path
        assert ra.type_refs == rb.type_refs, a.file_path
        assert ra.exports == rb.exports, a.file_path
        assert ra.endpoints == rb.endpoints, a.file_path
        assert ra.http_calls == rb.http_calls, a.file_path
    # Whole-object equality (also covers variable_types) as a backstop.
    assert threads == processes


class TestFanOutSelection:
    """_should_use_process_pool gates the process pool on file count + workers."""

    def test_uses_processes_at_threshold_with_workers(self) -> None:
        n = parser_phase_module._PROCESS_POOL_MIN_FILES
        assert parser_phase_module._should_use_process_pool(n, 8) is True

    def test_below_threshold_returns_false(self) -> None:
        n = parser_phase_module._PROCESS_POOL_MIN_FILES - 1
        assert parser_phase_module._should_use_process_pool(n, 8) is False

    def test_single_worker_returns_false(self) -> None:
        n = parser_phase_module._PROCESS_POOL_MIN_FILES + 100
        assert parser_phase_module._should_use_process_pool(n, 1) is False

    def test_zero_files_returns_false(self) -> None:
        assert parser_phase_module._should_use_process_pool(0, 8) is False


class TestFanOutFallback:
    """_parse_files falls back to threads on the documented triggers."""

    def test_below_threshold_stays_on_threads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def spy(files, max_workers):
            calls.append(len(files))
            return []

        monkeypatch.setattr(parser_phase_module, "_parse_with_processes", spy)
        files = _many_trivial_files(parser_phase_module._PROCESS_POOL_MIN_FILES - 1)

        result = parser_phase_module._parse_files(files, max_workers=8)

        assert calls == []  # process pool never consulted
        assert len(result) == len(files)
        assert all(isinstance(d, FileParseData) for d in result)

    def test_single_worker_stays_on_threads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def spy(files, max_workers):
            calls.append(len(files))
            return []

        monkeypatch.setattr(parser_phase_module, "_parse_with_processes", spy)
        files = _many_trivial_files(parser_phase_module._PROCESS_POOL_MIN_FILES + 5)

        result = parser_phase_module._parse_files(files, max_workers=1)

        assert calls == []  # single worker -> threads, despite the file count
        assert len(result) == len(files)

    def test_process_pool_failure_falls_back_to_threads(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(files, max_workers):
            raise RuntimeError("subprocesses forbidden here")

        monkeypatch.setattr(parser_phase_module, "_parse_with_processes", boom)
        files = _many_trivial_files(parser_phase_module._PROCESS_POOL_MIN_FILES)

        with caplog.at_level(
            logging.WARNING, logger="synaptiq.core.ingestion.parser_phase"
        ):
            result = parser_phase_module._parse_files(files, max_workers=2)

        expected = parser_phase_module._parse_with_threads(files, max_workers=2)
        assert result == expected  # thread fallback produced the full, correct result
        assert "falling back to thread pool" in caplog.text


class TestParseFanOutEquivalence:
    """The process-pool and thread-pool paths produce identical parse data."""

    def test_repo_corpus_equivalent_across_paths(self) -> None:
        files = _repo_python_corpus() + _curated_multi_language_corpus()
        # Guard: a meaningful corpus, not a trivial one.
        assert len(files) > 50

        threads = parser_phase_module._parse_with_threads(files, max_workers=4)
        processes = parser_phase_module._parse_with_processes(files, max_workers=4)

        _assert_parse_data_equal(threads, processes)

    def test_curated_corpus_exercises_key_fields(self) -> None:
        """The curated corpus actually populates the fields the equivalence
        test compares, so a future parser change can't silently gut coverage."""
        parsed = parser_phase_module._parse_with_threads(
            _curated_multi_language_corpus(), max_workers=2
        )
        by_path = {d.file_path: d.parse_result for d in parsed}

        py = by_path["svc/api.py"]
        assert py.symbols and py.imports and py.calls and py.exports
        assert any(kind == "extends" for _, kind, _ in py.heritage)

        ts = by_path["web/item.ts"]
        assert ts.symbols and ts.type_refs and ts.variable_types and ts.exports
        assert any(kind == "extends" for _, kind, _ in ts.heritage)

        rb = by_path["lib/dog.rb"]
        assert rb.symbols and rb.imports and rb.calls
        assert {"extends", "mixin"} <= {kind for _, kind, _ in rb.heritage}


class TestProcessParsingBothPaths:
    """process_parsing builds an identical graph whether Phase 1 fans out to
    threads or to processes (the graph-mutation phase is path-agnostic)."""

    @staticmethod
    def _add_file_nodes(graph: KnowledgeGraph, files: list[FileEntry]) -> None:
        for fe in files:
            graph.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, fe.path),
                    label=NodeLabel.FILE,
                    name=Path(fe.path).name,
                    file_path=fe.path,
                    language=fe.language,
                )
            )

    @staticmethod
    def _canonical(graph: KnowledgeGraph):
        nodes = sorted(
            (
                n.id,
                n.label.value,
                n.name,
                n.start_line,
                n.end_line,
                n.content,
                n.signature,
                n.class_name,
                n.is_exported,
                n.language,
                repr(sorted((n.properties or {}).items())),
            )
            for n in graph.iter_nodes()
        )
        defines = sorted(
            (r.source, r.target, r.id)
            for r in graph.get_relationships_by_type(RelType.DEFINES)
        )
        return nodes, defines

    def test_graph_identical_threads_vs_processes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = _repo_python_corpus(limit=40) + _curated_multi_language_corpus()

        # Thread path: default threshold (100) keeps this ~44-file batch on
        # threads.
        g_threads = KnowledgeGraph()
        self._add_file_nodes(g_threads, files)
        process_parsing(files, g_threads, max_workers=2)

        # Process path: drop the threshold so the same batch fans out to
        # worker processes.
        monkeypatch.setattr(parser_phase_module, "_PROCESS_POOL_MIN_FILES", 1)
        g_procs = KnowledgeGraph()
        self._add_file_nodes(g_procs, files)
        process_parsing(files, g_procs, max_workers=2)

        assert self._canonical(g_threads) == self._canonical(g_procs)
