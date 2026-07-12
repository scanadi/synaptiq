"""Tests for the import resolution phase (Phase 4)."""

from __future__ import annotations

import time
from pathlib import PurePosixPath

import pytest

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphNode,
    NodeLabel,
    RelType,
    generate_id,
)
from synaptiq.core.ingestion.imports import (
    _RubyBasenameIndex,
    build_file_index,
    process_imports,
    resolve_import_path,
)
from synaptiq.core.ingestion.parser_phase import FileParseData
from synaptiq.core.parsers.base import ImportInfo, ParseResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FILE_PATHS = [
    # Python files
    ("src/auth/validate.py", "python"),
    ("src/auth/utils.py", "python"),
    ("src/auth/__init__.py", "python"),
    ("src/models/user.py", "python"),
    ("src/models/__init__.py", "python"),
    ("src/app.py", "python"),
    # TypeScript files
    ("lib/index.ts", "typescript"),
    ("lib/utils.ts", "typescript"),
    ("lib/models/user.ts", "typescript"),
    ("lib/models/index.ts", "typescript"),
    # Ruby files
    ("app/main.rb", "ruby"),
    ("lib/foo.rb", "ruby"),
    ("app/services/user_service.rb", "ruby"),
    ("config/settings.rb", "ruby"),
]


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a KnowledgeGraph pre-populated with File nodes for testing."""
    g = KnowledgeGraph()
    for path, language in _FILE_PATHS:
        node_id = generate_id(NodeLabel.FILE, path)
        g.add_node(
            GraphNode(
                id=node_id,
                label=NodeLabel.FILE,
                name=path.rsplit("/", 1)[-1],
                file_path=path,
                language=language,
            )
        )
    return g


@pytest.fixture()
def file_index(graph: KnowledgeGraph) -> dict[str, str]:
    """Return the file index built from the fixture graph."""
    return build_file_index(graph)


# ---------------------------------------------------------------------------
# build_file_index
# ---------------------------------------------------------------------------


class TestBuildFileIndex:
    """build_file_index creates correct mapping from graph File nodes."""

    def test_build_file_index(self, graph: KnowledgeGraph) -> None:
        index = build_file_index(graph)

        assert len(index) == len(_FILE_PATHS)
        for path, _ in _FILE_PATHS:
            assert path in index
            assert index[path] == generate_id(NodeLabel.FILE, path)

    def test_build_file_index_empty_graph(self) -> None:
        g = KnowledgeGraph()
        index = build_file_index(g)
        assert index == {}

    def test_build_file_index_ignores_non_file_nodes(self) -> None:
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FOLDER, "src"),
                label=NodeLabel.FOLDER,
                name="src",
                file_path="src",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/app.py"),
                label=NodeLabel.FILE,
                name="app.py",
                file_path="src/app.py",
                language="python",
            )
        )
        index = build_file_index(g)
        assert len(index) == 1
        assert "src/app.py" in index


# ---------------------------------------------------------------------------
# resolve_import_path — Python
# ---------------------------------------------------------------------------


class TestResolvePythonRelativeImport:
    """from .utils import helper in src/auth/validate.py -> src/auth/utils.py."""

    def test_resolve_python_relative_import(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module=".utils", names=["helper"], is_relative=True)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "src/auth/utils.py")
        assert result == expected_id


class TestResolvePythonParentRelative:
    """from ..models import User in src/auth/validate.py -> src/models/__init__.py."""

    def test_resolve_python_parent_relative(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="..models", names=["User"], is_relative=True)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "src/models/__init__.py")
        assert result == expected_id

    def test_resolve_python_parent_relative_direct_module(self) -> None:
        """When models.py exists instead of models/__init__.py."""
        g = KnowledgeGraph()
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/auth/validate.py"),
                label=NodeLabel.FILE,
                name="validate.py",
                file_path="src/auth/validate.py",
                language="python",
            )
        )
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/models.py"),
                label=NodeLabel.FILE,
                name="models.py",
                file_path="src/models.py",
                language="python",
            )
        )
        index = build_file_index(g)

        imp = ImportInfo(module="..models", names=["User"], is_relative=True)
        result = resolve_import_path("src/auth/validate.py", imp, index)

        expected_id = generate_id(NodeLabel.FILE, "src/models.py")
        assert result == expected_id


class TestResolvePythonExternal:
    """import os or from os.path import join -> returns None (external)."""

    def test_resolve_python_external_import(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="os", names=[], is_relative=False)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)
        assert result is None

    def test_resolve_python_external_from_import(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="os.path", names=["join"], is_relative=False)
        result = resolve_import_path("src/auth/validate.py", imp, file_index)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_import_path — TypeScript / JavaScript
# ---------------------------------------------------------------------------


class TestResolveTsRelative:
    """import { foo } from './utils' in lib/index.ts -> lib/utils.ts."""

    def test_resolve_ts_relative(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="./utils", names=["foo"], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "lib/utils.ts")
        assert result == expected_id


class TestResolveTsDirectoryIndex:
    """import { User } from './models' in lib/index.ts -> lib/models/index.ts."""

    def test_resolve_ts_directory_index(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="./models", names=["User"], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "lib/models/index.ts")
        assert result == expected_id


class TestResolveTsExternal:
    """import express from 'express' -> returns None (external)."""

    def test_resolve_ts_external(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="express", names=["express"], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)
        assert result is None

    def test_resolve_ts_scoped_external(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="@types/node", names=[], is_relative=False)
        result = resolve_import_path("lib/index.ts", imp, file_index)
        assert result is None


# ---------------------------------------------------------------------------
# resolve_import_path — Ruby
# ---------------------------------------------------------------------------


class TestResolveRubyRequireRelative:
    """require_relative '../lib/foo' in app/main.rb -> lib/foo.rb."""

    def test_resolve_ruby_require_relative(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="../lib/foo", names=[], is_relative=True)
        result = resolve_import_path("app/main.rb", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "lib/foo.rb")
        assert result is not None
        assert result == expected_id

    def test_resolve_ruby_require_relative_same_dir(self) -> None:
        """require_relative './service' resolves within the same directory."""
        g = KnowledgeGraph()
        for path in ("app/main.rb", "app/service.rb"):
            g.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, path),
                    label=NodeLabel.FILE,
                    name=path.rsplit("/", 1)[-1],
                    file_path=path,
                    language="ruby",
                )
            )
        index = build_file_index(g)

        imp = ImportInfo(module="./service", names=[], is_relative=True)
        result = resolve_import_path("app/main.rb", imp, index)

        assert result == generate_id(NodeLabel.FILE, "app/service.rb")


class TestResolveRubyRequire:
    """require 'config/settings' -> config/settings.rb (project-root path)."""

    def test_resolve_ruby_require_project_path(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="config/settings", names=[], is_relative=False)
        result = resolve_import_path("app/main.rb", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "config/settings.rb")
        assert result == expected_id


class TestResolveRubyConvention:
    """Rails autoload convention: UserService -> app/services/user_service.rb."""

    def test_resolve_ruby_autoload_constant_convention(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(
            module="user_service", names=["UserService"], is_relative=False
        )
        result = resolve_import_path("app/main.rb", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "app/services/user_service.rb")
        assert result == expected_id

    def test_resolve_ruby_convention_from_feature_name(
        self, file_index: dict[str, str]
    ) -> None:
        """A snake_case feature without an explicit constant still resolves."""
        imp = ImportInfo(module="user_service", names=[], is_relative=False)
        result = resolve_import_path("app/main.rb", imp, file_index)

        expected_id = generate_id(NodeLabel.FILE, "app/services/user_service.rb")
        assert result == expected_id


class TestResolveRubyExternal:
    """Gem requires and missing files resolve to None."""

    def test_resolve_ruby_gem_require(self, file_index: dict[str, str]) -> None:
        imp = ImportInfo(module="rails", names=[], is_relative=False)
        result = resolve_import_path("app/main.rb", imp, file_index)
        assert result is None

    def test_resolve_ruby_missing_relative(
        self, file_index: dict[str, str]
    ) -> None:
        imp = ImportInfo(module="../lib/missing", names=[], is_relative=True)
        result = resolve_import_path("app/main.rb", imp, file_index)
        assert result is None


# ---------------------------------------------------------------------------
# process_imports — Integration
# ---------------------------------------------------------------------------


class TestProcessImportsCreatesRelationships:
    """process_imports creates IMPORTS edges in the graph."""

    def test_process_imports_creates_relationships(
        self, graph: KnowledgeGraph
    ) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1

        rel = imports_rels[0]
        assert rel.source == generate_id(NodeLabel.FILE, "src/auth/validate.py")
        assert rel.target == generate_id(NodeLabel.FILE, "src/auth/utils.py")
        assert rel.properties["symbols"] == "helper"

    def test_process_imports_relationship_id_format(
        self, graph: KnowledgeGraph
    ) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1
        assert imports_rels[0].id.startswith("imports:")
        assert "->" in imports_rels[0].id

    def test_process_imports_skips_external(
        self, graph: KnowledgeGraph
    ) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(module="os", names=["path"], is_relative=False),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 0

    def test_process_imports_multiple_files(
        self, graph: KnowledgeGraph
    ) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
            FileParseData(
                file_path="lib/index.ts",
                language="typescript",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module="./utils",
                            names=["foo"],
                            is_relative=False,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 2

    def test_process_imports_ruby_require_relative(
        self, graph: KnowledgeGraph
    ) -> None:
        parse_data = [
            FileParseData(
                file_path="app/main.rb",
                language="ruby",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module="../lib/foo",
                            names=[],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1
        rel = imports_rels[0]
        assert rel.source == generate_id(NodeLabel.FILE, "app/main.rb")
        assert rel.target == generate_id(NodeLabel.FILE, "lib/foo.rb")


class TestProcessImportsNoDuplicates:
    """Same import twice does not create duplicate edges."""

    def test_process_imports_no_duplicates(
        self, graph: KnowledgeGraph
    ) -> None:
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                        ImportInfo(
                            module=".utils",
                            names=["other_func"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1

    def test_process_imports_no_duplicates_across_parse_data(
        self, graph: KnowledgeGraph
    ) -> None:
        """Duplicates are also prevented across separate FileParseData entries
        for the same file (e.g. if the same file appears twice)."""
        parse_data = [
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
            FileParseData(
                file_path="src/auth/validate.py",
                language="python",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module=".utils",
                            names=["helper"],
                            is_relative=True,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, graph)

        imports_rels = graph.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1


# ---------------------------------------------------------------------------
# W2.5b equivalence + scale — Ruby basename resolution indexing
#
# ``_resolve_ruby_by_basename_reference`` below is a frozen, verbatim copy
# of the pre-W2.5b per-lookup scan+sort implementation (it used to be
# ``_resolve_ruby_by_basename`` in source). It is deliberately NOT imported
# from source -- the whole point is to pin the *old* behaviour independently
# of whatever ``_RubyBasenameIndex`` does now, so a future edit to the real
# implementation can't accidentally make this test compare an
# implementation against itself.
# ---------------------------------------------------------------------------


def _resolve_ruby_by_basename_reference(
    name: str, file_index: dict[str, str]
) -> str | None:
    """Frozen copy of the pre-W2.5b scan-the-whole-index-per-lookup
    implementation. Same sort key as today: the lexicographically-first
    *node id* (not path) among files sharing the target basename wins.
    """
    target = PurePosixPath(name).name
    if not target.endswith(".rb"):
        target = f"{target}.rb"

    matches = sorted(
        node_id
        for path, node_id in file_index.items()
        if PurePosixPath(path).name == target
    )
    return matches[0] if matches else None


def _build_large_ruby_file_index() -> dict[str, str]:
    """A 130+ file index with several basenames that are deliberately
    ambiguous (same basename, multiple directories) so the "which node id
    wins" tie-break is actually exercised.
    """
    file_index: dict[str, str] = {}

    ambiguous: dict[str, list[str]] = {
        "user_service": [
            "app/services",
            "lib/legacy",
            "vendor/gems/foo/lib",
            "z_shadow/services",
        ],
        "base_controller": [
            "app/controllers",
            "app/controllers/api",
            "lib/legacy/controllers",
        ],
        "http_client": ["lib/net", "app/clients", "vendor/http_client/lib"],
    }
    for basename, dirs in ambiguous.items():
        for d in dirs:
            path = f"{d}/{basename}.rb"
            file_index[path] = generate_id(NodeLabel.FILE, path)

    # Pad out to 130+ unambiguous files.
    for i in range(120):
        path = f"app/models/model_{i:03d}.rb"
        file_index[path] = generate_id(NodeLabel.FILE, path)

    return file_index


class TestRubyBasenameIndexEquivalence:
    """The indexed lookup must match the frozen per-lookup scan+sort
    reference on a large file index with ambiguous basenames.
    """

    def test_matches_reference_for_ambiguous_and_unique_names(self) -> None:
        file_index = _build_large_ruby_file_index()
        assert len(file_index) >= 100

        lookups = [
            "user_service",
            "UserService",
            "base_controller",
            "BaseController",
            "http_client",
            "HTTPClient",
            "model_000",
            "model_042",
            "model_119",
            "app/services/user_service",  # namespaced-looking input
            "does_not_exist",
            "does/not/exist_either",
        ]

        index = _RubyBasenameIndex(file_index)
        for name in lookups:
            expected = _resolve_ruby_by_basename_reference(name, file_index)
            actual = index.resolve(name)
            assert actual == expected, f"mismatch for {name!r}"

    def test_ambiguous_winner_is_lexicographically_first_node_id(self) -> None:
        """Pin the actual winning value (not just old==new) for one
        ambiguous basename, so a silent shift in tie-break key (e.g. path
        instead of node id) would be caught even if both implementations
        drifted together.
        """
        file_index = _build_large_ruby_file_index()
        expected_winner = min(
            generate_id(NodeLabel.FILE, f"{d}/user_service.rb")
            for d in ("app/services", "lib/legacy", "vendor/gems/foo/lib", "z_shadow/services")
        )

        index = _RubyBasenameIndex(file_index)
        assert index.resolve("user_service") == expected_winner
        assert _resolve_ruby_by_basename_reference("user_service", file_index) == expected_winner

    def test_end_to_end_through_process_imports(self) -> None:
        """The winner selection also holds through the full public API
        (resolve_import_path -> _resolve_ruby -> _RubyBasenameIndex), not
        just the indexed helper in isolation.
        """
        g = KnowledgeGraph()
        paths = [
            "app/services/user_service.rb",
            "lib/legacy/user_service.rb",
            "app/main.rb",
        ]
        for path in paths:
            g.add_node(
                GraphNode(
                    id=generate_id(NodeLabel.FILE, path),
                    label=NodeLabel.FILE,
                    name=path.rsplit("/", 1)[-1],
                    file_path=path,
                    language="ruby",
                )
            )

        parse_data = [
            FileParseData(
                file_path="app/main.rb",
                language="ruby",
                parse_result=ParseResult(
                    imports=[
                        ImportInfo(
                            module="user_service",
                            names=["UserService"],
                            is_relative=False,
                        ),
                    ],
                ),
            ),
        ]

        process_imports(parse_data, g)

        imports_rels = g.get_relationships_by_type(RelType.IMPORTS)
        assert len(imports_rels) == 1
        expected_winner = min(
            generate_id(NodeLabel.FILE, "app/services/user_service.rb"),
            generate_id(NodeLabel.FILE, "lib/legacy/user_service.rb"),
        )
        assert imports_rels[0].target == expected_winner


class TestRubyBasenameIndexScale:
    """The indexed lookup must stay fast across many repeated lookups
    against a large file index -- the scenario that made the old
    scan-the-whole-index-per-lookup implementation expensive.
    """

    def test_many_lookups_on_shared_index_complete_quickly(self) -> None:
        file_index = _build_large_ruby_file_index()
        # 600 lookups (120 distinct basenames, repeated 5x) against a
        # single shared index, mirroring how process_imports reuses one
        # _RubyBasenameIndex across an entire run.
        names = [f"model_{i:03d}" for i in range(120)] * 5
        assert len(names) == 600

        index = _RubyBasenameIndex(file_index)

        start = time.perf_counter()
        results = [index.resolve(name) for name in names]
        elapsed = time.perf_counter() - start

        assert all(r is not None for r in results)
        # Generous bound -- only needs to catch a regression back to an
        # O(files) scan per lookup, not chase a tight benchmark.
        assert elapsed < 2.0
