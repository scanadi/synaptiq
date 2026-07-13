"""Phase 4: Import resolution for Synaptiq.

Takes the FileParseData produced by the parsing phase and resolves import
statements to actual File nodes in the knowledge graph, creating IMPORTS
relationships between the importing file and the target file.
"""

from __future__ import annotations

import logging
import posixpath
import re
from collections import defaultdict
from pathlib import PurePosixPath

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from synaptiq.core.ingestion.parser_phase import FileParseData
from synaptiq.core.parsers.base import ImportInfo

logger = logging.getLogger(__name__)

_JS_TS_EXTENSIONS = (".ts", ".js", ".tsx", ".jsx", ".mjs", ".cjs")
_RUBY_EXTENSIONS = (".rb", ".rake", ".ru", ".gemspec", ".rbi")

#: Extensions whose stem forms a module name (stripped before computing forms).
_MODULE_EXTENSIONS = frozenset({".py", ".go", *_JS_TS_EXTENSIONS, *_RUBY_EXTENSIONS})
#: Basenames that name their *package* (parent dir) rather than a leaf module.
_PACKAGE_INIT_STEMS = frozenset({"__init__", "index"})


def _module_forms(spec: str) -> set[str]:
    """Normalize an import spec (or a path stem) into comparable module forms.

    Reuses the resolution normalization (dots↔slashes) so the added-file closure
    (:func:`~synaptiq.core.ingestion.incremental.plan_incremental`) can match an
    unchanged file's *unresolved* import string against a newly-added file's
    :func:`importable_identities`. Dots become slashes (Python dotted → path),
    leading relative markers are stripped, and every trailing path *suffix* is
    emitted (both slashed and dotted) so a source-root layout — ``src/pkg/foo``
    imported as ``pkg.foo`` — still matches. Deliberately over-broad: extra forms
    only cost a cheap, correct re-resolution, a missed form costs staleness.
    """
    if not spec:
        return set()
    normalized = spec.replace(".", "/").replace("\\", "/").strip("/")
    segments = [s for s in normalized.split("/") if s]
    forms: set[str] = set()
    for i in range(len(segments)):
        suffix = segments[i:]
        forms.add("/".join(suffix))
        forms.add(".".join(suffix))
    return forms


def importable_identities(path: str) -> set[str]:
    """Module forms an import could use to resolve to *path*.

    The inverse of :func:`resolve_import_path`, used by the incremental planner's
    added-file dependent closure: when *path* is a brand-new file, any unchanged
    file whose recorded ``unresolved_imports`` normalize (via :func:`_module_forms`)
    to one of these forms must be re-resolved so the now-satisfiable import links
    up. ``__init__.py`` / ``index.ts`` additionally name their parent package.
    """
    pure = PurePosixPath(path)
    noext = str(pure.with_suffix("")) if pure.suffix.lower() in _MODULE_EXTENSIONS else str(pure)
    forms = _module_forms(noext)
    if pure.stem in _PACKAGE_INIT_STEMS:
        parent = str(pure.parent)
        if parent and parent != ".":
            forms |= _module_forms(parent)
    return forms


def build_file_index(graph: KnowledgeGraph) -> dict[str, str]:
    """Build an index mapping file paths to their graph node IDs.

    Iterates over all :pyclass:`NodeLabel.FILE` nodes in the graph and
    returns a dict keyed by ``file_path`` with node ``id`` as value.

    Args:
        graph: The knowledge graph containing File nodes.

    Returns:
        A dict like ``{"src/auth/validate.py": "file:src/auth/validate.py:"}``.
    """
    file_nodes = graph.get_nodes_by_label(NodeLabel.FILE)
    return {node.file_path: node.id for node in file_nodes}

def _detect_source_roots(file_index: dict[str, str]) -> set[str]:
    """Detect Python source root directories (e.g. ``src/``) from the file index.

    A source root is a directory whose children have ``__init__.py`` but the
    directory itself does not, indicating a ``src/`` layout.
    """
    init_dirs: set[str] = set()
    for path in file_index:
        if path.endswith("/__init__.py"):
            init_dirs.add(str(PurePosixPath(path).parent))

    roots: set[str] = set()
    for d in init_dirs:
        parent = str(PurePosixPath(d).parent)
        if parent != "." and parent not in init_dirs:
            roots.add(parent)
    return roots


def resolve_import_path(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
    source_roots: set[str] | None = None,
    ruby_basename_index: _RubyBasenameIndex | None = None,
    go_package_index: _GoPackageIndex | None = None,
) -> str | None:
    """Resolve an import statement to the target file's node ID.

    Uses the importing file's path, the parsed :class:`ImportInfo`, and the
    index of all known project files to determine which file is being
    imported.  Returns ``None`` for external/unresolvable imports.

    Args:
        importing_file: Relative path of the file containing the import
            (e.g. ``"src/auth/validate.py"``).
        import_info: The parsed import information.
        file_index: Mapping of relative file paths to their graph node IDs.
        source_roots: Optional set of detected source roots for resolving
            imports in ``src/`` layout projects.
        ruby_basename_index: Optional shared :class:`_RubyBasenameIndex` for
            Ruby autoload-convention fallback resolution. Callers resolving
            many imports against the same *file_index* (e.g.
            :func:`process_imports`) should build one instance and pass it
            to every call so the underlying basename index is built at most
            once. When omitted, a throwaway instance is created per call.

    Returns:
        The node ID of the resolved target file, or ``None`` if the import
        cannot be resolved to a file in the project.
    """
    language = _detect_language(importing_file)

    if language == "python":
        return _resolve_python(importing_file, import_info, file_index, source_roots)
    if language in ("typescript", "javascript"):
        return _resolve_js_ts(importing_file, import_info, file_index)
    if language == "ruby":
        return _resolve_ruby(importing_file, import_info, file_index, ruby_basename_index)
    if language == "go":
        # A Go package import resolves to every .go file in the package
        # directory; this single-result entry point returns the first (see
        # _resolve_import_targets / process_imports for the full fan-out).
        targets = _resolve_go(import_info, file_index, go_package_index)
        return targets[0] if targets else None

    return None


def _resolve_import_targets(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
    source_roots: set[str] | None,
    ruby_basename_index: _RubyBasenameIndex | None,
    go_package_index: _GoPackageIndex | None,
) -> list[str]:
    """Return **all** target file node IDs an import resolves to.

    Go package imports fan out to every ``.go`` file in the target package
    directory (one IMPORTS edge per file, mirroring the file-level granularity
    of Python package resolution and letting :mod:`calls` resolve a
    ``pkg.Symbol`` call against whichever file of the package defines it).
    Every other language resolves to at most one file, so its single result is
    wrapped in a list (or an empty list when unresolved).
    """
    if _detect_language(importing_file) == "go":
        return _resolve_go(import_info, file_index, go_package_index)
    single = resolve_import_path(
        importing_file, import_info, file_index, source_roots, ruby_basename_index
    )
    return [single] if single else []

def process_imports(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
) -> None:
    """Resolve imports and create IMPORTS relationships in the graph.

    For each file's parsed imports, resolves the target file and creates
    an ``IMPORTS`` relationship from the importing file node to the target
    file node.  Duplicate edges (same source -> same target) are skipped.

    Args:
        parse_data: Parse results from the parsing phase.
        graph: The knowledge graph to populate with IMPORTS relationships.
    """
    file_index = build_file_index(graph)
    source_roots = _detect_source_roots(file_index)
    seen: set[tuple[str, str]] = set()
    # Shared across the whole run: the underlying basename->candidates index
    # is built at most once, lazily, on the first Ruby require that falls
    # through to the autoload-convention fallback (see _RubyBasenameIndex).
    ruby_basename_index = _RubyBasenameIndex(file_index)
    # Shared across the run for Go package resolution (built lazily on the
    # first Go import; see _GoPackageIndex).
    go_package_index = _GoPackageIndex(file_index)

    for fpd in parse_data:
        source_file_id = generate_id(NodeLabel.FILE, fpd.file_path)
        # Import module strings that resolved to no project file — stashed on the
        # File node so the manifest can record them (added-file closure fix). Only
        # this loop has both the parsed import and the resolution verdict, so it
        # is the one place that can capture "referenced but not present (yet)".
        unresolved: list[str] = []

        for imp in fpd.parse_result.imports:
            target_ids = _resolve_import_targets(
                fpd.file_path, imp, file_index, source_roots,
                ruby_basename_index, go_package_index,
            )
            if not target_ids:
                if imp.module:
                    unresolved.append(imp.module)
                continue
            for target_id in target_ids:
                # A Go package fans out to every file in its directory; never
                # link a file to itself.
                if target_id == source_file_id:
                    continue

                pair = (source_file_id, target_id)
                if pair in seen:
                    continue
                seen.add(pair)

                rel_id = f"imports:{source_file_id}->{target_id}"
                graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=RelType.IMPORTS,
                        source=source_file_id,
                        target=target_id,
                        properties={"symbols": ",".join(imp.names)},
                    )
                )

        if unresolved:
            source_node = graph.get_node(source_file_id)
            if source_node is not None:
                source_node.properties["unresolved_imports"] = sorted(set(unresolved))

def _detect_language(file_path: str) -> str:
    """Infer language from a file's extension."""
    suffix = PurePosixPath(file_path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in (".ts", ".tsx"):
        return "typescript"
    if suffix in (".js", ".jsx", ".mjs", ".cjs"):
        return "javascript"
    if suffix in _RUBY_EXTENSIONS:
        return "ruby"
    if suffix == ".go":
        return "go"
    return ""

def _resolve_python(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
    source_roots: set[str] | None = None,
) -> str | None:
    """Resolve a Python import to a file node ID.

    Handles:
    - Relative imports (``is_relative=True``): dot-prefixed module paths
      resolved relative to the importing file's directory.
    - Absolute imports: treated as dotted paths from the project root,
      with fallback to detected source roots (e.g. ``src/``).

    Returns ``None`` for external (not in file_index) imports.
    """
    if import_info.is_relative:
        return _resolve_python_relative(importing_file, import_info, file_index)
    return _resolve_python_absolute(import_info, file_index, source_roots)

def _resolve_python_relative(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve a relative Python import (``from .foo import bar``).

    The number of leading dots determines how many directory levels to
    traverse upward from the importing file's parent directory.

    ``from .utils import helper``  -> one dot  -> same directory
    ``from ..models import User``  -> two dots -> parent directory
    """
    module = import_info.module

    dot_count = 0
    for ch in module:
        if ch == ".":
            dot_count += 1
        else:
            break

    remainder = module[dot_count:]

    base = PurePosixPath(importing_file).parent
    for _ in range(dot_count - 1):
        base = base.parent

    if remainder:
        segments = remainder.split(".")
        target_dir = base / PurePosixPath(*segments)
    else:
        target_dir = base

    return _try_python_paths(str(target_dir), file_index)

def _resolve_python_absolute(
    import_info: ImportInfo,
    file_index: dict[str, str],
    source_roots: set[str] | None = None,
) -> str | None:
    """Resolve an absolute Python import (``from mypackage.auth import validate``).

    Converts the dotted module path to a filesystem path and looks it up
    in the file index.  If not found at the project root, tries each
    detected source root (e.g. ``src/mypackage/auth``).

    Returns ``None`` for external packages not present in the project.
    """
    module = import_info.module
    target_path = str(PurePosixPath(*module.split(".")))

    result = _try_python_paths(target_path, file_index)
    if result:
        return result

    if source_roots:
        for root in source_roots:
            result = _try_python_paths(f"{root}/{target_path}", file_index)
            if result:
                return result

    return None

def _try_python_paths(base_path: str, file_index: dict[str, str]) -> str | None:
    """Try common Python file resolution patterns for *base_path*.

    Checks in order:
    1. ``base_path.py`` (direct module file)
    2. ``base_path/__init__.py`` (package directory)
    """
    candidates = [
        f"{base_path}.py",
        f"{base_path}/__init__.py",
    ]
    for candidate in candidates:
        if candidate in file_index:
            return file_index[candidate]
    return None

def _resolve_js_ts(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
) -> str | None:
    """Resolve a JavaScript/TypeScript import to a file node ID.

    Relative imports (starting with ``./`` or ``../``) are resolved against
    the importing file's directory.  Bare specifiers (e.g. ``'express'``)
    are treated as external and return ``None``.
    """
    module = import_info.module

    if not module.startswith("."):
        return None

    base = PurePosixPath(importing_file).parent
    # posixpath.normpath collapses './' and '../' segments — PurePosixPath
    # keeps '..' literal, which would never match the file index.
    resolved_str = posixpath.normpath(str(base / module))

    return _try_js_ts_paths(resolved_str, file_index)

def _try_js_ts_paths(base_path: str, file_index: dict[str, str]) -> str | None:
    """Try common JS/TS file resolution patterns for *base_path*.

    Checks in order:
    1. ``base_path`` as-is (already has extension)
    2. ``base_path`` + each known extension (.ts, .js, .tsx, .jsx)
    3. ``base_path/index`` + each known extension
    """
    # 1. Exact match (import already includes extension).
    if base_path in file_index:
        return file_index[base_path]

    # 2. Try appending extensions.
    for ext in _JS_TS_EXTENSIONS:
        candidate = f"{base_path}{ext}"
        if candidate in file_index:
            return file_index[candidate]

    # 3. Try as directory with index file.
    for ext in _JS_TS_EXTENSIONS:
        candidate = f"{base_path}/index{ext}"
        if candidate in file_index:
            return file_index[candidate]

    return None

def _resolve_ruby(
    importing_file: str,
    import_info: ImportInfo,
    file_index: dict[str, str],
    basename_index: _RubyBasenameIndex | None = None,
) -> str | None:
    """Resolve a Ruby ``require``/``require_relative``/``autoload`` to a file node.

    Handles:
    - ``require_relative`` (``is_relative=True``): resolved against the
      importing file's directory (e.g. ``../lib/foo`` -> ``lib/foo.rb``).
    - ``require`` / ``autoload``: tried as a project-root-relative path first
      (e.g. ``config/settings`` -> ``config/settings.rb``), then via the Rails
      autoload naming convention — the required feature or the autoloaded
      constant is snake-cased and matched against file basenames
      (``UserService`` -> ``app/services/user_service.rb``).

    Returns ``None`` for external gems (``require "rails"``) and missing files.

    Args:
        basename_index: Optional pre-built/shared basename index for the
            autoload-convention fallback (see :class:`_RubyBasenameIndex`).
            A throwaway instance scoped to this call is used when omitted.
    """
    module = import_info.module

    if import_info.is_relative:
        base = PurePosixPath(importing_file).parent
        resolved = posixpath.normpath(str(base / module))
        return _try_ruby_paths(resolved, file_index)

    # require / autoload — first try the literal feature path against the root.
    result = _try_ruby_paths(module, file_index)
    if result:
        return result

    # Convention-based fallback: match by snake_cased basename. The required
    # feature itself may already be snake_case ("user_service"); autoload also
    # carries the CamelCase constant in ``names`` ("UserService").
    if basename_index is None:
        basename_index = _RubyBasenameIndex(file_index)

    candidates = [module, *import_info.names]
    for candidate in candidates:
        result = basename_index.resolve(_underscore(candidate))
        if result:
            return result

    return None

def _try_ruby_paths(base_path: str, file_index: dict[str, str]) -> str | None:
    """Try common Ruby file resolution patterns for *base_path*.

    Checks the path as-is (when it already carries a ``.rb`` suffix) and with
    ``.rb`` appended.
    """
    candidates = [base_path, f"{base_path}.rb"]
    for candidate in candidates:
        if candidate in file_index:
            return file_index[candidate]
    return None

def _ruby_basename_target(name: str) -> str:
    """Compute the ``.rb``-suffixed basename that *name* should match.

    ``name`` is a (possibly ``/``-namespaced) snake_case path; only the final
    segment is compared against each indexed file's basename, so
    ``user_service`` and ``app/services/user_service`` both target
    ``user_service.rb``.
    """
    target = PurePosixPath(name).name
    if not target.endswith(".rb"):
        target = f"{target}.rb"
    return target

def _build_ruby_basename_index(file_index: dict[str, str]) -> dict[str, list[str]]:
    """Group every *file_index* entry by basename for O(1) autoload lookups.

    Each bucket is sorted so that, exactly like the ``sorted(...)`` call it
    replaces, the lexicographically-first node ID wins when multiple files
    in different directories share a basename.
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    for path, node_id in file_index.items():
        buckets[PurePosixPath(path).name].append(node_id)
    for bucket in buckets.values():
        bucket.sort()
    return buckets

class _RubyBasenameIndex:
    """Memoized ``basename -> sorted [node_id, ...]`` index for Ruby autoload
    resolution.

    The previous implementation re-scanned and re-sorted the *entire* file
    index for every unresolved ``require`` (``O(files)`` per lookup). This
    class instead builds a ``basename -> sorted candidates`` grouping once
    and reuses it for every subsequent lookup (``O(1)`` after the first).

    Building the index is deferred until the first lookup — a require that
    resolves via a literal path never needs it — so projects with no
    autoload-style requires (or no Ruby at all) never pay for it. Callers
    that resolve many requires against the same *file_index* (a whole
    :func:`process_imports` run) should create a single instance and share
    it so the one-time build cost is paid at most once for the run.
    """

    def __init__(self, file_index: dict[str, str]) -> None:
        self._file_index = file_index
        self._buckets: dict[str, list[str]] | None = None

    def resolve(self, name: str) -> str | None:
        """Return the winning node ID for *name*'s basename target, if any."""
        if self._buckets is None:
            self._buckets = _build_ruby_basename_index(self._file_index)
        matches = self._buckets.get(_ruby_basename_target(name))
        return matches[0] if matches else None

def _underscore(name: str) -> str:
    """Convert a Ruby constant path to its conventional snake_case file path.

    Mirrors ActiveSupport's ``underscore``: ``UserService`` -> ``user_service``,
    ``HTTPClient`` -> ``http_client``, ``Admin::UserService`` ->
    ``admin/user_service``.
    """
    name = name.replace("::", "/")
    segments = []
    for segment in name.split("/"):
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", segment)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        segments.append(s.lower())
    return "/".join(segments)


def _resolve_go(
    import_info: ImportInfo,
    file_index: dict[str, str],
    go_package_index: _GoPackageIndex | None = None,
) -> list[str]:
    """Resolve a Go ``import`` path to the node IDs of the package's files.

    A Go import path is ``<module-prefix>/<package-dir>``; the package's
    surface is spread across *every* ``.go`` file in that directory, so the
    import resolves to all of them (one IMPORTS edge per file downstream).
    External/stdlib packages (``fmt``, ``net/http``) resolve to nothing.

    Args:
        go_package_index: Optional shared :class:`_GoPackageIndex`. A throwaway
            instance scoped to this call is built when omitted.
    """
    if go_package_index is None:
        go_package_index = _GoPackageIndex(file_index)
    return go_package_index.resolve(import_info.module)


def _build_go_package_index(file_index: dict[str, str]) -> dict[str, list[str]]:
    """Group every ``.go`` *file_index* entry by its directory.

    Each bucket is sorted for determinism. The directory keys are matched
    against import-path suffixes by :class:`_GoPackageIndex`.
    """
    buckets: dict[str, list[str]] = defaultdict(list)
    for path, node_id in file_index.items():
        if path.endswith(".go"):
            buckets[str(PurePosixPath(path).parent)].append(node_id)
    for bucket in buckets.values():
        bucket.sort()
    return buckets


class _GoPackageIndex:
    """Memoized ``package_dir -> sorted [.go node_id, ...]`` index for Go
    import resolution.

    A Go import path mirrors the on-disk package directory prefixed by the
    module path from ``go.mod``. The manifest is not a source file, so it never
    enters the graph; instead of parsing it, this index matches the import
    path's trailing segments against the discovered package directories. Go
    requires the import path to mirror the directory layout, so the **longest**
    path-suffix that names a real package directory is the target — stripping
    the module prefix falls out for free, and repos with no ``go.mod``
    (GOPATH-style) or nested modules resolve just the same.

    Building the directory grouping is deferred until the first lookup — a run
    with no Go imports never pays for it — and reused across the whole
    :func:`process_imports` run when a single instance is shared.
    """

    def __init__(self, file_index: dict[str, str]) -> None:
        self._file_index = file_index
        self._dirs: dict[str, list[str]] | None = None

    def resolve(self, import_path: str) -> list[str]:
        """Return the node IDs of every ``.go`` file in *import_path*'s package.

        Tries progressively shorter suffixes of the import path (longest first,
        i.e. most specific) and returns the first that names a known package
        directory. Returns ``[]`` for external packages.
        """
        if self._dirs is None:
            self._dirs = _build_go_package_index(self._file_index)
        if not import_path:
            return []
        parts = import_path.split("/")
        for i in range(len(parts)):
            suffix = "/".join(parts[i:])
            match = self._dirs.get(suffix)
            if match:
                return match
        return []
