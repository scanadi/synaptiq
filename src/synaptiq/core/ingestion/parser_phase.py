"""Phase 3: Code parsing for Synaptiq.

Takes file entries from the walker, parses each one with the appropriate
tree-sitter parser, and adds symbol nodes (Function, Class, Method, Interface,
TypeAlias, Enum) to the knowledge graph with DEFINES relationships from File
to Symbol.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from synaptiq.core.ingestion.walker import FileEntry
from synaptiq.core.parsers.base import LanguageParser, ParseResult

logger = logging.getLogger(__name__)

_KIND_TO_LABEL: dict[str, NodeLabel] = {
    "function": NodeLabel.FUNCTION,
    "class": NodeLabel.CLASS,
    "method": NodeLabel.METHOD,
    "interface": NodeLabel.INTERFACE,
    "type_alias": NodeLabel.TYPE_ALIAS,
    "enum": NodeLabel.ENUM,
    "module": NodeLabel.MODULE,
}

# Symbol kinds that are intentionally parsed but not materialized as graph
# nodes (no corresponding NodeLabel).  Skipped silently to avoid log noise.
_UNMAPPED_KINDS: frozenset[str] = frozenset({"constant"})


@dataclass
class FileParseData:
    """Parse results for a single file, kept for later phases."""

    file_path: str
    language: str
    parse_result: ParseResult
    content: str = ""
    # Populated by process_parsing's Phase 2 (graph mutation, parent
    # process) with the same node IDs assign_symbol_ids just computed while
    # creating the symbol nodes -- lets later phases (e.g. calls.py's
    # decorator-edge resolution) reuse them instead of recomputing via
    # assign_symbol_ids a second time over every symbol in the repo.
    # Stays ``None`` for FileParseData built outside process_parsing (direct
    # construction in tests, diff.py's scoped-diff path); a plain list of
    # str/None is picklable, and Phase 1 workers never set this field, so it
    # crosses the process boundary unset and is only populated afterwards in
    # the parent.
    symbol_ids: list[str | None] | None = None

# Parser instances are cached per (thread, language).  tree-sitter parser
# objects are stateful and documented as single-thread-at-a-time, so the
# cache is thread-local: the thread-pool fan-out gives each worker thread
# its own parser, and the process-pool fan-out gives each worker process
# its own copy of this module (hence its own ``_PARSER_LOCAL``, populated
# by that process's single task-running thread).  Either way no native
# ``Parser`` is ever touched by two threads at once.
_PARSER_LOCAL = threading.local()


def get_parser(language: str) -> LanguageParser:
    """Return the appropriate tree-sitter parser for *language*.

    Parser instances are cached per language *per thread* to avoid repeated
    instantiation while keeping each tree-sitter ``Parser`` confined to a
    single thread.

    Args:
        language: One of ``"python"``, ``"typescript"``, or ``"javascript"``.

    Returns:
        A :class:`LanguageParser` instance ready to parse source code.

    Raises:
        ValueError: If *language* is not supported.
    """
    cache: dict[str, LanguageParser] | None = getattr(_PARSER_LOCAL, "cache", None)
    if cache is None:
        cache = {}
        _PARSER_LOCAL.cache = cache

    cached = cache.get(language)
    if cached is not None:
        return cached

    if language == "python":
        from synaptiq.core.parsers.python_lang import PythonParser

        parser = PythonParser()

    elif language == "typescript":
        from synaptiq.core.parsers.typescript import TypeScriptParser

        parser = TypeScriptParser(dialect="typescript")

    elif language == "tsx":
        from synaptiq.core.parsers.typescript import TypeScriptParser

        parser = TypeScriptParser(dialect="tsx")

    elif language == "javascript":
        from synaptiq.core.parsers.typescript import TypeScriptParser

        parser = TypeScriptParser(dialect="javascript")

    elif language == "ruby":
        from synaptiq.core.parsers.ruby_lang import RubyParser

        parser = RubyParser()

    else:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Expected one of: python, typescript, javascript, ruby"
        )

    cache[language] = parser
    return parser

def assign_symbol_ids(symbols: list, file_path: str) -> list[str | None]:
    """Compute the final graph node ID for each symbol in *symbols*.

    Applies the same collision handling as :func:`process_parsing` —
    same-named duplicates get a ``#L{start_line}`` suffix — so any phase
    that needs a symbol's node ID (e.g. decorator CALLS edges) derives
    the exact ID the node was stored under.  Entries are ``None`` for
    unknown kinds or unresolvable collisions.
    """
    seen: set[str] = set()
    ids: list[str | None] = []
    for symbol in symbols:
        label = _KIND_TO_LABEL.get(symbol.kind)
        if label is None:
            ids.append(None)
            continue

        symbol_name = (
            f"{symbol.class_name}.{symbol.name}"
            if symbol.kind == "method" and symbol.class_name
            else symbol.name
        )
        symbol_id = generate_id(label, file_path, symbol_name)
        if symbol_id in seen:
            symbol_id = generate_id(label, file_path, f"{symbol_name}#L{symbol.start_line}")
            if symbol_id in seen:
                ids.append(None)
                continue
        seen.add(symbol_id)
        ids.append(symbol_id)
    return ids


def parse_file(file_path: str, content: str, language: str) -> FileParseData:
    """Parse a single file and return structured parse data.

    If parsing fails for any reason the returned :class:`FileParseData` will
    contain an empty :class:`ParseResult` so that downstream phases can
    safely skip it.

    Args:
        file_path: Relative path to the file (used for identification).
        content: Raw source code of the file.
        language: Language identifier (``"python"``, ``"typescript"``, etc.).

    Returns:
        A :class:`FileParseData` carrying the parse result.
    """
    try:
        parser = get_parser(language)
        result = parser.parse(content, file_path)
    except Exception:
        logger.warning("Failed to parse %s (%s), skipping", file_path, language, exc_info=True)
        result = ParseResult()

    return FileParseData(
        file_path=file_path, language=language, parse_result=result, content=content
    )


# Files handed to each process-pool task.  Chunking amortizes the per-task
# pickling/IPC overhead of shipping work to workers; a value in the 16-32
# range keeps chunks large enough to matter without starving workers on
# medium repos.
_PARSE_CHUNKSIZE = 32

# Minimum file count before the process pool earns its startup + IPC cost.
# Below this, spawning worker processes (and shipping file content to them)
# dominates the actual parse work, so ``process_parsing`` uses the
# in-process thread pool instead.  Small repos — and the existing unit
# tests, which parse a handful of files — therefore stay on threads.
_PROCESS_POOL_MIN_FILES = 100


def _should_use_process_pool(n_files: int, max_workers: int) -> bool:
    """Return whether process-parallel parsing is worth it for this batch.

    Processes only pay off with enough files to amortize the spawn + IPC
    overhead (``n_files >= _PROCESS_POOL_MIN_FILES``) and with more than
    one worker to spread the load across.  A single worker (e.g.
    ``--jobs 1`` or a one-core machine) always stays on threads.
    """
    return n_files >= _PROCESS_POOL_MIN_FILES and max_workers > 1


def _parse_with_threads(files: list[FileEntry], max_workers: int) -> list[FileParseData]:
    """Parse *files* on an in-process thread pool, preserving input order.

    tree-sitter releases the GIL during native parsing, so threads give
    some overlap; the pure-Python symbol extraction stays GIL-serialized
    (why the process pool exists for large repos).
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda f: parse_file(f.path, f.content, f.language),
                files,
            )
        )


def _parse_with_processes(files: list[FileEntry], max_workers: int) -> list[FileParseData]:
    """Parse *files* across worker processes, preserving input order.

    Uses an explicit ``spawn`` context: ``fork`` is unsafe once the parent
    has ever started threads (prior pools leave locks in an unknown state
    in the child), and ``spawn`` is macOS's default regardless.  Only
    picklable primitives — the path/content/language strings — cross the
    process boundary; :func:`parse_file` is a module-level function
    re-imported inside each worker, and the returned :class:`FileParseData`
    is a tree of plain, picklable dataclasses.  Files are chunked
    (:data:`_PARSE_CHUNKSIZE`) to amortize per-task IPC.  ``executor.map``
    yields results in submission order, so the returned list lines up with
    *files* exactly as the thread path does.
    """
    ctx = mp.get_context("spawn")
    paths = [f.path for f in files]
    contents = [f.content for f in files]
    languages = [f.language for f in files]
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        return list(
            executor.map(
                parse_file,
                paths,
                contents,
                languages,
                chunksize=_PARSE_CHUNKSIZE,
            )
        )


def _parse_files(files: list[FileEntry], max_workers: int) -> list[FileParseData]:
    """Parse every file, fanning out to processes when it's worth it.

    Falls back to the thread pool for small file sets, a single-worker
    limit, or when the process pool is unavailable / fails to start (e.g.
    a sandbox that forbids subprocesses, or a mid-run ``BrokenProcessPool``)
    — parallelism must never crash the pipeline.  On fallback the whole
    batch is re-parsed on threads, which is safe because :func:`parse_file`
    is pure.
    """
    if _should_use_process_pool(len(files), max_workers):
        try:
            return _parse_with_processes(files, max_workers)
        except Exception:
            logger.warning(
                "Process-parallel parsing unavailable (%d files, %d workers); "
                "falling back to thread pool",
                len(files),
                max_workers,
                exc_info=True,
            )
    return _parse_with_threads(files, max_workers)


def process_parsing(
    files: list[FileEntry],
    graph: KnowledgeGraph,
    max_workers: int | None = None,
) -> list[FileParseData]:
    """Parse every file and populate the knowledge graph with symbol nodes.

    Parsing is fanned out across worker *processes* on large repos (file
    count ``>= _PROCESS_POOL_MIN_FILES`` with more than one worker), which
    sidesteps the GIL for the pure-Python symbol extraction; it falls back
    to an in-process *thread* pool for small repos, a single-worker limit,
    or if the process pool is unavailable (see :func:`_parse_files`).
    Graph mutation always remains sequential in the parent since
    :class:`KnowledgeGraph` is not thread-safe.

    For each symbol discovered during parsing a graph node is created with
    the appropriate label (Function, Class, Method, etc.) and a DEFINES
    relationship is added from the owning File node to the new symbol node.

    Args:
        files: File entries produced by the walker phase.
        graph: The knowledge graph to populate.  File nodes are expected to
            already exist (created by the structure phase).
        max_workers: Maximum number of parallel parse workers (processes or
            threads).  Defaults to ``None``, which resolves
            ``current_limits().pool_workers`` at call time (``min(8,
            cpu_count)`` unless capped further by ``analyze --jobs``) —
            pass an explicit value to override.

    Returns:
        A list of :class:`FileParseData` objects that carry the full parse
        results (imports, calls, heritage, type_refs) for use by later phases.
    """
    if max_workers is None:
        from synaptiq.core.resources import current_limits

        max_workers = current_limits().pool_workers

    # Phase 1: Parse all files in parallel — processes on large repos,
    # threads on small repos or when the process pool is unavailable.
    all_parse_data = _parse_files(files, max_workers)

    # Phase 2: Graph mutation (sequential — not thread-safe).
    for file_entry, parse_data in zip(files, all_parse_data):
        file_id = generate_id(NodeLabel.FILE, file_entry.path)
        exported_names: set[str] = set(parse_data.parse_result.exports)
        symbol_ids = assign_symbol_ids(parse_data.parse_result.symbols, file_entry.path)
        # Carry the computed IDs forward on the returned FileParseData (see
        # the field docstring) so calls.py doesn't have to recompute them.
        parse_data.symbol_ids = symbol_ids

        # Build class -> base class names for storing on class nodes.
        class_bases: dict[str, list[str]] = {}
        for cls_name, kind, parent_name in parse_data.parse_result.heritage:
            if kind == "extends":
                class_bases.setdefault(cls_name, []).append(parent_name)

        for symbol, symbol_id in zip(parse_data.parse_result.symbols, symbol_ids):
            if symbol_id is None:
                if symbol.kind in _UNMAPPED_KINDS:
                    continue
                logger.warning(
                    "Unknown symbol kind %r for %s in %s, skipping",
                    symbol.kind,
                    symbol.name,
                    file_entry.path,
                )
                continue
            label = _KIND_TO_LABEL[symbol.kind]

            props: dict[str, Any] = {}
            if symbol.decorators:
                props["decorators"] = symbol.decorators
            if symbol.kind == "class" and symbol.name in class_bases:
                props["bases"] = class_bases[symbol.name]

            is_exported = symbol.name in exported_names

            graph.add_node(
                GraphNode(
                    id=symbol_id,
                    label=label,
                    name=symbol.name,
                    file_path=file_entry.path,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    content=symbol.content,
                    signature=symbol.signature,
                    class_name=symbol.class_name,
                    language=file_entry.language,
                    is_exported=is_exported,
                    properties=props,
                )
            )

            rel_id = f"defines:{file_id}->{symbol_id}"
            graph.add_relationship(
                GraphRelationship(
                    id=rel_id,
                    type=RelType.DEFINES,
                    source=file_id,
                    target=symbol_id,
                )
            )

    return all_parse_data
