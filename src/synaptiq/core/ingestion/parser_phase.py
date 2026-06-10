"""Phase 3: Code parsing for Synaptiq.

Takes file entries from the walker, parses each one with the appropriate
tree-sitter parser, and adds symbol nodes (Function, Class, Method, Interface,
TypeAlias, Enum) to the knowledge graph with DEFINES relationships from File
to Symbol.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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
}

@dataclass
class FileParseData:
    """Parse results for a single file, kept for later phases."""

    file_path: str
    language: str
    parse_result: ParseResult
    content: str = ""

# Parser instances are cached per (thread, language).  tree-sitter parser
# objects are stateful and documented as single-thread-at-a-time, and
# ``process_parsing`` calls this from a thread pool — sharing one parser
# instance across threads would race inside the native parse call.
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

    else:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Expected one of: python, typescript, javascript"
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

def process_parsing(
    files: list[FileEntry],
    graph: KnowledgeGraph,
    max_workers: int = 8,
) -> list[FileParseData]:
    """Parse every file and populate the knowledge graph with symbol nodes.

    Parsing is done in parallel using a thread pool (tree-sitter releases
    the GIL during C parsing).  Graph mutation remains sequential since
    :class:`KnowledgeGraph` is not thread-safe.

    For each symbol discovered during parsing a graph node is created with
    the appropriate label (Function, Class, Method, etc.) and a DEFINES
    relationship is added from the owning File node to the new symbol node.

    Args:
        files: File entries produced by the walker phase.
        graph: The knowledge graph to populate.  File nodes are expected to
            already exist (created by the structure phase).
        max_workers: Maximum number of threads for parallel parsing.

    Returns:
        A list of :class:`FileParseData` objects that carry the full parse
        results (imports, calls, heritage, type_refs) for use by later phases.
    """
    # Phase 1: Parse all files in parallel.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_parse_data = list(
            executor.map(
                lambda f: parse_file(f.path, f.content, f.language),
                files,
            )
        )

    # Phase 2: Graph mutation (sequential — not thread-safe).
    for file_entry, parse_data in zip(files, all_parse_data):
        file_id = generate_id(NodeLabel.FILE, file_entry.path)
        exported_names: set[str] = set(parse_data.parse_result.exports)
        symbol_ids = assign_symbol_ids(parse_data.parse_result.symbols, file_entry.path)

        # Build class -> base class names for storing on class nodes.
        class_bases: dict[str, list[str]] = {}
        for cls_name, kind, parent_name in parse_data.parse_result.heritage:
            if kind == "extends":
                class_bases.setdefault(cls_name, []).append(parent_name)

        for symbol, symbol_id in zip(parse_data.parse_result.symbols, symbol_ids):
            if symbol_id is None:
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
