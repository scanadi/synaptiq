"""Phase 5: Call tracing for Synaptiq.

Takes FileParseData from the parser phase and resolves call expressions to
target symbol nodes, creating CALLS relationships with confidence scores.

Resolution priority:
1. Same-file exact match (confidence 1.0)
2. Import-resolved match (confidence 1.0)
3. Global fuzzy match (confidence 0.5)
4. Receiver method resolution (confidence 0.8)
"""

from __future__ import annotations

import logging

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from synaptiq.core.ingestion.parser_phase import FileParseData, assign_symbol_ids
from synaptiq.core.ingestion.symbol_lookup import (
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)
from synaptiq.core.parsers.base import CallInfo

logger = logging.getLogger(__name__)

_CALLABLE_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
)

# Confidence assigned to weak references (object-literal shorthand) that
# only resolve via global fuzzy matching.  Low enough to read as "uncertain"
# in every consumer, but the edge still exists — dropping it entirely made
# dead-code detection flag symbols whose only reference was the shorthand.
_WEAK_REF_CONFIDENCE = 0.3

# Names that should never produce CALLS edges.  These are language builtins,
# stdlib utilities, framework hooks, and common JS/TS globals whose definitions
# do not exist in the user's codebase.  Filtering them before resolution
# prevents low-confidence global-fuzzy matches against short, common names.
_CALL_BLOCKLIST: frozenset[str] = frozenset({
    # Python builtins
    "print", "len", "range", "map", "filter", "sorted", "list", "dict",
    "set", "str", "int", "float", "bool", "type", "super", "isinstance",
    "issubclass", "hasattr", "getattr", "setattr", "open", "iter", "next",
    "zip", "enumerate", "any", "all", "min", "max", "sum", "abs", "round",
    "repr", "id", "hash", "dir", "vars", "input", "format", "tuple",
    "frozenset", "bytes", "bytearray", "memoryview", "object", "property",
    "classmethod", "staticmethod", "delattr", "callable", "compile", "eval",
    "exec", "globals", "locals", "breakpoint", "exit", "quit",
    # Python stdlib — common method names that collide with user-defined symbols
    "append", "extend", "update", "pop", "get", "items", "keys", "values",
    "split", "join", "strip", "replace", "startswith", "endswith", "lower",
    "upper", "encode", "decode", "read", "write", "close",
    # JS/TS built-in globals
    "console", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "JSON", "Array", "Object", "Promise", "Math", "Date", "Error", "Symbol",
    "parseInt", "parseFloat", "isNaN", "isFinite", "encodeURIComponent",
    "decodeURIComponent", "fetch", "require", "exports", "module",
    "document", "window", "process", "Buffer", "URL",
    # JS/TS dotted method names extracted as bare call names
    "log", "error", "warn", "info", "debug",
    "parse", "stringify",
    "assign", "freeze",
    "isArray", "from", "of",
    "resolve", "reject", "race",
    "floor", "ceil", "random",
    # React hooks
    "useState", "useEffect", "useRef", "useCallback", "useMemo",
    "useContext", "useReducer", "useLayoutEffect", "useImperativeHandle",
    "useDebugValue", "useId", "useTransition", "useDeferredValue",
})

# Ruby Kernel / Enumerable builtins and common framework macros whose
# definitions do not live in the user's codebase.  These are kept separate
# from ``_CALL_BLOCKLIST`` and applied ONLY to Ruby files: many of these names
# (``find``, ``select``, ``count``, ``merge``, ``send``, ``first``, ``sort`` …)
# are perfectly ordinary user-defined function/method names in Python and
# TS/JS, so blocklisting them globally would silently drop legitimate CALLS
# edges in non-Ruby codebases.
_RUBY_CALL_BLOCKLIST: frozenset[str] = frozenset({
    "puts", "p", "pp", "require_relative", "autoload", "load",
    "attr_accessor", "attr_reader", "attr_writer", "include", "prepend",
    "raise", "fail", "throw", "catch", "loop", "lambda", "proc", "send",
    "public_send", "respond_to?", "instance_variable_get",
    "instance_variable_set", "define_method", "method_missing",
    "new", "dup", "clone", "to_s", "to_sym", "to_a", "to_h", "to_i", "to_f",
    "each", "each_with_index", "each_with_object", "select", "reject",
    "reduce", "inject", "find", "detect", "collect", "flat_map", "sort",
    "sort_by", "group_by", "count", "first", "last", "push", "concat",
    "fetch", "merge", "key?", "nil?", "empty?", "blank?", "present?",
    "freeze!", "tap", "then", "yield_self",
})

# Go builtin functions whose definitions do not live in the user's codebase.
# Kept separate from ``_CALL_BLOCKLIST`` and applied ONLY to Go files: several
# of these names (``make``, ``new``, ``copy``, ``delete``, ``close``, ``min``,
# ``max``, ``clear``, ``append`` ...) are perfectly ordinary user-defined
# function names in Python/TS/Ruby, so blocklisting them globally would drop
# legitimate CALLS edges in non-Go codebases.
_GO_CALL_BLOCKLIST: frozenset[str] = frozenset({
    "append", "cap", "clear", "close", "complex", "copy", "delete", "imag",
    "len", "make", "max", "min", "new", "panic", "print", "println", "real",
    "recover",
})


def _build_import_cache(
    file_path: str,
    graph: KnowledgeGraph,
) -> dict[str, set[str]]:
    """Build {symbol_name -> set of imported file_paths} for a file.

    The special key ``"*"`` contains file paths from wildcard/full-module imports.
    """
    source_file_id = generate_id(NodeLabel.FILE, file_path)
    import_rels = graph.get_outgoing(source_file_id, RelType.IMPORTS)

    cache: dict[str, set[str]] = {}
    for rel in import_rels:
        target_node = graph.get_node(rel.target)
        if target_node is None:
            continue
        symbols_str = rel.properties.get("symbols", "")
        imported_names = {s.strip() for s in symbols_str.split(",") if s.strip()}
        if not imported_names:
            cache.setdefault("*", set()).add(target_node.file_path)
        else:
            for sym_name in imported_names:
                cache.setdefault(sym_name, set()).add(target_node.file_path)
    return cache


def _build_call_index_by_file(
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
) -> dict[str, dict[str, list[str]]]:
    """Group *call_index* candidates into ``{name: {file_path: [node_ids]}}``.

    ``resolve_call`` and ``_resolve_self_method`` both need to answer "is
    there a symbol named X defined in THIS file" -- previously a linear scan
    over every same-name candidate in the whole repo.  For a common name
    that recurs across many files (``run``, ``get``, ``process``, ...) that
    scan cost is O(repo-wide candidates) *per call site*.  Building this
    index once up front turns each same-file lookup into an O(1) dict hit.

    Iterates each name's candidates in *call_index*'s existing order, so
    every per-file bucket preserves the original candidate order -- required
    for callers whose resolution picks the first matching candidate.
    """
    by_file: dict[str, dict[str, list[str]]] = {}
    for name, candidate_ids in call_index.items():
        buckets: dict[str, list[str]] = {}
        for nid in candidate_ids:
            node = graph.get_node(nid)
            if node is None:
                continue
            buckets.setdefault(node.file_path, []).append(nid)
        by_file[name] = buckets
    return by_file


def resolve_call(
    call: CallInfo,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    caller_class_name: str | None = None,
    import_cache: dict[str, set[str]] | None = None,
    call_index_by_file: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[str | None, float]:
    """Resolve a call expression to a target node ID and confidence score.

    Resolution strategy (tried in order):

    1. **Same-file exact match** (confidence 1.0) -- the called symbol is
       defined in the same file as the caller.
    2. **Import-resolved match** (confidence 1.0) -- the called name was
       imported into this file; find the symbol in the imported file.
    3. **Global fuzzy match** (confidence 0.5) -- any symbol with this name
       anywhere in the codebase.  If multiple matches exist, the one sharing
       the longest directory prefix with the caller is preferred.
       Skipped when there are more than 5 candidates (too ambiguous).

    For method calls (``call.receiver`` is non-empty):
    - If the receiver is ``"self"`` or ``"this"``, look for a method with
      that name in the same class (same file, matching class_name).
    - Otherwise, try to resolve the method name globally.

    Args:
        call: The parsed call information.
        file_path: Path to the file containing the call.
        call_index: Mapping from symbol names to node IDs built by
            :func:`build_name_index`.
        graph: The knowledge graph.
        caller_class_name: Optional class name of the calling symbol,
            used to scope ``self``/``this`` method resolution.
        import_cache: Optional pre-built import cache for the file.
        call_index_by_file: Optional ``{name: {file_path: [node_ids]}}``
            index built by :func:`_build_call_index_by_file`.  When given,
            same-file resolution (step 1 below, and ``self``/``this``
            resolution) is an O(1) dict hit instead of scanning every
            same-name candidate across the repo.  Falls back to the linear
            scan over *candidate_ids* when omitted, so direct callers (e.g.
            tests) keep working unchanged.

    Returns:
        A tuple of ``(node_id, confidence)`` or ``(None, 0.0)`` if the
        call cannot be resolved.
    """
    name = call.name
    receiver = call.receiver

    if receiver in ("self", "this"):
        result = _resolve_self_method(
            name,
            file_path,
            call_index,
            graph,
            caller_class_name,
            call_index_by_file=call_index_by_file,
        )
        if result is not None:
            return result, 1.0

    # Without type info the receiver doesn't help — fall through to name-based resolution.
    candidate_ids = call_index.get(name, [])
    if not candidate_ids:
        return None, 0.0

    # 1. Same-file exact match.  With the auxiliary by-file index this is a
    # single dict hit; without it, fall back to the original linear scan
    # over every same-name candidate across the repo.
    if call_index_by_file is not None:
        same_file_ids = call_index_by_file.get(name, {}).get(file_path)
        if same_file_ids:
            return same_file_ids[0], 1.0
    else:
        for nid in candidate_ids:
            node = graph.get_node(nid)
            if node is not None and node.file_path == file_path:
                return nid, 1.0

    # 2. Import-resolved match.
    effective_cache = import_cache if import_cache is not None else _build_import_cache(
        file_path, graph
    )
    imported_target = _resolve_via_imports(name, candidate_ids, graph, effective_cache)
    if imported_target is not None:
        return imported_target, 1.0

    # 3. Global fuzzy match — skip when too many candidates (ambiguous).
    if len(candidate_ids) > 5:
        return None, 0.0
    return _pick_closest(candidate_ids, graph, caller_file_path=file_path), 0.5

def _resolve_self_method(
    method_name: str,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    caller_class_name: str | None = None,
    call_index_by_file: dict[str, dict[str, list[str]]] | None = None,
) -> str | None:
    """Find a method with *method_name* in the same file and class.

    When the receiver is ``self`` or ``this`` the target must be a Method
    node defined in the same file.  If *caller_class_name* is provided,
    candidates are further filtered to the same class.

    When *call_index_by_file* is given, the candidate list is pre-scoped to
    *file_path* (O(1) dict hit) instead of scanning every *method_name*
    candidate across the repo; the filtering loop below is otherwise
    unchanged (and still re-checks ``file_path`` so behavior is identical
    either way).
    """
    if call_index_by_file is not None:
        candidate_ids = call_index_by_file.get(method_name, {}).get(file_path, [])
    else:
        candidate_ids = call_index.get(method_name, [])

    fallback: str | None = None
    for nid in candidate_ids:
        node = graph.get_node(nid)
        if (
            node is not None
            and node.label == NodeLabel.METHOD
            and node.file_path == file_path
        ):
            if caller_class_name and node.class_name == caller_class_name:
                return nid
            if fallback is None:
                fallback = nid
    return fallback

def _resolve_via_imports(
    name: str,
    candidate_ids: list[str],
    graph: KnowledgeGraph,
    import_cache: dict[str, set[str]],
) -> str | None:
    """Check if *name* was imported and resolve to the target using cached data.

    Uses the pre-built *import_cache* (from :func:`_build_import_cache`)
    to avoid re-scanning IMPORTS relationships for every call in the same file.
    """
    if not import_cache:
        return None

    imported_file_paths = import_cache.get(name, set()) | import_cache.get("*", set())
    if not imported_file_paths:
        return None

    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path in imported_file_paths:
            return nid

    return None


def _common_prefix_len(a: str, b: str) -> int:
    """Return the length of the common directory prefix between two paths."""
    parts_a = a.split("/")
    parts_b = b.split("/")
    common = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            common += 1
        else:
            break
    return common


def _pick_closest(
    candidate_ids: list[str],
    graph: KnowledgeGraph,
    caller_file_path: str = "",
) -> str | None:
    """Pick the candidate sharing the longest directory prefix with the caller.

    Falls back to shortest file path when no caller path is provided.
    Returns ``None`` if no candidates can be resolved to actual nodes.
    """
    best_id: str | None = None
    best_score: tuple[int, int] = (-1, 0)

    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is None:
            continue
        if caller_file_path:
            prefix = _common_prefix_len(caller_file_path, node.file_path)
            score = (prefix, -len(node.file_path))
        else:
            score = (0, -len(node.file_path))
        if score > best_score:
            best_score = score
            best_id = nid

    return best_id


def _add_calls_edge(
    source_id: str,
    target_id: str,
    confidence: float,
    graph: KnowledgeGraph,
    seen: set[str],
) -> None:
    """Create a deduplicated CALLS relationship."""
    rel_id = f"calls:{source_id}->{target_id}"
    if rel_id not in seen:
        seen.add(rel_id)
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.CALLS,
                source=source_id,
                target=target_id,
                properties={"confidence": confidence},
            )
        )

def _resolve_receiver_method(
    receiver: str,
    method_name: str,
    source_id: str,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    seen: set[str],
) -> None:
    """Resolve ``Receiver.method()`` to the METHOD node and create a CALLS edge.

    Looks for a METHOD node whose ``name`` matches *method_name* and whose
    ``class_name`` matches *receiver*.  Searches same-file first, then
    globally.
    """
    same_file_match: str | None = None
    global_match: str | None = None

    for nid in call_index.get(method_name, []):
        node = graph.get_node(nid)
        if (
            node is not None
            and node.label == NodeLabel.METHOD
            and node.class_name == receiver
        ):
            if node.file_path == file_path:
                same_file_match = nid
                break
            elif global_match is None:
                global_match = nid

    target = same_file_match or global_match
    if target is not None:
        _add_calls_edge(source_id, target, 0.8, graph, seen)


def _build_var_type_map(parse_data: list[FileParseData]) -> dict[str, dict[str, str]]:
    """Build a per-file {var_name → type_name} map from variable type info.

    Used to resolve receiver method calls like ``pool.acquire()`` where
    ``pool`` was declared as ``const pool = new Pool()``.
    """
    result: dict[str, dict[str, str]] = {}
    for fpd in parse_data:
        if fpd.parse_result.variable_types:
            file_map: dict[str, str] = {}
            for vt in fpd.parse_result.variable_types:
                file_map.setdefault(vt.var_name, vt.type_name)
            result[fpd.file_path] = file_map
    return result


def process_calls(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
) -> None:
    """Resolve call expressions and create CALLS relationships in the graph.

    For each call expression in the parse data:

    1. Determine which symbol in the file *contains* the call (by line
       number range).
    2. Resolve the call to a target symbol node.
    3. Create a CALLS relationship from the containing symbol to the
       target, with a ``confidence`` property.

    Skips calls where:
    - The call name is in the blocklist (builtins/stdlib/globals).
    - The containing symbol cannot be determined.
    - The target cannot be resolved.
    - A relationship with the same ID already exists (deduplication).

    Args:
        parse_data: File parse results from the parser phase.
        graph: The knowledge graph to populate with CALLS relationships.
    """
    call_index = build_name_index(graph, _CALLABLE_LABELS)
    call_index_by_file = _build_call_index_by_file(call_index, graph)
    file_sym_index = build_file_symbol_index(graph, _CALLABLE_LABELS)
    var_type_map = _build_var_type_map(parse_data)
    seen: set[str] = set()

    for fpd in parse_data:
        import_cache = _build_import_cache(fpd.file_path, graph)

        # Ruby's Enumerable/Kernel and Go's builtin names collide with ordinary
        # user function names in other languages, so only fold each in for its
        # own language's files.
        if fpd.language == "ruby":
            blocklist = _CALL_BLOCKLIST | _RUBY_CALL_BLOCKLIST
        elif fpd.language == "go":
            blocklist = _CALL_BLOCKLIST | _GO_CALL_BLOCKLIST
        else:
            blocklist = _CALL_BLOCKLIST

        for call in fpd.parse_result.calls:
            # Builtin/stdlib names never resolve as call targets, but their
            # callback arguments are real references — `rows.map(formatRow)`
            # must still link formatRow or dead-code false-flags it.
            blocklisted = (
                call.name in blocklist and call.receiver not in ("self", "this")
            )

            source_id = find_containing_symbol(
                call.line, fpd.file_path, file_sym_index
            )
            if source_id is None:
                # Module-level call outside any function/class body.
                # Attribute to the File node so the target still gets a
                # CALLS edge (prevents false-positive dead-code flags).
                source_id = generate_id(NodeLabel.FILE, fpd.file_path)

            if not blocklisted:
                # Determine caller class for self/this resolution.
                caller_class_name: str | None = None
                if call.receiver in ("self", "this"):
                    source_node = graph.get_node(source_id)
                    if source_node is not None:
                        caller_class_name = source_node.class_name

                target_id, confidence = resolve_call(
                    call, fpd.file_path, call_index, graph,
                    caller_class_name=caller_class_name,
                    import_cache=import_cache,
                    call_index_by_file=call_index_by_file,
                )
                if target_id is not None:
                    # Weak references (object-literal shorthand) resolved only
                    # by global fuzzy matching keep their edge — removing it
                    # would let dead-code flag genuinely-referenced symbols —
                    # but at a confidence that marks them clearly uncertain.
                    if call.is_weak_ref and confidence < 0.8:
                        confidence = _WEAK_REF_CONFIDENCE
                    _add_calls_edge(source_id, target_id, confidence, graph, seen)

            # Callback arguments: bare identifiers passed as arguments
            # (e.g. map(transform, items), Depends(get_db)).
            for arg_name in call.arguments:
                if arg_name in blocklist:
                    continue
                arg_call = CallInfo(name=arg_name, line=call.line)
                arg_id, arg_conf = resolve_call(
                    arg_call, fpd.file_path, call_index, graph,
                    import_cache=import_cache,
                    call_index_by_file=call_index_by_file,
                )
                if arg_id is not None:
                    _add_calls_edge(source_id, arg_id, arg_conf * 0.8, graph, seen)

            # Receiver: link to the class and resolve the method on it.
            receiver = call.receiver
            if blocklisted:
                continue
            if receiver and receiver not in ("self", "this"):
                receiver_call = CallInfo(name=receiver, line=call.line)
                recv_id, recv_conf = resolve_call(
                    receiver_call, fpd.file_path, call_index, graph,
                    import_cache=import_cache,
                    call_index_by_file=call_index_by_file,
                )
                if recv_id is not None:
                    _add_calls_edge(source_id, recv_id, recv_conf, graph, seen)

                # Use type-inferred name when available (e.g., pool → Pool).
                resolved_receiver = (
                    var_type_map.get(fpd.file_path, {}).get(receiver, receiver)
                )

                _resolve_receiver_method(
                    resolved_receiver, call.name, source_id, fpd.file_path,
                    call_index, graph, seen,
                )

        # Decorators are implicit calls — @cost_decorator on a function is
        # equivalent to calling cost_decorator(func).  Create CALLS edges
        # from the decorated symbol to the decorator definition.  IDs come
        # from the same assignment logic as the parser phase so collision
        # suffixes (#L) attach the edge to the right duplicate.  The parser
        # phase already computed these while creating the symbol nodes
        # (process_parsing's Phase 2) and carries them on the FileParseData
        # it returns — reuse them instead of recomputing over every symbol
        # in the repo a second time.  Falls back to recomputing for
        # FileParseData built outside process_parsing (direct construction
        # in tests, diff.py's scoped-diff path).
        symbol_ids = (
            fpd.symbol_ids
            if fpd.symbol_ids is not None
            else assign_symbol_ids(fpd.parse_result.symbols, fpd.file_path)
        )
        for symbol, source_id in zip(fpd.parse_result.symbols, symbol_ids):
            if not symbol.decorators or source_id is None:
                continue

            for dec_name in symbol.decorators:
                # Strip the base name for dotted decorators (e.g. "app.route" → "route")
                # but also try the full dotted name.
                base_name = dec_name.rsplit(".", 1)[-1] if "." in dec_name else dec_name
                call_obj = CallInfo(name=base_name, line=symbol.start_line)
                target_id, confidence = resolve_call(
                    call_obj, fpd.file_path, call_index, graph,
                    import_cache=import_cache,
                    call_index_by_file=call_index_by_file,
                )
                if target_id is None and "." in dec_name:
                    # Try full dotted name as well.
                    call_obj = CallInfo(name=dec_name, line=symbol.start_line)
                    target_id, confidence = resolve_call(
                        call_obj, fpd.file_path, call_index, graph,
                        import_cache=import_cache,
                        call_index_by_file=call_index_by_file,
                    )
                if target_id is None:
                    continue

                rel_id = f"calls:{source_id}->{target_id}"
                if rel_id in seen:
                    continue
                seen.add(rel_id)

                graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=RelType.CALLS,
                        source=source_id,
                        target=target_id,
                        properties={"confidence": confidence},
                    )
                )
