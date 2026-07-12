"""Phase 10: Dead code detection for Synaptiq.

Scans the knowledge graph to find unreachable symbols (functions, methods,
classes) that have zero incoming CALLS relationships and are not entry points,
exported, constructors, test functions, or dunder methods.  Flags them by
setting ``is_dead = True`` on the corresponding graph node.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath

from synaptiq.core.graph.graph import KnowledgeGraph
from synaptiq.core.graph.model import GraphNode, NodeLabel, RelType

logger = logging.getLogger(__name__)

_SYMBOL_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
)

_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"__init__", "__new__", "constructor"})


def _is_constructor(name: str, file_path: str) -> bool:
    """Return ``True`` if *name* is a constructor for its language.

    ``__init__``/``__new__``/``constructor`` are language-agnostic; Ruby's
    constructor ``initialize`` is only treated as such in ``.rb`` files (it is
    an ordinary, dead-able method name in other languages).
    """
    if name in _CONSTRUCTOR_NAMES:
        return True
    return name == "initialize" and file_path.endswith(".rb")

# Ruby metaprogramming hooks the interpreter/framework calls indirectly.  They
# never have a direct CALLS edge but are not dead.
_RUBY_METAPROGRAMMING_NAMES: frozenset[str] = frozenset({
    "method_missing", "respond_to_missing?", "const_missing",
    "method_added", "singleton_method_added", "inherited",
    "included", "extended", "prepended", "coerce",
})

# Macro prefixes whose ``"{macro}:{name}"`` decorator entries (recorded by the
# Ruby parser) name methods Ruby/Rails invokes indirectly: ``attr_*`` accessors
# and Rails lifecycle callbacks.  A method whose owning type carries a matching
# decorator is exempt from dead-code flagging.
_RUBY_MACRO_PREFIXES: tuple[str, ...] = (
    "attr_accessor", "attr_reader", "attr_writer",
    "before_action", "after_action", "around_action",
    "before_filter", "after_filter", "around_filter",
    "before_save", "after_save", "around_save",
    "before_create", "after_create", "around_create",
    "before_update", "after_update", "around_update",
    "before_destroy", "after_destroy", "around_destroy",
    "before_validation", "after_validation",
    "after_commit", "after_rollback", "after_initialize",
    "after_find", "after_touch",
)

def _is_test_class(name: str) -> bool:
    """Return ``True`` if *name* follows pytest class convention (``Test*``).

    Matches names starting with ``Test`` where the next character is uppercase,
    e.g. ``TestHandleQuery``, ``TestBulkLoad``.
    """
    return len(name) > 4 and name.startswith("Test") and name[4].isupper()

def _has_test_suffix(filename: str) -> bool:
    """Return ``True`` if the filename has a test/spec/stories suffix.

    Matches Jest/Vitest (``*.test.ts``, ``*.spec.tsx``) and Storybook
    (``*.stories.tsx``) naming conventions common in TypeScript/React projects.
    """
    base = filename
    for ext in (".ts", ".tsx", ".js", ".jsx", ".mts", ".mjs"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base.endswith((".test", ".spec", ".stories"))


def _is_test_file(file_path: str) -> bool:
    """Return ``True`` if the file is in a test directory or is a test file.

    Uses path component matching rather than substring matching to avoid
    false positives (e.g. ``contest/`` matching ``/test``).
    """
    parts = PurePosixPath(file_path).parts
    filename = parts[-1] if parts else ""
    return (
        "tests" in parts
        or "test" in parts
        or "spec" in parts
        or "__tests__" in parts
        or "__mocks__" in parts
        or "__fixtures__" in parts
        or any(p.startswith("test_") for p in parts)
        or file_path.endswith("conftest.py")
        or filename.endswith(("_spec.rb", "_test.rb"))
        or _has_test_suffix(filename)
    )

def _is_dunder(name: str) -> bool:
    """Return ``True`` if *name* is a dunder (double-underscore) method.

    Dunders start and end with ``__`` and have at least one character in
    between (e.g. ``__str__``, ``__repr__``).
    """
    return name.startswith("__") and name.endswith("__") and len(name) > 4

def _is_type_referenced(graph: KnowledgeGraph, node_id: str, label: NodeLabel) -> bool:
    """Return ``True`` if *node_id* is a class with incoming USES_TYPE edges.

    Classes referenced via type annotations (enums, dataclasses, Protocol
    classes) are not dead — they are actively used as types.  This check
    is restricted to CLASS nodes; a function used only in a type annotation
    is legitimately unused.
    """
    if label != NodeLabel.CLASS:
        return False
    return graph.has_incoming(node_id, RelType.USES_TYPE)

_NON_FRAMEWORK_DECORATORS: frozenset[str] = frozenset({
    "functools.wraps",
    "functools.lru_cache",
    "functools.cached_property",
    "functools.cache",
})

_FRAMEWORK_DECORATOR_NAMES: frozenset[str] = frozenset({
    "task", "shared_task", "periodic_task", "job",
    "receiver", "on_event", "handler",
    "validator", "field_validator", "root_validator", "model_validator",
    "contextmanager", "asynccontextmanager",
    "fixture",
    "route", "endpoint", "command",
    "hybrid_property",
})

def _has_framework_decorator(node: GraphNode) -> bool:
    """Return ``True`` if *node* has a framework decorator (dotted or undotted)."""
    decorators: list[str] = node.properties.get("decorators", [])
    return any(
        dec in _FRAMEWORK_DECORATOR_NAMES or ("." in dec and dec not in _NON_FRAMEWORK_DECORATORS)
        for dec in decorators
    )

def _has_property_decorator(node: GraphNode) -> bool:
    """Return ``True`` if *node* is a ``@property`` (accessed as attribute, not called)."""
    decorators: list[str] = node.properties.get("decorators", [])
    return "property" in decorators

_TYPING_STUB_DECORATORS: frozenset[str] = frozenset({
    "overload", "typing.overload",
    "abstractmethod", "abc.abstractmethod",
})

def _has_typing_stub_decorator(node: GraphNode) -> bool:
    """Return ``True`` if *node* is an ``@overload`` or ``@abstractmethod`` stub."""
    decorators: list[str] = node.properties.get("decorators", [])
    return any(d in _TYPING_STUB_DECORATORS for d in decorators)

_ENUM_BASES: frozenset[str] = frozenset({
    "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag",
})

_FRAMEWORK_MODEL_BASES: frozenset[str] = frozenset({
    # Pydantic
    "BaseModel", "BaseSettings",
    # Django ORM
    "Model", "Manager",
    # SQLAlchemy
    "Base", "DeclarativeBase",
    # TypeORM
    "BaseEntity",
    # unittest
    "TestCase",
    # Rails (heritage stores the last namespace segment, so e.g.
    # ``ActiveRecord::Base`` is captured as ``Base``, already listed above).
    "ApplicationRecord", "ApplicationController", "ApplicationJob",
    "ApplicationMailer", "ActiveRecord", "ActionController",
})

_VITE_PLUGIN_HOOKS: frozenset[str] = frozenset({
    "resolveId", "load", "transform", "buildStart", "buildEnd",
    "closeBundle", "configResolved", "configureServer", "handleHotUpdate",
    "renderChunk", "generateBundle", "writeBundle", "options",
    "manualChunks", "renderStart", "renderError", "footer", "banner",
    "intro", "outro", "moduleParsed",
})

_CONFIG_CALLBACK_PATTERNS: frozenset[str] = frozenset({
    "configure", "getLoadContext", "beforeAll", "onError",
    "onShellError", "onShellReady", "onAllReady",
    "esbuildOptions", "onSuccess", "viteFinal", "onwarn",
})

def _is_enum_class(node: GraphNode, label: NodeLabel) -> bool:
    """Return ``True`` if *node* is an enum class (members accessed via dot, not called)."""
    if label != NodeLabel.CLASS:
        return False
    bases: list[str] = node.properties.get("bases", [])
    return bool(_ENUM_BASES & set(bases))

def _is_framework_model_class(node: GraphNode, label: NodeLabel) -> bool:
    """Return ``True`` if *node* extends a framework model base class.

    Framework models (Pydantic BaseModel, Django Model, etc.) are
    instantiated and used by the framework's metaclass machinery.
    They appear uncalled in the graph because the framework does the
    calling, not user code.
    """
    if label != NodeLabel.CLASS:
        return False
    bases: list[str] = node.properties.get("bases", [])
    return bool(_FRAMEWORK_MODEL_BASES & set(bases))

def _is_python_public_api(name: str, file_path: str) -> bool:
    """Return ``True`` if *name* is a public symbol in an ``__init__.py`` file."""
    return file_path.endswith("__init__.py") and not name.startswith("_")

def _is_config_file_hook(name: str, file_path: str) -> bool:
    """Return ``True`` if *name* is a known framework config hook in a config file."""
    if name in _CONFIG_CALLBACK_PATTERNS:
        return True
    if name in _VITE_PLUGIN_HOOKS and (
        "vite.config" in file_path
        or "vitest.config" in file_path
        or "tsup.config" in file_path
        or "rollup.config" in file_path
    ):
        return True
    return False

def _is_framework_entry_file(file_path: str) -> bool:
    """Return ``True`` if the file is a framework entry point where all symbols are used."""
    basename = PurePosixPath(file_path).name
    return basename in (
        "entry.server.tsx", "entry.server.ts", "entry.server.js",
        "entry.client.tsx", "entry.client.ts", "entry.client.js",
    )

def _is_exempt(
    name: str, is_entry_point: bool, is_exported: bool, file_path: str = ""
) -> bool:
    """Return ``True`` if the symbol is exempt from dead-code flagging.

    A symbol is exempt when ANY of the following hold:

    - It is marked as an entry point.
    - It is marked as exported (may be used externally).
    - It is a constructor (``__init__`` / ``__new__`` / ``constructor``).
    - It is a test function (name starts with ``test_``).
    - It is a test class (name starts with ``Test``).
    - It lives in a test file (fixtures, helpers are not dead code).
    - It is a dunder method (``__str__``, ``__repr__``, etc.).
    - It is a Ruby metaprogramming hook (``method_missing``, etc.).
    - It is a public symbol in a Python ``__init__.py`` file.
    """
    return (
        is_entry_point
        or is_exported
        or _is_constructor(name, file_path)
        or name.startswith("test_")
        or _is_test_class(name)
        or _is_test_file(file_path)
        or _is_dunder(name)
        or name in _RUBY_METAPROGRAMMING_NAMES
        or _is_python_public_api(name, file_path)
        or _is_config_file_hook(name, file_path)
        or _is_framework_entry_file(file_path)
    )


def is_symbol_exempt_from_dead_code(
    name: str, is_entry_point: bool, is_exported: bool, file_path: str = ""
) -> bool:
    """Public reuse hook: the exact exemption predicate used by dead-code flagging.

    Exposed (W3.2d) so the storage layer's scoped ``is_dead`` recount on the
    incremental delta path can decide deadness with the *same* logic instead of
    forking it. Delegates verbatim to the internal :func:`_is_exempt`, which
    :func:`process_dead_code` also uses. A symbol with zero incoming CALLS is
    dead iff this predicate returns ``False``.
    """
    return _is_exempt(name, is_entry_point, is_exported, file_path)


def _clear_override_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods that override a non-dead base class method.

    When ``A extends B`` and ``B.method`` is called, ``A.method`` (the
    override) has zero incoming CALLS and gets flagged dead.  This pass
    detects that situation and clears ``is_dead`` on the override.

    Returns the number of overrides un-flagged.
    """
    # Build a mapping: class_name -> set of method names that are NOT dead.
    alive_methods_by_class: dict[str, set[str]] = {}
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead and method.class_name:
            alive_methods_by_class.setdefault(method.class_name, set()).add(method.name)

    # Build child -> parent class mapping from EXTENDS relationships.
    child_to_parents: dict[str, list[str]] = {}
    for rel in graph.get_relationships_by_type(RelType.EXTENDS):
        child_node = graph.get_node(rel.source)
        parent_node = graph.get_node(rel.target)
        if child_node and parent_node:
            child_to_parents.setdefault(child_node.name, []).append(parent_node.name)

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue

        parent_classes = child_to_parents.get(method.class_name, [])
        for parent_name in parent_classes:
            alive_in_parent = alive_methods_by_class.get(parent_name, set())
            if method.name in alive_in_parent:
                method.is_dead = False
                cleared += 1
                logger.debug("Un-flagged override: %s.%s", method.class_name, method.name)
                break

    return cleared

def _clear_protocol_conformance_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods on classes that structurally conform to a Protocol.

    When a Protocol defines methods ``{m1, m2, m3}`` and a concrete class
    implements all of those methods without an explicit EXTENDS edge
    (structural subtyping), the concrete methods may be flagged dead
    because CALLS edges resolve to the Protocol's stubs, not the
    concrete implementations.

    This pass:

    1. Finds Protocol classes (annotated with ``is_protocol`` in properties).
    2. Collects their non-dunder method names as the required interface.
    3. Finds non-Protocol classes whose methods are a superset.
    4. Un-flags dead methods whose name is in the protocol interface.

    Returns the number of methods un-flagged.
    """
    protocol_methods: dict[str, set[str]] = {}
    for cls_node in graph.get_nodes_by_label(NodeLabel.CLASS):
        if not cls_node.properties.get("is_protocol"):
            continue
        methods = set()
        for method in graph.get_nodes_by_label(NodeLabel.METHOD):
            if method.class_name == cls_node.name and not _is_dunder(method.name):
                methods.add(method.name)
        if methods:
            protocol_methods[cls_node.name] = methods

    if not protocol_methods:
        return 0

    class_methods: dict[str, set[str]] = {}
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if method.class_name:
            class_methods.setdefault(method.class_name, set()).add(method.name)

    clearable: dict[str, set[str]] = {}
    for proto_name, required in protocol_methods.items():
        for cls_name, methods in class_methods.items():
            if cls_name == proto_name:
                continue
            if required <= methods:  # structural conformance
                clearable.setdefault(cls_name, set()).update(required)

    if not clearable:
        return 0

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue
        names_to_clear = clearable.get(method.class_name)
        if names_to_clear and method.name in names_to_clear:
            method.is_dead = False
            cleared += 1
            logger.debug(
                "Un-flagged protocol conformance: %s.%s",
                method.class_name,
                method.name,
            )

    return cleared

def _clear_protocol_stub_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods on Protocol classes.

    Protocol stubs define the interface contract — they are never called
    directly (calls resolve to concrete implementations).  Flagging them
    as dead is always a false positive.

    Returns the number of methods un-flagged.
    """
    protocol_class_names: set[str] = set()
    for cls_node in graph.get_nodes_by_label(NodeLabel.CLASS):
        if cls_node.properties.get("is_protocol"):
            protocol_class_names.add(cls_node.name)

    if not protocol_class_names:
        return 0

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue
        if method.class_name in protocol_class_names:
            method.is_dead = False
            cleared += 1
            logger.debug("Un-flagged protocol stub: %s.%s", method.class_name, method.name)

    return cleared

def _clear_alive_class_method_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods on classes or object-literal variables that are alive.

    In TypeScript/JavaScript, class methods are typically called via instance
    references (``obj.method()``, ``this.method()``).  Without type-flow
    analysis the call resolver cannot link these calls to their targets,
    so methods appear to have zero incoming CALLS edges.

    This pass un-flags non-private methods on alive classes, since they are
    very likely called via instances.  Methods starting with ``_`` (private
    by convention) are left flagged.

    Also covers the ESLint/Babel visitor pattern where methods are defined
    inside an object literal assigned to a variable (``const Service = { ... }``).
    If the owning variable name matches an alive function or is referenced
    by alive code in the same file, the methods are un-flagged.

    Returns the number of methods un-flagged.
    """
    dead_class_names: set[str] = set()
    alive_class_names: set[str] = set()
    for cls_node in graph.get_nodes_by_label(NodeLabel.CLASS):
        if cls_node.is_dead:
            dead_class_names.add(cls_node.name)
        else:
            alive_class_names.add(cls_node.name)

    # Also consider alive functions whose name matches a class_name — these
    # cover the pattern where a const variable holds an object literal:
    #   const Service = { sendOTP() {} }
    # and an alive function elsewhere references Service.
    alive_function_names: set[str] = set()
    for func_node in graph.get_nodes_by_label(NodeLabel.FUNCTION):
        if not func_node.is_dead:
            alive_function_names.add(func_node.name)

    # Build a set of object-literal "class" names that are referenced by alive
    # code in the same file.  A method's class_name is the variable name
    # (e.g., "Service" for `const Service = { sendOTP() {} }`).
    # Check if any alive symbol in the same file references that name in its content.
    alive_obj_literal_names: set[str] = _find_alive_object_literal_names(graph)

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue
        if method.name.startswith("_"):
            continue
        if method.class_name in alive_class_names:
            method.is_dead = False
            cleared += 1
            logger.debug(
                "Un-flagged alive-class method: %s.%s", method.class_name, method.name
            )
        elif (
            method.class_name not in dead_class_names
            and method.class_name in (alive_function_names | alive_obj_literal_names)
        ):
            method.is_dead = False
            cleared += 1
            logger.debug(
                "Un-flagged alive-object-literal method: %s.%s",
                method.class_name,
                method.name,
            )

    return cleared


def _find_alive_object_literal_names(graph: KnowledgeGraph) -> set[str]:
    """Find object-literal variable names referenced by alive code.

    For each dead Method whose class_name is not a known class, check if
    any alive symbol (Function/Method) in the same file mentions the
    class_name in its content.  This covers patterns like:

        function createVisitor() {
            const visitors = { enter() {}, exit() {} };
            return visitors;
        }

    where ``visitors`` is alive because ``createVisitor`` returns it.
    """
    # Collect all class_names from dead methods that are NOT real classes.
    class_names_set: set[str] = set()
    for cls_node in graph.get_nodes_by_label(NodeLabel.CLASS):
        class_names_set.add(cls_node.name)

    # Map: (file_path, obj_name) pairs to check
    candidates: dict[str, set[str]] = {}  # file_path -> set of obj names
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if method.is_dead and method.class_name and method.class_name not in class_names_set:
            candidates.setdefault(method.file_path, set()).add(method.class_name)

    if not candidates:
        return set()

    alive_names: set[str] = set()
    for label in (NodeLabel.FUNCTION, NodeLabel.METHOD):
        for node in graph.get_nodes_by_label(label):
            if node.is_dead or not node.content:
                continue
            obj_names = candidates.get(node.file_path)
            if not obj_names:
                continue
            for obj_name in obj_names:
                if obj_name in node.content:
                    alive_names.add(obj_name)

    return alive_names


def _clear_inner_function_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag inner functions whose containing function is alive.

    In TypeScript/React, handler functions are often defined as arrow
    functions inside a component.  These appear as separate Function nodes
    but are only referenced within their parent's JSX output.  If the
    parent function is alive, the inner function is likely used.

    Returns the number of functions un-flagged.
    """
    alive_ranges: dict[str, list[tuple[int, int]]] = {}
    for node in graph.get_nodes_by_label(NodeLabel.FUNCTION):
        if not node.is_dead and node.start_line and node.end_line:
            alive_ranges.setdefault(node.file_path, []).append(
                (node.start_line, node.end_line)
            )

    cleared = 0
    for node in graph.get_nodes_by_label(NodeLabel.FUNCTION):
        if not node.is_dead or not node.start_line or not node.end_line:
            continue
        ranges = alive_ranges.get(node.file_path)
        if not ranges:
            continue
        for parent_start, parent_end in ranges:
            if parent_start < node.start_line and node.end_line < parent_end:
                node.is_dead = False
                cleared += 1
                logger.debug("Un-flagged inner function: %s in %s", node.name, node.file_path)
                break

    return cleared


def _clear_ruby_macro_method_false_positives(graph: KnowledgeGraph) -> int:
    """Un-flag methods named by an ``attr_*`` accessor or Rails callback macro.

    The Ruby parser records macro arguments on the owning type's ``decorators``
    as ``"{macro}:{name}"`` (e.g. ``attr_reader:name``, ``before_action:auth``).
    Such methods are invoked by Ruby/Rails indirectly and have no direct CALLS
    edge, so a hand-written ``def name`` / ``def auth`` would otherwise be
    flagged dead.  This pass un-flags those methods.

    Returns the number of methods un-flagged.
    """
    # Map (file_path, owning_type_name) -> set of macro-named methods.
    macro_methods: dict[tuple[str, str], set[str]] = {}
    for label in (NodeLabel.CLASS, NodeLabel.MODULE):
        for node in graph.get_nodes_by_label(label):
            decorators: list[str] = node.properties.get("decorators", [])
            for dec in decorators:
                prefix, sep, target = dec.partition(":")
                if sep and prefix in _RUBY_MACRO_PREFIXES and target:
                    macro_methods.setdefault((node.file_path, node.name), set()).add(target)

    if not macro_methods:
        return 0

    cleared = 0
    for method in graph.get_nodes_by_label(NodeLabel.METHOD):
        if not method.is_dead or not method.class_name:
            continue
        names = macro_methods.get((method.file_path, method.class_name))
        if names and method.name in names:
            method.is_dead = False
            cleared += 1
            logger.debug(
                "Un-flagged Ruby macro method: %s.%s", method.class_name, method.name
            )

    return cleared


def process_dead_code(graph: KnowledgeGraph) -> int:
    """Detect dead (unreachable) symbols and flag them in the graph.

    A symbol is considered dead when **all** of the following are true:

    1. It has zero incoming ``CALLS`` relationships.
    2. It is not an entry point (``is_entry_point == False``).
    3. It is not exported (``is_exported == False``).
    4. It is not a class constructor (``__init__`` / ``__new__``).
    5. It is not a test function (name starts with ``test_``).
    6. It is not a test class (name starts with ``Test``).
    7. It is not in a test file (fixtures/helpers are exempt).
    8. It is not a dunder method (name starts and ends with ``__``).
    9. It is not a class referenced via type annotations (``USES_TYPE``).
    10. It does not have a framework-registration decorator.
    11. It is not a ``@property`` method.
    12. It is not an ``@overload`` or ``@abstractmethod`` stub.
    13. It is not an enum class (extends ``Enum``, ``IntEnum``, etc.).

    After the initial pass, five additional passes reduce false positives:

    - **Override pass**: un-flags method overrides whose base class method
      is called (resolves dynamic dispatch false positives).
    - **Protocol conformance pass**: un-flags methods on classes that
      structurally conform to a Protocol interface.
    - **Protocol stub pass**: un-flags methods on Protocol classes
      themselves (stubs are never called directly).
    - **Alive class method pass**: un-flags non-private methods on classes
      that are alive (instance call resolution gap).
    - **Inner function pass**: un-flags inner functions defined inside
      alive parent functions (React handler pattern).

    For each dead symbol the function sets ``node.is_dead = True``.

    Args:
        graph: The knowledge graph to scan and mutate.

    Returns:
        The total number of symbols flagged as dead.
    """
    dead_count = 0

    for label in _SYMBOL_LABELS:
        for node in graph.get_nodes_by_label(label):
            if _is_exempt(node.name, node.is_entry_point, node.is_exported, node.file_path):
                continue
            if graph.has_incoming(node.id, RelType.CALLS):
                continue
            if _is_type_referenced(graph, node.id, label):
                continue
            if _has_framework_decorator(node):
                continue
            if _has_property_decorator(node):
                continue
            if _has_typing_stub_decorator(node):
                continue
            if _is_enum_class(node, label):
                continue
            if _is_framework_model_class(node, label):
                continue

            node.is_dead = True
            dead_count += 1
            logger.debug("Dead symbol: %s (%s)", node.name, node.id)

    # Second pass: un-flag overrides of called base-class methods.
    cleared = _clear_override_false_positives(graph)
    dead_count -= cleared

    # Third pass: un-flag methods on classes that structurally conform to a Protocol.
    protocol_cleared = _clear_protocol_conformance_false_positives(graph)
    dead_count -= protocol_cleared

    # Fourth pass: un-flag Protocol class stubs (interface contracts, never called directly).
    stub_cleared = _clear_protocol_stub_false_positives(graph)
    dead_count -= stub_cleared

    # Fifth pass: un-flag non-private methods on alive classes (instance call resolution gap).
    alive_class_cleared = _clear_alive_class_method_false_positives(graph)
    dead_count -= alive_class_cleared

    # Sixth pass: un-flag inner functions defined inside alive parent functions.
    inner_cleared = _clear_inner_function_false_positives(graph)
    dead_count -= inner_cleared

    # Seventh pass: un-flag Ruby methods named by attr_*/Rails-callback macros.
    ruby_macro_cleared = _clear_ruby_macro_method_false_positives(graph)
    dead_count -= ruby_macro_cleared

    return dead_count
