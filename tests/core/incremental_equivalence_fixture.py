"""Seeded polyglot fixture + randomized edit-script generator (W3.2f).

The equivalence harness needs a repository that (a) is *polyglot* — Python,
TypeScript, Ruby, and Go, with a ``go.mod`` and a REST endpoint/client pair —
(b) is *rich in cross-file structural edges* (import-resolved CALLS, IMPORTS,
EXTENDS, IMPLEMENTS, MIXES_IN, USES_TYPE) so the strict-core equivalence is
actually exercised, and (c) can be *deterministically mutated* by every edit
operation the incremental design must survive:

    body-only edit, signature change, rename symbol, add symbol, delete symbol,
    add file (incl. the "imported-later" closure case), delete file,
    move file (delete + add), no-op touch.

The design is a **structured model**, not source-text munging: a
:class:`RepoState` holds hand-authored *structural* files (the rich edge zoo,
mirroring the proven ``go_project`` / ``ruby_project`` / ``test_incremental_build``
fixtures) plus generator-owned *leaf* files (a uniform funcs-and-calls model the
per-symbol ops manipulate). :meth:`RepoState.render` produces the full on-disk
tree; :func:`sync_disk` writes only what changed. :class:`EditScriptGenerator`
turns a seed into a deterministic list of :class:`Step` s (each a burst of ops),
so a failing example is reproducible from the seed alone.

This module is a test *helper* (not ``test_``-prefixed → never collected). It is
pure/deterministic: same seed ⇒ same script ⇒ same trees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from posixpath import basename, dirname

# ---------------------------------------------------------------------------
# Language surface
# ---------------------------------------------------------------------------

LANGS = ("py", "ts", "go", "rb")
_EXT = {"py": ".py", "ts": ".ts", "go": ".go", "rb": ".rb"}
_COMMENT = {"py": "#", "ts": "//", "go": "//", "rb": "#"}
_LEAF_DIR = {"py": "py", "ts": "ts", "rb": "rb"}  # go leaves get their own package dir
GO_MODULE = "example.com/proj"
GO_MOD_PATH = "go/go.mod"


@dataclass
class Leaf:
    """A generator-owned leaf file: functions + cross-file import-and-call refs.

    ``calls`` are ``(target_path, symbol)`` pairs this file imports and calls
    (the first function issues the calls), producing import-resolved CALLS +
    IMPORTS edges; ``salt`` drives a pure body-only edit (a rendered comment +
    each function's return literal); ``extra_args`` bumps a function's arity for
    a signature (identity) change.
    """

    lang: str
    stem: str
    funcs: list[str]
    salt: int = 0
    extra_args: dict[str, int] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    @property
    def dir(self) -> str:
        return f"go/{self.stem}" if self.lang == "go" else _LEAF_DIR[self.lang]

    @property
    def path(self) -> str:
        return f"{self.dir}/{self.stem}{_EXT[self.lang]}"


@dataclass
class Struct:
    """A hand-authored structural file, mutated only as whole text + a salt line."""

    lang: str
    base: str
    salt: int = 0


# ---------------------------------------------------------------------------
# Renderers (one per language; deterministic)
# ---------------------------------------------------------------------------


def _py_module(path: str) -> str:
    return path[: -len(".py")].replace("/", ".")


def _render_py_leaf(leaf: Leaf) -> str:
    lines = [f"# salt {leaf.salt}"]
    for tp, sym in leaf.calls:
        lines.append(f"from {_py_module(tp)} import {sym}")
    lines.append("")
    for i, fn in enumerate(leaf.funcs):
        args = ", ".join(["a"] + [f"x{j}" for j in range(leaf.extra_args.get(fn, 0))])
        lines.append(f"def {fn}({args}):")
        if i == 0:
            for _, sym in leaf.calls:
                lines.append(f"    {sym}(1)")
        lines.append(f"    return {leaf.salt}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_ts_leaf(leaf: Leaf) -> str:
    lines = [f"// salt {leaf.salt}"]
    for tp, sym in leaf.calls:
        lines.append(f"import {{ {sym} }} from './{basename(tp)[: -len('.ts')]}';")
    lines.append("")
    for i, fn in enumerate(leaf.funcs):
        args = ", ".join([f"a{j}: number" for j in range(1 + leaf.extra_args.get(fn, 0))])
        body = " ".join(f"{sym}();" for _, sym in leaf.calls) if i == 0 else ""
        lines.append(f"export function {fn}({args}): number {{ {body} return {leaf.salt}; }}")
    return "\n".join(lines) + "\n"


def _render_go_leaf(leaf: Leaf) -> str:
    lines = [f"// salt {leaf.salt}", f"package {leaf.stem}", ""]
    imports = sorted({dirname(tp) for tp, _ in leaf.calls})
    if imports:
        lines.append("import (")
        for d in imports:
            lines.append(f'\t"{GO_MODULE}/{d}"')
        lines.append(")")
        lines.append("")
    for i, fn in enumerate(leaf.funcs):
        args = ", ".join([f"a{j} int" for j in range(1 + leaf.extra_args.get(fn, 0))])
        lines.append(f"func {fn}({args}) int {{")
        if i == 0:
            for tp, sym in leaf.calls:
                lines.append(f"\t{basename(dirname(tp))}.{sym}()")
        lines.append(f"\treturn {leaf.salt}")
        lines.append("}")
    return "\n".join(lines) + "\n"


def _render_rb_leaf(leaf: Leaf) -> str:
    lines = [f"# salt {leaf.salt}"]
    for tp, _ in leaf.calls:
        lines.append(f'require_relative "{basename(tp)[: -len(".rb")]}"')
    lines.append("")
    for i, fn in enumerate(leaf.funcs):
        args = ", ".join([f"a{j}" for j in range(1 + leaf.extra_args.get(fn, 0))])
        lines.append(f"def {fn}({args})")
        if i == 0:
            for _, sym in leaf.calls:
                lines.append(f"  {sym}(1)")
        lines.append(f"  {leaf.salt}")
        lines.append("end")
        lines.append("")
    return "\n".join(lines) + "\n"


_RENDER = {
    "py": _render_py_leaf,
    "ts": _render_ts_leaf,
    "go": _render_go_leaf,
    "rb": _render_rb_leaf,
}


# ---------------------------------------------------------------------------
# Hand-authored structural base (the rich cross-file edge zoo)
# ---------------------------------------------------------------------------

_PY_MODELS = (
    "class Account:\n"
    "    def save(self):\n"
    "        return 1\n\n\n"
    "def make_user(name):\n"
    "    acct = Account()\n"
    "    return acct\n"
)
_PY_SERVICE = (
    "from py.models import make_user, Account\n\n\n"
    "class Service(Account):\n"
    "    def run(self):\n"
    "        return make_user('x')\n"
)
_PY_HANDLER = (
    "from py.service import Service\n\n\ndef handle():\n    s = Service()\n    return s.run()\n"
)
_PY_REPORT = "from py.models import make_user\n\n\ndef report():\n    return make_user('y')\n"
_PY_REST = (
    "import requests\n\n\n"
    "@app.get('/ping')\n"
    "def ping():\n"
    "    return 'pong'\n\n\n"
    "def call_ping():\n"
    "    return requests.get('/ping')\n"
)
# The imported-later case: imports a module that does NOT exist at base index
# time; an add_file op can later create py/ghost.py and this must re-resolve.
_PY_GHOST_CLIENT = (
    "from py.ghost import ghost_symbol\n\n\ndef use_ghost():\n    return ghost_symbol()\n"
)

_TS_TYPES = "export interface Repo {\n  find(): string;\n}\n\nexport class Store {}\n"
_TS_APP = (
    "import { Repo, Store } from './types';\n\n"
    "export class App implements Repo {\n"
    "  find(): string { return 'x'; }\n"
    "  make(): Store { return new Store(); }\n"
    "}\n"
)

_GO_MOD = f"module {GO_MODULE}\n\ngo 1.21\n"
_GO_BASE = "package base\n\n// Base is embedded by User.\ntype Base struct {\n\tID int\n}\n"
_GO_USER = (
    "package models\n\n"
    'import "example.com/proj/go/base"\n\n'
    "type User struct {\n\tName string\n\tbase.Base\n}\n\n"
    "func NewUser() *User {\n\treturn &User{}\n}\n\n"
    "func (u *User) Display() string {\n\treturn u.Name\n}\n"
)
# No cross-file REST client here: a client → an *unchanged* endpoint in another
# file is deferred by design (D7), and load_graph drops the ``rest_link`` marker
# that would let the comparator exclude it — so the harness keeps REST self-
# contained (py/rest_self.py) and Go stays a cross-package CALLS + embedding case.
_GO_MAIN = (
    "package main\n\n"
    'import "example.com/proj/go/models"\n\n'
    "func main() {\n"
    "\tu := models.NewUser()\n"
    "\t_ = u.Display()\n"
    "}\n"
)

_RB_GREETER = (
    "module Greeter\n"
    "  def greet(name)\n"
    '    "Hello, #{name}!"\n'
    "  end\n\n"
    "  def unused_greet\n"
    '    "never called"\n'
    "  end\n"
    "end\n"
)
_RB_USER = (
    'require_relative "greeter"\n\n'
    "class User\n"
    "  include Greeter\n\n"
    "  attr_reader :name\n\n"
    "  def initialize(name)\n"
    "    @name = name\n"
    "  end\n\n"
    "  def display\n"
    "    greet(@name)\n"
    "  end\n"
    "end\n"
)
_RB_APP_CONTROLLER = "class ApplicationController\n  def authenticate\n    true\n  end\nend\n"
_RB_USERS_CONTROLLER = (
    'require_relative "application_controller"\n'
    'require_relative "user"\n\n'
    "class UsersController < ApplicationController\n"
    "  def show\n"
    "    authenticate\n"
    '    User.new("alice").display\n'
    "  end\n"
    "end\n"
)

_STRUCTURAL_BASE: dict[str, Struct] = {
    "py/models.py": Struct("py", _PY_MODELS),
    "py/service.py": Struct("py", _PY_SERVICE),
    "py/handler.py": Struct("py", _PY_HANDLER),
    "py/report.py": Struct("py", _PY_REPORT),
    "py/rest_self.py": Struct("py", _PY_REST),
    "py/ghost_client.py": Struct("py", _PY_GHOST_CLIENT),
    "ts/types.ts": Struct("ts", _TS_TYPES),
    "ts/app.ts": Struct("ts", _TS_APP),
    "go/base/base.go": Struct("go", _GO_BASE),
    "go/models/user.go": Struct("go", _GO_USER),
    "go/main.go": Struct("go", _GO_MAIN),
    "rb/greeter.rb": Struct("rb", _RB_GREETER),
    "rb/user.rb": Struct("rb", _RB_USER),
    "rb/application_controller.rb": Struct("rb", _RB_APP_CONTROLLER),
    "rb/users_controller.rb": Struct("rb", _RB_USERS_CONTROLLER),
}

# The module identity a ghost import is waiting for → the path add_file recreates
# to exercise the imported-later closure fix. Keyed by language.
GHOST_TARGET = "py/ghost.py"
GHOST_SYMBOL = "ghost_symbol"


def _render_struct(s: Struct) -> str:
    return f"{_COMMENT[s.lang]} salt {s.salt}\n{s.base}"


# ---------------------------------------------------------------------------
# Leaf seed — a small cross-linked set per language so the FIRST edit already
# has real dependents (renaming a provider pulls a consumer into the closure).
# ---------------------------------------------------------------------------


def _seed_leaves() -> dict[str, Leaf]:
    leaves: dict[str, Leaf] = {}

    def add(leaf: Leaf) -> None:
        leaves[leaf.path] = leaf

    # Python: prov -> cons chain (cons imports+calls prov.Fn0).
    add(Leaf("py", "prov_a", ["Fn0", "Fn1"], salt=1))
    add(Leaf("py", "cons_a", ["Fn2"], salt=1, calls=[("py/prov_a.py", "Fn0")]))
    add(Leaf("py", "leaf_a", ["Fn3"], salt=1))
    # TypeScript.
    add(Leaf("ts", "prov_t", ["Fn0", "Fn1"], salt=1))
    add(Leaf("ts", "cons_t", ["Fn2"], salt=1, calls=[("ts/prov_t.ts", "Fn0")]))
    # Go (each leaf is its own package).
    add(Leaf("go", "prova", ["Fn0", "Fn1"], salt=1))
    add(Leaf("go", "consa", ["Fn2"], salt=1, calls=[("go/prova/prova.go", "Fn0")]))
    # Ruby.
    add(Leaf("rb", "prov_r", ["Fn0", "Fn1"], salt=1))
    add(Leaf("rb", "cons_r", ["Fn2"], salt=1, calls=[("rb/prov_r.rb", "Fn0")]))
    return leaves


# ---------------------------------------------------------------------------
# Repo state
# ---------------------------------------------------------------------------


class RepoState:
    """The mutable repository model an edit script walks.

    Holds structural files (whole-text + salt) and leaf files (structured), a
    monotonic counter for fresh unique names/paths (keeps ids stable + the
    script reproducible), and the set of just-deleted provider paths so
    ``add_file`` can recreate one — the delete-then-readd shape that re-links a
    dangling module import (the imported-later closure).
    """

    def __init__(self) -> None:
        self.structural: dict[str, Struct] = {p: replace(s) for p, s in _STRUCTURAL_BASE.items()}
        self.leaves: dict[str, Leaf] = _seed_leaves()
        self.counter: int = 100
        self.deleted_paths: list[str] = []

    # -- rendering ---------------------------------------------------------
    def render(self) -> dict[str, str]:
        tree: dict[str, str] = {GO_MOD_PATH: _GO_MOD}
        for path, s in self.structural.items():
            tree[path] = _render_struct(s)
        for path, leaf in self.leaves.items():
            tree[path] = _RENDER[leaf.lang](leaf)
        return tree

    # -- helpers -----------------------------------------------------------
    def _fresh(self) -> int:
        self.counter += 1
        return self.counter

    def all_symbols(self) -> list[tuple[str, str]]:
        """Every ``(leaf_path, func)`` currently defined — call targets."""
        return [(leaf.path, fn) for leaf in self.leaves.values() for fn in leaf.funcs]


# ---------------------------------------------------------------------------
# Edit operations (each mutates a RepoState, returns a human-readable detail
# string, or None when it does not apply to the current state)
# ---------------------------------------------------------------------------


def _pick(rng: random.Random, items: list):
    return rng.choice(sorted(items, key=repr)) if items else None


def op_body_only(state: RepoState, rng: random.Random) -> str | None:
    leaf_paths = list(state.leaves)
    struct_paths = list(state.structural)
    target = _pick(rng, leaf_paths + struct_paths)
    if target is None:
        return None
    if target in state.leaves:
        state.leaves[target].salt += 1
    else:
        state.structural[target].salt += 1
    return f"body_only({target})"


def op_signature_change(state: RepoState, rng: random.Random) -> str | None:
    leaf = _pick(rng, list(state.leaves.values()))
    if leaf is None or not leaf.funcs:
        return None
    fn = _pick(rng, leaf.funcs)
    leaf.extra_args[fn] = leaf.extra_args.get(fn, 0) + 1
    return f"signature_change({leaf.path}:{fn})"


def op_rename_symbol(state: RepoState, rng: random.Random) -> str | None:
    leaf = _pick(rng, list(state.leaves.values()))
    if leaf is None or not leaf.funcs:
        return None
    old = _pick(rng, leaf.funcs)
    new = f"Fn{state._fresh()}"
    leaf.funcs[leaf.funcs.index(old)] = new
    if old in leaf.extra_args:
        leaf.extra_args[new] = leaf.extra_args.pop(old)
    # 50/50: update consumers to the new name (else leave a now-stale call).
    if rng.random() < 0.5:
        for other in state.leaves.values():
            other.calls = [
                (tp, new if (tp == leaf.path and sym == old) else sym) for tp, sym in other.calls
            ]
    return f"rename_symbol({leaf.path}:{old}->{new})"


def op_add_symbol(state: RepoState, rng: random.Random) -> str | None:
    leaf = _pick(rng, list(state.leaves.values()))
    if leaf is None:
        return None
    new = f"Fn{state._fresh()}"
    leaf.funcs.append(new)
    return f"add_symbol({leaf.path}:{new})"


def op_delete_symbol(state: RepoState, rng: random.Random) -> str | None:
    candidates = [leaf for leaf in state.leaves.values() if len(leaf.funcs) >= 2]
    leaf = _pick(rng, candidates)
    if leaf is None:
        return None
    fn = _pick(rng, leaf.funcs)
    leaf.funcs.remove(fn)
    leaf.extra_args.pop(fn, None)
    return f"delete_symbol({leaf.path}:{fn})"


def _new_leaf(state: RepoState, rng: random.Random, lang: str, stem: str) -> Leaf:
    funcs = [f"Fn{state._fresh()}" for _ in range(rng.randint(1, 2))]
    # Import+call a random existing SAME-LANGUAGE symbol (a real cross-file edge).
    # The new leaf is not in ``state.leaves`` yet, so it can never target itself;
    # Go leaves each own a package dir, so any existing target is a valid import.
    same_lang = [
        (path, fn) for path, leaf in state.leaves.items() if leaf.lang == lang for fn in leaf.funcs
    ]
    target = _pick(rng, same_lang)
    calls = [target] if target is not None else []
    return Leaf(lang, stem, funcs, salt=1, calls=calls)


def op_add_file(state: RepoState, rng: random.Random) -> str | None:
    # Imported-later closure: if the ghost target is absent AND a ghost client
    # still imports it, (re)create it so the dangling import re-links.
    ghost_client_present = any(
        GHOST_SYMBOL in s.base and GHOST_TARGET[: -len(".py")].replace("/", ".") in s.base
        for s in state.structural.values()
    )
    if GHOST_TARGET not in state.structural and ghost_client_present and rng.random() < 0.5:
        state.structural[GHOST_TARGET] = Struct("py", f"def {GHOST_SYMBOL}():\n    return 42\n")
        return f"add_file(ghost:{GHOST_TARGET})"
    # Otherwise recreate a just-deleted leaf path (re-links its importers) or a
    # fresh leaf importing an existing symbol.
    lang = _pick(rng, list(LANGS))
    stem = f"gen_{lang}_{state._fresh()}"
    leaf = _new_leaf(state, rng, lang, stem)
    state.leaves[leaf.path] = leaf
    return f"add_file({leaf.path})"


#: Structural files a delete may remove — never go.mod, never the providers whose
#: heritage / interface / REST edges anchor the strict core (deleting those would
#: still be *valid*, but keeping them keeps the edge zoo rich across a long script).
_DELETABLE_STRUCTURAL = frozenset({"py/report.py", "py/handler.py", GHOST_TARGET})


def op_delete_file(state: RepoState, rng: random.Random) -> str | None:
    # Any generator-owned leaf, plus a few non-critical structural files. Deleting
    # a leaf that others import is intentional: its importers become depth-1
    # dependents and re-resolve (their import goes unresolved) — a full rebuild
    # reproduces that exactly. go.mod and the edge-anchoring providers are kept.
    deletable = list(state.leaves) + [p for p in state.structural if p in _DELETABLE_STRUCTURAL]
    target = _pick(rng, deletable)
    if target is None:
        return None
    state.deleted_paths.append(target)
    state.leaves.pop(target, None)
    state.structural.pop(target, None)
    return f"delete_file({target})"


def op_move_file(state: RepoState, rng: random.Random) -> str | None:
    # Move (= delete + re-add) any leaf to a fresh path: its path-derived symbol
    # ids all change, so its importers become depth-1 dependents — the delete + add
    # shape the strict core must reproduce.
    leaf = _pick(rng, list(state.leaves.values()))
    if leaf is None:
        return None
    old_path = leaf.path
    moved = replace(
        leaf,
        stem=f"gen_{leaf.lang}_{state._fresh()}",
        extra_args=dict(leaf.extra_args),
        funcs=list(leaf.funcs),
        calls=list(leaf.calls),
    )
    state.leaves.pop(old_path)
    state.leaves[moved.path] = moved
    state.deleted_paths.append(old_path)
    return f"move_file({old_path}->{moved.path})"


def op_noop_touch(state: RepoState, rng: random.Random) -> str | None:
    # A no-op: pick any file and change nothing (identical content ⇒ no work).
    target = _pick(rng, list(state.leaves) + list(state.structural))
    if target is None:
        return None
    return f"noop_touch({target})"


ALL_OPS = (
    op_body_only,
    op_signature_change,
    op_rename_symbol,
    op_add_symbol,
    op_delete_symbol,
    op_add_file,
    op_delete_file,
    op_move_file,
    op_noop_touch,
)


# ---------------------------------------------------------------------------
# Edit-script generation
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One burst of edits applied together, then checked for equivalence.

    ``tree`` is the full rendered repo *after* this step's ops — an immutable
    snapshot the harness writes to disk, so the generator owns all mutation and
    the harness just replays.
    """

    ops: list[str]  # human-readable op details (for failure reproduction)
    tree: dict[str, str]


@dataclass
class Script:
    """A reproducible edit script: the base tree + a sequence of snapshotted steps."""

    seed: int
    base_tree: dict[str, str]
    steps: list[Step]


class EditScriptGenerator:
    """Deterministically turn a seed into a base repo + a list of edit steps.

    A script = 3-8 random ops spread across 2-5 steps (design §11 / the plan's
    "randomized edit scripts"). :meth:`build` applies the ops step by step,
    snapshotting the rendered tree after each burst, so the harness renders/writes
    each snapshot and asserts equivalence — the generator owns all mutation.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def build(self) -> Script:
        state = RepoState()
        base_tree = state.render()
        total_ops = self.rng.randint(3, 8)
        num_steps = self.rng.randint(2, min(5, total_ops))
        # Partition total_ops into num_steps parts, each >= 1.
        counts = [1] * num_steps
        for _ in range(total_ops - num_steps):
            counts[self.rng.randrange(num_steps)] += 1

        steps: list[Step] = []
        for count in counts:
            applied: list[str] = []
            attempts = 0
            while len(applied) < count and attempts < count * 6:
                attempts += 1
                op = self.rng.choice(ALL_OPS)
                detail = op(state, self.rng)
                if detail is not None:
                    applied.append(detail)
            # Snapshot even a fully no-op burst — the "nothing changed" path is a
            # case worth exercising (unchanged files must be skipped).
            steps.append(Step(ops=applied or ["<no-op burst>"], tree=state.render()))
        return Script(seed=self.seed, base_tree=base_tree, steps=steps)


# ---------------------------------------------------------------------------
# Disk sync
# ---------------------------------------------------------------------------


def write_tree(root: Path, tree: dict[str, str]) -> None:
    """Write every file in *tree* under *root* (initial materialization)."""
    for rel, content in tree.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def sync_disk(root: Path, old_tree: dict[str, str], new_tree: dict[str, str]) -> None:
    """Apply the *diff* between two rendered trees to disk.

    Writes added + content-changed files and unlinks removed ones — the exact
    filesystem delta the watcher/analyze would observe between two saves.
    """
    for rel, content in new_tree.items():
        if old_tree.get(rel) != content:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
    for rel in old_tree:
        if rel not in new_tree:
            (root / rel).unlink(missing_ok=True)
