"""Golden equivalence harness for the single-pass parser refactor (W2.2).

Proves the *live* single-pass ``PythonParser`` / ``RubyParser`` produce a
byte-identical :class:`ParseResult` to the *frozen two-pass reference* for a
broad corpus:

* every ``src/**/*.py`` file in the repo (real, diverse Python),
* every Ruby fixture under ``tests/fixtures/`` (real Ruby),
* the ``CODE`` sample constants used by the existing parser test modules
  (collected programmatically by reflection), and
* a curated set of snippets that exercise the tricky extraction paths where a
  naive fold of the second (call-extraction) pass into the main walk could
  drift: decorators with call arguments, calls in default parameter values,
  ``except`` / ``raise`` variants, ``__all__`` exports, chained/nested calls,
  class superclasses containing calls, Ruby ``locals_`` scope transitions
  (memoization ``||=``, sequential locals, block/method scope boundaries),
  ``require`` hidden inside an assignment RHS, mixins, ``attr_*`` macros, ...

The reference lives in ``tests/core/golden_reference`` and is an exact copy of
each parser *before* the refactor. While both sides are the two-pass
implementation the harness is a tautology; once the live parser becomes
single-pass it turns into the equivalence contract.

Equality is full structural :class:`ParseResult` equality — every field
(symbols, imports, calls **including order and duplicates**, type_refs,
heritage, exports, variable_types, endpoints, http_calls) must match exactly.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import ModuleType

import pytest

from synaptiq.core.parsers.base import ParseResult
from synaptiq.core.parsers.python_lang import PythonParser
from synaptiq.core.parsers.ruby_lang import RubyParser
from tests.core import test_parser_python, test_parser_ruby
from tests.core.golden_reference.reference_python_lang import (
    PythonParser as RefPythonParser,
)
from tests.core.golden_reference.reference_ruby_lang import (
    RubyParser as RefRubyParser,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Corpus collection
# ---------------------------------------------------------------------------


def _files(root: Path, glob: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in sorted(root.rglob(glob)):
        out.append((str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8")))
    return out


def _code_constants(module: ModuleType) -> list[tuple[str, str]]:
    """Pull class-level ``CODE`` string constants out of a test module.

    The existing parser tests keep their representative sample sources on the
    test classes as a ``CODE`` attribute; those are collected here so the
    harness rides on exactly the sources the suite already trusts.
    """
    out: list[tuple[str, str]] = []
    for cls_name, cls in vars(module).items():
        if not inspect.isclass(cls):
            continue
        code = cls.__dict__.get("CODE")
        if isinstance(code, str) and code.strip():
            out.append((f"{module.__name__}:{cls_name}.CODE", code))
    return out


# --- Curated Python snippets: the fold-sensitive constructs -----------------

PYTHON_SNIPPETS: list[tuple[str, str]] = [
    ("py/empty", ""),
    ("py/whitespace", "\n\n   \n\t\n"),
    ("py/comment_only", "# just a comment\n# another\n"),
    ("py/module_level_call", "setup()\nconfigure(app)\n"),
    (
        "py/decorator_with_call_args",
        '@app.route("/x", methods=["GET"])\n'
        "def index():\n"
        "    return render()\n",
    ),
    (
        "py/decorator_call_and_body_calls",
        "@server.list_tools()\n"
        "async def list_tools():\n"
        "    prepare()\n"
        "    return build()\n",
    ),
    (
        "py/default_value_calls",
        "def f(x=compute(), y=make(default())):\n"
        "    return x\n",
    ),
    (
        "py/lambda_default_with_call",
        "def f(cb=lambda: g()):\n"
        "    return cb()\n",
    ),
    (
        "py/except_variants",
        "def run():\n"
        "    try:\n"
        "        do()\n"
        "    except ValueError:\n"
        "        handle_a()\n"
        "    except (KeyError, IndexError):\n"
        "        handle_b()\n"
        "    except OSError as e:\n"
        "        handle_c(e)\n"
        "    except (TypeError, RuntimeError) as e:\n"
        "        handle_d(e)\n",
    ),
    (
        "py/raise_variants",
        "def run():\n"
        "    if a:\n"
        "        raise ValueError\n"
        "    if b:\n"
        "        raise RuntimeError('boom')\n"
        "    raise build_error()\n",
    ),
    (
        "py/all_exports_list",
        '__all__ = ["Foo", "bar", "baz"]\n'
        "def bar():\n"
        "    pass\n",
    ),
    (
        "py/all_exports_tuple_and_call",
        "__all__ = ('Alpha', 'Beta')\n"
        "extra = build_all()\n",
    ),
    (
        "py/chained_and_nested_calls",
        "def run():\n"
        "    obj.method1().method2()\n"
        "    outer(inner(deep()))\n"
        "    self.logger.info('x')\n"
        "    get_user().save()\n",
    ),
    (
        "py/keyword_and_identifier_args",
        "def wire():\n"
        "    register(Depends(get_db))\n"
        "    items = map(transform, values)\n"
        "    run(callback=handler, other=thing)\n",
    ),
    (
        "py/superclass_with_call_and_subscript",
        "class Foo(make_base()):\n"
        "    pass\n"
        "class Bar(Generic[T]):\n"
        "    pass\n"
        "class Baz(module.Base):\n"
        "    pass\n",
    ),
    (
        "py/nested_scopes",
        "class A:\n"
        "    def m(self):\n"
        "        def inner():\n"
        "            helper()\n"
        "        class Local:\n"
        "            def n(self):\n"
        "                deep()\n"
        "        return inner\n"
        "def outer():\n"
        "    class C:\n"
        "        def meth(self):\n"
        "            pass\n",
    ),
    (
        "py/def_in_if_inside_class",
        "class A:\n"
        "    if True:\n"
        "        def conditional(self):\n"
        "            work()\n"
        "    else:\n"
        "        def other(self):\n"
        "            work2()\n",
    ),
    (
        "py/variable_annotation_with_call",
        "config: AppConfig = load_config()\n"
        "result: AuthResult = authenticate(user)\n"
        "plain = just_a_call()\n",
    ),
    (
        "py/decorated_class_and_methods",
        "@dataclass\n"
        "class Config:\n"
        "    name: str\n"
        "    @property\n"
        "    def upper(self) -> str:\n"
        "        return transform(self.name)\n"
        "    @staticmethod\n"
        "    @cache\n"
        "    def build():\n"
        "        return construct()\n",
    ),
    (
        "py/comprehensions_with_calls",
        "def run(items):\n"
        "    return [transform(i) for i in fetch(items) if valid(i)]\n",
    ),
    (
        "py/typed_params_and_returns",
        "def handle(user: User, config: Config = default()) -> Response:\n"
        "    result: AuthResult = check(user)\n"
        "    return wrap(result)\n",
    ),
    (
        "py/with_and_walrus",
        "def run():\n"
        "    with open_file() as f:\n"
        "        if (n := read(f)):\n"
        "            process(n)\n",
    ),
]


# --- Curated Ruby snippets: locals_ scope transitions & macro paths ---------

RUBY_SNIPPETS: list[tuple[str, str]] = [
    ("rb/empty", ""),
    ("rb/whitespace", "\n\n   \n"),
    ("rb/comment_only", "# just a comment\n# another\n"),
    ("rb/top_level_calls", "puts 'hi'\nsetup\nconfigure app\n"),
    (
        "rb/memoization_operator_assignment",
        "def cache\n"
        "  @cache ||= build_cache\n"
        "end\n",
    ),
    (
        "rb/sequential_locals",
        "def m\n"
        "  step = 1\n"
        "  count += step\n"
        "  count\n"
        "  total = compute\n"
        "  total\n"
        "end\n",
    ),
    (
        "rb/local_vs_call",
        "def m\n"
        "  count = compute\n"
        "  count\n"
        "  fresh\n"
        "end\n",
    ),
    (
        "rb/params_not_calls",
        "def m(value, other = 1)\n"
        "  value\n"
        "  other\n"
        "  real_call\n"
        "end\n",
    ),
    (
        "rb/block_and_do_block",
        "items.map { |i| transform(i) }\n"
        "items.each do |row|\n"
        "  row\n"
        "  process row\n"
        "end\n",
    ),
    (
        "rb/nested_blocks_scope",
        "def m\n"
        "  outer.each do |a|\n"
        "    a\n"
        "    inner.map { |b| combine(a, b) }\n"
        "    leaked\n"
        "  end\n"
        "  a\n"
        "end\n",
    ),
    (
        "rb/chained_calls",
        "a.b.c(x)\n"
        "user&.name\n"
        "Foo.create(attrs)\n",
    ),
    (
        "rb/assignment_rhs_calls",
        "def m\n"
        "  result = fetch_data\n"
        "  result\n"
        "  config = Settings.load(path)\n"
        "  config\n"
        "end\n",
    ),
    (
        "rb/constant_with_call_rhs",
        "CONFIG = build_config\n"
        "TABLE = Registry.build(:main)\n",
    ),
    (
        "rb/require_hidden_in_assignment",
        # The reference two-pass walk records NO import for a require buried in
        # an assignment RHS (the symbol walk never descends into assignments);
        # the single-pass walk must match that exactly, not newly emit one.
        'loaded = require "json"\n'
        'ok = require_relative "./thing"\n',
    ),
    (
        "rb/superclass_with_call",
        "class Foo < Base.for(:widget)\n"
        "  def m\n"
        "    work\n"
        "  end\n"
        "end\n",
    ),
    (
        "rb/class_with_macros_and_calls",
        "class User < ApplicationRecord\n"
        "  include Trackable\n"
        "  extend Findable\n"
        "  prepend Auditing\n"
        "  attr_accessor :name, :email\n"
        "  attr_reader :id\n"
        "  before_save :normalize\n"
        "  after_commit :notify, :reindex\n"
        "  require 'json'\n"
        "  DEFAULT = compute_default\n"
        "  def save\n"
        "    validate\n"
        "    persist\n"
        "  end\n"
        "end\n",
    ),
    (
        "rb/singleton_methods",
        "class Repo\n"
        "  def self.find(id)\n"
        "    query(id)\n"
        "  end\n"
        "  def self.all\n"
        "    fetch_all\n"
        "  end\n"
        "end\n"
        "def Foo.bar\n"
        "  standalone\n"
        "end\n",
    ),
    (
        "rb/if_then_else_ensure_calls",
        "def m\n"
        "  if cond\n"
        "    a_call\n"
        "  else\n"
        "    b_call\n"
        "  end\n"
        "  begin\n"
        "    risky\n"
        "  rescue => e\n"
        "    recover\n"
        "  ensure\n"
        "    cleanup\n"
        "  end\n"
        "end\n",
    ),
    (
        "rb/block_attached_call_nested_calls",
        "def m\n"
        "  records.each do |r|\n"
        "    transform(r).each { |x| emit(x) }\n"
        "  end\n"
        "end\n",
    ),
    (
        "rb/nested_class_in_method",
        "def m\n"
        "  x = 1\n"
        "  class Local\n"
        "    def inner\n"
        "      work\n"
        "    end\n"
        "  end\n"
        "  x\n"
        "end\n",
    ),
    (
        "rb/interpolation_and_operators",
        "def m\n"
        '  greeting = "hi #{name}"\n'
        "  a + b\n"
        "  compute\n"
        "end\n",
    ),
    (
        "rb/predicate_bang_setter_names",
        "def valid?\nend\n"
        "def save!\nend\n"
        "def name=(v)\nend\n",
    ),
    (
        "rb/syntax_error_partial",
        "class Good\n"
        "  def ok\n"
        "    work\n"
        "  end\n"
        "end\n"
        "@@@ bad tokens @@@\n",
    ),
    (
        "rb/non_ascii_content",
        "# émojis 🎉🚀 and åäö accented\n"
        'GREETING = "héllo wörld"\n'
        "class Greeter\n"
        "  # ünïcödé comment\n"
        "  def hello(name)\n"
        '    puts "héllo, #{name}"\n'
        "  end\n"
        "end\n",
    ),
]


def _python_corpus() -> list[tuple[str, str]]:
    return (
        _files(SRC_DIR, "*.py")
        + _code_constants(test_parser_python)
        + PYTHON_SNIPPETS
    )


def _ruby_corpus() -> list[tuple[str, str]]:
    return (
        _files(FIXTURES_DIR, "*.rb")
        + _code_constants(test_parser_ruby)
        + RUBY_SNIPPETS
    )


PYTHON_CORPUS = _python_corpus()
RUBY_CORPUS = _ruby_corpus()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _assert_parse_equal(label: str, reference: ParseResult, live: ParseResult) -> None:
    """Field-by-field equality with a precise message on the first divergence."""
    for f in dataclasses.fields(ParseResult):
        ref_val = getattr(reference, f.name)
        live_val = getattr(live, f.name)
        if ref_val != live_val:
            detail = [
                f"{label}: field '{f.name}' diverged",
                f"  reference ({len(ref_val)}): {ref_val!r}",
                f"  live      ({len(live_val)}): {live_val!r}",
            ]
            if len(ref_val) == len(live_val):
                for i, (r, x) in enumerate(zip(ref_val, live_val)):
                    if r != x:
                        detail.append(f"  first mismatch at index {i}:")
                        detail.append(f"    reference: {r!r}")
                        detail.append(f"    live:      {x!r}")
                        break
            raise AssertionError("\n".join(detail))


@pytest.mark.parametrize(
    "label,source", PYTHON_CORPUS, ids=[c[0] for c in PYTHON_CORPUS]
)
def test_python_parser_equivalence(label: str, source: str) -> None:
    reference = RefPythonParser().parse(source, label)
    live = PythonParser().parse(source, label)
    _assert_parse_equal(label, reference, live)


@pytest.mark.parametrize(
    "label,source", RUBY_CORPUS, ids=[c[0] for c in RUBY_CORPUS]
)
def test_ruby_parser_equivalence(label: str, source: str) -> None:
    reference = RefRubyParser().parse(source, label)
    live = RubyParser().parse(source, label)
    _assert_parse_equal(label, reference, live)


def test_corpus_is_substantial() -> None:
    """Guard against the corpus silently collapsing to nothing."""
    # 58 src files + reflected CODE constants + curated snippets.
    assert len(PYTHON_CORPUS) >= 60
    # Ruby fixtures + reflected CODE constants + curated snippets.
    assert len(RUBY_CORPUS) >= 25
