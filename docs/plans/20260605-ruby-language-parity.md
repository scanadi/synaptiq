# Ruby Language Support — Full Parity

## Overview
Add Ruby (`.rb`) as a first-class language in Synaptiq's ingestion pipeline, at
**full parity** with the existing Python and TypeScript support. After this work,
`synaptiq analyze` on a Ruby/Rails/Sinatra codebase produces a knowledge graph
with symbols, imports, calls, class inheritance, **module mixins**, framework
entry points, REST endpoint linking, and dead-code detection — queryable through
the same CLI commands and MCP tools as Python/TS.

**Problem it solves:** Synaptiq currently ignores `.rb` files entirely (the walker
drops any extension not in `SUPPORTED_EXTENSIONS`), so Ruby codebases get no code
intelligence.

**How it integrates:** The architecture is parity-friendly. Parsers emit a
language-agnostic `ParseResult` IR (`core/parsers/base.py`); the 11 downstream
phases consume the IR + graph, not the AST. Adding Ruby is therefore:
1. One new parser (`ruby_lang.py`) that emits the existing IR dataclasses.
2. Plumbing (extension map, parser dispatch, dependency).
3. Two new graph enum members (`MODULE`, `MIXES_IN`) for Ruby's module/mixin model.
4. Per-language branches in the 4 language-coupled phases (`imports`, `processes`,
   `dead_code`, `rest_linking`) plus the call blocklist.

## Context (from discovery)

**Files/components involved:**
- Plumbing: `pyproject.toml`, `src/synaptiq/config/languages.py`,
  `src/synaptiq/core/ingestion/parser_phase.py` (`get_parser`, `_KIND_TO_LABEL`)
- New parser: `src/synaptiq/core/parsers/ruby_lang.py` (+ tests)
- Graph model: `src/synaptiq/core/graph/model.py` (`NodeLabel`, `RelType`)
- MCP schema doc: `src/synaptiq/mcp/resources.py` (`get_schema`, hand-written)
- Language-coupled phases: `imports.py`, `heritage.py`, `processes.py`,
  `dead_code.py`, `rest_linking.py`, `calls.py`

**Related patterns found:**
- `python_lang.py` (614 lines) and `typescript.py` (831 lines) are the reference
  parser implementations; both subclass `LanguageParser` and implement `parse()`.
- Parser tests mirror: `tests/core/test_parser_python.py`,
  `tests/core/test_parser_typescript.py`, `tests/core/test_parser_phase.py`.
- The walker assigns `FileEntry.language` via `config.languages.get_language()`,
  so a new extension entry auto-flows through walker → parser_phase.
- Heritage uses `(class_name, kind, parent_name)` tuples; `kind ∈ {extends,
  implements}` maps via `_KIND_TO_REL`. `_HERITAGE_LABELS = (CLASS, INTERFACE)`.
- Entry-point detection (`processes._matches_framework_pattern`) and dead-code
  exemptions (`dead_code.py`) branch on language/extension and decorator props.
- REST extraction (`rest_linking.extract_rest_info_from_source`) is a regex pass
  branching on `language`; endpoints/HTTP calls also flow from the parser IR
  (`EndpointInfo`, `HttpCallInfo`).

**Dependencies identified:**
- `tree-sitter-ruby` is published on PyPI (mirrors `tree-sitter-python`/`-javascript`).
- **Kuzu schema auto-derives from `NodeLabel`** (`_NODE_TABLE_NAMES`,
  `_LABEL_TO_TABLE`, `_LABEL_MAP` are comprehensions over the enum), and **all
  relationships share one `REL TABLE GROUP`** with `rel_type` as a string column.
  → Adding `NodeLabel.MODULE` + `RelType.MIXES_IN` requires **no manual Kuzu DDL**;
  only the enums + the hand-written `resources.get_schema()` text need editing.

## Development Approach
- **Testing approach: TDD (tests first)** — write failing parser/phase tests, then
  implement until green, per task.
- Complete each task fully before moving to the next.
- Make small, focused changes.
- **CRITICAL: every task MUST include new/updated tests.** Unit tests for new and
  modified functions, success + error/edge scenarios.
- **CRITICAL: all tests must pass before starting the next task.**
- **CRITICAL: update this plan file when scope changes during implementation.**
- Run tests after each change. Maintain backward compatibility (Python/TS behavior
  must not regress).

## Testing Strategy
- **Unit tests**: required for every task. New parser → dedicated
  `tests/core/test_parser_ruby.py`. Phase changes → extend the relevant phase test.
- **No UI/Playwright e2e** in this project. The closest equivalent is a
  full-pipeline integration test that indexes a small Ruby fixture project and
  asserts on the resulting graph — added in Task 12.
- Lint/format gate: `uv run ruff check src/ tests/` and `uv run ruff format` must
  pass before each task is considered done.

## Progress Tracking
- Mark completed items with `[x]` immediately when done.
- Add newly discovered tasks with ➕ prefix.
- Document blockers with ⚠️ prefix.
- Keep the plan in sync with actual work.

## Solution Overview

**High-level approach:** Build the Ruby parser to emit the existing IR, extend the
graph vocabulary minimally for Ruby's module/mixin semantics, then teach the four
language-coupled phases about Ruby idioms.

**Key design decisions:**
- **Modules & mixins:** model Ruby `module` as a `NodeLabel.MODULE` node and
  `include`/`extend`/`prepend` as `RelType.MIXES_IN` edges (chosen over reusing
  CLASS+IMPLEMENTS for Ruby-idiomatic precision). Class inheritance
  (`class A < B`) stays `EXTENDS`. The parser emits a new heritage `kind="mixin"`.
- **Type extraction skipped:** plain Ruby has no type annotations; `types.py` is
  left untouched and `TypeRef` emission is out of scope (documented as future
  Sorbet/RBS work).
- **Constructor name:** Ruby's `initialize` is added to constructor exemption sets
  alongside `__init__`/`constructor`.
- **Metaprogramming pragmatism:** dead-code detection cannot follow `send`/
  `method_missing`/`define_method`. We add conservative exemptions (attr-generated
  accessors, Rails callbacks, `method_missing`) rather than attempting to resolve
  dynamic dispatch.

**How it fits:** No changes to the daemon, search, embeddings, community, coupling,
or storage backends beyond the auto-derived schema tables. CLI/MCP surfaces are
unchanged except the schema description string.

## Technical Details

**IR mapping (Ruby AST → `ParseResult`):**
| Ruby construct | IR output |
|---|---|
| `def foo` (top-level / in module function context) | `SymbolInfo(kind="function")` |
| `class Foo` | `SymbolInfo(kind="class")`; `def`s inside → `kind="method"`, `class_name="Foo"` |
| `module Bar` | `SymbolInfo(kind="module")`; `def`s inside → `kind="method"`, `class_name="Bar"` |
| `class Foo < Base` | heritage `("Foo", "extends", "Base")` |
| `include M` / `extend M` / `prepend M` | heritage `("Foo", "mixin", "M")` |
| `require "x"` / `require_relative "./x"` | `ImportInfo(module, is_relative)` |
| `obj.method(args)` / bare `method` | `CallInfo(name, receiver, arguments)` |
| `attr_accessor :x` etc. | recorded for dead-code exemption (decorator-like prop) |
| Rails `get "/x" => ...` / Sinatra `get "/x" do` | `EndpointInfo` (parser + regex pass) |
| `Net::HTTP` / `HTTParty.get` / `Faraday` | `HttpCallInfo` |

**New `_KIND_TO_LABEL` entry:** `"module": NodeLabel.MODULE` in `parser_phase.py`.

**New heritage wiring:** `_KIND_TO_REL["mixin"] = RelType.MIXES_IN`;
`_HERITAGE_LABELS` gains `NodeLabel.MODULE` so mixin targets resolve.

**Processing flow:** unchanged pipeline order; Ruby files simply route to the new
parser and the phases recognize Ruby via `language == "ruby"` / `.rb` suffix.

## What Goes Where
- **Implementation Steps** (`[ ]`): all code, schema enum, and test changes in this repo.
- **Post-Completion** (no checkboxes): manual validation against a real Rails repo,
  PyPI dependency surface review, and the `synaptiq` skill `SKILL.md` schema note.

## Implementation Steps

### Task 1: Plumbing — dependency, extension map, parser dispatch

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/synaptiq/config/languages.py`
- Modify: `src/synaptiq/core/ingestion/parser_phase.py`
- Modify: `tests/core/test_config.py` (existing — already covers `get_language`/`is_supported`/`SUPPORTED_EXTENSIONS`)
- Create: `tests/core/test_parser_ruby.py` (stub asserting `get_parser("ruby")` works)

- [x] write test (in `tests/core/test_config.py`): `get_language("foo.rb") == "ruby"`, `is_supported("foo.rb")`
- [x] **Decision (resolved): route Ruby special files.** `get_language` currently keys on `Path.suffix` only. Extend it to also recognize suffix-less / non-`.rb` Ruby files by basename so they parse as Ruby (and Task 9 entry heuristics on `Rakefile`/`config.ru` become live).
- [x] write tests (in `tests/core/test_config.py`): special filenames map to `"ruby"` — `Rakefile`, `Gemfile`, `Guardfile`, `Capfile`, `Vagrantfile`, `Brewfile`, `Podfile`; and special extensions — `.rake`, `.gemspec`, `.ru`, `.rbi`
- [x] add a `SPECIAL_FILENAMES: dict[str, str]` map (`{"Rakefile": "ruby", "Gemfile": "ruby", "Guardfile": "ruby", "Capfile": "ruby", "Vagrantfile": "ruby", "Brewfile": "ruby", "Podfile": "ruby"}`) to `config/languages.py`; have `get_language`/`is_supported` consult `Path.name` against it before falling back to suffix lookup
- [x] add `".rake": "ruby"`, `".gemspec": "ruby"`, `".ru": "ruby"`, `".rbi": "ruby"` to `SUPPORTED_EXTENSIONS`
- [x] write test: `parser_phase.get_parser("ruby")` returns a `LanguageParser` (will fail until parser exists — see Task 3 note)
- [x] add `tree-sitter-ruby>=0.23.0` to `pyproject.toml` dependencies; `uv sync --all-extras`
- [x] add `".rb": "ruby"` to `SUPPORTED_EXTENSIONS`
- [x] add `ruby` branch in `get_parser()` importing `RubyParser` (temporary minimal stub class is fine until Task 3); also extend the `ValueError` message to list `ruby`
- [x] run tests + `ruff check` — language/dispatch tests must pass before next task

### Task 2: Graph model — add `MODULE` label and `MIXES_IN` relationship

**Files:**
- Modify: `src/synaptiq/core/graph/model.py`
- Modify: `src/synaptiq/mcp/resources.py` (`get_schema`)
- Modify: `tests/core/test_graph_model.py` (existing — enum tests live here, NOT `test_model.py`)
- Modify: `tests/core/test_graph.py` and/or storage test for schema creation
- Modify: `tests/mcp/` schema resource test (if present)

- [ ] write test (in `tests/core/test_graph_model.py`): `NodeLabel.MODULE.value == "module"`, `RelType.MIXES_IN.value == "mixes_in"`
- [ ] write test: adding a `MODULE` node + a `MIXES_IN` rel round-trips through the Kuzu backend (verifies auto-derived `Module` table + REL TABLE GROUP accept the new types)
- [ ] write test: `resources.get_schema()` mentions `Module` and `MIXES_IN`
- [ ] add `MODULE = "module"` to `NodeLabel` and `MIXES_IN = "mixes_in"` to `RelType`
- [ ] update `resources.get_schema()` — add `Module` to the node-labels block (confirm that block exists to edit) and `MIXES_IN` to the relationship-types block
- [ ] run full storage + model + mcp tests — must pass before next task

### Task 3: Ruby parser — symbol extraction (methods, classes, modules, constants)

**Files:**
- Create: `src/synaptiq/core/parsers/ruby_lang.py`
- Modify: `src/synaptiq/core/ingestion/parser_phase.py` (`_KIND_TO_LABEL["module"]`)
- Modify: `tests/core/test_parser_ruby.py`

- [ ] write tests: top-level `def`, `class` with methods, `module` with methods, nested classes, singleton/`self.` class methods, constants — assert `SymbolInfo` kind/name/class_name/lines/signature
- [ ] implement `RubyParser(LanguageParser)` using `tree_sitter_ruby`, walking `method`, `singleton_method`, `class`, `module`, `assignment`(constants) nodes
- [ ] emit `kind="function"|"class"|"method"|"module"`; set `class_name` for methods nested in class/module
- [ ] add `"module": NodeLabel.MODULE` to `_KIND_TO_LABEL` in `parser_phase.py`
- [ ] handle parse failures gracefully (return empty `ParseResult`, matching existing parsers)
- [ ] write error/edge tests: empty file, syntax error, deeply nested modules
- [ ] run tests + ruff — must pass before next task

### Task 4: Ruby parser — import extraction (`require` / `require_relative` / `autoload`)

**Files:**
- Modify: `src/synaptiq/core/parsers/ruby_lang.py`
- Modify: `tests/core/test_parser_ruby.py`

- [ ] write tests: `require "json"`, `require_relative "../lib/foo"`, `autoload :Bar, "bar"`; assert `ImportInfo.module`, `is_relative`, names
- [ ] implement import extraction from `call` nodes whose method is `require`/`require_relative`/`autoload`/`load`
- [ ] set `is_relative=True` for `require_relative`
- [ ] write edge tests: dynamic require (non-literal arg) ignored safely
- [ ] run tests + ruff — must pass before next task

### Task 5: Ruby parser — call extraction (receivers, `self`, blocks)

**Files:**
- Modify: `src/synaptiq/core/parsers/ruby_lang.py`
- Modify: `tests/core/test_parser_ruby.py`

- [ ] write tests: `foo()`, `obj.bar(x)`, `self.baz`, bare call `helper`, chained calls, block/proc arg callbacks → `CallInfo(name, receiver, arguments)`
- [ ] implement call extraction from `call`/`method_call`/`command` nodes (handle paren-less calls)
- [ ] populate `receiver` (`self`, identifier, or constant) and bare-identifier `arguments`
- [ ] write edge tests: operator methods, safe-navigation `&.`, no false call for local var reference
- [ ] run tests + ruff — must pass before next task

### Task 6: Ruby parser — heritage & mixins (`<`, include/extend/prepend)

**Files:**
- Modify: `src/synaptiq/core/parsers/ruby_lang.py`
- Modify: `tests/core/test_parser_ruby.py`

- [ ] write tests: `class A < B` → `("A","extends","B")`; `include M`/`extend M`/`prepend M` inside a class/module → `("A","mixin","M")`
- [ ] implement superclass extraction from `class` node superclass field
- [ ] implement mixin extraction from `include`/`extend`/`prepend` call nodes, attributing to the enclosing class/module
- [ ] capture `attr_accessor`/`attr_reader`/`attr_writer` symbol names into a node property (for dead-code exemption in Task 10)
- [ ] write edge tests: multiple includes, namespaced parents (`A < Foo::Bar`)
- [ ] run tests + ruff — must pass before next task

### Task 7: Heritage phase — wire `mixin` kind → `MIXES_IN`

**Files:**
- Modify: `src/synaptiq/core/ingestion/heritage.py`
- Modify: `tests/core/test_heritage.py`

- [ ] write tests: Ruby class with `include M` produces a `MIXES_IN` edge from class/module to module node; `class A < B` produces `EXTENDS`
- [ ] add `"mixin": RelType.MIXES_IN` to `_KIND_TO_REL`
- [ ] add `NodeLabel.MODULE` to `_HERITAGE_LABELS` so mixin/parent modules resolve
- [ ] verify same-file-preference resolution still works for module targets
- [ ] write edge test: unresolved external module mixin is skipped without error
- [ ] run tests + ruff — must pass before next task

### Task 8: Imports phase — `_resolve_ruby` (require_relative + Rails autoload)

**Files:**
- Modify: `src/synaptiq/core/ingestion/imports.py`
- Modify: `tests/core/test_imports.py`

- [ ] write tests: `require_relative "../lib/foo"` resolves to `lib/foo.rb` in the file index (assert NON-None); Rails-style implicit `UserService` → `app/services/user_service.rb` (snake_case convention)
- [ ] add `.rb → "ruby"` mapping in `_detect_language` (imports.py:136)
- [ ] add the dispatch arm `if language == "ruby": return _resolve_ruby(...)` in `resolve_import_path` (imports.py:85-92) — **separate checkbox: without this arm Ruby imports silently return None with no error**
- [ ] implement `_resolve_ruby`: relative resolution from importing file dir; `require` against project roots; convention-based (underscore) name→path mapping for autoload-style references
- [ ] write edge tests: gem requires (`require "rails"`) resolve to None (external); missing file → None
- [ ] run tests + ruff — must pass before next task

### Task 9: Processes phase — Ruby framework entry points

**Files:**
- Modify: `src/synaptiq/core/ingestion/processes.py`
- Modify: `tests/core/test_processes.py`

- [ ] write tests: Rails controller actions (methods in `*_controller.rb` classes ending `Controller`), Sinatra route blocks, rake tasks, RSpec `describe`/`it` blocks flagged as entry points
- [ ] add `_RUBY_ENTRY_*` constants (controller/job/mailer suffixes, `app/controllers` paths) and Ruby branch in `_matches_framework_pattern`
- [ ] **condition for the Ruby branch: `language == "ruby" or node.file_path.endswith(".rb")` — do NOT include the `""` empty-language fallback the Python/TS branches use, or it will double-match those files**
- [ ] add Ruby entry filenames/conventions to entry-file heuristics: `*_spec.rb`, `*_test.rb`, plus `Rakefile`, `config.ru`, `*.rake` (these now route to Ruby per Task 1's special-files decision, so the heuristics are live)
- [ ] write edge tests: a plain private method with incoming calls is NOT an entry point
- [ ] run tests + ruff — must pass before next task

### Task 10: Dead-code phase — Ruby exemptions (metaprogramming, Rails, tests)

**Files:**
- Modify: `src/synaptiq/core/ingestion/dead_code.py`
- Modify: `tests/core/test_dead_code.py`

- [ ] write tests: `initialize` exempt (constructor); `method_missing`/`respond_to_missing?` exempt; attr-generated accessors exempt; Rails callbacks (`before_action`, `after_save`, etc.) targets exempt; `*_spec.rb`/`*_test.rb` files treated as test files
- [ ] add `"initialize"` to `_CONSTRUCTOR_NAMES`
- [ ] add Ruby test-file detection to `_is_test_file` (`*_spec.rb`, `*_test.rb`, `spec/`, dirs)
- [ ] add Ruby metaprogramming/framework exemptions (method_missing, attr accessors recorded in Task 6, Rails model bases like `ApplicationRecord`)
- [ ] write edge tests: a genuinely unused private Ruby method IS flagged dead
- [ ] run tests + ruff — must pass before next task

### Task 11: REST linking phase — Ruby endpoints & HTTP calls

**Files:**
- Modify: `src/synaptiq/core/ingestion/rest_linking.py`
- Modify: `src/synaptiq/core/parsers/ruby_lang.py` (emit `EndpointInfo`/`HttpCallInfo` where AST-detectable)
- Modify: `tests/core/test_rest_linking.py`
- Modify: `tests/core/test_parser_ruby.py`

- [ ] write tests: Sinatra `get "/users/:id" do` and Rails `get "/users/:id" => "users#show"` → `EndpointInfo`; `Net::HTTP`/`HTTParty.get`/`Faraday` → `HttpCallInfo`
- [ ] add `.rb → "ruby"` branch in `rest_linking._detect_language` (rest_linking.py:323) — **separate, load-bearing checkbox: the extraction loop does `if not language: continue`, so a missing arm silently drops ALL Ruby endpoints/HTTP calls, even AST-emitted ones**
- [ ] add a `language == "ruby"` regex pass in `extract_rest_info_from_source`
- [ ] add Ruby endpoint/HTTP regex patterns; normalize `:id` path params
- [ ] write edge tests: non-route DSL methods not misdetected
- [ ] run tests + ruff — must pass before next task

### Task 12: Call blocklist + full-pipeline integration test

**Files:**
- Modify: `src/synaptiq/core/ingestion/calls.py` (`_CALL_BLOCKLIST`)
- Create: `tests/fixtures/ruby_project/` (small Rails/Sinatra-flavored sample; `tests/fixtures/` does not yet exist)
- Create: `tests/e2e/test_ruby_pipeline.py` (model structure/assertions on existing `tests/e2e/test_full_pipeline.py`)

- [ ] write integration test: index `tests/fixtures/ruby_project/` end-to-end; assert Function/Class/Module/Method nodes, CALLS, IMPORTS, EXTENDS, MIXES_IN edges, entry points, and dead-code results (must run only after Task 3+ — the Task 1 stub produces zero symbols)
- [ ] add Ruby builtins/Kernel methods to `_CALL_BLOCKLIST` (`puts`, `print`, `p`, `require`, `require_relative`, `attr_accessor`, `attr_reader`, `attr_writer`, `include`, `extend`, `prepend`, `raise`, `loop`, `lambda`, `proc`, `send`, `freeze`, `new`, `to_s`, `to_sym`, `each`, `map`, `select`, etc.)
- [ ] write test asserting blocklisted Ruby builtins produce no CALLS edges
- [ ] run full test suite — must pass before next task

### Task 13: Verify acceptance criteria
- [ ] verify all Overview requirements implemented (symbols, imports, calls, extends, mixins, entry points, REST, dead code)
- [ ] verify Python/TS behavior unchanged (no regressions in existing parser/phase tests)
- [ ] run full suite: `uv run pytest`
- [ ] run fast suite: `uv run pytest tests/core/ tests/cli/ tests/mcp/`
- [ ] lint/format clean: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
- [ ] manual smoke: `uv run synaptiq analyze` on `tests/fixtures/ruby_project/` then `synaptiq status`/`query`

### Task 14: [Final] Documentation
- [ ] update `CLAUDE.md` (Parsers section, supported languages, new `MODULE`/`MIXES_IN` graph vocabulary)
- [ ] update `README.md` language support list
- [ ] update the `synaptiq` skill `SKILL.md` schema section to document `Module` node + `MIXES_IN` edge
- [ ] move this plan to `docs/plans/completed/`

## Post-Completion
*Items requiring manual intervention or external systems — informational only*

**Manual verification:**
- Run `synaptiq analyze` against a real medium/large Rails app and sanity-check
  call-graph density, dead-code false-positive rate (metaprogramming-heavy code is
  the risk area), and import-resolution coverage for autoloaded constants.
- Compare entry-point detection against the app's actual routes/jobs.

**External / future work (out of scope here):**
- Sorbet `sig` / RBS type extraction to populate `USES_TYPE` edges for Ruby.
- ERB/HAML template linking to controller actions.
- Gem dependency graph (Gemfile/`.gemspec`) ingestion.
- Confirm `tree-sitter-ruby` wheels are available for all supported platforms in CI
  before release; update `.github/workflows/publish.yml` only if a build step needs it.
