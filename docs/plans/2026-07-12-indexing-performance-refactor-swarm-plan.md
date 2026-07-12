# Indexing Performance Refactor — Swarm Execution Plan

**Date:** 2026-07-12
**Status:** Ready for execution
**Source:** Full-codebase performance analysis (4-agent sweep; findings verified against source at commit `c1b025b`, v1.5.1)
**Goal:** Materially faster indexing and lower CPU usage with **zero loss of functionality**.

---

## 1. Mission and non-negotiables

Every work package below must satisfy:

1. **No functionality loss.** All existing CLI commands, MCP tools, and output formats behave identically unless the package explicitly adds a new flag/option. Graph contents (nodes, edges, properties) must be equivalent before/after, verified by tests.
2. **Full test suite green:** `uv run pytest` passes. Lint clean: `uv run ruff check src/ tests/`.
3. **Measured, not assumed.** Perf packages must report before/after numbers from the W0 benchmark harness in their PR description.
4. **Conventional commits** (`feat:`, `fix:`, `refactor:`, `perf:`, `test:`), one PR per work package, PR title = package ID + title.
5. **Isolation:** each agent works in its own git worktree/branch off `main`. Never touch files owned by another in-flight package (see conflict matrix §4).

---

## 2. Ground truth (why these packages exist)

Findings from the 2026-07-12 analysis, with source references:

| # | Finding | Evidence |
|---|---------|----------|
| G1 | No incremental indexing: Phase 0 "reserved — not implemented"; every analyze is a full re-index; `bulk_load` wipes the DB | `pipeline.py:7,96-97`, `kuzu_backend.py:1160-1206` |
| G2 | Watcher global phase runs the FULL pipeline every `30s + build_time` under sustained editing; dirty flag cleared **before** build so mid-build edits immediately requeue; no skip-if-clean; no in-flight guard | `watcher.py:31,151-180` |
| G3 | Full-corpus FTS rebuild (all 11 node tables) on **every file save** via `apply_reindex`; 3 of 11 tables (Folder/Community/Process) are never searched but always indexed | `pipeline.py:235`, `kuzu_backend.py:1208-1228,1451-1463`, `_SEARCHABLE_TABLES` at `kuzu_backend.py:45-47` |
| G4 | Symbol extraction is 100% manual Python recursion (zero `tree_sitter.Query`), GIL-serialized despite `ThreadPoolExecutor(8)`; Python and Ruby walk each tree **twice**; ~50-65% of parse CPU | `parser_phase.py:206`, `python_lang.py:54-57`, `ruby_lang.py:87-88`; TS single-pass template at `typescript.py:88-90,138-139` |
| G5 | `analyze` never calls `set_profile` → interactive profile = **all cores** for Kuzu and ONNX; no `--jobs`/`--threads` flag; first-run ONNX embedding (~19k encodes) is the largest single cost | `cli/main.py:164-253`, `resources.py:82-83`, `embedder.py:30-39,103-105` |
| G6 | Kuzu COPY forced `PARALLEL=false` (embedded newlines in code content break the parallel CSV reader); 384-float vectors serialized as decimal strings into CSV | `kuzu_backend.py:1229-1252,1249,1394` |
| G7 | rest_linking: nested http-call × endpoint loop with a full `graph.iter_nodes()` scan inside on match; unconditional 2-3 regexes/line over every file | `rest_linking.py:284-290,307-329,405-407` |
| G8 | Watcher row inserts auto-commit per row, no prepared-statement reuse | `kuzu_backend.py:1465-1547` |
| G9 | `deduplicate_flows` is O(flows²); Ruby import resolution scans+sorts the file index per unresolved require; `assign_symbol_ids` recomputed in calls phase | `processes.py:281-316`, `imports.py:336-375`, `calls.py:491` |
| G10 | Two full builds can run concurrently in one server process (watcher global phase + socket `reindex`); only the DB commit is lock-serialized, not the CPU build | `watcher.py:165`, `cli/main.py:834`, `socket_server.py:42` |
| G11 | Coupling phase is one `git log` subprocess (GIL-releasing) that runs serially at the end; `{imports, heritage, types, rest_linking}` are mutually independent after parsing; hard serial tail is community→processes→dead_code | `coupling.py:52-67`, dependency analysis §7 |
| G12 | Bug: Ruby parser slices Python `str` by tree-sitter **byte** offsets → content/signature drift on non-ASCII files | `ruby_lang.py:155,192,219,251,282`; correct pattern at `python_lang.py:106-108` |
| G13 | Bug/waste: MCP `cycles` tool materializes the entire graph + runs SCC on every call | `mcp/tools.py:1129-1145` |
| G14 | Strategic risk: KuzuDB abandoned upstream — verified 2026-07-12 via GitHub API (`archived: true`, last push 2025-10-10) and PyPI (frozen at 0.11.3). Ecosystem forked, not dead: LadybugDB ("the KuzuDB successor", enterprise-backed), Vela fork (adds concurrent multi-writer), RyuGraph, bighorn. `StorageBackend` Protocol is the insulation layer | `storage/base.py`; GitHub/PyPI APIs |

> Line numbers are as of commit `c1b025b`. Agents MUST re-verify each reference before editing — do not patch blind.

---

## 3. Execution model

- **Waves run in order** (W0 → W1 → W2 → W3). A wave starts only when its gate (§8) passes.
- **Within a wave, packages run in parallel**, one agent per package, **except where the conflict matrix (§4) forces sequencing.**
- Each agent: fresh worktree off `main` → implement → test → benchmark (if perf) → PR. Packages are sized for one agent context each.
- **Verification protocol for perf packages:** run the W0 harness before change (on `main`) and after change (on branch), same machine, embeddings skipped unless the package targets embeddings. Include the two `--profile` outputs in the PR.
- **Coordination:** the orchestrator (main session) reviews each PR against acceptance criteria before merge; merge order within a wave follows the conflict matrix.

## 4. File-conflict matrix (parallelism constraints)

| File | Packages touching it | Rule |
|------|----------------------|------|
| `core/ingestion/watcher.py` | W1.1 (all of G2+G10) | Single package owns it — debounce + guard merged into W1.1 |
| `core/ingestion/pipeline.py` | W0.1 (timings), W1.2 (FTS defer), W2.4 (phase overlap) | W0.1 merges first; W1.2 and W2.4 are in different waves — no conflict |
| `core/storage/kuzu_backend.py` | W1.3 (FTS tables), W1.6 (txn/prepared), W2.3 (Parquet COPY) | W1.3 and W1.6 touch disjoint functions but same file: **sequence W1.3 → W1.6** (W1.3 merges first). W2.3 is next wave |
| `cli/main.py` | W0.1 (`--profile`), W1.4 (`--jobs`) | **Sequence W0.1 → W1.4** |
| `core/resources.py` | W1.4 only | — |
| `core/ingestion/rest_linking.py` | W1.5 only | — |
| `core/ingestion/parser_phase.py` | W2.1 only | — |
| `core/parsers/python_lang.py`, `ruby_lang.py` | W2.2 (single-pass), B1 (byte offsets, ruby only) | **B1 merges before W2.2** (bug fix first, then refactor rides on fixed baseline) |
| `core/ingestion/{processes,imports,calls}.py` | W2.5 (three independent sub-fixes) | May be one agent or three; files are disjoint |
| `mcp/tools.py` | B2 only | — |

---

## 5. Wave 0 — Measurement first (1 package, blocks everything)

### W0.1 — Per-phase timing + benchmark harness
- **Why:** No observability exists; every later package must prove its win.
- **Files:** `core/ingestion/pipeline.py`, `cli/main.py`, new `scripts/bench_index.py`, new test.
- **Changes:**
  1. Capture per-phase wall-time in `run_pipeline` (wrap each `report(...)` pair; store `dict[str, float]` on `PipelineResult.phase_timings`).
  2. `synaptiq analyze --profile` prints a phase-timing table (rich) after the run; also written into `meta.json` under `stats.phase_timings`.
  3. `scripts/bench_index.py`: runs `analyze` N times (default 3) against a target path with `--no-embeddings` and once with embeddings, prints median per-phase table + total; `--json` output for PR pasting.
- **Acceptance:** `--profile` shows all 12 phases + storage + embeddings; timings sum ≈ `duration_seconds`; no behavior change without the flag; tests cover `PipelineResult.phase_timings` population.
- **Risk:** minimal.

---

## 6. Wave 1 — Steady-state CPU + quick wins (8 packages)

### W1.1 — Watcher: quiescence debounce, skip-if-clean, single-flight builds
- **Files:** `core/ingestion/watcher.py` (+ small hook in `cli/main.py` socket reindex path for the shared guard).
- **Changes:**
  1. **Quiescence debounce:** global phase fires only after N seconds (default 30, configurable) with **no new changes**; a change during the wait resets the timer. (Replaces: clear-flag-then-build.)
  2. **Skip-if-clean:** before building, compute a cheap fingerprint (sorted changed-file set + content hashes accumulated since last successful commit); if the fingerprint matches the last committed build, skip.
  3. **Single-flight guard:** one in-process `asyncio.Lock`/flag shared by the watcher global phase and the socket-delivered `reindex` handler; a second trigger while building coalesces into one follow-up run (never queues more than one).
  4. Cap: if quiescence never occurs (pathological churn), force a build after a max-staleness ceiling (default 10 min) so the index can't stay stale forever.
- **Owner decision (default chosen, override welcome):** debounce=30s quiescence, max-staleness=600s.
- **Acceptance:** simulated edit-burst test proves exactly one global build after quiescence; concurrent trigger test proves single-flight; unchanged-repo trigger performs zero pipeline work; existing watcher tests pass.
- **Risk:** low-medium (async timing tests). **Expected win:** removes most steady-state CPU during active editing.

### W1.2 — Stop full FTS rebuild on every file save
- **Files:** `core/ingestion/pipeline.py` (`apply_reindex`), `core/ingestion/watcher.py` (flag plumb), `core/storage/kuzu_backend.py` (no-op if nothing needed).
- **Changes:** remove `rebuild_fts_indexes()` from the per-save path; mark FTS dirty; the W1.1 global phase (or an idle timer ≤60s) performs one rebuild. Search must degrade gracefully in the window: BM25 serves the last-built corpus (documented, acceptable — vectors already behave this way, see `watcher` notes).
- **Acceptance:** per-save path performs no FTS rebuild (assert via backend spy in test); FTS reflects changes after the next global phase; hybrid search never errors in the stale window.
- **Risk:** medium (stale-window semantics — mirror the existing stale-embedding precedent). **Expected win:** per-save cost drops from O(corpus) to O(file).

### W1.3 — FTS only for searchable tables
- **Files:** `core/storage/kuzu_backend.py:1208-1228,1451-1463`.
- **Changes:** iterate `_SEARCHABLE_TABLES` (8) instead of `_NODE_TABLE_NAMES` (11) in both `_create_fts_indexes` and `rebuild_fts_indexes`; guard `query_fts` against the removed indexes (verify no caller searches Folder/Community/Process — confirmed by `_SEARCHABLE_TABLES` usage).
- **Acceptance:** rebuild builds exactly 8 indexes; all search tests pass.
- **Risk:** minimal. **Win:** ~25% of every FTS rebuild. **Sequencing:** merge before W1.6 (same file).

### W1.4 — `--jobs` flag + polite default CPU usage for `analyze`
- **Files:** `cli/main.py`, `core/resources.py`, `core/ingestion/parser_phase.py`/`walker.py` (accept worker count).
- **Changes:**
  1. `synaptiq analyze --jobs N` caps: Kuzu threads, ONNX embed threads, parse/walk worker pools. Default `0` = current all-cores behavior… **except** embed threads default to `max(2, cores - 2)` so a foreground index leaves the machine usable.
  2. Env vars still override (documented precedence: flag > env > profile).
  3. Replace hardcoded `max_workers=8` (`walker.py:154`, `parser_phase.py:183`) with a value derived from the resolved limits.
- **Owner decision (default chosen):** default embed cap `cores - 2`; full all-cores available via `--jobs 0`/env.
- **Acceptance:** `--jobs 2` measurably bounds CPU (assert thread-pool sizes via unit tests, not load tests); defaults documented in `--help`; README note.
- **Risk:** low. **Win:** directly addresses the CPU complaint.

### W1.5 — rest_linking: index the match loop, pre-filter the scan
- **Files:** `core/ingestion/rest_linking.py`.
- **Changes:**
  1. Build once: `(file_path, function_name) → node_id` dict and `name → [node_ids]` (reuse `symbol_lookup` helpers if applicable) — kills the `iter_nodes()` scan at `:405-407`.
  2. Bucket endpoints by `(http_method, normalized_url)` → O(1) candidate lookup instead of the nested loop at `:307-308`.
  3. Cheap substring pre-filter per file before the per-line regex pass (skip files containing none of: `fetch`, `axios`, `requests`, `http`, `@app`, `HTTParty`, `Faraday`, `RestClient`, `Typhoeus`, route verbs, etc. — derive the exact set from the regexes so the filter is provably conservative).
- **Acceptance:** identical REST links on the test fixtures (byte-for-byte edge set — add a golden test first if missing); regression test for the pre-filter's conservativeness (a file matching any regex must pass the filter).
- **Risk:** low with golden test. **Win:** high on service-heavy repos.

### W1.6 — Transactions + prepared statements on the row-insert path
- **Files:** `core/storage/kuzu_backend.py:1465-1547` (+ `apply_reindex` caller).
- **Changes:** wrap `add_nodes`/`add_relationships` batches in explicit `BEGIN TRANSACTION`/`COMMIT`; create prepared statements once per table/rel-pair and reuse across rows; keep the row-by-row fallback semantics identical on error (rollback + surface).
- **Acceptance:** watcher-path integration test still green; micro-benchmark in PR (insert 1k nodes) shows the multiplier; crash-safety: failed batch leaves DB consistent (test with induced failure).
- **Risk:** low-medium (transaction error paths). **Sequencing:** after W1.3.

### W1.7 — Small CPU hygiene (bundled)
- **Files:** `cli/main.py:712-728` (stdin poll), `core/ingestion/walker.py:60` (redundant gitignore re-filter on the git path).
- **Changes:** stdin HUP poll interval 1s → 10s (or `poll(-1)` if POLLHUP-only proves reliable on macOS+Linux); drop the redundant `should_ignore` re-check when the file list came from `git ls-files --exclude-standard` (keep it for extra user patterns from `.synaptiq` config if any — verify before removing).
- **Acceptance:** serve/proxy shutdown-on-hup still works (existing daemon tests); walker returns identical file sets on fixtures.
- **Risk:** low.

### W1.8 — Pin kuzu to the final upstream release (bundled with W1.3/W1.7 agent)
- **Files:** `pyproject.toml`, `uv.lock`.
- **Changes:** `kuzu>=0.11.0` → `kuzu==0.11.3` with a comment explaining why (upstream archived 2025-10-10; a floating floor against a dead upstream invites accidental breakage with no possible benefit; synaptiq installs as an isolated tool so an exact pin is safe). Refresh the lock.
- **Acceptance:** `uv sync` resolves; full suite green on the pinned version.
- **Risk:** none.

---

## 7. Wave 2 — Structural speed (5 packages)

### W2.1 — Process-parallel parsing
- **Files:** `core/ingestion/parser_phase.py`.
- **Changes:** swap `ThreadPoolExecutor` → `ProcessPoolExecutor` for the parse fan-out (`parse_file` is self-contained; `FileParseData`/`ParseResult` are picklable dataclasses). Chunk files (e.g., 32/task) to amortize IPC; keep graph mutation sequential in the parent; workers must `os.nice(5)` when the W1.4 polite mode is active; thread-local parser cache logic becomes per-process (simpler). Fallback to threads on platforms where fork/spawn is problematic (document choice: `spawn` for safety; measure overhead).
- **Acceptance:** identical graph output vs thread version on full fixture set (golden diff of node+edge sets); W0 benchmark shows parse-phase speedup ≈ core count on a large fixture; memory ceiling documented (content is shipped to workers — measure RSS).
- **Risk:** medium (pickling edge cases, spawn overhead on small repos — auto-fallback to threads below a file-count threshold).

### W2.2 — Single-pass tree walks for Python and Ruby
- **Files:** `core/parsers/python_lang.py:54-57`, `core/parsers/ruby_lang.py:87-88` (template: `typescript.py:88-90,138-139`).
- **Changes:** fold call extraction into the main `_walk` so each AST is traversed once. Preserve the documented no-double-count semantics at scope boundaries (`python_lang.py:68-71`) and Ruby's `locals_` scope tracking (`ruby_lang.py:395-439`).
- **Acceptance:** byte-identical `ParseResult` (symbols, calls incl. order-insensitive comparison, heritage, type_refs, exports) on the entire test corpus vs `main` — write the comparison harness first.
- **Risk:** medium (subtle extraction semantics — the golden harness is the safety net). **Sequencing:** after B1 merges.

### W2.3 — Arrow/Parquet COPY for bulk load and embeddings
- **Files:** `core/storage/kuzu_backend.py:1229-1404`.
- **Changes:** replace temp-CSV + `COPY (PARALLEL=false)` with `COPY FROM` a pyarrow Table (or Parquet temp file) for nodes, rels, and embeddings (vector column as `list<float>` → `FLOAT[384]`, eliminating per-float stringification at `:1394`). **Spike first:** verify the pinned Kuzu version supports the Arrow/Parquet path with multiline strings and fixed-size float lists; if unsupported, fall back to Parquet-file COPY; if that also fails, document and close as blocked (Kuzu is frozen upstream — see G14). Adds `pyarrow` dependency (weigh: it's heavy; make it an optional extra `synaptiq[fast-load]` with CSV fallback if size matters).
- **Acceptance:** identical DB contents (row counts + spot-check queries per table); benchmark shows COPY-stage speedup; crash-safe `.rebuild` swap behavior unchanged; CSV path retained as fallback and still tested.
- **Risk:** medium — gated by the spike.

### W2.4 — Overlap independent phases
- **Files:** `core/ingestion/pipeline.py`.
- **Changes:** start coupling's `git log` subprocess (GIL-releasing, G11) in a background thread right after the walk; join before storage load. Optionally also warm the fastembed model load in parallel with late CPU phases (it's I/O+init heavy). Do **not** thread pure-Python phases (no win under GIL; graph is not thread-safe) — dependency facts in G11.
- **Acceptance:** identical results incl. `coupled_pairs`; wall-clock improvement on repos with long git history; failures in the background thread surface exactly as they do today.
- **Risk:** low (background-thread failure paths must match current behavior).

### W2.6 — Storage successor spike (evaluation only, no migration)
- **Why:** Kuzu upstream is archived (G14). Owner directive 2026-07-12: the successor bet must *improve* the product, not just replace like-for-like. Two forks offer improvement; score them.
- **Candidates:** **LadybugDB** (DuckDB-lineage engineering + enterprise backing → durability and likely faster bulk load) and **Vela-Engineering/kuzu** (concurrent multi-writer → could eventually retire Synaptiq's primary/proxy daemon layer, whose entire existence is a workaround for Kuzu's single-writer model, and shrink write-lock staleness windows). RyuGraph and bighorn are parity-only fallbacks — evaluate only if both leaders fail.
- **Scoring rubric (each item pass/fail + notes):** PyPI wheels published + cadence; Python 3.12/3.13/3.14 support; storage-format compatibility with kuzu 0.11.3 files (or documented migration); Python API compatibility (drop-in for `kuzu_backend.py`?); COPY FROM Arrow/Parquet with multiline strings (unblocks/simplifies W2.3); FTS + HNSW parity; multi-writer semantics (Vela) — real serializability or marketing; license; bus-factor/backing signals; benchmark: bulk_load + hybrid query on the W0 harness vs pinned kuzu.
- **Deliverable:** `docs/plans/2026-XX-storage-successor-evaluation.md` with scores, a recommendation, and an adoption cost estimate. **No production code changes.**
- **Acceptance:** both leaders scored against the full rubric with evidence links; recommendation is falsifiable (states what would change it).
- **Risk:** none (isolated spike). **Gate:** feeds the Wave 3 design review — W3.1 must state which engine the incremental design assumes and keep the Protocol engine-portable either way.

### W2.7 — Replace Kuzu with LadybugDB in full (OWNER DECISION 2026-07-12)
- **Why:** W2.6's evaluation designated LadybugDB (PyPI `ladybug`, MIT, wheels cp310-3.14 incl. Windows, API drop-in, 7/7 Synaptiq-critical ops pass hands-on, 3× faster COPY micro-bench). Owner overrode the cautious dual-backend path: with a near-zero install base there is nothing to migrate — replace outright, no legacy kuzu support.
- **Scope:** swap `kuzu==0.11.3` → `ladybug` (pin the hands-on-tested version, floor-pin acceptable given live upstream) in pyproject + lock; port `kuzu_backend.py` (expected mostly import/module renames given drop-in API — verify every Synaptiq-critical behavior: rel table group + rel_type filtering, properties_json, FTS CREATE/DROP/QUERY, HNSW create/drop/query + pinning constraint, CSV COPY incl. PARALLEL flag semantics, crash-safe `.rebuild` swap, transactions + prepared statements (W1.6), generation counter, DOUBLE[] legacy fallback removal is allowed — no legacy DBs exist); update resources.py env-var naming only if the engine's knobs differ (keep `SYNAPTIQ_KUZU_*` as deprecated aliases for one release); existing on-disk kuzu indexes must fail-safe into a clean full reindex (verify `open_with_recovery` handles a foreign/unreadable DB dir). Update CLAUDE.md + README storage references.
- **Explicitly allowed:** renaming the backend module/class; dropping kuzu-specific workarounds that LadybugDB obsoletes (document each drop in the commit).
- **Gate:** ENTIRE suite incl. e2e green; bench_index before/after (storage phase — evaluation predicts an improvement); manual smoke: `analyze` on a repo with a stale kuzu-era `.synaptiq` dir rebuilds cleanly.
- **Sequencing:** after W2.3 merges (same file). W2.3's Arrow work is retargeted/ported as part of this package if its kuzu-era form doesn't carry over — LadybugDB has native Arrow interop, so the Arrow path may get simpler, not harder.
- **Risk:** medium-high (whole storage surface) — mitigated by the 1018-test suite + e2e as the gate and git-revert as escape hatch.

### W2.5 — Algorithmic fixes bundle (3 independent sub-fixes, may split across agents)
- **a. processes:** `deduplicate_flows` O(n²) → inverted index `node → [flow_idx]`, compare only overlapping flows; top-k slice instead of full sort at `processes.py:238-240`.
- **b. imports (Ruby):** precompute `basename → [file_ids]` once instead of scan+sort per unresolved require (`imports.py:336-375`).
- **c. calls:** make the call index `name → {file: [ids]}` for O(1) same-file resolution; accept `assign_symbol_ids` results from parser_phase instead of recomputing (`calls.py:491`, touches `parser_phase.py` return contract — coordinate if W2.1 is in flight: **merge W2.1 first**).
- **Acceptance (each):** identical edges/flows on fixtures; complexity fix demonstrated with a synthetic large fixture in tests.
- **Risk:** low.

---

## 8. Wave 3 — Incremental indexing (the headline; design-first)

> Single large package with a mandatory design review gate. Do not start implementation until the design doc is approved by the owner.

### W3.1 — Design doc (blocks W3.2)
- Deliverable: `docs/plans/2026-XX-XX-incremental-indexing-design.md` covering:
  - Per-file manifest in `.synaptiq/` (content hash → file's symbols/edges provenance), diffing against the walk.
  - Change scoping: re-parse changed files; re-resolve imports/calls/heritage/types for changed files **plus dependents** (files whose edges point at changed symbols — obtainable from the graph itself: reverse IMPORTS/CALLS closure, depth-bounded).
  - Global phases policy: community/processes/dead_code recomputed when the affected-symbol set is non-trivial (threshold) or on the max-staleness timer; coupling on git-HEAD change only.
  - Storage deltas: scoped delete+insert (exists: `remove_nodes_by_file`, W1.6 batched inserts) vs full `bulk_load` when change ratio > threshold (e.g., >30% files → full rebuild is cheaper).
  - Which engine the design assumes (input: W2.6 recommendation); the Protocol stays engine-portable either way.
  - FTS/vector index policy in incremental mode; `meta.json` versioning; fallback to full rebuild on manifest corruption/version mismatch; `--full` escape hatch (already a flag, currently a no-op — G1).
### W3.2 — Implementation per approved design
- **Acceptance (end-state):** touching 1 file and re-running `analyze` re-parses O(1) files and completes in seconds on a ~19k-symbol repo; equivalence test: incremental result graph ≡ full-rebuild graph on randomized edit scripts (property-based test); all staleness windows documented.
- **Sub-packages defined at design time:** manifest, scoping, storage deltas, global-phase policy, equivalence harness.

---

## 9. Bug-fix track (parallel with Wave 1; independent files)

### B1 — Ruby byte-offset slicing (correctness)
- **Files:** `core/parsers/ruby_lang.py:155,192,219,251,282`.
- **Changes:** slice via encoded bytes (`content.encode` once + `bytes[start_byte:end_byte].decode`) or track the source as `bytes` like `python_lang.py:106-108` does. Add non-ASCII Ruby fixtures (comments + identifiers with UTF-8).
- **Acceptance:** correct content/signature/lines for non-ASCII fixtures; existing Ruby tests green. **Merge before W2.2.**

### B2 — Cache MCP `cycles` per index version
- **Files:** `mcp/tools.py:1129-1145`.
- **Changes:** compute SCCs once per index generation (key: `meta.json` `last_indexed_at` or a monotonic index version), cache result like dead-code/communities; invalidate on reindex commit.
- **Acceptance:** second `cycles` call performs no `load_graph`; results identical; invalidation test across a reindex.

### B3 — (absorbed into W1.1 item 3 — single-flight guard covers G10)

---

## 10. Wave gates

- **Gate W0→W1:** W0.1 merged; baseline numbers recorded in `docs/plans/benchmarks/2026-07-baseline.md` (bench on a real large repo + the synaptiq repo itself).
- **Gate W1→W2:** all W1 packages + B1/B2 merged; steady-state check: watcher under a scripted edit-burst shows ≤1 global rebuild per quiescence period; suite green.
- **Gate W2→W3:** W2 merged; first-index benchmark shows cumulative ≥2× improvement vs baseline (expected: parse ~cores×, COPY 2-5× on its stage); W3.1 design approved by owner.
- **Release policy:** version bump + release after each wave completes (patch for W1, minor for W2 and W3), per `scripts/release.sh` flow.

## 11. Wave 4 — Product improvements (owner directive: improve, don't just match)

Specs written at the W3 gate; listed here so waves 1-3 keep their seams open. Priority order:

1. **W4.1 Lazy background embeddings** — commit the graph first, embed afterward; index queryable in seconds instead of minutes. `analyze` gains `--embeddings=lazy|sync|off` (default lazy). Builds on W1.4's threading seams and the existing stale-vector precedent.
2. **W4.2 Go language support** — the highest-value coverage gap for real users (polyglot monorepos). New `go_lang.py` parser via tree-sitter-go extending `BaseParser`, plus calls/imports/heritage semantics; single-pass walk from day one (post-W2.2 template). SQL stays out of scope: it needs schema/lineage modeling, not a call graph — revisit only as a distinct node-type design.
3. **W4.3 Adopt storage successor** (per W2.6 recommendation + W3 design) — if Vela-style multi-writer wins, follow-up: simplify/retire the primary/proxy daemon layer (`core/daemon/`), removing a whole class of lock/staleness failure modes.
4. **W4.4 `--fast-embeddings` (model2vec/static)** — orders-of-magnitude cheaper vectors for CI and low-power machines; quality-tiered alongside bge-small.
5. **W4.5 Index freshness in MCP** — every tool response carries index age + dirty state so agents can trust or request a refresh; pairs with incremental indexing (W3) to make refresh cheap.

## 12. Explicitly out of scope (deferred)

- **tree-sitter Query API rewrite** of extraction (higher ceiling than W2.1/W2.2 but riskier; re-evaluate after W2 benchmarks).
- **Rust/PyO3 extraction extension** (only if post-W3 profiles still show extraction dominating).
- **Hand-rolled DuckDB/SQLite re-platform** (dominated by the fork path per W2.6 analysis).
- **SQL language support** (different analysis model; see W4.2 note).

## 13. Agent operating notes

- Re-read every cited line range before editing (line numbers drift).
- Golden-output harnesses come **first** in W1.5, W2.1, W2.2 — equivalence is the contract.
- Do not run benchmarks or full `analyze` on the owner's machine while another index is running; coordinate via the orchestrator.
- Tests live under `tests/core/`, `tests/cli/`, `tests/mcp/`; async tests use `asyncio_mode=auto`. Style: ruff `E,F,I,N,W`, line length 100, Python 3.11 target.
- When a package's assumptions don't match the code you find, stop and report to the orchestrator — do not improvise scope.
