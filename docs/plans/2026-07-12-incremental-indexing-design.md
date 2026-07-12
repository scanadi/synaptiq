# Incremental Indexing — Design (W3.1)

**Date:** 2026-07-12
**Status:** Design for review (blocks W3.2 implementation)
**Source plan:** `docs/plans/2026-07-12-indexing-performance-refactor-swarm-plan.md` §8 (W3)
**Baseline:** `docs/plans/benchmarks/2026-07-baseline.md` (incl. lvlp-app production validation)
**Code state:** `main` @ `335f12d` (v1.5.1 + Waves 0–2). **Engine: LadybugDB** (kuzu is gone; the plan text at §8 predates W2.7).

---

## 0. TL;DR — the verdict

> **Recommend the leaner design.** Waves 1–2 already delivered "analyze in seconds instead of minutes,"
> so incremental indexing is no longer a *latency* feature — it is a **watcher-CPU / battery** feature and a
> **scale headroom** feature. The plan's headline mechanism ("reverse IMPORTS/CALLS closure, depth-bounded,
> re-resolve dependents") is **more than the numbers justify**. A **symbol-level manifest diff + depth-1 scoped
> re-resolution of high-confidence edges only + deferred FTS/HNSW + threshold-gated global phases**, with the
> already-fast full rebuild retained as the **consolidation pass and correctness safety net**, hits every honest
> target with far less machinery and far less equivalence risk.

The numbers that drive this (from the lvlp-app production validation, 790k LOC / 3,514 files / 22,689 symbols /
115,684 rels / ~26k embeddings):

| Path | Today | Target | Leaner design (projected) |
|---|---|---|---|
| Cold first index, no embeddings | **10.7s** | (keep) | unchanged — first index is always full |
| Warm full re-analyze (embeddings reused) | **15.8s** | (keep) | unchanged — this is the safety net |
| **Watcher per-burst rebuild, 1-file edit** | **~11s CPU** every quiescence window | **≤2s end-to-end** | **~0.1–0.8s** (scoped; no repo-size term except an O(symbols) in-memory index build) |
| Repo 5–10× lvlp-app, 1-file edit | ~60–120s per burst | seconds | scoped work is O(change), not O(repo) |
| Battery / sustained-edit CPU | full pipeline + full FTS + full HNSW per burst | minimize | one parse + a bounded delta; FTS/HNSW deferred |

**Why not the full impact-closure the plan sketched?** Strict equivalence on the graph's *low-confidence* edges
(global-fuzzy CALLS at confidence 0.5 / weak-ref 0.3, `calls.py:44,250`) and on the genuinely-global phases
(Leiden communities, BFS processes) would reintroduce exactly the **unbounded fan-out** the whole refactor is
killing: adding one symbol named `run` would force re-resolution of every file that calls `run`. Those edges are
*already marked uncertain*, and a full rebuild is now cheap enough to be the periodic reconciler. So the
incremental path should be **correct-and-cheap on the high-confidence structural core** and **bounded-stale on the
uncertain fringe**, with staleness closed by the same debounced consolidation the watcher already runs.

Everything below justifies and specifies that design, then decomposes it into six one-context W3.2 sub-packages
with a conflict matrix, and closes with the open decisions for the lead.

---

## 1. Ground truth has shifted — read this before the mechanism

The W3 pitch in the swarm plan ("touching 1 file completes in seconds on a ~19k-symbol repo") is **already true**
for the *user-facing* `analyze` command after Waves 1–2. The remaining, honest problems are:

1. **The watcher's global tier is a full rebuild.** Under sustained editing the debounced global phase
   (`watcher.py:187` `_GlobalPhaseScheduler`) fires once per 30s quiescence window (or the 600s ceiling) and each
   firing runs `build_full_index` → full `run_pipeline` → `commit_full_index` → `bulk_load` (`pipeline.py:368,407`,
   `watcher.py:378` `_on_build`). That is **~11s of CPU for a one-line edit** on lvlp-app, and it repeats every
   window. This is the battery/fan complaint and the thing incremental indexing must fix.

2. **Scale.** 11s is fine at 790k LOC; at 5–10× it is 60–120s, and the watcher would never keep up. Scoped work
   removes the repo-size term.

3. **The immediate tier is already scoped — but lossy and subtly buggy.** The per-batch file-local tier
   (`watcher.py:433–444`) already does `parse_files` (phases 2–7 for the changed files only, `pipeline.py:261`) +
   `apply_reindex` (`pipeline.py:284`: `remove_nodes_by_file` then `add_nodes`/`add_relationships`). It deliberately
   skips FTS, embeddings, and every global phase, and it does **not** re-resolve edges from unchanged files — so
   between saves the graph is missing inbound edges and has stale communities/dead-code. The full rebuild is what
   makes it correct again. Two concrete defects in this existing path (both masked today only because the periodic
   full rebuild wipes the DB) that the incremental design must fix:

   - **Inbound-edge loss.** `remove_nodes_by_file(A)` is `DETACH DELETE` per table (`ladybug_backend.py:658–686`),
     which deletes edges *into* A's symbols (e.g. `C → A.foo` CALLS, `C → file:A` IMPORTS). `apply_reindex`
     re-inserts only A's *outbound* subgraph, so those inbound edges vanish until the next full rebuild.
   - **Edge duplication.** `_insert_relationship` is `CREATE`, **not** `MERGE` (`ladybug_backend.py:2057–2085`;
     contrast `_insert_node`'s `MERGE`, `:2004–2035`). Re-parsing a file re-emits its folder→folder / folder→file
     `CONTAINS` edges (`structure.py:74–110`), which survived the `DETACH DELETE` (folder nodes have a different
     `file_path`), so `add_relationships` **duplicates** them. Harmless today (wiped every ~30s); **accumulates
     permanently** the moment we stop doing periodic full wipes.

**Implication:** the incremental design is not greenfield. It is (a) making the *global tier* scoped instead of
full, and (b) hardening the *immediate tier* into a correct, idempotent delta so it can stand without a full-wipe
safety net every 30 seconds.

---

## 2. Engine assumed: LadybugDB

Per W2.7 the backend is **LadybugDB** (`ladybug`, MIT, PyPI). The `StorageBackend` Protocol
(`core/storage/base.py`) stays the portability seam — every new capability this design needs is added to the
Protocol first and implemented in `ladybug_backend.py`, so a future engine swap re-implements a small, named delta
surface rather than the pipeline. Engine-specific facts this design relies on, all verified in the current backend:

- `bulk_load` builds a sibling `.rebuild` DB and atomically swaps (`ladybug_backend.py:1375–1431`) — crash-safe.
- `add_nodes` = batched-transactional idempotent `MERGE` upsert (W1.6, `:591–604,2004–2035`).
- `add_relationships` = batched-transactional **non-idempotent `CREATE`** (`:606–613,2057–2085`) — **must be
  fixed or bypassed** for deltas (see §6).
- `remove_nodes_by_file` = per-table `DETACH DELETE` + embedding cleanup (`:658–686`).
- `store_embeddings` drops the HNSW index, `MERGE`s vectors (does **not** wipe the table), recreates HNSW
  (`:1151–1212`); `load_embeddings` snapshots `{node_id: (text_sha, vec)}` (`:1214–1234`).
- `rebuild_fts_indexes` = `DROP`+`CREATE` FTS over the 8 `_SEARCHABLE_TABLES` (`:1433–1460`) — **no incremental
  FTS API** (see §7).
- `load_graph` materializes the whole graph into memory (`:929–989`) — an O(nodes+rels) read that includes every
  symbol's full `content`; the design **avoids** it on the hot path (see §5.4).
- `.synaptiq/` holds the single-file DB (`kuzu`, kept for legacy recovery, `cli/main.py:265`) + `meta.json`
  (`pipeline.py:419`), plus transient `.wal`/`.shadow`/`.rebuild`. `open_with_recovery` wipes+rebuilds only on
  verified corruption (`:244–320`); `_verify_schema` rejects old schemas (`:82`, keys off `properties_json`).

---

## 3. Design overview (the leaner mechanism)

```
                          ┌─────────────────────────── file change(s) ───────────────────────────┐
                          ▼                                                                         │
   ┌──────────────────────────────────────────┐                                                    │
   │ 1. PARSE changed files (phase 2–7 inputs) │  parse_file × |changed|      O(change)             │
   └───────────────┬──────────────────────────┘                                                    │
                   ▼                                                                                 │
   ┌──────────────────────────────────────────┐                                                    │
   │ 2. MANIFEST DIFF (per symbol / per edge)  │  old provenance vs new       O(change)             │
   │    → added / removed / body-only-changed  │                                                    │
   └───────────────┬──────────────────────────┘                                                    │
                   ▼                                                                                 │
   ┌──────────────────────────────────────────┐                                                    │
   │ 3. SCOPE PLAN                              │  from manifest + reverse edges                     │
   │    reparse set  = changed files           │                                                    │
   │    reresolve-in = depth-1 dependents IFF   │  only when symbol *identity set* changed           │
   │                   identity set changed     │  (add/remove/rename/export/class/heritage-sig)     │
   │    file add/del → importer re-resolution   │                                                    │
   │    change ratio → full-rebuild fallback?   │  ratio > threshold ⇒ hand off to full bulk_load    │
   └───────────────┬──────────────────────────┘                                                    │
                   ▼                                                                                 │
   ┌──────────────────────────────────────────┐                                                    │
   │ 4. RESOLVE edges (imports/calls/heritage/  │  against a GLOBAL name/file index rebuilt from     │
   │    types) for reparse ∪ reresolve-in       │  the manifest (in-memory dict, no I/O parse)       │
   │    → GraphDelta{nodes±, edges±}            │  high-confidence edges only; fuzzy = best-effort   │
   └───────────────┬──────────────────────────┘                                                    │
                   ▼                                                                                 │
   ┌──────────────────────────────────────────┐                                                    │
   │ 5. APPLY GraphDelta to storage (one txn)   │  surgical: SET body-only, add/remove symbols,      │
   │    + scoped dead_code recount              │  delete-by-endpoint+type then idempotent insert    │
   │    + write manifest                        │  edges; recount is_dead for touched targets        │
   └───────────────┬──────────────────────────┘                                                    │
                   ▼                                                                                 │
   ┌──────────────────────────────────────────┐   mark stale: FTS, HNSW, communities, processes     │
   │ graph is queryable NOW (≤~0.8s)           │   ── consolidated later ──▼                         │
   └────────────────────────────────────────────────────────────────────────────────────────────── │
                                                                                                     ▼
   ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ CONSOLIDATION (debounced 30s-quiescence / 600s ceiling, OR analyze without a running daemon):        │
   │   • coupling  → only if git HEAD moved                                                               │
   │   • community + processes → recompute if accumulated affected-symbol ratio > threshold, else defer   │
   │   • FTS rebuild (8 tables) + HNSW rebuild → once, over the accumulated change set                    │
   │   • if accumulated change ratio > 30% files ⇒ just run the existing full bulk_load (cheaper + simplest)│
   └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The immediate delta (steps 1–5) is the ≤2s path. Consolidation folds the genuinely-global work into the watcher's
**existing** debounce (`_GlobalPhaseScheduler`), so there is no new timer and no new staleness class — FTS,
vectors, communities, and processes already lag per-save today and are documented as such (`pipeline.py:296–331`,
`watcher.py:1–36`). We are **narrowing** what consolidation must redo, not adding a new lag.

---

## 4. Per-file manifest

### 4.1 Purpose

The manifest is the provenance record that lets us answer, without re-reading the whole repo:

1. **What changed?** content-hash per file → the changed-file set (already computed cheaply; the watcher hashes
   content at `watcher.py:87,454`).
2. **What did each file contribute?** the exact symbol IDs and outbound-edge endpoints a file produced, so a
   *symbol-level* diff (not just "file changed") drives surgical storage updates and the identity-change test.
3. **Who depends on a changed file?** the reverse-IMPORTS map and the name→defining-files / name→referencing-files
   maps, so the depth-1 dependent set is a lookup, not a scan.
4. **Is the manifest still trustworthy?** a version + a whole-index fingerprint, so corruption or a schema/tool
   bump falls back to a full rebuild.

### 4.2 Where it lives

**Recommendation: a dedicated table inside the LadybugDB database**, not a sidecar file. Rationale:

- **Atomicity with the graph.** The manifest must never disagree with the stored nodes/edges. A table is written
  in the same transaction as the delta (§6) and rides the same crash-safe `.rebuild` swap on a full build
  (`ladybug_backend.py:1375`). A sidecar JSON is a second thing to keep consistent and a second thing to corrupt
  independently.
- **No new corruption-recovery surface.** `open_with_recovery` already wipes+rebuilds an unreadable DB
  (`:244`); a manifest *table* inside it is covered for free. A sidecar needs its own "missing/corrupt → force
  full" handling (which we still add as defense-in-depth, §8).
- **Scale.** 22,689 symbols is a lot of JSON to parse on every cold `analyze`; a table is queried, and the hot
  in-memory indexes (§5.3) are built from a single scan.

**Trade-off / fallback:** if storing the manifest in-DB complicates the `.rebuild` swap or the Protocol
(LadybugDB has no schema migration for a new user table beyond `CREATE NODE TABLE`), fall back to
`.synaptiq/manifest.json` (or a small `manifest.sqlite`). This is **Decision D3** for the lead. Either way the
*logical* schema below is identical; only the sink changes.

### 4.3 Logical schema

Per file (keyed by repo-relative path):

```
FileManifest {
  path:            str            # repo-relative, posix
  content_sha:     str            # sha256 of raw bytes (== watcher._content_hash)
  language:        str
  symbol_ids:      list[str]      # every node id this file defined (function:…, class:…, method:…, module:…, file:…)
  symbol_sigs:     dict[str,str]  # id → identity fingerprint: sha256(name|kind|class_name|signature|is_exported)
                                  #   — changes iff the symbol's *identity* (not its body) changed
  out_edges:       list[EdgeRef]  # outbound edges this file's resolution produced (imports/calls/heritage/types),
                                  #   as (rel_type, src_id, tgt_id, confidence_bucket) — for exact delete-before-insert
}
```

Index-level (one row / small blob):

```
IndexManifest {
  manifest_version: int          # bump when this schema or the scoping rules change ⇒ force full
  tool_version:     str          # synaptiq __version__ at build time
  git_head:         str|None     # HEAD sha coupling was computed against (or None if not a git repo)
  full_fingerprint: str          # sha256 over sorted(path→content_sha) — the "am I consistent" check
  consolidated_at:  iso8601      # last time FTS/HNSW/community/process were made fresh
  pending:          {            # accumulated-since-consolidation, drives thresholds + staleness reporting
    affected_symbols: int,
    changed_files:    int,
    fts_dirty:        bool,
    hnsw_dirty:       bool,
    community_dirty:  bool,
    process_dirty:    bool,
  }
}
```

`symbol_sigs` is the crux of the leaner design: **body-only edits leave every `symbol_sigs[id]` unchanged**, so
the planner (§5) can prove no dependent needs re-resolution and skip all closure work.

### 4.4 When it's written

- **Full build** (`bulk_load` path, first index / `--full` / ratio-fallback / consolidation-as-full): the pipeline
  emits the complete manifest from the in-memory graph it already holds — a cheap by-product of a build it is
  already doing. (New hook alongside `write_meta`, `pipeline.py:419`.)
- **Incremental delta:** the changed files' `FileManifest` rows are rewritten and `IndexManifest.full_fingerprint`
  / `pending` updated, inside the same storage transaction as the delta.

---

## 5. Change scoping — precise rules

### 5.1 Classify each changed file via the symbol-level diff

For each changed file A (new parse vs manifest):

- **`added`**: symbol ids present now, absent before.
- **`removed`**: present before, absent now.
- **`identity-changed`**: id present in both but `symbol_sigs` differ (rename keeps id only if the id is
  path+name-derived — a rename changes the id, so a rename = `removed` old + `added` new; see node-id format
  `graph/model.py:46`).
- **`body-only`**: id present in both, `symbol_sigs` equal (content/line-range changed but identity did not).

Define **identity-set-changed(A)** := `added(A) ∪ removed(A) ∪ identity-changed(A)` is non-empty.

### 5.2 The two-question scoping rule

> **Q1 — Whose *outbound* edges must be recomputed?** Always the changed files themselves (their calls/imports/
> heritage/types depend on their own new text). This is unconditional and O(change).
>
> **Q2 — Whose edges *into* a changed file must be recomputed (the "dependent closure")?** **Only when
> identity-set-changed(A).** If A's edit is body-only, every inbound edge `C → A.sym` still points at a symbol
> with the same identity, so it remains correct and needs no work. When identity *did* change, the bounded
> depth-1 dependent set is:
>
> ```
> dependents(A) = { files that IMPORT A }                              # reverse-IMPORTS, from manifest/storage
>               ∪ { files whose parse references a name in added(A)     # they may now link to a new symbol
>                   or removed(A) }                                     # name→referencing-files, from manifest
> ```
>
> Re-resolve **only the outbound edges of `dependents(A)`**, and **only the high-confidence kinds** (same-file /
> import-resolved CALLS at confidence 1.0, IMPORTS, EXTENDS/IMPLEMENTS/MIXES_IN, USES_TYPE). Fuzzy/global CALLS
> (confidence 0.5/0.3) are **not** chased across dependents — they are best-effort in the delta and reconciled at
> consolidation (§7, §9).

**Why depth-1 and not transitive.** Adding `C → A.bar` changes C's *outbound* edge set but **not C's symbol
identities** (`symbol_sigs(C)` unchanged), so nothing that imports C needs re-resolution. Edge changes do not
propagate to further identity changes — the closure is provably depth-1 for structural edges. The only phase that
could propagate transitively is reachability, and **dead-code here is not transitive**: `process_dead_code` flags a
symbol iff it has **zero incoming CALLS** and is unexempt (`dead_code.py:640`, `graph.has_incoming`), a purely
*local* in-degree predicate. So a symbol's `is_dead` can flip only if *its own* incoming-CALLS crossed zero — a set
we already know from the edge delta (§5.5). No BFS, no transitive closure.

### 5.3 When a name-index rebuild suffices vs when closure is needed

The plan asked exactly this. The answer falls out of §5.2:

| Change | Global name-index rebuild? | Dependent closure (Q2)? |
|---|---|---|
| **Body-only edit** (the common case: edit a function body) | **Yes** (cheap, in-memory) — needed so A's *own* re-resolved calls can find their targets repo-wide | **No** — inbound edges unchanged |
| **Add/remove/rename symbol** | Yes | **Yes**, depth-1, high-confidence only |
| **File added / deleted** | Yes (file-index changes) | Yes, but scoped to **importers** (IMPORTS resolution depends only on the *set of paths*, `imports.py:65–105`, not on other files' contents) |
| **Pure whitespace / comment (no symbol lines move)** | not even needed if the parse is byte-identical | No |

The "global name-index" is `build_name_index` (`symbol_lookup.py:17`) + `build_file_index` (`imports.py:31`) —
plain `dict` builds over the symbol set. **Crucially these are rebuilt from the manifest (in memory), not from a
re-parse and not from `load_graph`.** Building `{name: [ids]}` and `{path: id}` for 22,689 symbols is a few tens of
ms of dict work; at 200k symbols (a 10× repo) it is still sub-second and holds resident in the daemon. This is the
only term in the immediate path that scales with repo size, and it is the cheap kind of scaling.

### 5.4 What we deliberately do *not* do on the hot path

- **No `load_graph`.** Materializing the whole graph (`ladybug_backend.py:929`) reads every symbol's `content`;
  at lvlp-app scale that is ~1–3s and grows with the repo — it would blow the budget by itself. All whole-graph
  inputs the resolvers need (name-index, file-index, reverse-IMPORTS) come from the manifest instead.
- **No whole-graph community / process recompute.** Deferred to threshold-gated consolidation (§9).
- **No FTS or HNSW rebuild.** Deferred (§7).

### 5.5 Producing the `GraphDelta`

The resolvers (`process_imports`, `process_calls`, `process_heritage`, `process_types`) currently take a
`KnowledgeGraph` and mutate it. For the delta path they run over a **small in-memory graph seeded with only the
reparse ∪ dependents nodes**, but resolve against the **manifest-backed global name/file index**. Output is a
`GraphDelta`:

```
GraphDelta {
  nodes_upsert:  list[GraphNode]      # added + body-only-changed + identity-changed (SET/MERGE by id)
  nodes_remove:  list[str]            # removed symbol ids (+ deleted-file symbol ids)
  edges_add:     list[GraphRelationship]
  edges_remove:  list[EdgeRef]        # (rel_type, src, tgt) tuples to delete before re-insert — from old manifest out_edges
  dead_recount:  set[str]             # symbol ids whose incoming-CALLS may have crossed 0 (targets of added/removed CALLS ∪ nodes_upsert ∪ nodes_remove)
}
```

`edges_remove` comes from the **old** manifest `out_edges` of every file we re-resolve (delete exactly what that
file previously contributed), which is what makes the storage apply idempotent without a global `DETACH DELETE`
(§6). This also repairs the current inbound-edge-loss defect: we no longer `DETACH DELETE` a whole file's node set;
we surgically remove only genuinely-removed symbols (which correctly cascades *their* now-dangling inbound edges)
and leave surviving symbols — and their inbound edges — in place.

---

## 6. Storage deltas

### 6.1 New Protocol method: `apply_graph_delta(delta) -> None`

Added to `StorageBackend` (`core/storage/base.py`) and implemented in `ladybug_backend.py`. One explicit
transaction (reuse `_write_batch`'s BEGIN/COMMIT/rollback discipline, `:615–637`), applied in order:

1. **`edges_remove`** — `MATCH (a)-[r:CodeRelation]->(b) WHERE a.id=$s AND b.id=$t AND r.rel_type=$k DELETE r`.
   Deletes exactly the edges the re-resolved files previously contributed.
2. **`nodes_remove`** — per-table `DETACH DELETE` by id (not by file) + embedding cleanup for those ids. Cascades
   the inbound edges of *genuinely removed* symbols (correct: they are now dangling).
3. **`nodes_upsert`** — existing idempotent `MERGE` node path (`_insert_node`, `:2004`). Body-only changes become
   a `SET` of `content`/`start_line`/`end_line`; no edge impact.
4. **`edges_add`** — **idempotent** insert (see 6.2).
5. **`dead_recount`** — for each id, `MATCH ()-[r:CodeRelation {rel_type:'calls'}]->(n {id:$id})` count; set
   `is_dead` per the *same* exemption predicate as `process_dead_code` (reuse it — do **not** fork the logic).
6. Rewrite the changed `FileManifest` rows + `IndexManifest` (fingerprint, `pending`) in the same transaction.

### 6.2 Fix edge non-idempotency (also fixes a latent bug in today's watcher path)

`_insert_relationship` is `CREATE` (`:2083`). For the delta path, edge inserts must be **idempotent** so a re-added
`CONTAINS`/`IMPORTS`/etc. does not duplicate. Two options (**Decision D6**):

- **(a) MERGE the edge** keyed on `(src, tgt, rel_type)`: `MATCH (a),(b) … MERGE (a)-[r:CodeRelation {rel_type:$k}]->(b) SET r.confidence=…`.
  Cleanest; makes `add_relationships` safe to call repeatedly. Verify LadybugDB supports `MERGE` on a rel with a
  property key (the rel-group model stores logical kind in `rel_type`, so the merge key is a property, not a label).
- **(b) delete-then-create**: rely on step-1 `edges_remove` to guarantee absence, then `CREATE`. Works, but only
  if `edges_remove` provenance is perfect; MERGE is more robust.

Recommend **(a)** and, as a **standalone precursor fix**, apply it to the existing `apply_reindex` path too — the
folder-CONTAINS duplication (§1) is a real latent bug the moment periodic full wipes stop.

### 6.3 Scoped delete+insert vs full `bulk_load` — the ratio threshold

When many files change at once (branch switch, `git pull`, formatter run), the scoped path loses to the full
build (`bulk_load` is a bulk COPY + one FTS pass, and it is now only ~11s). **Fall back to full `bulk_load` when
the accumulated change ratio crosses a threshold.** Recommended trigger (**Decision D4**):

```
if changed_files / total_files > 0.30  OR  affected_symbols / total_symbols > 0.40:
      run the existing full build (build_full_index → commit_full_index)   # simplest, and cheaper past the knee
else: apply_graph_delta + defer consolidation
```

The 30% figure mirrors the plan's suggestion; it must be **measured** in W3.2 (the crossover depends on the
per-file delta cost vs the ~11s full cost and shifts with repo size). Until measured, 30% files is a safe,
conservative default that keeps large change-sets on the well-tested full path.

---

## 7. FTS and vector-index policy in incremental mode

### 7.1 The FTS floor — quantified, with the honest caveat

**LadybugDB has no incremental FTS update.** `rebuild_fts_indexes` is `DROP`+`CREATE` over 8 tables
(`ladybug_backend.py:1433–1460`), re-tokenizing `name`+`content`+`signature` for **every** symbol in each table.
There is no "add one document" call.

We have **no direct lvlp-app FTS-phase measurement** — the baseline doc explicitly flagged that "Loading to
storage" was never split into schema/COPY/FTS/swap (`benchmarks/2026-07-baseline.md:33`), and W3.2 must add that
split. From the anchors we do have:

- On the **synaptiq repo (1,974 symbols)**, building the 8 empty FTS indexes on open costs **~1.9s**
  (`benchmarks/2026-07-baseline.md:65`, W2.7 note) — this is almost entirely **fixed per-index `DROP`+`CREATE`
  ceremony** (~0.2–0.4s × 8), since the tables are empty.
- W1.3 dropping 3 of 11 FTS indexes saved **~1.16s** of `bulk_load` (`:46`) — consistent with ~0.3–0.4s of fixed
  ceremony per index.
- The **content-proportional** tokenization cost is *on top* of that ceremony and scales with total indexed text.

At **22,689 symbols (~11.5× synaptiq)** the content term grows ~11.5×. Even under the conservative reading that
synaptiq's content term is small, the fixed ceremony alone (~1.9s) **already meets or exceeds the entire ≤2s
budget**, and any non-trivial content scaling pushes a full 8-table rebuild into the **~2–3s+** range. **Bounded
conclusion: a synchronous full FTS rebuild per burst does *not* fit the ≤2s target at lvlp-app scale, and gets
worse with repo size.**

**Design response — defer FTS, don't try to fit it.** BM25 is already documented as lagging per-save
(`pipeline.py:296–331`, `apply_reindex` "leaves FTS stale"; `watcher.py:5–17`); exact-name and graph traversal are
unaffected because the *graph* is updated immediately, and `fts_search` catches per-table failures so a stale index
degrades results but never errors (`ladybug_backend.py:1044–1060`). So:

- The immediate delta path **marks `fts_dirty`** and does **no** FTS work → the ≤2s budget is not gated by FTS.
- **Consolidation** does one FTS rebuild over the accumulated change set (in practice: one full `rebuild_fts_indexes`
  today; a true incremental-FTS is impossible on this engine). This is the **same** FTS rebuild the watcher
  already does every quiescence window — we are not adding cost, we are removing it from the hot path.
- Staleness bound is exactly the existing one: 30s quiescence, 600s ceiling (`watcher.py:62,67`). Document it.

W3.2 **must** produce the storage sub-phase split so this estimate becomes a measurement; if FTS turns out cheaper
than bounded, the policy is unchanged (deferral is still correct, just less critical).

### 7.2 Vector index (HNSW)

`store_embeddings` already **reuses** unchanged vectors via `text_sha` (`pipeline.py:238,350`, `embedder.embed_graph(previous=…)`)
and `MERGE`s rather than wipes (`ladybug_backend.py:1151`). The unavoidable O(all-vectors) cost is the **HNSW
index rebuild** (drop + `CREATE_VECTOR_INDEX`, `:1181–1204`) — there is no incremental HNSW insert exposed.
Policy: **the immediate delta path does no embedding work and marks `hnsw_dirty`.** Embeddings + HNSW ride the
**already-lazy/background** embedding path (W4.1 direction; today's `_on_build` re-embeds under the consolidation).
`vector_search` falls back to a full `array_cosine_similarity` scan when the index is absent/stale
(`:1236–1253`), so semantic search degrades-not-errors in the window — same precedent as FTS. HNSW rebuild is thus
**never** on the ≤2s path.

---

## 8. Versioning, corruption, and full-rebuild fallback

The manifest is an **optimization with a guaranteed correct fallback**: any doubt ⇒ full rebuild (which is only
~11s and always correct). Triggers that force a full build and discard the manifest:

1. **No manifest** (first index, or upgraded-from-pre-manifest install). → full build, emit manifest.
2. **`manifest_version` mismatch** — the schema or the scoping rules changed between synaptiq versions. → full.
   (Do **not** attempt manifest migration; the source of truth is always the code, cf. `open_with_recovery`'s
   "rebuildable derived artifact" stance, `:253`.)
3. **`full_fingerprint` disagrees with the walk** — recompute `sha256(sorted(path→content_sha))` from the current
   walk and compare to the stored fingerprint before trusting any delta. Mismatch (e.g. files changed while the
   daemon was down, a git operation, a corrupt/partial manifest) ⇒ full. This is the single check that makes the
   incremental path safe against "the world moved under us."
4. **DB corruption** — already handled by `open_with_recovery` (`:244`); a manifest-in-DB is wiped with it.
5. **Manifest read/parse error** (sidecar variant) — treat as (1).
6. **Change ratio over threshold** (§6.3) — not corruption, just economics.

`meta.json` gains `manifest_version` and mirrors `pending`/`consolidated_at` so `synaptiq status` can show index
freshness (dovetails with W4.5). `IndexManifest.git_head` gates coupling (§9).

---

## 9. Global-phase policy

| Phase | Input | Global? | Incremental policy |
|---|---|---|---|
| **structure** (File/Folder/CONTAINS) | changed files | No | In the delta (surgical; idempotent CONTAINS via §6.2). |
| **imports / calls / heritage / types** | changed ∪ depth-1 dependents, global name/file index | Local w/ global index | Scoped re-resolution (§5). High-confidence exact; fuzzy best-effort. |
| **rest_linking** | http-call sites × endpoints | Cross-file | Re-run for changed files' call sites against the endpoint index; endpoints are keyed (W1.5), so a changed file re-links against all endpoints and a changed endpoint re-links matching clients. If cost is non-trivial, **defer to consolidation** (it is not a correctness-critical structural edge). **Decision D7.** |
| **community** (Leiden) | whole CALLS graph | **Yes, and unstable** | **Defer.** Leiden is seeded-deterministic (`community.py:176`, seed=42) but *discontinuous*: a few edge changes can renumber/relabel many communities, so recomputing per burst is both costly and produces churn in `MEMBER_OF`/labels. Recompute only at consolidation, and only if `pending.affected_symbols / total > threshold` (else keep last partition — a symbol added since is simply unclustered until then). **Decision D5.** |
| **processes** (entry points + BFS) | entry points, CALLS graph | **Yes** | **Defer**, same threshold gate. New/changed entry points and flows appear at consolidation. |
| **dead_code** | incoming CALLS per symbol | Local (in-degree) | **In the delta**, scoped to `dead_recount` (§5.5, §6.1). Not deferred — it is cheap and locally exact. |
| **coupling** (COUPLED_WITH) | `git log --since=6mo` | Depends only on git history | **Recompute only when `git HEAD` changed.** Unstaged edits (what the watcher mostly sees) never change coupling. Store `git_head` in the manifest; on consolidation, if HEAD moved, run the existing overlapped git-log path (`pipeline.py:141–221`, `coupling.py`); else reuse stored COUPLED_WITH. **Decision D8.** |

**Consolidation = the watcher's existing debounced global phase, made scoped.** `_on_build` (`watcher.py:378`)
changes from "always full rebuild" to: if `pending` ratio > threshold ⇒ full `bulk_load` (simplest); else
recompute only the dirty global phases + FTS + HNSW over the accumulated set, then clear `pending`. The debounce,
single-flight `RebuildCoordinator` (`watcher.py:110`), and skip-if-clean fingerprint (`:228`) are unchanged.

---

## 10. `--full` escape hatch — made real

`analyze --full` currently sets `full=True` which reaches a **no-op** Phase 0 (`cli/main.py:193,288`,
`pipeline.py:107–109` "Currently Phase 0 is a no-op regardless of this flag"). After W3.2:

- **`analyze` (default):** incremental — walk, hash, load manifest, verify fingerprint; if any fallback trigger
  (§8) fires, transparently do a full build; else `apply_graph_delta` + defer/trigger consolidation.
- **`analyze --full`:** force the current full `run_pipeline` → `bulk_load`, unconditionally, and rewrite the
  manifest from scratch. The always-correct reset button and the equivalence oracle.
- **First index (no manifest):** implicitly full.
- **Server-delegated reindex** (`cli/main.py:247` `_reindex_via_server`, `socket_server`): carries the `full` flag
  through so a client `--full` forces a full rebuild in the daemon; default requests an incremental consolidation.

---

## 11. Equivalence contract

**Contract:** *After consolidation, the incremental index is byte-for-byte equal to a `--full` rebuild on the
high-confidence structural core, and equal-or-bounded-stale on the uncertain fringe.*

Precisely, partition the graph:

- **Strict-equal set (must match `--full` exactly, post-consolidation):** all nodes and their properties; CONTAINS;
  IMPORTS; EXTENDS/IMPLEMENTS/MIXES_IN; USES_TYPE; CALLS at confidence ≥ 0.8 (same-file, import-resolved, receiver);
  `is_dead`; `is_entry_point`; MEMBER_OF and Process/STEP_IN_PROCESS **after** a consolidation that recomputed them;
  COUPLED_WITH when HEAD matches.
- **Bounded-stale set (may differ *between* consolidations, converges *at* consolidation):** CALLS at confidence
  ≤ 0.5 (global-fuzzy) and 0.3 (weak-ref); communities/processes/FTS/HNSW while their `*_dirty` flag is set. The
  bound is the documented staleness window (30s/600s).

**Testing (W3.2f — the harness is the deliverable's spine):**

- **Property-based, randomized edit scripts** (Hypothesis). Generate a repo, index it full; apply a random edit
  script (add/remove/rename symbol, edit body, add/delete file, move code between files, non-ASCII edits); run the
  incremental path; force a consolidation; assert `incremental_consolidated == full_rebuild` on the strict-equal
  set. This is the plan's "randomized edit scripts" acceptance (§8) made concrete.
- **Immediate-delta invariants** (no consolidation): no duplicate edges (regression for §1/§6.2); no orphaned
  inbound edges to surviving symbols; `is_dead` locally exact; manifest fingerprint matches the walk.
- **Fallback tests:** version bump ⇒ full; fingerprint mismatch ⇒ full; ratio over threshold ⇒ full; corrupt
  manifest ⇒ full; each leaves a graph equal to `--full`.
- **Reuse the existing golden-diff infrastructure** (`tests/core/golden_reference/`, `diff.py`'s `diff_graphs`) as
  the comparison engine, and the existing watcher async-timing tests (`tests/core/test_watcher.py`) for the
  consolidation debounce.

The `--full` rebuild is the **oracle** in every equivalence test — cheap enough (~11s, or seconds on the small test
repos) to run as the reference on each property example.

---

## 12. W3.2 sub-package breakdown

Sized one agent-context each. Dependency edges are explicit; the two roots (a, d) are independent and start in
parallel.

### W3.2a — Manifest core
- **Scope:** `FileManifest`/`IndexManifest` dataclasses; read/write (in-DB table per §4.2, or sidecar per D3);
  fingerprint computation; version/corruption gating (§8 triggers 1,2,3,5); emit-on-full-build hook next to
  `write_meta`.
- **Files:** new `core/ingestion/manifest.py`; `core/storage/base.py` + `ladybug_backend.py` (only if in-DB —
  a `Manifest` node table + read/write methods); `pipeline.py` (emit hook); `meta.json` fields.
- **Depends on:** nothing (foundation).
- **Risk:** low–medium (in-DB storage choice interacts with the `.rebuild` swap — spike D3 first).
- **Tests:** round-trip; fingerprint stability; version-mismatch/corrupt ⇒ `None`/force-full signal.

### W3.2b — Change-detection & scope planner
- **Scope:** the pure planner (§5): symbol-level diff, classify body-only vs identity-changed, compute reparse set
  + depth-1 dependents + file add/del importer set + change ratio. No storage, no resolution — a function from
  `(changed_files, new_parses, manifest) → ScopePlan`.
- **Files:** new `core/ingestion/incremental.py` (planner half).
- **Depends on:** W3.2a (manifest types).
- **Risk:** medium (the closure rules are the intellectual core; get Q1/Q2 exactly right).
- **Tests:** unit tests over synthetic manifests for each row of the §5.3 table; adversarial edit cases.

### W3.2c — Scoped resolution → `GraphDelta`
- **Scope:** run imports/calls/heritage/types resolvers over the reparse∪dependents mini-graph against a
  manifest-backed global name/file index; emit `GraphDelta` (§5.5). Requires light refactors so resolvers accept an
  injected global index instead of scanning a whole `KnowledgeGraph` (the functions already isolate index
  construction: `build_name_index`, `build_file_index`, `_build_call_index_by_file`).
- **Files:** `core/ingestion/incremental.py` (apply half); minor signatures in `calls.py`/`imports.py`/`heritage.py`/`types.py`
  (accept external index — additive, keep current callers working, mirrors the `symbol_ids`/`call_index_by_file`
  optional-param pattern already there, `calls.py:167,563`).
- **Depends on:** W3.2b.
- **Risk:** medium–high (resolver reuse; confidence-bucket fidelity).
- **Tests:** delta edges == full-rebuild edges on the strict-equal set for scripted edits (pre-storage, in-memory).

### W3.2d — Storage delta ops + edge idempotency
- **Scope:** `apply_graph_delta` (§6.1) on the Protocol + LadybugDB; idempotent edge insert (§6.2, MERGE) and the
  standalone `apply_reindex` duplication fix; scoped `is_dead` recount reusing the dead-code exemption predicate;
  `edges_remove`/`nodes_remove`-by-id primitives.
- **Files:** `core/storage/base.py`, `core/storage/ladybug_backend.py`; small reuse hook from `dead_code.py`
  (export the exemption predicate).
- **Depends on:** the `GraphDelta` shape from W3.2c (define the dataclass jointly, up front, so a and c can proceed
  against a frozen interface) — otherwise **independent of a/b** and can start immediately on the edge-idempotency
  fix.
- **Risk:** medium (transaction correctness; MERGE-on-rel-property spike against LadybugDB).
- **Tests:** delta leaves DB == full-rebuild (strict set); no duplicate edges; induced-failure rollback (mirror
  `_write_batch` crash tests); embedding cleanup on node removal.

### W3.2e — Global-phase policy, consolidation, CLI wiring
- **Scope:** coupling git-HEAD gate; community/processes threshold-defer; make consolidation scoped in `_on_build`;
  ratio-fallback to full `bulk_load`; `analyze` default→incremental + real `--full`; server-delegated `full` flag.
- **Files:** `core/ingestion/pipeline.py`, `core/ingestion/watcher.py`, `cli/main.py` (+ `socket_server`/client if
  the flag plumb needs it).
- **Depends on:** W3.2b/c/d.
- **Risk:** medium (watcher async timing; threshold defaults — leave them env-overridable like `MAX_STALENESS_SECONDS`).
- **Tests:** coupling skipped when HEAD unchanged; community recompute only past threshold; `--full` forces full;
  consolidation debounce unchanged (extend `test_watcher.py`).

### W3.2f — Equivalence harness
- **Scope:** the property-based randomized-edit-script suite (§11) + fallback tests + immediate-delta invariants,
  using `--full` as oracle and `diff_graphs` as comparator.
- **Files:** new `tests/core/test_incremental_equivalence.py`; fixtures.
- **Depends on:** a–e (but scaffold the generators early, in parallel).
- **Risk:** low (test-only) / high value — this is the gate the plan's W3.2 acceptance names.

### Conflict / sequencing matrix

| File | Sub-packages touching it | Rule |
|---|---|---|
| `core/ingestion/manifest.py` (new) | W3.2a | Sole owner. |
| `core/ingestion/incremental.py` (new) | W3.2b, W3.2c | Same agent or sequential b→c; one new file, split by planner/apply halves. |
| `core/storage/base.py` | W3.2a (if in-DB manifest), W3.2d | **Sequence a→d** if a adds manifest methods; else d-only. |
| `core/storage/ladybug_backend.py` | W3.2a (if in-DB), W3.2d | **Sequence a→d.** Disjoint funcs but same file. |
| `calls.py/imports.py/heritage.py/types.py` | W3.2c | Sole owner (additive optional params). |
| `dead_code.py` | W3.2d (export predicate) | W3.2d-only; additive export. |
| `core/ingestion/pipeline.py` | W3.2a (emit hook), W3.2e (default path) | **Sequence a→e.** |
| `core/ingestion/watcher.py` | W3.2e | Sole owner (single-package rule, as in W1.1). |
| `cli/main.py` | W3.2e | Sole owner. |
| `GraphDelta` dataclass | W3.2c defines, W3.2d consumes | **Freeze the interface first** in a tiny shared commit so c and d parallelize. |

**Critical path:** `{a, d-edge-fix}` ∥ → `b` → `c` → `d-apply` → `e` → `f`. Realistic parallelism: a+d-edge-idempotency
first (independent), then b, then c∥d-apply against the frozen `GraphDelta`, then e, with f's generators scaffolded
throughout.

---

## 13. Decision points for the lead

| # | Decision | Recommendation | Why / what would change it |
|---|---|---|---|
| **D1** | Leaner (depth-1, deferred consolidation) vs full impact-closure re-resolution | **Leaner.** | 11s full rebuild is the safety net; full closure reintroduces unbounded fan-out for edges already marked uncertain. Change it only if a user need for *always-fresh fuzzy edges / communities* emerges. |
| **D2** | `analyze` default = incremental or full? | **Incremental when a valid manifest exists; full otherwise; `--full` forces full.** | Matches the plan's "`--full` made real"; incremental is transparent and always has a correct fallback. |
| **D3** | Manifest sink: in-DB table vs sidecar JSON/sqlite | **In-DB table** (atomic, rides crash-safe swap, covered by `open_with_recovery`). | Spike LadybugDB user-table + `.rebuild`-swap interaction first; if it complicates the swap, fall back to `manifest.json` with its own force-full-on-corrupt. |
| **D4** | Change-ratio threshold for full fallback | **>30% files OR >40% symbols → full.** | Conservative; **must be measured** in W3.2 (crossover shifts with repo size and per-file delta cost). |
| **D5** | Community/processes on incremental | **Defer to consolidation; recompute only if affected-symbol ratio > ~10–15%.** | Leiden is discontinuous → per-burst churn + cost. Tighten/loosen after measuring label churn. |
| **D6** | Edge idempotency | **MERGE on `(src,tgt,rel_type)`; also fix `apply_reindex` now.** | Delete-then-CREATE works but is provenance-fragile. Spike `MERGE`-on-rel-property in LadybugDB. |
| **D7** | rest_linking in incremental | **Re-run for changed files' call sites; defer if the keyed re-link is non-trivial at scale.** | It's a convenience edge, not structural — deferral is low-harm. Decide after W1.5's keyed cost is re-measured at lvlp scale. |
| **D8** | Coupling recompute trigger | **git HEAD change only.** | Coupling depends solely on 6-month git history; unstaged edits don't change it. Uncontroversial. |
| **D9** | Equivalence contract scope | **Strict on high-confidence core + bounded-stale on fuzzy/global.** | Strict-on-everything = unbounded fan-out. Revisit only if fuzzy-edge staleness proves user-visible. |
| **D10** | Manifest resident in the daemon? | **Yes for `serve`/`watch` (holds name-index resident → ~0.1s hot path); reload for one-shot `analyze` (≤0.5s at lvlp scale).** | The only repo-size term on the hot path is the in-memory index build; keeping it resident removes it for the watcher, the primary target. |

---

## 14. Open risks

1. **FTS floor is estimated, not measured.** W3.2 must split storage timing (schema/COPY/FTS/swap) at lvlp scale
   to confirm deferral is necessary (it is correct regardless, but the urgency depends on the number).
2. **LadybugDB `MERGE`-on-rel-property** (D6) and **in-DB manifest table + `.rebuild` swap** (D3) are both spikes —
   settle them before c/d/a lock their interfaces.
3. **Fuzzy-edge staleness could surprise users** if they query low-confidence CALLS between consolidations. Mitigate
   with W4.5 freshness reporting and the documented window; the graph itself never *errors*, only lags.
4. **Rename-heavy or move-heavy refactors** produce large `removed`+`added` sets and may cross the ratio threshold
   often — acceptable (they fall back to the fast full build), but worth watching in the equivalence property tests.
5. **Determinism of deferred communities:** `MEMBER_OF` for symbols added since the last consolidation is absent
   (not wrong) until consolidation; ensure consumers (MCP `overview`, community labels) tolerate an unclustered
   symbol, matching how they already tolerate the per-save stale window.

---

*End W3.1. Implementation (W3.2) must not begin until this design is approved by the owner (plan §8 gate).*

## Approval (2026-07-12, lead as owner-delegate per blanket execution instruction)

APPROVED with conditions: leaner design accepted (D1); all recommendations D2-D10 accepted as written; spikes D3 (in-DB manifest vs .rebuild swap) and D6 (MERGE-on-rel-property support) must complete before interface freeze; the two latent apply_reindex bugs (inbound-edge deletion, duplicate CONTAINS on re-emit) are P0 inside W3.2d. Owner may overrule before release.
