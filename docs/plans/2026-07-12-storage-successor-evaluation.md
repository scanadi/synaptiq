# Storage Successor Evaluation — LadybugDB vs Vela fork (W2.6)

**Status:** research/evaluation spike — no production code changed.
**Date:** 2026-07-12
**Author:** W2.6 (Synaptiq performance refactor)
**Scope:** Evaluate successors to the archived KuzuDB for Synaptiq's `StorageBackend`
(`src/synaptiq/core/storage/base.py`). Current pin: `kuzu==0.11.3` (pyproject.toml,
final upstream release; upstream repo archived 2025-10-10).

---

## 1. TL;DR — Recommendation

**Designate LadybugDB (`pip install ladybug`, MIT, github.com/LadybugDB/ladybug) as
Synaptiq's storage successor, but do not adopt it in production yet.** Keep the
`kuzu==0.11.3` pin as the shipping backend and, in parallel, land a `LadybugBackend`
behind the existing `StorageBackend` Protocol, so a switch is a one-line backend
selection rather than a port. LadybugDB is the only candidate that passes the full
rubric: it publishes standard PyPI wheels across CPython 3.10–3.14 (including Windows
and musllinux), is API drop-in for Synaptiq's bindings, reproduces every
Synaptiq-critical operation in hands-on testing (node/rel-group schema, prepared
statements, `PARALLEL=false` CSV COPY, BM25 FTS, and cosine HNSW vector search all pass
out of the box), is MIT-licensed, company-backed with commercial support, and — matching
the owner's "improve the product" directive — adds capabilities Kuzu never had (pluggable
`backend='auto'` storage with Arrow/Parquet/DuckDB interop, multi-label nodes, subgraph
isolation, opt-in `enable_multi_writes`). **The Vela fork is not viable as a drop-in:**
it is not on PyPI (wheels ship off-GitHub under the colliding `kuzu` name), its extension
registry returns **404 for the current release so BM25 and vector search are both broken**
on the wheel tested, it carries bus-factor risk (VC-firm side project, ~38 stars), and its
one differentiator — concurrent writes — is **intra-process only** and therefore does not
let Synaptiq retire its daemon layer.

**Falsifiability — this recommendation flips if any of the following proves true:**
(a) LadybugDB fails to publish a wheel for a CPython version Synaptiq must target, or goes
unmaintained (no release across two-plus of its ~monthly cycles) → re-evaluate RyuGraph and
staying on pinned 0.11.3; (b) a hands-on port reveals a Cypher / `rel_type` / FTS / vector
behavioral divergence the Protocol cannot absorb; (c) a W3 design genuinely requires
**cross-process** multi-writer — which **neither fork provides** (both keep Kuzu's exclusive
file lock), so that requirement points away from these embedded forks toward a client/server
engine, not toward Vela.

---

## 2. What Synaptiq actually depends on (the port surface)

From `src/synaptiq/core/storage/kuzu_backend.py` and `base.py`, the load-bearing Kuzu
API surface is:

| Feature | Kuzu API used | Synaptiq call site |
|---|---|---|
| Embedded DB + resource caps | `kuzu.Database(path, read_only, max_num_threads, buffer_pool_size)` | `initialize()` |
| Cypher w/ params + prepared stmts | `Connection.execute(q, parameters=…)`, `Connection.prepare()` | reads + batched writes |
| Single rel table group | `CREATE REL TABLE GROUP CodeRelation(FROM…TO…, rel_type STRING, …)`; queries filter on `r.rel_type` (never `[:CALLS]`) | `_create_schema`, all traversals |
| BM25 full-text search | `INSTALL/LOAD fts`, `CREATE_FTS_INDEX`, `QUERY_FTS_INDEX`, `DROP_FTS_INDEX` | `fts_search`, `rebuild_fts_indexes` |
| Vector search (HNSW) | `INSTALL/LOAD vector`, `CREATE_VECTOR_INDEX(... metric:='cosine')`, `QUERY_VECTOR_INDEX`, `FLOAT[384]` column | `store_embeddings`, `vector_search` |
| Bulk load | `COPY <table> FROM 'file.csv' (HEADER=false, PARALLEL=false)` | `_csv_copy`, `bulk_load` |
| Scalars | `array_cosine_similarity`, `levenshtein`, `CAST(x AS FLOAT[dim])` | fallback scan, `fuzzy_search` |
| Txns / schema | `BEGIN/COMMIT/ROLLBACK`, `CALL TABLE_INFO`, `ALTER TABLE ADD` | `_write_batch`, migration |
| Single-writer workaround | none — `core/daemon/` (fcntl lock + Unix-socket primary/proxy) exists **because** Kuzu is single-writer across processes | `serve`/`mcp`/`watch` |

**Key de-risking fact:** the `.synaptiq/` index is a **rebuildable derived artifact** —
`synaptiq analyze` reconstructs it from source via `bulk_load` (which already builds a fresh
`.rebuild` DB from scratch). Synaptiq is not a system of record. Therefore **on-disk
storage-format compatibility with existing kuzu 0.11.3 databases is low-impact**: the
migration path for any successor is simply "bump the backend, run `synaptiq analyze`." This
materially lowers the weight of the storage-compat rubric row for Synaptiq specifically.

A backend swap touches ~12 files that reference the backend or raw Cypher
(`cli/main.py`, `core/ingestion/{pipeline,watcher}.py`, `core/search/{hybrid,pagerank}.py`,
`core/embeddings/embedder.py`, `mcp/{server,tools,resources,suggest}.py`, `core/resources.py`),
but almost all go through the `StorageBackend` Protocol; the concentrated work is a new
`kuzu_backend.py` sibling.

---

## 3. Scoring table (both leaders × full rubric)

Legend: ✅ pass · ⚠️ partial/caveat · ❌ fail. Evidence links in §7.

| Criterion | LadybugDB (`ladybug` 0.18.1) | Vela fork (`kuzu` 0.12.0-vela) |
|---|---|---|
| **On PyPI / install** | ✅ `pip install ladybug` (standard PyPI) | ❌ Not on PyPI. Wheels only as GitHub-release assets under package name `kuzu` 0.12.0 → **name-collides with real `kuzu`**; install needs a pinned release URL / custom index |
| **Wheel Python coverage** | ✅ cp310–cp314, `requires_python <3.15,>=3.10` | ⚠️ cp311–cp314 (no cp310) |
| **Platform coverage** | ✅ macOS arm64+x86_64, manylinux, **musllinux**, **Windows amd64+arm64** | ❌ macOS arm64 + manylinux aarch64/x86_64 only; **no Windows, no musllinux**; macOS floor 11.0 |
| **Release cadence** | ✅ ~monthly (v0.16.1 May 4 → 0.17.0 → 0.17.1 → 0.18.0 Jul 1 → 0.18.1 Jul 10) | ⚠️ frequent but all `0.12.0-vela.<hash>` pre-releases; ~1–2/wk in May–Jun |
| **API drop-in (Python bindings)** | ✅ `import ladybug as lb`; `lb.Database`/`lb.Connection`/`conn.execute`; **same** `prepare+execute` DeprecationWarning Synaptiq already suppresses | ✅ identical `import kuzu`; same class/method surface as 0.11.3 |
| **FTS (BM25) parity** | ✅ **passes hands-on** (LOAD + CREATE_FTS_INDEX + QUERY_FTS_INDEX) | ❌ **`INSTALL fts` → HTTP 404** from own registry; BM25 broken on tested wheel |
| **Vector (HNSW cosine) parity** | ✅ **passes hands-on** (CREATE_VECTOR_INDEX cosine + QUERY_VECTOR_INDEX) | ❌ **`INSTALL vector` → HTTP 404**; semantic search broken on tested wheel |
| **CSV COPY (`PARALLEL=false`)** | ✅ passes, embedded newlines OK | ✅ passes, embedded newlines OK |
| **Rel-table-group + `rel_type` filter** | ✅ passes | ✅ passes |
| **Storage-format compat w/ 0.11.3** | ⚠️ `.lbug` format; documented export/import migration. **Low impact** (Synaptiq re-indexes) | ⚠️ 0.12.0 format; not verified. **Low impact** (re-index) |
| **Cross-process multi-writer** (would retire `core/daemon/`) | ❌ single-writer file lock (even with `enable_multi_writes=True`) | ❌ single-writer file lock (default) |
| **Intra-process concurrent writes** | ✅ with `enable_multi_writes=True` (default off) | ✅ default on |
| **Bulk-load perf (10k nodes/50k rels, micro)** | ✅ nodes 0.148s / rels 0.093s / scan 0.004s | ✅ nodes 0.241s / rels 0.143s / scan 0.005s (baseline 0.11.3: 0.446 / 0.097 / 0.005) |
| **License** | ✅ MIT | ✅ MIT |
| **Backing / bus factor** | ✅ Ladybug Memory (Arun Sharma, ex-Meta "Dragon" lead); commercial support offered | ⚠️ Vela Partners (VC firm) side project; explicitly built for their own agent-memory product |
| **Community momentum** | ✅ ~1.4k stars, 6.1k commits, active roadmap | ❌ ~38 stars, 6 forks |
| **Product-improvement upside** ("improve, not like-for-like") | ✅ pluggable `backend='auto'` (Arrow/Parquet/DuckDB "graph lakehouse"), multi-label nodes, subgraph isolation (`CREATE GRAPH`/`USE`), `AsyncConnection` | ⚠️ intra-process concurrent writes + crash-safety flags (`throw_on_wal_replay_failure`, `enable_checksums`) — real, but not Synaptiq's bottleneck |

**Rubric outcome: LadybugDB passes; Vela fails on packaging (not on PyPI, name collision),
on FTS + vector (both broken on the current wheel), and on bus factor.**

---

## 4. Hands-on findings (reproducible)

All tests run in throwaway venvs (macOS arm64, CPython 3.13.7). Scripts live in the
scratchpad; commands and outputs below are verbatim.

### 4.1 Install & import
- **LadybugDB:** `pip install ladybug` → `ladybug 0.18.1`. Live PyPI JSON `summary`:
  *"Highly scalable, extremely fast, easy-to-use embeddable graph database"*,
  `home_page: github.com/LadybugDB/ladybug`. (Note: the `ladybug` PyPI name now resolves to
  the graph DB, not the older building-energy tool — confirmed by installing and running it.)
  Exposed API: `Database, Connection, AsyncConnection, PreparedStatement, QueryResult,
  ArrowQueryResult, ArrowRelTableLayout, CSRResult, Type`.
- **Vela:** not on PyPI (`pip index versions kuzu-vela` → no distribution; `kuzu` on PyPI is
  upstream-only, frozen at 0.11.3). Installed the GitHub-release wheel
  `kuzu-0.12.0-cp313-cp313-macosx_11_0_arm64.whl` → imports as `kuzu 0.12.0`.

### 4.2 Functional parity (7 Synaptiq-critical operations)

| Check | LadybugDB | Vela |
|---|---|---|
| CREATE NODE TABLE + REL TABLE GROUP | PASS | PASS |
| parameterized insert + `prepare()` | PASS | PASS |
| `rel_type`-filtered CodeRelation traversal | PASS | PASS |
| CSV COPY (`HEADER=false, PARALLEL=false`, embedded newline) | PASS | PASS |
| FTS: LOAD + CREATE_FTS_INDEX + QUERY_FTS_INDEX (BM25) | **PASS** | **FAIL** |
| Vector: CREATE_VECTOR_INDEX cosine + QUERY_VECTOR_INDEX (HNSW) | **PASS** | **FAIL** |
| `array_cosine_similarity` + `levenshtein` | PASS | PASS |

**Vela extension failure (verbatim):**
```
fts FAILED: RuntimeError: IO exception: HTTP Returns: 404, Failed to download extension:
  "fts" from https://vela-engineering.github.io/kuzu/v0.12.0/osx_arm64/fts/libfts.kuzu_extension.
vector FAILED: RuntimeError: IO exception: HTTP Returns: 404, Failed to download extension:
  "vector" from https://vela-engineering.github.io/kuzu/v0.12.0/osx_arm64/vector/libvector.kuzu_extension.
```
Registry root returns HTTP 200, but every extension artifact path 404s
(osx_arm64, linux_amd64, linux_x86_64, and even v0.11.3/osx_arm64). This is the exact
unfinished item flagged in Vela PR #17: *"Publish the v0.12.0 registry artifacts before
cutting packages that install extensions."* The package was cut; the artifacts were not
published. Because Synaptiq's `_create_schema` swallows `INSTALL/LOAD` failures, adopting
this wheel would **silently** gut BM25 + vector search (two of the three hybrid-search
pillars) with no error surfaced to the user.

### 4.3 Bulk-load micro-benchmark (10k nodes / 50k rels, CSV COPY — Synaptiq's hot path)
```
baseline kuzu 0.11.3 : nodes_copy=0.446s  rels_copy=0.097s  rel_scan=0.005s
Vela fork    0.12.0   : nodes_copy=0.241s  rels_copy=0.143s  rel_scan=0.005s
LadybugDB    0.18.1   : nodes_copy=0.148s  rels_copy=0.093s  rel_scan=0.004s
```
Interpretation: **no regression** from either fork; all three are within the same envelope
at this scale. LadybugDB is marginally fastest on node COPY. These are small absolute numbers
on a micro-run and should not be read as a full-indexing benchmark — they only establish that
the CSV-COPY path is not a porting hazard.

---

## 5. The Vela multi-writer verdict (does it let us retire `core/daemon/`?)

**Verdict: NO. Vela's "concurrent multi-writer" is real but INTRA-PROCESS only. It does not
relax Kuzu's cross-process exclusive file lock, so Synaptiq's daemon layer must stay.**

Vela's headline pitch is "concurrent multi-writer support … for architectures where multiple
AI agents write simultaneously." The blog/README give no mechanism. PR #17 shows real
engineering (checkpoint draining across active transactions, WAL-replay hardening, a C++/Python
concurrency stress tool) but never states process-vs-thread scope. Direct two-process and
two-thread tests resolve it:

**Cross-process (two OS processes open the same on-disk DB read-write):**
```
BASELINE kuzu 0.11.3 : worker A opened=False "Could not set lock on file"; B ok → SINGLE-WRITER
VELA fork 0.12.0     : worker B opened=False "Could not set lock on file"; A ok → SINGLE-WRITER
LadybugDB 0.18.1     : worker B opened=False "Could not set lock on file"; A ok → SINGLE-WRITER
   (LadybugDB with enable_multi_writes=True → STILL single-writer file lock)
```

**Intra-process (one `Database`, two threads, two `Connection`s, writing concurrently):**
```
BASELINE kuzu 0.11.3 : thread B → "Cannot start a new write transaction ... Only one write
                        transaction at a time is allowed"  → REJECTED/SERIALIZED (3000/6000)
VELA fork 0.12.0     : both threads commit, 6000/6000, overlapping intervals → CONCURRENT OK
LadybugDB (mw=True)  : both threads commit, 6000/6000 → CONCURRENT OK  (default off → serialized)
```

**Constructor evidence** (no cross-process/shared-writer knob on either fork):
- Vela adds only `throw_on_wal_replay_failure=True`, `enable_checksums=True` (crash-safety for
  its concurrent checkpointing).
- LadybugDB adds `enable_multi_writes=False` (intra-process opt-in) and `backend='auto'`
  (pluggable storage). Neither exposes a multi-process writer mode.

**Why this matters for the product claim:** `src/synaptiq/core/daemon/` (fcntl `lock.py`,
`rwlock.py`, `socket_client.py`, `socket_server.py`) exists to arbitrate multiple MCP
**processes** sharing one index, precisely because Kuzu takes an exclusive file lock per
process. Both forks keep that lock. The intra-process concurrency they add is not Synaptiq's
constraint — Synaptiq already serializes writes within a process via `AsyncRWLock`, and its
ingestion pipeline is single-process. **So "adopt Vela → retire the daemon" is false**, and
Vela's sole differentiator delivers little to Synaptiq's actual architecture.

---

## 6. Adoption plan — work items (no time estimates)

Prudent path: **stay on `kuzu==0.11.3`; pre-stage LadybugDB behind the Protocol; adopt on a
trigger (§8).** Work items, in dependency order:

**A. Proof-of-port (spike, behind the Protocol)**
- [ ] Add `LadybugBackend` as a sibling of `KuzuBackend`, implementing the `StorageBackend`
      Protocol via `import ladybug as lb` (the Cypher/DDL is identical; changes are the module
      alias and the `Database`/`Connection` constructors).
- [ ] Add a backend-selection seam (env var or config) so `KuzuBackend` stays default and
      `LadybugBackend` is opt-in.
- [ ] Reuse the existing DeprecationWarning suppression (`prepare + execute`) — LadybugDB emits
      the same warning.
- [ ] Confirm FTS index naming, `QUERY_FTS_INDEX`/`QUERY_VECTOR_INDEX` result columns
      (`node.*`, `score`, `distance`), and `CAST(x AS FLOAT[dim])` semantics match 0.11.3.

**B. Migration / rebuild**
- [ ] Wire the successor so `synaptiq analyze` writes a `.lbug`-format index; treat the format
      change as a full re-index (no data migration needed — the index is derived).
- [ ] Add a version/format stamp so an old kuzu index triggers an automatic rebuild rather than
      a silent empty result.

**C. Test / CI**
- [ ] Parametrize the storage test suite (`tests/core/`) over both backends.
- [ ] Add a CI job that installs `ladybug` from PyPI and runs the search/traverse/bulk_load tests.
- [ ] Pin a known-good `ladybug` version; watch the `requires_python` ceiling (`<3.15`).

**D. Packaging**
- [ ] Switch the dependency from `kuzu==0.11.3` to a pinned `ladybug==<x.y.z>` (standard PyPI
      resolution; no custom index — unlike Vela).
- [ ] Keep the Neo4j optional-extra path untouched.

**E. Product-improvement follow-ons (optional, post-adoption — the "improve, not like-for-like" upside)**
- [ ] Evaluate `backend='auto'` / Arrow/Parquet export for sharing indexes or querying them from DuckDB.
- [ ] Evaluate multi-label nodes to simplify the per-label node-table fan-out in `kuzu_backend.py`.
- [ ] Evaluate subgraph isolation (`CREATE GRAPH … / USE`) for multi-repo separation inside one DB.

**Explicitly out of scope / not solved by this swap:** retiring `core/daemon/` — neither
candidate provides cross-process multi-writer. If W3 needs that, it is a separate architectural
decision (client/server engine), not a fork choice.

---

## 7. Evidence links

- Kuzu archived (final 0.11.3, uploaded 2025-10-10T13:36Z): uv.lock `kuzu` entry; PyPI `kuzu` tops out at 0.11.3.
- LadybugDB: https://ladybugdb.com/ · https://github.com/LadybugDB/ladybug · https://docs.ladybugdb.com/client-apis/python · PyPI `ladybug` 0.18.1 (`requires_python <3.15,>=3.10`).
- LadybugDB analysis: https://thedataquarry.com/blog/from-kuzu-to-ladybug/ ("shape is intentionally the same as Kuzu … continuity for existing users"; migration "pretty straightforward" via export/import); https://gdotv.com/blog/kuzu-legacy-embedded-graph-database-landscape/ (Arun Sharma stewardship; "Native full text search and vector index"; "Serializable ACID transactions").
- Vela fork: https://github.com/Vela-Engineering/kuzu · release `v0.12.0-vela.2efa20b` (2026-06-14) · PR #17 "default concurrent writes" (checkpoint draining; registry-artifact caveat) · https://vela.partners/blog/kuzudb-ai-agent-memory-graph-database · extension registry 404s under https://vela-engineering.github.io/kuzu/.
- Fork landscape: https://szarnyasg.org/posts/kuzu-forks/ (LadybugDB, RyuGraph, bighorn, Vela).

---

## 8. Decision triggers (what event should cause action)

**Adopt LadybugDB (execute §6) when the first of these fires:**
1. **Python-version wall:** Synaptiq needs to target a CPython version for which upstream
   `kuzu==0.11.3` has **no wheel** (0.11.3 wheels stop being published / built for a new
   CPython). This is the most likely near-term trigger — upstream is archived, so the first
   unavailable interpreter wheel forces the move. LadybugDB already ships cp310–cp314.
2. **A W3 design item that LadybugDB uniquely enables** — e.g. Arrow/Parquet index export,
   multi-label node consolidation, or subgraph-per-repo — becomes a committed requirement.
3. **A 0.11.3 defect** (crash, corruption, security) with no fix path, since upstream is frozen.

**Do NOT let these trigger adoption of the Vela fork:**
- A desire to retire `core/daemon/` — Vela does not provide cross-process multi-writer (proven
  in §5). Only reconsider Vela if it **both** publishes its extension registry (FTS + vector
  install cleanly) **and** ships to PyPI under a non-colliding name **and** adds a documented
  cross-process writer mode.

**Re-open the whole evaluation if:**
- LadybugDB goes quiet (no release across two-plus of its ~monthly cycles) or drops a platform
  Synaptiq ships on → evaluate RyuGraph (predictable-labs) and bighorn (Kineviz), the
  documented fallbacks (not deep-dived here because a leader passed), and the option of simply
  holding on pinned 0.11.3.

---

## Appendix — fallbacks (not deep-dived; a leader passed)

- **RyuGraph** (predictable-labs): Kuzu fork, "built for speed with vector search and full-text
  search built in." Nearest technical analog to LadybugDB; evaluate first if LadybugDB fails a trigger.
- **bighorn** (Kineviz): Kuzu fork tied to the GraphXR visualization platform; embedded +
  standalone-server modes. Server mode is the only surveyed option that could, in principle,
  address cross-process access — worth a look only if W3 makes multi-process writing a hard requirement.
