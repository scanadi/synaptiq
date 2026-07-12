# Baseline — pre-optimization (Wave 0 gate)

**Date:** 2026-07-12 · **Machine:** 10-core Apple Silicon laptop, macOS, quiet (load ~2.0, no concurrent test runs)
**Code state:** `9cf9d83` = main `c1b025b` (v1.5.1) + W0.1 instrumentation only — **no perf packages included** (run from the frozen W0.1 worktree so W1.3/W1.5/W1.7 do not contaminate the "before").
**Target:** the synaptiq repo itself — 122 files, 1,974 symbols, 9,603 relationships, 2,095 embeddings.
**Commands:** `uv run python scripts/bench_index.py . --runs 3 --json` and `... --runs 1 --embeddings --json`

## Medians, no embeddings (3 runs; total 6.37s)

| Phase | Seconds | % |
|---|---|---|
| Loading to storage | 5.990 | **94.1%** |
| Parsing code | 0.151 | 2.4% |
| Walking files | 0.080 | 1.3% |
| Analyzing git history | 0.048 | 0.8% |
| Detecting communities | 0.032 | 0.5% |
| Linking REST endpoints | 0.025 | 0.4% |
| Tracing calls | 0.016 | 0.2% |
| Detecting execution flows | 0.013 | 0.2% |
| all others | <0.01 | — |

## Single run, with embeddings (total 55.67s)

| Phase | Seconds | % |
|---|---|---|
| Generating embeddings | 49.248 | **88.5%** |
| Loading to storage | 6.056 | 10.9% |
| everything else | ~0.37 | 0.7% |

## Interpretation

1. **Embeddings dominate first-index wall time** (49s for 2,095 symbols ≈ 42 symbols/s) — confirms G5 and makes W4.1 (lazy background embeddings) the largest product-feel lever; W1.4's thread cap governs the CPU footprint of this phase.
2. **"Loading to storage" is ~6s nearly independent of run** for only ~12k rows — dominated by fixed overhead (schema create incl. the 121-subtable `CodeRelation` group, FTS index creation, rebuild-swap), not row volume. Small-repo `analyze` latency is therefore mostly storage-fixed-cost; W2.3 (Arrow/Parquet COPY) work should first **split this phase's timing into schema / COPY / FTS / swap** before optimizing, and W1.3 (already merged) trims the FTS slice.
3. **Parse phase is negligible at this scale** (0.15s) — W2.1/W2.2 only pay off on large repos (thousands of files); keep them, but they are not small-repo levers.
4. Comparison directive: every perf package's "after" must be measured with the same commands on the same machine, quiet, ≥3 runs (embeddings runs may stay at 1 — variance is low relative to magnitude).

Raw JSON: session task output `bpzm9odjy` (numbers transcribed above verbatim).

---

# After Wave 1 (same machine, quiet, same commands, main @ W1.2 merge)

| Metric | Baseline | After W1 | Δ |
|---|---|---|---|
| Total, no embeddings (median of 3) | 6.37s | 5.20s | **-18%** |
| Loading to storage (median) | 5.99s | 4.83s | **-19%** (W1.3: 6 fewer FTS builds per bulk_load) |
| Total, with embeddings (1 run) | 55.7s | 61.5s | +10% |
| Generating embeddings (1 run) | 49.2s | 56.2s | **+14% by design** (W1.4 polite default: cores-2 ONNX threads; `--jobs 0` restores all-cores) |
| Parsing code | 0.151s | 0.167s | noise |

**Not visible in a full-index benchmark (the actual Wave-1 targets):** watcher continuous-rebuild loop → single rebuild after 30s quiescence with 600s ceiling and skip-if-clean (W1.1); per-save cost drops from O(corpus) FTS rebuild to O(file) (W1.2); incremental insert path ~47× via transactions + prepared statements, 13.9s → 0.29s per 1k nodes (W1.6, agent micro-benchmark); rest_linking phase 16-43× at scale (W1.5, agent synthetic benchmark at N=300/1000); MCP `cycles` now cached per index generation (B2).

---

# After Wave 2 (same machine, quiet, same commands, main @ W2.7 merge — now on LadybugDB)

| Metric | Baseline | After W1 | After W2 | Δ vs baseline |
|---|---|---|---|---|
| Total, no embeddings (median of 3) | 6.37s | 5.20s | **3.02s** | **-53%** |
| Loading to storage | 5.99s | 4.83s | **2.58s** | **-57%** (W2.3 Arrow COPY + FTS-on-empty skip + W2.7 LadybugDB) |
| Analyzing git history (visible) | 0.048s | — | **0.001s** | fully overlapped (W2.4) |
| Parsing code | 0.151s | 0.167s | 0.255s | small repo stays on thread path (process pool gated ≥100 files); Δ within load noise |
| Generating embeddings (1 run) | 49.2s | 56.2s | 70.2s | ⚠ unexplained +25% vs W1 — same model/threads; suspected thermal/load artifact after hours of agent activity; **re-measure on cool machine before reading anything into it** |

**Not visible here (Wave-2 wins at scale):** process-parallel parsing 6.36× on a 500-file corpus (W2.1, threshold-gated); single-pass walks ~1.2× on extraction (W2.2, honest below-estimate result — old symbol walk was already light); calls-phase symbol-ID reuse + O(1) same-file resolution (W2.5c); flow-dedup and Ruby-import quadratic fixes (W2.5ab). The real `analyze` CLI additionally skips ~1.9s of empty-FTS build on open (W2.7) that this bench's timed phases don't include.

**Caveat — these numbers were measured WITH pyarrow installed** (the `fast-load` / dev extra). A default `pip install synaptiq` has no pyarrow and takes the CSV COPY path, so the "Loading to storage" numbers above will be **higher** until pyarrow is installed or promoted to a core dependency (owner decision pending).
