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
