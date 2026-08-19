# Holographic Market Engine — Technical Data Brief
**For: NVIDIA engineering review (via William)**
**Prepared: August 2026 — every number below is from a run executed and logged this week, not a projection.**

---

## 0. What this document is

A factual technical summary for anyone writing about this engine — architecture,
verified performance, and cuML integration specifics. Section 6 explicitly lists
what is **not** built yet, so nothing downstream (write-ups, case studies) claims
more than the code actually does.

---

## 1. Architecture — 6-stage GPU pipeline

Order-book state is modeled as a dynamic graph over instruments; trade routing
across instruments defines a gauge connection on that graph, and the engine
computes the connection's curvature (a Yang–Mills action) directly on-device,
end to end, every tick.

| Stage | What it does | Library |
|---|---|---|
| I. Normalized Laplacian | Cross-impact tensor → graph Laplacian, CSR assembly | cuSPARSE |
| II. LOBPCG Fiedler vector | k=4 eigenpair solve, pruning mask | cuBLAS SpMM |
| III. Incidence matrices | Edge/triangle incidence (B₁, B₂) | custom CUDA kernels |
| IV. Hodge decomposition | Full spectral solve — splits order-flow into exact / harmonic / co-exact components; co-exact is the tradeable signal | cuSOLVER (`Ssyevd`) |
| V. Curvature extraction | Active-loop detection where `\|γ_e\| > ε` | custom CUDA kernel |
| VI. Regime clustering | DBSCAN over a rolling `[128 × 4]` feature window (Yang-Mills action, max/mean curl, Betti-1), refit every 10 signal ticks | **cuML** (`cuml::DBSCAN`), fully on-device |

Stage VI is the current cuML integration: the feature window is built and stays
resident in VRAM, `cuml::DBSCAN::fit()` runs on a dedicated CUDA stream so the
refit never blocks the per-tick hot path (stages I–V), and the clustering result
is read back only for logging/decision use — no host round-trip on the compute
path itself.

---

## 2. Environment

- GPU: NVIDIA A100-SXM4-40GB (also verified on L40S)
- CUDA 12.5, cuML 24.10.00, driver via conda-forge `rapids-holographic` env
- C++20 / CUDA, Boost.Asio + Beast for networking, no Python in the hot path
- Live venue: Binance USDS-M Futures (Testnet verified; architecture is
  venue-agnostic at the feed/gateway boundary)

---

## 3. Performance — verified, post-fix

### 3.1 The honest baseline

An earlier internal benchmark claimed "451.9M rows in 51.5s (~8.7M rows/sec)."
That number was **found to be wrong** and has been retracted: the CSV-replay
producer was silently dropping updates whenever the GPU consumer fell behind,
instead of backpressuring. On that run, 846.6M of 1.298B total updates (65%)
were silently discarded — the 8.7M rows/sec figure measured how fast data was
read and thrown away, not processed.

**Fix**: producer now blocks (spin-waits) on a full ring instead of dropping.
`updates_dropped` reads `0` on every run since.

### 3.2 Corrected throughput (default config, `HOLO_GPU_BATCH=2048`)

Run on 14 days of real Binance L2 tick data (4 instruments: BTC/ETH/SOL/BNB):

```
Rows read        : 51,164,388
Updates dropped  : 0
Elapsed          : 95.10 s
Throughput       : 538,006 rows/sec  (1,076,012 LOB updates/sec)
Signals executed : 28,879
Sharpe (per-sig) : 0.0393
Win Rate         : 52.0%
Terminal PnL     : 5,120.12 bps
```

This is genuine end-to-end throughput through the full 6-stage pipeline
(Hodge decomposition + cuSOLVER SVD + cuML DBSCAN included), not a memcpy
benchmark.

### 3.3 Batch-size sweep — speed vs. signal quality is a real, measured tradeoff

`HOLO_GPU_BATCH` controls how many raw LOB updates accumulate before each GPU
pipeline invocation. Swept on the same 14-day dataset:

| Batch | Elapsed | Signals | Sharpe | PnL (bps) | Win rate |
|---|---|---|---|---|---|
| 1024 | 187.0s | 57,868 | 0.0367 | 9,589.5 | 51.8% |
| 2048 | 94.2s | 29,538 | 0.0373 | 4,974.2 | 51.8% |
| 4096 | 48.4s | 14,887 | 0.0286 | 1,917.9 | 51.3% |
| 8192 | 30.6s | 7,691 | 0.0086 | 297.9 | 50.3% |
| 16384 | 22.2s | 3,946 | -0.0192 | -343.4 | 48.9% |

Interpretation: 1024→2048 costs almost nothing in per-signal quality (PnL/signal
is ~0.166–0.168 bps in both) — the difference is pure sampling volume. Beyond
4096, both volume *and* per-signal quality degrade together — the pipeline is
sampling the order-book graph too infrequently to catch the signal cleanly, not
just less often. This is a genuine, disclosable methodology tradeoff, not a free
speedup — see `infra/batch_sweep.sh` / `infra/batch_sweep_chart.py` in the repo
for the reproducible methodology.

---

## 4. Latency budget (single-GPU, per tick)

| Stage | Engine | Measured |
|---|---|---|
| PCIe DMA, LOB frame | pinned host → device | ~0.5 μs |
| Laplacian assembly | cuSPARSE | ~50 μs |
| LOBPCG Fiedler, 30 iter | cuBLAS SpMM | ~200 μs |
| Hodge decomposition | cuSOLVER `Ssyevd` | ~500 μs |
| Curvature/signal extraction | custom kernel | ~10 μs |
| **Full cycle, book update → signal** | | **< 1 ms** |
| Regime clustering (async, every 10 cycles) | cuML DBSCAN | ~5–20 ms |

---

## 5. Engineering-rigor note (relevant context, not a claim about the numbers above)

This week's process also surfaced and fixed a genuine concurrency bug: a
lock-free ring buffer publishing its write index (via `fetch_add`, `acq_rel`)
*before* the payload write it was meant to guard — a reader racing in during
that window could observe stale data from a full ring rotation earlier. Root-
caused via elimination (ruled out hypervisor steal-time, per-tick
`cudaMalloc`/`cudaFree` overhead, and stdout buffering, in that order, before
finding the actual bug) and fixed; verified via a 10-minute stability run with
zero recurrence. Full methodology is in the repo's commit history
(`debug/signal-latency-spike` branch, merged to `main`).

---

## 6. What is NOT built yet — explicit, so nobody overclaims

- **cuDF** for order-book ingest — not integrated; ingest is currently raw
  C++/mmap CSV replay (backtest) or a Boost.Beast WS feed (live).
- **UMAP** dimensionality reduction ahead of clustering — not built.
- **HDBSCAN** — Stage VI currently uses `cuml::DBSCAN`, not HDBSCAN.
- **XGBoost** prediction layer on top of regime clusters — not built.
- **FPGA / AF_XDP / kernel-bypass networking** — not present anywhere in the
  codebase. Networking is standard Boost.Asio/Beast.
- **Core pinning (`isolcpus`/`taskset`)** — not implemented.

These are a legitimate near-term roadmap (William's stated interest — KDE,
KMeans, UMAP, HDBSCAN, XGBoost — maps directly onto extending Stage VI), just
not yet shipped.

---

## 7. Repo

`github.com/nathaniel-brodetsky/holographic_market` — `main` branch, current
as of this brief. Relevant files for a write-up:
- `engine/ai/cuml_clustering.cu` / `.cuh` — the cuML DBSCAN integration
- `engine/math/hodge_kernel.cu` — Hodge decomposition, cuSOLVER usage
- `infra/batch_sweep.sh`, `infra/batch_sweep_chart.py` — the sweep methodology
  in §3.3, fully reproducible
