# Holographic Market Architecture — v0.3.0

> *"The curvature of a gauge connection is not a metaphor for arbitrage. It is arbitrage."*

---

## I. What This Is

A bare-metal GPU execution engine for **topological arbitrage detection** in live cryptocurrency limit order books. No indicators. No heuristics. Pure differential geometry computed on an NVIDIA GPU in under one millisecond per cycle.

**Phase I** — zero-allocation C++20 execution kernel, 64-byte aligned SPSC ring buffer.  
**Phase II** — CUDA topological pipeline: normalized Laplacian → LOBPCG Fiedler vector → Hodge-De Rham decomposition → Yang-Mills curvature extraction.  
**Phase III** — live Binance L2 WebSocket feed handler (implemented, wired in). cuML DBSCAN on-device regime clustering is scaffolded (`engine/ai/cuml_clustering.{cu,cuh}`, `TopologyClusterer`) but **not yet implemented** — every method is currently an empty stub, and the class is not instantiated anywhere in `main_live.cpp`/`main_backtest.cpp`/`cuda_pipeline.cu`. The live pipeline currently runs stages ①–⑤ only (Laplacian → LOBPCG → Hodge decomposition → signal extraction); stage ⑥ below is a design target, not a running component.  
**Phase IV** *(current)* — live execution engine / maker OMS (`engine/net/execution_engine.hpp`, `oms_core.hpp`, `binance_gateway.hpp`, `user_data_feed.hpp`), talking to **Binance Futures Testnet** (`testnet.binancefuture.com`) via WebSocket order routing. Includes authoritative order-state tracking from the User Data Stream (incl. a margin-call callback), a reject-burst circuit breaker plus a separate PnL drawdown kill-switch (`check_drawdown`), and a legging risk manager that waits a fixed 50ms (`kLegTimeout`) for the resting leg to fill before hedging the remainder at market (`hedge_remaining`). This is wired into `main_live.cpp` against the exchange testnet — it has not been run against mainnet or with real capital.

---

## II. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOLOGRAPHIC ENGINE v0.3                              │
│                                                                         │
│  [Binance WSS fstream.binance.com]                                      │
│        │  TLS 1.3 / Beast WebSocket                                     │
│        │  simdjson zero-copy parse                                      │
│        ▼                                                                │
│  [BinanceFeedHandler — boost::asio io_context]                          │
│        │  try_push → DynamicSpscRingBuffer<LobUpdate>                   │
│        ▼  (64-byte aligned, lock-free SPSC)                             │
│  [CPU Core 2 — LOB Consumer]                                            │
│        │  apply() → LobSoA (SoA, alignas(64), float32[N×D])            │
│        ▼                                                                │
│    cudaHostRegister (pinned, zero-copy PCIe DMA)                        │
│        │                                                                │
│  ┌─────▼──────────────────────────────────────────────────────────┐    │
│  │                     GPU PIPELINE (CudaPipeline)                │    │
│  │                                                                │    │
│  │  ① build_normalized_laplacian                                  │    │
│  │     cuSPARSE CSR  ·  kernel_build_cross_impact_edges           │    │
│  │     L = D^{-½}(D−W)D^{-½}  ∈ ℝ^{N×N}                        │    │
│  │                    │                                           │    │
│  │  ② lobpcg_solve                                                │    │
│  │     LOBPCG + cuBLAS SpMM  ·  k=4 eigenpairs                   │    │
│  │     Fiedler v₂ → 99th-percentile prune mask                   │    │
│  │                    │                                           │    │
│  │  ③ build_incidence_matrices                                    │    │
│  │     B₁ ∈ ℝ^{|V|×|E|}  ·  B₂ ∈ ℝ^{|E|×|T|}                  │    │
│  │     Δ₁ = B₁ᵀB₁ + B₂B₂ᵀ  ∈ ℝ^{|E|×|E|}                      │    │
│  │                    │                                           │    │
│  │  ④ compute_hodge_decomposition                                 │    │
│  │     cuSOLVER Ssyevd  ·  full spectral decomp of Δ₁            │    │
│  │     ω = dα ⊕ δβ ⊕ γ   (Helmholtz-Hodge)                      │    │
│  │     γ ∈ ker(Δ₁)  →  harmonic arbitrage flow                   │    │
│  │                    │                                           │    │
│  │  ⑤ extract_arbitrage_signal                                    │    │
│  │     S_YM[A] = tr(F∧⋆F) / 4g²                                 │    │
│  │     d_arb_signal → active loops {e : |γₑ| > ε}               │    │
│  │                    │                                           │    │
│  │  ⑥ TopologyClusterer (cuML DBSCAN)                             │    │
│  │     ML::dbscanFit on d_arb_signal — device only               │    │
│  │     Regime labels stay on GPU                                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                    [Signal Consumer / Order Router]                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## III. Mathematical Foundations

### III.1 Cross-Impact Tensor as Gauge Connection

$$\mathcal{W}_{ij} = \sum_{d=0}^{D-1} \frac{(\mu_i^d - \mu_j^d)(\mathrm{OIB}_i^d - \mathrm{OIB}_j^d)}{|\mu_i^d - \mu_j^d| + \varepsilon}$$

where $\mu_i^d = \frac{b_i^d + a_i^d}{2}$ is mid-price at depth $d$ and $\mathrm{OIB}_i^d = \frac{Q_i^{\mathrm{bid},d} - Q_i^{\mathrm{ask},d}}{Q_i^{\mathrm{bid},d} + Q_i^{\mathrm{ask},d}}$ is order imbalance.

Normalized graph Laplacian: $\mathcal{L} = D^{-1/2}(D - W)D^{-1/2} \in \mathbb{R}^{N \times N}$

### III.2 LOBPCG Fiedler Extraction

$$\mathbf{x}^{(k+1)} = \underset{\mathbf{x} \in \mathrm{span}(\mathbf{X}^{(k)}, \mathbf{W}^{(k)}, \mathbf{P}^{(k)})}{\arg\min}\, \mathcal{R}(\mathbf{x}), \qquad \mathcal{R}(\mathbf{x}) = \frac{\mathbf{x}^T \mathcal{L} \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$$

Convergence bound: $\tan\angle(\mathbf{x}^{(k)}, \mathbf{v}_2) \leq \left(\frac{\lambda_3 - \lambda_2}{\lambda_3 + \lambda_2}\right)^k \cdot C_0$

### III.3 Hodge-De Rham Decomposition

$$\Delta_1 = B_1^T B_1 + B_2 B_2^T \in \mathbb{R}^{|E| \times |E|}$$
$$\omega = \underbrace{d\alpha}_{\text{exact}} \oplus \underbrace{\delta\beta}_{\text{co-exact}} \oplus \underbrace{\gamma}_{\text{harmonic}}$$

$\gamma \in \ker(\Delta_1)$: first de Rham cohomology class. $\beta_1 = \dim\ker(\Delta_1)$: first Betti number = count of independent arbitrage cycles.

### III.4 Yang-Mills Action

$$S_{YM}[A] = \frac{1}{4g^2} \sum_e F_e^2, \qquad F_e = (\gamma_e - \delta_e) + \gamma_e \cdot \delta_e - \delta_e \cdot \gamma_e$$

$S_{YM} \to 0$: market approaches flat connection (zero arbitrage).  
$S_{YM} \gg 0$: structural curvature present.

### III.5 cuML DBSCAN Regime Clustering — *design spec, not yet implemented*

> **Status: not implemented.** `engine/ai/cuml_clustering.cu` currently contains only empty method bodies, and `TopologyClusterer` is not instantiated anywhere in the live or backtest binaries. The description below is the intended design, kept here as a spec for the eventual implementation — treat any numbers in III.6 for this stage as a target, not a measurement.

On-device DBSCAN over a rolling window of $(S_{YM}, \max|\gamma|, \bar{|\gamma|}, \beta_1)$ feature vectors. Identifies macro-regimes of structural arbitrage without D→H copies on the signal path.

$$\text{DBSCAN}(\varepsilon, m_{\min}): \mathbb{R}^{|W| \times 4} \to \{-1, 0, 1, \ldots, K\}^{|W|}$$

### III.6 Latency Budget

| Stage | Target |
|---|---|
| PCIe DMA (16 KB LOB) | ~0.5 μs |
| Laplacian assembly (cuSPARSE) | ~50 μs |
| LOBPCG Fiedler (30 iter) | ~200 μs |
| Hodge decomp (cuSOLVER Ssyevd) | ~500 μs |
| Signal extraction | ~10 μs |
| **Total pipeline** | **< 1 ms** |
| DBSCAN (async, every 10 cycles) | ~5–20 ms *(target — stage not implemented, see III.5)* |

---

## IV. Repository Structure

```
holographic_market/
├── infra/
│   ├── provision.sh                      # Idempotent VM bootstrap (Ubuntu)
│   ├── provision_ubuntu.sh / provision_fedora.sh
│   ├── download_binance_aggtrades.py / _v2.py   # AggTrades → LOB CSV downloader
│   ├── download_binance_data.py
│   ├── download_historical_range.sh / _v2.sh
│   ├── build_sharpe_grid.sh / _v2.sh     # Parameter-sweep backtest runner
│   ├── run_wfo.sh                        # Walk-forward optimization driver
│   ├── plot_backtest.py                  # PnL / drawdown chart → data/backtest_chart.png
│   └── quant_tearsheet.py                # PDF tear sheet → data/Institutional_Tear_Sheet.pdf
├── engine/                               # C++20 / CUDA engine (was `cpp_engine/`)
│   ├── CMakeLists.txt
│   ├── app/
│   │   ├── main_backtest.cpp             # CSV backtest entrypoint
│   │   └── main_live.cpp                 # Live engine entrypoint (Binance Futures Testnet)
│   ├── core/
│   │   ├── lob_core.hpp
│   │   ├── lockfree_ring_buffer.hpp
│   │   ├── memory_arena.hpp
│   │   └── backtest_stats.hpp
│   ├── math/                             # Phase II: topological pipeline
│   │   ├── cuda_pipeline.cu / .cuh
│   │   ├── cuda_utils.cuh
│   │   ├── gpu_lob_mirror.cu
│   │   ├── hodge_kernel.cu / .cuh
│   │   ├── lobpcg_solver.cu / .cuh
│   │   └── floer_homology.cu / .cuh
│   ├── net/                              # Phase III/IV: feed + execution
│   │   ├── binance_feed.hpp              # Phase III: L2 WebSocket feed handler
│   │   ├── binance_gateway.hpp           # Phase IV: WS-API order routing
│   │   ├── user_data_feed.hpp            # Phase IV: authoritative order/margin state
│   │   ├── oms_core.hpp                  # Phase IV: order/position bookkeeping
│   │   ├── execution_engine.hpp          # Phase IV: legging risk mgr, circuit breaker
│   │   ├── signal_router.hpp
│   │   ├── csv_replay.hpp                # Backtest feed replay
│   │   └── symbols.hpp                   # k_feed_n_instruments = 4
│   └── ai/
│       └── cuml_clustering.cu / .cuh     # Phase III: DBSCAN — STUB, not implemented
├── research/
│   └── analyze_wfo.py                    # Walk-forward optimization analysis
├── src/                                  # Python research (gauge-theory prototyping)
│   ├── gauge_theory_alpha.py
│   ├── spectral_pruning.py
│   └── tensor_compression.py
├── CUDA_SETUP.md                         # nvcc / PATH / cuML troubleshooting
├── DEPENDENCIES.md
├── requirements.txt
├── run_research.py
└── README.md
```

---

## V. Quick Start (VM)

```bash
# 1. Clone
git clone https://github.com/<org>/holographic_market.git
cd holographic_market/engine

# 2. Configure + build (Release; CUDA archs for T4/A100/RTX30xx/L4/L40S)
# If nvcc/CUDA isn't on your PATH, see CUDA_SETUP.md first.
# CMAKE_CUDA_ARCHITECTURES defaults to 89 if unset — override for your GPU
# (T4=75, A100=80, RTX30xx/L4=86, L40S/RTX40xx=89).
rm -rf build/
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DBOOST_ROOT=/usr/local \
    -DBoost_NO_SYSTEM_PATHS=ON

cmake --build build --parallel $(nproc)

# 3. Get test data — downloads Binance AggTrades for the given date into
#    data/test_data.csv (see infra/download_binance_aggtrades_v2.py --help
#    for --out, --symbols, --chunksize, --max-retries, --no-checksum)
cd ..
python3 infra/download_binance_aggtrades_v2.py --date 2024-01-01

# 4a. Run the CSV backtest
#     --mode {fastest|realtime|throttled} (default fastest), --gpu-device <N> (default 0),
#     --out-dir <path> for advanced_metrics.csv (default ../data)
./engine/build/bin/holographic_backtest data/test_data.csv --mode fastest --gpu-device 0

# 4b. Generate charts / tear sheet from the backtest output
python3 infra/plot_backtest.py        # → data/backtest_chart.png
python3 infra/quant_tearsheet.py      # → data/Institutional_Tear_Sheet.pdf

# 5. Run the live engine (Phase IV — Binance Futures TESTNET, not mainnet)
# Requires:
export HOLO_API_KEY=...
export HOLO_API_SECRET=...
# Optional overrides: HOLO_ORDER_SIZE_USD, HOLO_MAX_POSITION_USD, HOLO_MAX_DRAWDOWN_USD
./engine/build/bin/holographic_live
```

### Optional CMake flags

| Flag | Default | Effect |
|---|---|---|
| `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"` | — | Target SM architectures (T4=75, A100=80, RTX30xx/L4=86, L40S=89) |
| `-DBOOST_ROOT=/usr/local` | — | Point at a manually-built Boost 1.83+ (required: system, json) |

---

## VI. Dependencies

See [`DEPENDENCIES.md`](DEPENDENCIES.md) for the full manifest.  
`infra/provision.sh` installs everything automatically on Ubuntu 22.04.

**Summary:**

| Layer | Key deps |
|---|---|
| Toolchain | GCC 12, CMake 3.26+, Ninja |
| CUDA | Toolkit 12.x, cuBLAS, cuSPARSE, cuSOLVER |
| Networking | Boost 1.83 (Asio + Beast), OpenSSL 3 |
| Parsing | simdjson 3.9+, fast_float 6.x |
| ML | RAPIDS cuML 24.x (via conda), RAFT |
| Python | NumPy, SciPy, JAX, NetworkX, Matplotlib |

---

## VII. Publication Roadmap

**Target venues:**

1. **arXiv:q-fin.CP** — *"Topological Arbitrage Detection via Hodge Decomposition of Limit Order Book Flows"*
2. **Journal of Computational Finance** — peer-reviewed
3. **SIAM Journal on Financial Mathematics** — mathematical rigor track

**Remaining requirements for submission:**

1. Real market data validation (Binance WebSocket) ← **WS feed handler and the Phase IV maker OMS/execution engine are implemented and wired in against Binance Futures Testnet, currently configured for N = 4 instruments (`k_feed_n_instruments` in `engine/net/symbols.hpp`), not the N ≥ 100 target. Testnet only — not yet run against mainnet or real capital. This item does not include the DBSCAN regime-clustering stage, which remains unimplemented (see III.5).**
2. Statistical backtest: Sharpe ratio of harmonic-flow-ranked portfolios vs. benchmark
3. Comparison with PCA-based cross-impact models
4. Formal discrete proof of Arbitrage-Curvature Correspondence
5. Complexity analysis: LOBPCG $\mathcal{O}(k \cdot \mathrm{nnz})$ vs. dense $\mathcal{O}(N^3)$

---

## VIII. References

1. Atiyah, M.F. & Bott, R. (1983). The Yang-Mills equations over Riemann surfaces. *Phil. Trans. R. Soc. Lond. A* **308**, 523–615.
2. Knyazev, A.V. (2001). Toward the optimal preconditioned eigensolver: LOBPCG. *SIAM J. Sci. Comput.* **23**(2), 517–541.
3. Lim, L.H. (2020). Hodge Laplacians on graphs. *SIAM Review* **62**(3), 685–715.
4. Eckmann, B. (1944). Harmonische Funktionen und Randwertaufgaben in einem Komplex. *Comment. Math. Helv.* **17**, 240–255.
5. Fiedler, M. (1973). Algebraic connectivity of graphs. *Czech. Math. J.* **23**(98), 298–305.
6. NVIDIA Corporation (2024). cuSPARSE Library User's Guide. v12.x.
7. NVIDIA Corporation (2024). cuSOLVER Library User's Guide. v11.x.
8. Cont, R., Kukanov, A. & Stoikov, S. (2014). The price impact of order book events. *J. Financial Econometrics* **12**(1), 47–88.
9. Witten, E. (1989). Quantum field theory and the Jones polynomial. *Comm. Math. Phys.* **121**(3), 351–399.
10. Hamilton, R.S. (1982). Three-manifolds with positive Ricci curvature. *J. Diff. Geom.* **17**(2), 255–306.