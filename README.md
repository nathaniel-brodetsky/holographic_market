# Holographic Market Architecture — v0.5.0

> *"The curvature of a gauge connection is not a metaphor for arbitrage. It is arbitrage."*

---

## I. What This Is

A bare-metal GPU execution engine for **topological arbitrage detection** in live cryptocurrency limit order books. No indicators. No heuristics. Pure differential geometry computed on an NVIDIA GPU in under one millisecond per cycle.

**Phase I** — zero-allocation C++20 kernel: 64-byte aligned SPSC ring buffer.  
**Phase II** — CUDA topological pipeline: Laplacian → LOBPCG → Hodge-De Rham → Yang-Mills curvature.  
**Phase III** — live Binance L2 WebSocket feed + cuML DBSCAN on-device regime clustering.  
**Phase IV** — multi-asset cross-instrument signal routing.  
**Phase V** *(current)* — full refactor to single-owner `LiveNode` design; async execution gateway.

---

## II. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                   HOLOGRAPHIC ENGINE v0.5                            │
│                                                                      │
│  fstream.binance.com:443                                             │
│       │  TLS 1.2 / Beast WebSocket                                   │
│       ▼                                                              │
│  [BinanceWssClient]  ── raw JSON frames ──►  [FeedHandler]          │
│                                                   │  parse + push    │
│                                                   ▼                  │
│                                     DynamicSpscRingBuffer<LobUpdate> │
│                                                   │  lob_drain_thread│
│                                                   ▼                  │
│                                              [LobSoA]                │
│                                     (SoA, alignas(64), float32[N×D])│
│                                                   │                  │
│                              cudaHostRegister (pinned, zero-copy DMA)│
│                                                   │                  │
│  ┌────────────────────────────────────────────────▼───────────────┐  │
│  │                    GPU PIPELINE (CudaPipeline)                 │  │
│  │                                                                │  │
│  │  ① build_normalized_laplacian   L = D^{-½}(D−W)D^{-½}        │  │
│  │  ② lobpcg_solve                 Fiedler v₂ → prune mask       │  │
│  │  ③ build_incidence_matrices     B₁, B₂, Δ₁                   │  │
│  │  ④ compute_hodge_decomposition  ω = dα ⊕ δβ ⊕ γ              │  │
│  │  ⑤ extract_arbitrage_signal     S_YM[A] = tr(F∧⋆F) / 4g²     │  │
│  │  ⑥ TopologyClusterer (cuML)     DBSCAN regime labels on-device│  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │  SignalRecord (host copy)              │
│                              ▼                                        │
│                        [AlphaModel]                                   │
│                    synthesize curl flows                              │
│                    select top-K edges                                 │
│                              │  RoutedEdge[]                          │
│                              ▼                                        │
│                        [RiskManager]  ← pre_trade_check               │
│                              │  approved edges                        │
│                              ▼                                        │
│                  [BinanceFapiGateway]                                 │
│                  Boost.Asio coroutines                                │
│                  POST /fapi/v1/order                                  │
│                  record_fill → RiskManager                            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## III. Repository Structure

```
engine/
├── common/                     # No business logic, no external deps beyond STL
│   ├── types.hpp               # Side, LobUpdate, RoutedEdge, instrument catalogue
│   ├── memory_arena.hpp        # Zero-heap bump allocator
│   ├── lockfree_ring_buffer.hpp
│   └── thread_utils.hpp        # pin_thread, now_ns, sleep_ns
│
├── data_feed/                  # Ingestion layer
│   ├── lob_core.hpp            # LobSoA — structure-of-arrays order book
│   ├── binance_wss.hpp         # Raw WebSocket client (no business logic)
│   ├── feed_handler.hpp        # JSON parser → LobUpdate → ring push
│   └── csv_replay.hpp          # Backtest data source (CSV → ring buffer)
│
├── compute/                    # Heavy math — unchanged internally
│   ├── cuda_pipeline.cuh/.cu    # Facade: run_continuous, last_signal
│   ├── cuda_utils.cuh
│   ├── hodge_kernel.cuh/.cu
│   ├── lobpcg_solver.cuh/.cu
│   ├── floer_homology.cuh/.cu
│   ├── gpu_lob_mirror.cuh/.cu
│   └── ai/
│       └── cuml_clustering.cuh/.cu
│
├── trading/                    # Business logic
│   ├── alpha_model.hpp         # SignalRecord + LobSoA → RoutedEdge[]
│   │                           # (replaces HostCurlBuffer + synth_curl + SignalRouter)
│   ├── risk_manager.hpp        # Position tracking, PnL, pre-trade limits
│   │                           # (extracted from ExecutionEngine)
│   └── backtest_stats.hpp      # Sharpe/drawdown/win-rate evaluator
│
├── execution/                  # Order transport
│   ├── execution_gateway.hpp   # Abstract interface: submit(RoutedEdge)
│   └── binance_fapi.hpp        # Boost.Asio coroutines → POST /fapi/v1/order
│
└── app/
    ├── live_node.hpp           # Owns and wires all subsystems
    ├── main_live.cpp           # ~20 lines: config + gateway + node.run()
    └── main_backtest.cpp
```

### What changed from v0.4

| Before | After |
|---|---|
| `engine/net/binance_feed.hpp` | split → `data_feed/binance_wss.hpp` (transport) + `data_feed/feed_handler.hpp` (parse) |
| `engine/core/lob_core.hpp` | moved → `data_feed/lob_core.hpp` |
| `HostCurlBuffer` + `synth_curl_from_signal` + `SignalRouter` in `main_live.cpp` | → `trading/alpha_model.hpp` |
| `ExecutionEngine::update_paper_position` (mixed HTTP + position logic) | position tracking → `trading/risk_manager.hpp`, transport → `execution/binance_fapi.hpp` |
| `ExecutionEngine` monolith | → `ExecutionGateway` interface + `BinanceFapiGateway` impl |
| `main_live.cpp` — 300+ lines | → `app/live_node.hpp` (orchestration) + `main_live.cpp` (~20 lines) |
| `now_ns()`, `pin_thread()` duplicated in `main_live.cpp` | → `common/thread_utils.hpp` |
| `Side`, `LobUpdate`, `RoutedEdge` scattered across `net/` headers | → `common/types.hpp` |

---

## IV. Mathematical Foundations

### IV.1 Cross-Impact Tensor as Gauge Connection

$$\mathcal{W}_{ij} = \sum_{d=0}^{D-1} \frac{(\mu_i^d - \mu_j^d)(\mathrm{OIB}_i^d - \mathrm{OIB}_j^d)}{|\mu_i^d - \mu_j^d| + \varepsilon}$$

where $\mu_i^d = \frac{b_i^d + a_i^d}{2}$ and $\mathrm{OIB}_i^d = \frac{Q_i^{\mathrm{bid},d} - Q_i^{\mathrm{ask},d}}{Q_i^{\mathrm{bid},d} + Q_i^{\mathrm{ask},d}}$.

Normalized Laplacian: $\mathcal{L} = D^{-1/2}(D - W)D^{-1/2} \in \mathbb{R}^{N \times N}$

### IV.2 LOBPCG Fiedler Extraction

$$\mathbf{x}^{(k+1)} = \underset{\mathbf{x} \in \mathrm{span}(\mathbf{X}^{(k)}, \mathbf{W}^{(k)}, \mathbf{P}^{(k)})}{\arg\min}\, \mathcal{R}(\mathbf{x}), \qquad \mathcal{R}(\mathbf{x}) = \frac{\mathbf{x}^T \mathcal{L} \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$$

### IV.3 Hodge-De Rham Decomposition

$$\Delta_1 = B_1^T B_1 + B_2 B_2^T \in \mathbb{R}^{|E| \times |E|}$$
$$\omega = \underbrace{d\alpha}_{\text{exact}} \oplus \underbrace{\delta\beta}_{\text{co-exact}} \oplus \underbrace{\gamma}_{\text{harmonic}}$$

$\gamma \in \ker(\Delta_1)$: first de Rham cohomology class.  
$\beta_1 = \dim\ker(\Delta_1)$: first Betti number = independent arbitrage cycles.

### IV.4 Yang-Mills Action

$$S_{YM}[A] = \frac{1}{4g^2} \sum_e F_e^2, \qquad F_e = (\gamma_e - \delta_e) + \gamma_e \cdot \delta_e - \delta_e \cdot \gamma_e$$

$S_{YM} \to 0$: market approaches flat connection (zero arbitrage).  
$S_{YM} \gg 0$: structural curvature present → signal routed.

### IV.5 Latency Budget

| Stage | Target |
|---|---|
| PCIe DMA (16 KB LOB) | ~0.5 μs |
| Laplacian assembly (cuSPARSE) | ~50 μs |
| LOBPCG Fiedler (30 iter) | ~200 μs |
| Hodge decomp (cuSOLVER Ssyevd) | ~500 μs |
| Signal extraction + AlphaModel | ~10 μs |
| **Total pipeline** | **< 1 ms** |
| DBSCAN (async, every 10 cycles) | ~5–20 ms |

---

## V. Quick Start

```bash
# 1. Clone
git clone https://github.com/<org>/holographic_market.git
cd holographic_market

# 2. Provision (Ubuntu 22.04, requires root, ~15-20 min first run)
sudo bash infra/provision.sh

# 3. Build
cd engine
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DCUML_ROOT=/opt/miniforge3/envs/rapids-holographic
cmake --build build -j$(nproc)

# 4. Run live (paper trading, 30 s window)
./build/bin/holographic_live

# 5. Profile
nsys profile --trace=cuda,nvtx,osrt -o holo_profile ./build/bin/holographic_live
nsys-ui holo_profile.nsys-rep
```

### CMake flags

| Flag | Default | Effect |
|---|---|---|
| `-DENABLE_ASAN=ON` | OFF | AddressSanitizer + UBSan |
| `-DENABLE_TSAN=ON` | OFF | ThreadSanitizer |
| `-DCMAKE_CUDA_ARCHITECTURES=89` | native | SM target (L4=89, A100=80) |

---

## VI. Dependencies

| Layer | Key deps |
|---|---|
| Toolchain | GCC 12, CMake 3.26+, Ninja |
| CUDA | Toolkit 12.x, cuBLAS, cuSPARSE, cuSOLVER |
| Networking | Boost 1.83 (Asio + Beast), OpenSSL 3 |
| Parsing | Boost.JSON (monotonic_resource, zero-heap steady-state) |
| ML | RAPIDS cuML 24.x (conda), RAFT |
| Python | NumPy, SciPy, JAX, NetworkX, Matplotlib |

---

## VII. References

1. Atiyah & Bott (1983). Yang-Mills equations over Riemann surfaces. *Phil. Trans. R. Soc. Lond. A* **308**.
2. Knyazev (2001). Toward the optimal preconditioned eigensolver: LOBPCG. *SIAM J. Sci. Comput.* **23**(2).
3. Lim (2020). Hodge Laplacians on graphs. *SIAM Review* **62**(3).
4. Eckmann (1944). Harmonische Funktionen und Randwertaufgaben. *Comment. Math. Helv.* **17**.
5. Fiedler (1973). Algebraic connectivity of graphs. *Czech. Math. J.* **23**(98).
6. NVIDIA (2024). cuSPARSE / cuSOLVER Library User's Guide v12.x.
7. Cont, Kukanov & Stoikov (2014). Price impact of order book events. *J. Financial Econometrics* **12**(1).
8. Witten (1989). Quantum field theory and the Jones polynomial. *Comm. Math. Phys.* **121**(3).
