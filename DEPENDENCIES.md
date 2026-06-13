# Holographic Market Architecture — Dependency Manifest

## Runtime & Build Host Requirements

| Component | Min Version | Notes |
|-----------|-------------|-------|
| Linux kernel | 5.15+ | `io_uring`, huge-pages, `SO_BUSY_POLL` |
| glibc | 2.35+ | AVX-512 dispatch, `clock_gettime` VDSO |
| GCC | 11.4+ | C++20 coroutines, `std::atomic` wait (Ubuntu 22.04 ships 11.4) |
| Clang/LLVM | 16+ | Alternative toolchain; required for TSAN |
| CMake | 3.22+ | Native in Ubuntu 22.04. Requires explicit `-std=c++20` for NVCC (see Known Gotchas) |
| Ninja | 1.11+ | Recommended generator |
| CUDA Toolkit | 12.2+ | `cudaHostRegister`, NVTX r3, nvcc |
| NVIDIA Driver | 535+ | CUDA 12.x compatible |

---

## C++ System Libraries

| Library | Version | Purpose | Install |
|---------|---------|---------|---------|
| **Boost** | 1.81+ | Asio (async I/O), Beast (WebSocket/TLS), System | **Must build from source** (see warning below) |
| **OpenSSL** | 3.x | TLS 1.3 for Binance WSS | `libssl-dev` / `openssl-devel` |
| **simdjson** | 3.9+ | Zero-copy JSON parse of L2 feed frames | `libsimdjson-dev` / source |
| **fast_float** | 6.x | Locale-free `from_chars` for price strings | Header-only, vendored |
| **pthreads** | POSIX | Feed handler + consumer thread pinning | System |
| **libatomic** | glibc | 128-bit CAS on older x86 | `libatomic1` |

> ⚠️ **CRITICAL BOOST WARNING for Ubuntu 22.04:**
> `apt-get install libboost-all-dev` installs version **1.74**. The C++20 Asio
> Coroutines require Boost **1.81+**. You **must** download and build Boost
> manually, then point CMake to it:
> ```bash
> cmake ... -DBOOST_ROOT=/usr/local -DBoost_NO_SYSTEM_PATHS=ON
> ```

---

## NVIDIA GPU Libraries (CUDA ecosystem)

| Library | Version | Purpose |
|---------|---------|---------|
| **cuBLAS** | 12.x | GEMM for LOBPCG subspace operations |
| **cuSPARSE** | 12.x | SpMM for Laplacian-vector products |
| **cuSOLVER** | 11.x | `Ssyevd` full spectral decomp (Hodge Δ₁) |
| **NVTX** (r3) | 3.x | Nsight profiler range annotations |
| **cuML** | 24.x (RAPIDS) | `ML::dbscanFit` on-device clustering |
| **RAFT** | 24.x | cuML transitive dep; distance primitives |

### RAPIDS cuML Installation (Ubuntu/Debian)

```bash
# RAPIDS 24.x requires CUDA 12.x + Python 3.10–3.11
pip install \
  --extra-index-url https://pypi.nvidia.com \
  cuml-cu12==24.*

# OR via conda (recommended for full C++ headers)
conda install -c rapidsai -c conda-forge \
  cuml=24.10 cuda-version=12.2
```

> The C++ headers (`cuml/cluster/dbscan.hpp`, `raft/core/handle.hpp`) ship with
> the **conda** package. The PyPI wheel does **not** include them.
> Set `CUML_ROOT=$(conda info --base)/envs/<env>` when invoking CMake.

### Lightweight Builds (No cuML)

If deploying in a cloud environment where configuring the massive NVIDIA RAPIDS
conda environment is impractical, you can stub out `cuml_clustering.cu`. The
engine will gracefully bypass **Phase VI (Regime detection)** while keeping the
core arbitrage topology pipeline intact.

---

## Python (Research / Visualisation / Downloader Layer)

```text
numpy>=2.1.0
scipy>=1.14.0
sparse>=0.15.0
tensornetwork>=0.4.6
jax>=0.4.35
jaxlib>=0.4.35
networkx>=3.4.0
matplotlib>=3.9.0
pandas>=2.0.0
requests>=2.31.0
```

Full install:

```bash
python3 -m venv .venv_holographic
source .venv_holographic/bin/activate
pip install -r requirements.txt
```

> `pandas` and `requests` are required by `infra/download_binance_data.py` to
> fetch and parse historical L2 data for the fast C++ backtester.

---

## VM Provisioning (Ubuntu 22.04 LTS)

See `infra/provision.sh` for the full idempotent bootstrap. Quick reference:

```bash
# Toolchain
apt-get install -y \
  build-essential gcc-12 g++-12 clang-16 \
  cmake ninja-build git pkg-config \
  libssl-dev \
  libsimdjson-dev \
  libatomic1 \
  numactl hwloc linux-tools-generic

# CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y \
  cuda-toolkit-12-4 \
  libcublas-dev-12-4 \
  libcusparse-dev-12-4 \
  libcusolver-dev-12-4
```

> **Note:** `libboost-all-dev` is intentionally **omitted** — Ubuntu 22.04 ships
> Boost 1.74 which is incompatible. Build Boost 1.81+ from source instead.

---

## Known Gotchas (Cloud Deployments & Ubuntu 22.04)

If deploying on AWS, GCP, or cloud environments like Brev.dev, observe the
following critical fixes:

1. **Binance WebSocket Firewall Blocks:** Corporate cloud networks frequently
   block outbound traffic on port `9443` (Binance Spot WSS). Switch the target
   to Binance Futures (`fstream.binance.com`) using port `443` in
   `binance_feed.hpp` to bypass network filters.

2. **CMake CUDA20 Bug:** CMake 3.22 does not recognise the `CUDA20` language
   dialect. Remove `CMAKE_CUDA_STANDARD` from `CMakeLists.txt` and pass
   `-std=c++20` directly via `target_compile_options`.

3. **NVCC `cc1plus` Linker Error:** Force NVCC to use the correct host compiler:
   ```bash
   -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++
   ```

4. **LTO vs CUDA Separable Compilation:** Link-Time Optimisation (`-flto`)
   strips PTX `fatbinData` when combined with CUDA Separable Compilation. Ensure
   LTO is disabled:
   ```cmake
   set(HAS_LTO OFF)
   ```

---

## Complete Reliable CMake Build Command

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++ \
    -DBOOST_ROOT=/usr/local \
    -DBoost_NO_SYSTEM_PATHS=ON

cmake --build build --parallel $(nproc)
```

---

## Optional / Profiling

| Tool | Purpose |
|------|---------|
| Nsight Systems (`nsys`) | GPU+CPU timeline, PCIe bandwidth |
| Nsight Compute (`ncu`) | Kernel-level roofline, occupancy |
| Valgrind + Massif | Host heap profiling |
| `perf stat` | CPU PMU counters (cache misses, IPC) |
| `numactl` | NUMA-aware thread/memory placement |
| `ethtool` | Kernel bypass / RSS queue tuning |
