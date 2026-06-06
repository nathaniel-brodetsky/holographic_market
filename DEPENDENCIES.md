# Holographic Market Architecture — Dependency Manifest

## Runtime & Build Host Requirements

| Component | Min Version | Notes |
|-----------|-------------|-------|
| Linux kernel | 5.15+ | `io_uring`, huge-pages, `SO_BUSY_POLL` |
| glibc | 2.35+ | AVX-512 dispatch, `clock_gettime` VDSO |
| GCC | 12+ | C++20 coroutines, `std::atomic` wait |
| Clang/LLVM | 16+ | alternative toolchain; required for TSAN |
| CMake | 3.26+ | `cmake_path`, native CUDA arch detection |
| Ninja | 1.11+ | recommended generator |
| CUDA Toolkit | 12.2+ | `cudaHostRegister`, NVTX r3, nvcc |
| NVIDIA Driver | 535+ | CUDA 12.x compatible |

---

## C++ System Libraries

| Library | Version | Purpose | Install |
|---------|---------|---------|---------|
| **Boost** | 1.83+ | Asio (async I/O), Beast (WebSocket/TLS), System | see below |
| **OpenSSL** | 3.x | TLS 1.3 for Binance WSS | `libssl-dev` / `openssl-devel` |
| **simdjson** | 3.9+ | Zero-copy JSON parse of L2 feed frames | `libsimdjson-dev` / source |
| **fast_float** | 6.x | Locale-free `from_chars` for price strings | header-only, vendored |
| **pthreads** | POSIX | Feed handler + consumer thread pinning | system |
| **libatomic** | glibc | 128-bit CAS on older x86 | `libatomic1` |

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

### RAPIDS cuML installation (Ubuntu/Debian)

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
> the conda package. The PyPI wheel does **not** include them.
> Set `CUML_ROOT=$(conda info --base)/envs/<env>` when invoking CMake.

---

## Python (research / visualisation layer)

```
numpy>=2.1.0
scipy>=1.14.0
sparse>=0.15.0
tensornetwork>=0.4.6
jax>=0.4.35
jaxlib>=0.4.35
networkx>=3.4.0
matplotlib>=3.9.0
```

Full install:
```bash
python3 -m venv .venv_holographic
source .venv_holographic/bin/activate
pip install -r requirements.txt
```

---

## VM Provisioning (Ubuntu 22.04 LTS)

See `infra/provision.sh` for the full idempotent bootstrap.
Quick reference:

```bash
# Toolchain
apt-get install -y \
  build-essential gcc-12 g++-12 clang-16 \
  cmake ninja-build git pkg-config \
  libboost1.83-all-dev \
  libssl-dev \
  libsimdjson-dev \
  libatomic1 \
  numactl hwloc linux-tools-generic

# CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y \
  cuda-toolkit-12-5 \
  libcublas-dev-12-5 \
  libcusparse-dev-12-5 \
  libcusolver-dev-12-5
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