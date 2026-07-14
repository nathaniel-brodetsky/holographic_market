#!/usr/bin/env bash
# infra/provision_ubuntu.sh
# Idempotent bootstrap — Ubuntu 22.04 LTS (Jammy), x86-64, NVIDIA GPU.
# sudo bash infra/provision_ubuntu.sh
set -euo pipefail
IFS=$'\n\t'

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[PROVISION]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[FATAL]${NC} $*"; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo bash infra/provision_ubuntu.sh"
[[ "$(dpkg --print-architecture)" == "amd64" ]] || die "amd64 only"

CODENAME=$(lsb_release -sc 2>/dev/null || echo "jammy")
CUDA_VERSION="12-5"
RAPIDS_VERSION="24.10"
PYTHON_VERSION="3.11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log "Ubuntu ${CODENAME} · CUDA ${CUDA_VERSION} · RAPIDS ${RAPIDS_VERSION} · repo ${REPO_ROOT}"

# ─── 0. system update ────────────────────────────────────────────────────────
log "Phase 0: system update"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
ok "system updated"

# ─── 1. toolchain ────────────────────────────────────────────────────────────
log "Phase 1: toolchain"
apt-get install -y -qq \
    software-properties-common apt-transport-https ca-certificates \
    curl wget gnupg lsb-release git git-lfs \
    build-essential gcc-12 g++-12 \
    clang-16 clang-format-16 clang-tidy-16 \
    llvm-16 libc++-16-dev libc++abi-16-dev \
    ninja-build pkg-config \
    gdb valgrind linux-tools-generic \
    numactl hwloc libhwloc-dev libnuma-dev \
    pciutils ethtool iproute2 \
    libatomic1 cpufrequtils

if ! cmake --version 2>/dev/null | grep -qE "3\.[2-9][6-9]|3\.[3-9][0-9]"; then
    log "CMake: installing from Kitware APT"
    wget -qO /usr/share/keyrings/kitware-archive-keyring.gpg \
        https://apt.kitware.com/keys/kitware-archive-latest.asc
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
        https://apt.kitware.com/ubuntu/ ${CODENAME} main" \
        > /etc/apt/sources.list.d/kitware.list
    apt-get update -qq && apt-get install -y -qq cmake
fi

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12
ok "toolchain: $(gcc --version | head -1)"

# ─── 2. CUDA Toolkit ─────────────────────────────────────────────────────────
log "Phase 2: CUDA Toolkit ${CUDA_VERSION}"
if ! dpkg -l 2>/dev/null | grep -q "cuda-toolkit-${CUDA_VERSION}"; then
    TMP=$(mktemp /tmp/cuda-keyring-XXXXXX.deb)
    wget -qO "${TMP}" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i "${TMP}" && rm -f "${TMP}"
    apt-get update -qq
    apt-get install -y -qq \
        "cuda-toolkit-${CUDA_VERSION}" \
        "libcublas-dev-${CUDA_VERSION}" \
        "libcusparse-dev-${CUDA_VERSION}" \
        "libcusolver-dev-${CUDA_VERSION}" \
        "cuda-nvtx-${CUDA_VERSION}"
fi
cat > /etc/profile.d/cuda.sh << 'CUDA_ENV'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
CUDA_ENV
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
ok "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'nvcc — relogin to refresh PATH')"

# ─── 3. C++ libraries ────────────────────────────────────────────────────────
log "Phase 3: Boost · OpenSSL · simdjson"
apt-get install -y -qq \
    libboost-all-dev \
    libssl-dev \
    libsimdjson-dev \
    zlib1g-dev

FFDIR="${REPO_ROOT}/cpp_engine/third_party/fast_float/include/fast_float"
if [[ ! -f "${FFDIR}/fast_float.h" ]]; then
    log "vendoring fast_float 6.1.0"
    mkdir -p "${FFDIR}"
    wget -qO "${FFDIR}/fast_float.h" \
        "https://github.com/fastfloat/fast_float/releases/download/v6.1.0/fast_float.h"
fi
ok "C++ libs ready"

# ─── 4. Python venv ───────────────────────────────────────────────────────────
log "Phase 4: Python ${PYTHON_VERSION} venv"
apt-get install -y -qq \
    "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" \
    "python${PYTHON_VERSION}-venv" python3-pip

VENV="${REPO_ROOT}/.venv_holographic"
[[ -d "${VENV}" ]] || "python${PYTHON_VERSION}" -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip wheel setuptools -q
"${VENV}/bin/pip" install -r "${REPO_ROOT}/requirements.txt" -q
ok "venv: ${VENV}"

# ─── 5. Miniforge + RAPIDS cuML ──────────────────────────────────────────────
log "Phase 5: Miniforge + RAPIDS ${RAPIDS_VERSION} (cuML C++ headers)"
if ! command -v conda &>/dev/null; then
    TMP=$(mktemp /tmp/miniforge-XXXXXX.sh)
    wget -qO "${TMP}" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "${TMP}" -b -p /opt/miniforge3 && rm -f "${TMP}"
    ln -sf /opt/miniforge3/bin/conda /usr/local/bin/conda
fi

CONDA="/opt/miniforge3/bin/conda"
ENV_NAME="rapids-holographic"
if ! "${CONDA}" env list | grep -q "${ENV_NAME}"; then
    "${CONDA}" create -y -n "${ENV_NAME}" \
        -c rapidsai -c conda-forge -c nvidia \
        "cuml=${RAPIDS_VERSION}" "cuda-version=12.2" "python=${PYTHON_VERSION}" \
        || warn "cuML conda install failed — GPU absent or driver mismatch"
fi

CUML_ROOT="/opt/miniforge3/envs/${ENV_NAME}"
if [[ -d "${CUML_ROOT}/include/cuml" ]]; then
    ok "cuML headers: ${CUML_ROOT}/include/cuml"
    cat > /etc/profile.d/cuml.sh << CUML_ENV
export CUML_ROOT=${CUML_ROOT}
export CMAKE_PREFIX_PATH=${CUML_ROOT}:\${CMAKE_PREFIX_PATH:-}
CUML_ENV
else
    warn "cuML headers not found — set -DCUML_ROOT= manually in CMake"
fi

# ─── 6. system tuning ────────────────────────────────────────────────────────
log "Phase 6: system tuning"
echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo never > /sys/kernel/mm/transparent_hugepage/defrag  2>/dev/null || true
command -v cpufreq-set &>/dev/null && cpufreq-set -g performance 2>/dev/null || true
sysctl -w net.core.busy_poll=50              2>/dev/null || true
sysctl -w net.core.busy_read=50              2>/dev/null || true
sysctl -w net.ipv4.tcp_low_latency=1         2>/dev/null || true
sysctl -w net.core.rmem_max=134217728        2>/dev/null || true
sysctl -w net.core.wmem_max=134217728        2>/dev/null || true
sysctl -w net.core.netdev_max_backlog=250000 2>/dev/null || true
sysctl -w net.ipv4.tcp_fastopen=3            2>/dev/null || true
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -pm 1 2>/dev/null || true
    nvidia-smi --auto-boost-default=0 2>/dev/null || true
    ok "NVIDIA persistence ON"
fi
ok "system tuned"

# ─── 7. build ────────────────────────────────────────────────────────────────
log "Phase 7: cmake configure + build"
cd "${REPO_ROOT}/cpp_engine"
mkdir -p build

CUML_FLAG=""
[[ -d "/opt/miniforge3/envs/${ENV_NAME}" ]] && \
    CUML_FLAG="-DCUML_ROOT=/opt/miniforge3/envs/${ENV_NAME}"

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_CXX_COMPILER=g++-12 \
    ${CUML_FLAG} \
    2>&1 | tee build/cmake_configure.log

cmake --build build -j"$(nproc)" 2>&1 | tee build/cmake_build.log
ok "binary: ${REPO_ROOT}/cpp_engine/build/bin/holographic_bench"

# ─── summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  UBUNTU PROVISION COMPLETE${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Run:     cd ${REPO_ROOT}/cpp_engine && ./build/bin/holographic_bench"
echo "  Python:  source ${VENV}/bin/activate && python run_research.py"
echo "  Profile: nsys profile --trace=cuda,nvtx,osrt -o holo ./build/bin/holographic_bench"
echo ""
