#!/usr/bin/env bash
# infra/provision.sh
# Idempotent bootstrap for Ubuntu 22.04 LTS VM (x86-64, NVIDIA GPU).
# Run as root or via sudo.  Safe to re-run.
set -euo pipefail
IFS=$'\n\t'

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[PROVISION]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[FATAL]${NC} $*"; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo bash infra/provision.sh"

UBUNTU_CODENAME=$(lsb_release -sc 2>/dev/null || echo "jammy")
ARCH=$(dpkg --print-architecture)
[[ "$ARCH" == "amd64" ]] || die "Only amd64 supported. Got: $ARCH"

CUDA_VERSION="12-5"
RAPIDS_VERSION="24.10"
PYTHON_VERSION="3.11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log "Host: Ubuntu ${UBUNTU_CODENAME} / ${ARCH}"
log "CUDA target: ${CUDA_VERSION}"
log "RAPIDS target: ${RAPIDS_VERSION}"
log "Repo root: ${REPO_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# 0. System update
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 0: system update"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
ok "System updated"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Core toolchain
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 1: toolchain (gcc-12, clang-16, cmake, ninja)"
apt-get install -y -qq \
    software-properties-common apt-transport-https ca-certificates \
    curl wget gnupg lsb-release git git-lfs \
    build-essential gcc-12 g++-12 \
    clang-16 clang-format-16 clang-tidy-16 \
    llvm-16 libc++-16-dev libc++abi-16-dev \
    ninja-build pkg-config autoconf automake libtool \
    gdb valgrind linux-tools-generic \
    numactl hwloc libhwloc-dev libnuma-dev \
    pciutils ethtool iproute2 strace ltrace libatomic1

if ! cmake --version 2>/dev/null | grep -qE "3\.[2-9][6-9]|3\.[3-9][0-9]"; then
    log "Installing CMake from Kitware APT"
    wget -q -O /usr/share/keyrings/kitware-archive-keyring.gpg \
        https://apt.kitware.com/keys/kitware-archive-latest.asc
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
        https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/kitware.list
    apt-get update -qq
    apt-get install -y -qq cmake
fi

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-12
ok "Toolchain: $(gcc --version | head -1)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. CUDA Toolkit
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 2: CUDA Toolkit ${CUDA_VERSION}"
if ! dpkg -l 2>/dev/null | grep -q "cuda-toolkit-${CUDA_VERSION}"; then
    TMP_DEB="$(mktemp /tmp/cuda-keyring-XXXXXX.deb)"
    wget -q -O "${TMP_DEB}" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    dpkg -i "${TMP_DEB}"
    rm -f "${TMP_DEB}"
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
ok "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH yet — relogin')"

# ─────────────────────────────────────────────────────────────────────────────
# 3. C++ libraries: Boost, OpenSSL, simdjson, fast_float
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 3: C++ libraries (Boost, OpenSSL, simdjson)"
apt-get install -y -qq \
    libboost-all-dev \
    libssl-dev \
    libsimdjson-dev \
    zlib1g-dev

FAST_FLOAT_VERSION="6.1.0"
FAST_FLOAT_DIR="${REPO_ROOT}/cpp_engine/third_party/fast_float/include/fast_float"
if [[ ! -f "${FAST_FLOAT_DIR}/fast_float.h" ]]; then
    log "Vendoring fast_float ${FAST_FLOAT_VERSION}"
    mkdir -p "${FAST_FLOAT_DIR}"
    wget -q -O "${FAST_FLOAT_DIR}/fast_float.h" \
        "https://github.com/fastfloat/fast_float/releases/download/v${FAST_FLOAT_VERSION}/fast_float.h"
fi
ok "C++ libs ready"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Python venv + research deps
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 4: Python ${PYTHON_VERSION} venv"
apt-get install -y -qq \
    "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" \
    "python${PYTHON_VERSION}-venv" python3-pip

VENV="${REPO_ROOT}/.venv_holographic"
[[ -d "${VENV}" ]] || "python${PYTHON_VERSION}" -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip wheel setuptools -q
"${VENV}/bin/pip" install -r "${REPO_ROOT}/requirements.txt" -q
ok "Python venv: ${VENV}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Miniforge + RAPIDS cuML (C++ headers)
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 5: Miniforge + RAPIDS ${RAPIDS_VERSION} conda env"
if ! command -v conda &>/dev/null; then
    log "Installing Miniforge"
    TMP_MF="$(mktemp /tmp/miniforge-XXXXXX.sh)"
    wget -q -O "${TMP_MF}" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "${TMP_MF}" -b -p /opt/miniforge3
    rm -f "${TMP_MF}"
    ln -sf /opt/miniforge3/bin/conda /usr/local/bin/conda
fi

CONDA_BIN="/opt/miniforge3/bin/conda"
CONDA_ENV_NAME="rapids-holographic"

if ! "${CONDA_BIN}" env list | grep -q "${CONDA_ENV_NAME}"; then
    log "Creating conda env: ${CONDA_ENV_NAME}"
    "${CONDA_BIN}" create -y -n "${CONDA_ENV_NAME}" \
        -c rapidsai -c conda-forge -c nvidia \
        "cuml=${RAPIDS_VERSION}" \
        "cuda-version=12.2" \
        "python=${PYTHON_VERSION}" \
        || warn "cuML conda install failed — GPU not present or unsupported driver"
fi

CUML_ROOT="/opt/miniforge3/envs/${CONDA_ENV_NAME}"
if [[ -d "${CUML_ROOT}/include/cuml" ]]; then
    ok "cuML C++ headers: ${CUML_ROOT}/include/cuml"
    cat > /etc/profile.d/cuml.sh << CUML_ENV
export CUML_ROOT=${CUML_ROOT}
export CMAKE_PREFIX_PATH=${CUML_ROOT}:\${CMAKE_PREFIX_PATH:-}
CUML_ENV
else
    warn "cuML headers absent at ${CUML_ROOT}/include/cuml"
    warn "Set -DCUML_ROOT=<path> manually when invoking CMake"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. System tuning
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 6: system tuning"
echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo never > /sys/kernel/mm/transparent_hugepage/defrag  2>/dev/null || true
command -v cpupower &>/dev/null && cpupower frequency-set -g performance 2>/dev/null || true
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
    ok "NVIDIA persistence mode ON"
fi
ok "System tuned"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Configure & build
# ─────────────────────────────────────────────────────────────────────────────
log "Phase 7: cmake configure + build"
cd "${REPO_ROOT}/cpp_engine"
mkdir -p build

CUML_ROOT_FLAG=""
[[ -d "/opt/miniforge3/envs/${CONDA_ENV_NAME}" ]] && \
    CUML_ROOT_FLAG="-DCUML_ROOT=/opt/miniforge3/envs/${CONDA_ENV_NAME}"

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_CXX_COMPILER=g++-12 \
    ${CUML_ROOT_FLAG} \
    2>&1 | tee build/cmake_configure.log

cmake --build build -j"$(nproc)" 2>&1 | tee build/cmake_build.log
ok "Binary: ${REPO_ROOT}/cpp_engine/build/bin/holographic_bench"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  PROVISION COMPLETE${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Run the engine:"
echo "    cd ${REPO_ROOT}/cpp_engine && ./build/bin/holographic_bench"
echo ""
echo "  Python research env:"
echo "    source ${VENV}/bin/activate && python run_research.py"
echo ""
echo "  Nsight profile:"
echo "    nsys profile --trace=cuda,nvtx,osrt -o holographic_profile \\"
echo "         ./build/bin/holographic_bench"
echo ""
PROVEOF