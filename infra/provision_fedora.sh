#!/usr/bin/env bash
# infra/provision_fedora.sh
# Idempotent bootstrap — Fedora 39/40, x86-64, NVIDIA GPU.
# sudo bash infra/provision_fedora.sh
set -euo pipefail
IFS=$'\n\t'

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[PROVISION]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[FATAL]${NC} $*"; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo bash infra/provision_fedora.sh"
command -v dnf &>/dev/null || die "dnf not found — Fedora only"

FEDORA_VERSION=$(rpm -E %fedora)
CUDA_VERSION="12-5"
RAPIDS_VERSION="24.10"
PYTHON_VERSION="3.11"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log "Fedora ${FEDORA_VERSION} · CUDA ${CUDA_VERSION} · RAPIDS ${RAPIDS_VERSION} · repo ${REPO_ROOT}"

# ─── 0. system update ────────────────────────────────────────────────────────
log "Phase 0: system update"
dnf upgrade -y -q
ok "system updated"

# ─── 1. toolchain ────────────────────────────────────────────────────────────
log "Phase 1: toolchain"
dnf install -y -q \
    gcc gcc-c++ \
    clang clang-tools-extra \
    cmake ninja-build \
    git git-lfs \
    wget curl \
    make autoconf automake libtool \
    pkgconf-pkg-config \
    gdb valgrind \
    numactl numactl-devel \
    hwloc hwloc-devel \
    pciutils ethtool iproute \
    libatomic \
    perf \
    cpufrequtils \
    kernel-tools

ok "toolchain: $(gcc --version | head -1)"
ok "cmake:     $(cmake --version | head -1)"

# ─── 2. CUDA Toolkit ─────────────────────────────────────────────────────────
log "Phase 2: CUDA Toolkit ${CUDA_VERSION}"
if ! rpm -q "cuda-toolkit-${CUDA_VERSION}" &>/dev/null; then
    # NVIDIA CUDA repo for Fedora
    REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo"
    # Fedora 40 still uses fedora39 repo
    dnf config-manager --add-repo "${REPO_URL}" 2>/dev/null || \
        wget -qO /etc/yum.repos.d/cuda-fedora.repo "${REPO_URL}"

    dnf module disable -y nvidia-driver 2>/dev/null || true
    dnf install -y -q \
        "cuda-toolkit-${CUDA_VERSION}" \
        "libcublas-devel-${CUDA_VERSION}" \
        "libcusparse-devel-${CUDA_VERSION}" \
        "libcusolver-devel-${CUDA_VERSION}" \
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

# Boost (Fedora ships 1.83+ in F39/40)
dnf install -y -q \
    boost-devel \
    boost-system \
    boost-program_options \
    openssl-devel \
    zlib-devel

# simdjson — Fedora 39/40 ships simdjson in repos
if dnf info simdjson-devel &>/dev/null 2>&1; then
    dnf install -y -q simdjson-devel
else
    # Build from source if not in repos
    log "simdjson: building from source"
    TMP_DIR=$(mktemp -d)
    git clone --depth 1 --branch v3.9.3 \
        https://github.com/simdjson/simdjson.git "${TMP_DIR}/simdjson"
    cmake -S "${TMP_DIR}/simdjson" -B "${TMP_DIR}/simdjson/build" \
        -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build "${TMP_DIR}/simdjson/build"
    cmake --install "${TMP_DIR}/simdjson/build"
    rm -rf "${TMP_DIR}"
fi

# fast_float (header-only, vendored)
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

# Fedora may ship python3.11 or python3.13 depending on release
PYTHON_BIN=""
for v in python3.11 python3.12 python3.13 python3; do
    if command -v "${v}" &>/dev/null; then
        PYTHON_BIN="${v}"
        break
    fi
done
[[ -n "${PYTHON_BIN}" ]] || die "No python3 found"

dnf install -y -q \
    python3-devel \
    python3-pip \
    python3-virtualenv \
    2>/dev/null || true

# Install exact version if needed
if ! command -v "python${PYTHON_VERSION}" &>/dev/null; then
    warn "python${PYTHON_VERSION} not found, using ${PYTHON_BIN}"
    PYTHON_VERSION_CMD="${PYTHON_BIN}"
else
    PYTHON_VERSION_CMD="python${PYTHON_VERSION}"
fi

VENV="${REPO_ROOT}/.venv_holographic"
[[ -d "${VENV}" ]] || "${PYTHON_VERSION_CMD}" -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip wheel setuptools -q
"${VENV}/bin/pip" install -r "${REPO_ROOT}/requirements.txt" -q
ok "venv: ${VENV} ($(${VENV}/bin/python --version))"

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
        "cuml=${RAPIDS_VERSION}" "cuda-version=12.2" "python=3.11" \
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

# ─── 6. NVIDIA driver check ──────────────────────────────────────────────────
log "Phase 6: NVIDIA driver"
if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found — installing NVIDIA driver"
    # Fedora: use RPM Fusion
    dnf install -y -q \
        "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-${FEDORA_VERSION}.noarch.rpm" \
        "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-${FEDORA_VERSION}.noarch.rpm" \
        2>/dev/null || true
    dnf install -y -q akmod-nvidia xorg-x11-drv-nvidia-cuda || \
        warn "akmod-nvidia install failed — install driver manually"
fi
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -pm 1 2>/dev/null || true
    nvidia-smi --auto-boost-default=0 2>/dev/null || true
    ok "NVIDIA: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
fi

# ─── 7. system tuning ────────────────────────────────────────────────────────
log "Phase 7: system tuning"
echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo never > /sys/kernel/mm/transparent_hugepage/defrag  2>/dev/null || true
# cpupower ships in kernel-tools on Fedora
command -v cpupower &>/dev/null && cpupower frequency-set -g performance 2>/dev/null || true
sysctl -w net.core.busy_poll=50              2>/dev/null || true
sysctl -w net.core.busy_read=50              2>/dev/null || true
sysctl -w net.ipv4.tcp_low_latency=1         2>/dev/null || true
sysctl -w net.core.rmem_max=134217728        2>/dev/null || true
sysctl -w net.core.wmem_max=134217728        2>/dev/null || true
sysctl -w net.core.netdev_max_backlog=250000 2>/dev/null || true
sysctl -w net.ipv4.tcp_fastopen=3            2>/dev/null || true

# Persist sysctl across reboots
cat > /etc/sysctl.d/99-holographic.conf << 'SYSCTL'
net.core.busy_poll              = 50
net.core.busy_read              = 50
net.ipv4.tcp_low_latency        = 1
net.core.rmem_max               = 134217728
net.core.wmem_max               = 134217728
net.core.netdev_max_backlog     = 250000
net.ipv4.tcp_fastopen           = 3
SYSCTL
ok "sysctl persisted to /etc/sysctl.d/99-holographic.conf"
ok "system tuned"

# ─── 8. SELinux note ─────────────────────────────────────────────────────────
SELINUX_STATUS=$(getenforce 2>/dev/null || echo "unknown")
if [[ "${SELINUX_STATUS}" == "Enforcing" ]]; then
    warn "SELinux is Enforcing — cudaHostRegister may fail on some kernels"
    warn "If you see 'permission denied' errors: setenforce 0  (or add policy)"
fi

# ─── 9. build ────────────────────────────────────────────────────────────────
log "Phase 8: cmake configure + build"
cd "${REPO_ROOT}/cpp_engine"
mkdir -p build

CUML_FLAG=""
[[ -d "/opt/miniforge3/envs/${ENV_NAME}" ]] && \
    CUML_FLAG="-DCUML_ROOT=/opt/miniforge3/envs/${ENV_NAME}"

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    ${CUML_FLAG} \
    2>&1 | tee build/cmake_configure.log

cmake --build build -j"$(nproc)" 2>&1 | tee build/cmake_build.log
ok "binary: ${REPO_ROOT}/cpp_engine/build/bin/holographic_bench"

# ─── summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  FEDORA PROVISION COMPLETE${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Run:     cd ${REPO_ROOT}/cpp_engine && ./build/bin/holographic_bench"
echo "  Python:  source ${VENV}/bin/activate && python run_research.py"
echo "  Profile: nsys profile --trace=cuda,nvtx,osrt -o holo ./build/bin/holographic_bench"
echo ""
if [[ "${SELINUX_STATUS}" == "Enforcing" ]]; then
    echo -e "${YELLOW}  NOTE: SELinux Enforcing — run 'setenforce 0' if CUDA errors appear${NC}"
    echo ""
fi
