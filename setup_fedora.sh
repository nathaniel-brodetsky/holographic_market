#!/usr/bin/env bash
set -euo pipefail

echo "[HOLOGRAPHIC-SETUP] Phase 0: System Update & Explicit Base Toolchain"
dnf install -y gcc gcc-c++ make cmake ninja-build git wget htop openblas-devel python3.13 python3.13-devel

echo "[HOLOGRAPHIC-SETUP] Phase 1: Python Virtual Env"
python3.13 -m venv .venv_holographic --clear

echo "[HOLOGRAPHIC-SETUP] Phase 2: Installing core dependencies"
.venv_holographic/bin/pip install --upgrade pip
.venv_holographic/bin/pip install -r requirements.txt

echo "[HOLOGRAPHIC-SETUP] Setup complete!"