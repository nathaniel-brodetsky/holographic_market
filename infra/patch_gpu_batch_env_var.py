#!/usr/bin/env python3
"""
Makes k_gpu_batch configurable via HOLO_GPU_BATCH env var (default 2048,
matching the current hardcoded value), instead of a compile-time
constant -- so sweeping batch sizes doesn't need a recompile per value,
matching the existing HOLO_THRESHOLD (signal_router.hpp) / HOLO_K_ALPHA
(main_backtest.cpp) pattern already used in this codebase.

Run this on the instance, not in the sandbox.
Usage: python3 patch_gpu_batch_env_var.py
"""
import sys
from pathlib import Path

TARGET = Path.home() / "holographic_market" / "engine" / "app" / "main_backtest.cpp"

OLD = '''    constexpr std::uint64_t k_gpu_batch = 2048U;'''

NEW = '''    // Configurable via HOLO_GPU_BATCH (default 2048, the original hardcoded
    // value) instead of a compile-time constant -- lets a batch-size sweep
    // run without a recompile per value. Larger batches amortize the GPU
    // pipeline's fixed per-call overhead over more raw rows (faster), at
    // the direct cost of sampling the order-book graph less often (the
    // strategy sees fewer, coarser snapshots of the market) -- this is a
    // real methodology tradeoff, not a free speedup; see the batch-size
    // sweep comparison chart for how much it actually costs in Sharpe/PnL.
    const std::uint64_t k_gpu_batch = [] {
        if (const char* env = std::getenv("HOLO_GPU_BATCH")) {
            const auto v = std::strtoull(env, nullptr, 10);
            if (v > 0ULL) return v;
        }
        return 2048ULL;
    }();'''


def main():
    if not TARGET.exists():
        sys.exit(f"FATAL: {TARGET} not found.")

    src = TARGET.read_text()
    count = src.count(OLD)
    if count == 0:
        sys.exit("FATAL: exact old_str not found. File has drifted from what this patch "
                  "expects -- paste the current relevant section instead of proceeding blindly.")
    if count > 1:
        sys.exit(f"FATAL: old_str matched {count} times, expected exactly 1. "
                  f"Aborting with no changes made.")

    backup = TARGET.with_suffix(TARGET.suffix + ".bak")
    backup.write_text(src)
    TARGET.write_text(src.replace(OLD, NEW))
    print("edit applied OK")
    print(f"Backup of pre-patch file saved to {backup}")
    print(f"Patched: {TARGET}")


if __name__ == "__main__":
    main()
