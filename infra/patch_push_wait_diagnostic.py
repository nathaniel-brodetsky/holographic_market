#!/usr/bin/env python3
"""
Diagnostic: measures total time the CSV-replay producer spends spin-
waiting on a full ring buffer (the while(!try_push()) loop added by the
no-silent-drop fix), to see how much of the backtest's wall-clock time
is genuine backpressure wait vs. something else (LOB apply cost, signal
routing, main-loop overhead).

Note: calling steady_clock::now() around every single push (~100M+
times on a large file) has its own non-trivial overhead -- this is a
one-off diagnostic measurement, not something to leave permanently in
a hot path this tight.

Run this on the instance, not in the sandbox.
Usage: python3 patch_push_wait_diagnostic.py
"""
import sys
from pathlib import Path

ENGINE = Path.home() / "holographic_market" / "engine"
CSV_REPLAY = ENGINE / "net" / "csv_replay.hpp"
MAIN_BACKTEST = ENGINE / "app" / "main_backtest.cpp"

EDITS = [
    (
        CSV_REPLAY,
        '''#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>''',
        '''#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>'''
    ),
    (
        CSV_REPLAY,
        '''    std::atomic<std::uint64_t> updates_pushed{0U};
    std::atomic<std::uint64_t> updates_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};''',
        '''    std::atomic<std::uint64_t> updates_pushed{0U};
    std::atomic<std::uint64_t> updates_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};
    std::atomic<std::uint64_t> push_wait_ns{0U};  // diagnostic: total time spent
                                                    // spin-waiting on a full ring'''
    ),
    (
        CSV_REPLAY,
        '''            while (!ring_.try_push(bid)) {
                __builtin_ia32_pause();
            }
            metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);

            while (!ring_.try_push(ask)) {
                __builtin_ia32_pause();
            }
            metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);''',
        '''            {
                const auto wait_start = std::chrono::steady_clock::now();
                while (!ring_.try_push(bid)) {
                    __builtin_ia32_pause();
                }
                const auto wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_start).count();
                metrics_.push_wait_ns.fetch_add(static_cast<std::uint64_t>(wait_ns),
                                                 std::memory_order_relaxed);
            }
            metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);

            {
                const auto wait_start = std::chrono::steady_clock::now();
                while (!ring_.try_push(ask)) {
                    __builtin_ia32_pause();
                }
                const auto wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_start).count();
                metrics_.push_wait_ns.fetch_add(static_cast<std::uint64_t>(wait_ns),
                                                 std::memory_order_relaxed);
            }
            metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);'''
    ),
    (
        MAIN_BACKTEST,
        '''    std::printf("  Updates dropped  : %llu\\n", static_cast<unsigned long long>(rm.updates_dropped.load()));''',
        '''    std::printf("  Updates dropped  : %llu\\n", static_cast<unsigned long long>(rm.updates_dropped.load()));
    std::printf("  Push wait (diag) : %.2f s (producer time spent blocked on a full ring)\\n",
                 static_cast<double>(rm.push_wait_ns.load()) / 1e9);'''
    ),
]


def main():
    for path, _, _ in EDITS:
        if not path.exists():
            sys.exit(f"FATAL: {path} not found.")

    changed = {}
    for i, (path, old, new) in enumerate(EDITS, 1):
        src = changed.get(path, path.read_text())
        count = src.count(old)
        if count == 0:
            sys.exit(f"FATAL: edit {i}/{len(EDITS)} on {path.name} -- exact old_str not "
                      f"found. File has drifted -- paste the current relevant section "
                      f"instead of proceeding blindly.")
        if count > 1:
            sys.exit(f"FATAL: edit {i}/{len(EDITS)} on {path.name} -- old_str matched "
                      f"{count} times, expected exactly 1. Aborting with no changes made.")
        changed[path] = src.replace(old, new)
        print(f"edit {i}/{len(EDITS)} applied OK ({path.name})")

    for path, src in changed.items():
        backup = path.with_suffix(path.suffix + ".bak2")
        backup.write_text(path.read_text())
        path.write_text(src)
        print(f"patched {path} (backup: {backup})")


if __name__ == "__main__":
    main()
