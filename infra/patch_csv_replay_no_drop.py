#!/usr/bin/env python3
"""
Fixes csv_replay.hpp: the producer thread silently dropped a LobUpdate
whenever the ring buffer was full (try_push() failing -> immediately
counted as dropped and moved on), instead of waiting for the consumer to
catch up. For an offline backtest this is the wrong tradeoff --
correctness (processing every row of market data) matters far more than
raw wall-clock speed, and losing 65% of the data (846.6M of 1.298B
updates on the full 5-month dataset) silently produces both a wrong
backtest result AND a throughput number that's 3x inflated (measuring
how fast 35% of the data was processed, not the real dataset).

This never showed up on small single-day test files (rows read in the
1-3M range) because the ring (8,388,608 slots) never filled at that
scale -- the discrepancy only appears at sustained, large-scale
throughput, exactly the scenario in the William email.

Fix: spin-wait (matching the busy-wait idiom already used elsewhere in
this codebase, e.g. CudaPipeline::run_continuous's poll loop) until
try_push() succeeds, instead of dropping on the first failure. Trades
some wall-clock time for a complete, honest run -- updates_dropped
should read 0 after this fix.

Run this on the instance, not in the sandbox.
Usage: python3 patch_csv_replay_no_drop.py
"""
import sys
from pathlib import Path

TARGET = Path.home() / "holographic_market" / "engine" / "net" / "csv_replay.hpp"

OLD = '''            if (ring_.try_push(bid)) [[likely]]
                metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);
            else
                metrics_.updates_dropped.fetch_add(1U, std::memory_order_relaxed);

            if (ring_.try_push(ask)) [[likely]]
                metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);
            else
                metrics_.updates_dropped.fetch_add(1U, std::memory_order_relaxed);'''

NEW = '''            // BUG FIX: try_push() failing used to be treated as "drop this
            // update and move on" -- for an OFFLINE backtest that's the
            // wrong tradeoff. Losing data silently both corrupts the
            // backtest result (missing fills/quotes) and inflates the
            // reported throughput number (it measures how fast a
            // fraction of the data was processed, not the real
            // dataset) -- this is what caused 846.6M of 1.298B updates
            // (65%) to be silently dropped on the full 5-month run,
            // invisible on small single-day files where the ring never
            // filled. Spin-wait for the consumer instead of dropping;
            // updates_dropped should read 0 from now on for any file
            // this actually finishes reading.
            while (!ring_.try_push(bid)) {
                __builtin_ia32_pause();
            }
            metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);

            while (!ring_.try_push(ask)) {
                __builtin_ia32_pause();
            }
            metrics_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);'''


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
