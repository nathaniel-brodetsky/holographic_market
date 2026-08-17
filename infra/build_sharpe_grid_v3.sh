#!/usr/bin/env bash
# Build a (day, alpha, threshold) -> backtest-metrics grid ONCE.
#
# Same design as build_sharpe_grid_v2.sh, extended to sweep HOLO_THRESHOLD
# (the minimum |curl| for a signal to be routed at all -- see
# net/signal_router.hpp's active_threshold()) jointly with HOLO_K_ALPHA,
# instead of alpha alone. Every (day, alpha, threshold) combination is run
# EXACTLY ONCE; research/analyze_wfo_2d.py then does all walk-forward
# calibration/statistics on this table in Python, no more binary
# invocations needed.
#
# Runtime warning: this is ALPHA_GRID x THRESHOLD_GRID x n_days backtest
# runs. At ~2.4s/run on an A100, 9 alphas x 7 thresholds x 14 days is
# ~63 x 14 = 882 runs, ~35 minutes. Narrow either grid below if you want
# a faster first pass.
#
# Usage: ./infra/build_sharpe_grid_v3.sh
# Requires: data/wfo_cache_v2/*.csv already populated and the backtest
#           binary built.
set -euo pipefail
cd ~/holographic_market

DATA_DIR="data/wfo_cache_v2"
OUT="data/sharpe_grid_v3.csv"
ALPHA_GRID=(0.0005 0.001 0.002 0.003 0.005 0.008 0.01 0.02 0.03)
THRESHOLD_GRID=(5 10 15 20 25 30 35)
BIN="./engine/build/bin/holographic_backtest"

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not found/executable. Build it first:"
    echo "    cd engine/build && cmake --build . -j\$(nproc)"
    exit 1
fi

echo "date,alpha,threshold,sharpe,signals,win_rate,mean_bps,pnl_bps,updates_dropped" > "$OUT"

run_one() {
    local csv="$1" alpha="$2" threshold="$3"
    local out
    out=$(HOLO_K_ALPHA="$alpha" HOLO_THRESHOLD="$threshold" "$BIN" "$csv" 2>&1) || { echo "NAN NAN NAN NAN NAN NAN"; return; }
    # "No signals passed filter" is a real, valid outcome (high threshold
    # can legitimately zero out a day) -- not a failure. Treat missing
    # metric lines as 0 signals / NAN sharpe rather than a run failure,
    # so a high-threshold sweep doesn't look like it crashed.
    local signals winrate meanbps sharpe pnl dropped
    signals=$(echo "$out" | grep "Signals executed" | grep -oE '[0-9]+')
    winrate=$(echo "$out" | grep "Win Rate"          | grep -oE '[0-9]+\.[0-9]+')
    meanbps=$(echo "$out" | grep "Mean net"           | grep -oE '[0-9.-]+' | tail -1)
    sharpe=$(echo "$out"  | grep "Sharpe"              | grep -oE '[0-9.-]+' | tail -1)
    pnl=$(echo "$out"     | grep "Terminal PnL"        | grep -oE '[0-9.-]+' | head -1)
    dropped=$(echo "$out" | grep "Updates dropped"     | grep -oE '[0-9]+')
    echo "${sharpe:-NAN} ${signals:-0} ${winrate:-NAN} ${meanbps:-NAN} ${pnl:-NAN} ${dropped:-NAN}"
}

shopt -s nullglob
csvs=("$DATA_DIR"/*.csv)
total=$(( ${#csvs[@]} * ${#ALPHA_GRID[@]} * ${#THRESHOLD_GRID[@]} ))
done_count=0

if [ "${#csvs[@]}" -eq 0 ]; then
    echo "FATAL: no CSVs found in $DATA_DIR. Run infra/download_historical_range_v2.sh first."
    exit 1
fi

echo "Sweeping ${#ALPHA_GRID[@]} alphas x ${#THRESHOLD_GRID[@]} thresholds x ${#csvs[@]} days = $total runs"

for csv in "${csvs[@]}"; do
    day=$(basename "$csv" .csv)
    for alpha in "${ALPHA_GRID[@]}"; do
        for threshold in "${THRESHOLD_GRID[@]}"; do
            done_count=$((done_count+1))
            echo "[$done_count/$total] $day alpha=$alpha threshold=$threshold"
            read -r sharpe signals winrate meanbps pnl dropped <<< "$(run_one "$csv" "$alpha" "$threshold")"
            echo "$day,$alpha,$threshold,$sharpe,$signals,$winrate,$meanbps,$pnl,$dropped" >> "$OUT"
        done
    done
done

echo ""
echo "Grid written to $OUT ($total rows, ${#csvs[@]} days x ${#ALPHA_GRID[@]} alphas x ${#THRESHOLD_GRID[@]} thresholds)"
