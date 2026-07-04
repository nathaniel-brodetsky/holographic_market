#!/usr/bin/env bash
# Build a (day, alpha) -> backtest-metrics grid ONCE.
#
# run_wfo.sh re-runs the C++ backtest for the same (day, alpha) pair over and
# over, once for every sliding window it happens to fall into. That's fine
# for ~20 days but blows up once you're validating on months of history.
#
# This script instead runs every cached day against every alpha in the grid
# EXACTLY ONCE, and writes the results to data/sharpe_grid.csv. All downstream
# walk-forward analysis (research/analyze_wfo.py) then just indexes into this
# table in Python — no more binary invocations needed.
#
# Usage: ./infra/build_sharpe_grid.sh
# Requires: data/wfo_cache/*.csv already populated (see
#           infra/download_historical_range.sh) and the backtest binary built.
set -euo pipefail
cd ~/holographic_market

DATA_DIR="data/wfo_cache"
OUT="data/sharpe_grid.csv"
ALPHA_GRID=(0.0005 0.001 0.002 0.003 0.005 0.008 0.01 0.02 0.03)
BIN="./engine/build/bin/holographic_backtest"

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not found/executable. Build it first:"
    echo "    cd engine/build && cmake --build . -j\$(nproc)"
    exit 1
fi

echo "date,alpha,sharpe,signals,win_rate,mean_bps,pnl_bps,updates_dropped" > "$OUT"

run_one() {
    local csv="$1" alpha="$2"
    local out
    out=$(HOLO_K_ALPHA="$alpha" "$BIN" "$csv" 2>&1) || { echo "NAN NAN NAN NAN NAN NAN"; return; }
    local signals winrate meanbps sharpe pnl dropped
    signals=$(echo "$out" | grep "Signals executed" | grep -oE '[0-9]+')
    winrate=$(echo "$out" | grep "Win Rate"          | grep -oE '[0-9]+\.[0-9]+')
    meanbps=$(echo "$out" | grep "Mean net"           | grep -oE '[0-9.-]+' | tail -1)
    sharpe=$(echo "$out"  | grep "Sharpe"              | grep -oE '[0-9.-]+' | tail -1)
    pnl=$(echo "$out"     | grep "Terminal PnL"        | grep -oE '[0-9.-]+' | head -1)
    dropped=$(echo "$out" | grep "Updates dropped"     | grep -oE '[0-9]+')
    echo "${sharpe:-NAN} ${signals:-NAN} ${winrate:-NAN} ${meanbps:-NAN} ${pnl:-NAN} ${dropped:-NAN}"
}

shopt -s nullglob
csvs=("$DATA_DIR"/*.csv)
total=$(( ${#csvs[@]} * ${#ALPHA_GRID[@]} ))
done_count=0

if [ "${#csvs[@]}" -eq 0 ]; then
    echo "FATAL: no CSVs found in $DATA_DIR. Run infra/download_historical_range.sh first."
    exit 1
fi

for csv in "${csvs[@]}"; do
    day=$(basename "$csv" .csv)
    for alpha in "${ALPHA_GRID[@]}"; do
        done_count=$((done_count+1))
        echo "[$done_count/$total] $day alpha=$alpha"
        read -r sharpe signals winrate meanbps pnl dropped <<< "$(run_one "$csv" "$alpha")"
        echo "$day,$alpha,$sharpe,$signals,$winrate,$meanbps,$pnl,$dropped" >> "$OUT"
    done
done

echo ""
echo "Grid written to $OUT ($total rows, ${#csvs[@]} days x ${#ALPHA_GRID[@]} alphas)"
