#!/usr/bin/env bash
# Sweeps HOLO_GPU_BATCH across several values on the same CSV, saving each
# run's advanced_metrics.csv separately (data/batch_sweep/batch_<N>.csv)
# so the comparison chart script can plot every equity curve.
#
# Usage: ./batch_sweep.sh <csv_path> [batch1 batch2 ...]
# Default batches: 1024 2048 4096 8192 16384
set -euo pipefail
cd ~/holographic_market

CSV="${1:?usage: batch_sweep.sh <csv_path> [batch1 batch2 ...]}"
shift || true
BATCHES=("${@:-1024 2048 4096 8192 16384}")
if [ "${#BATCHES[@]}" -eq 1 ] && [[ "${BATCHES[0]}" == *" "* ]]; then
    read -ra BATCHES <<< "${BATCHES[0]}"
fi

BIN="./engine/build/bin/holographic_backtest"
OUT_DIR="data/batch_sweep"
mkdir -p "$OUT_DIR"

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not found/executable." >&2
    exit 1
fi

echo "batch,elapsed_s,push_wait_s,signals,sharpe,pnl_bps,win_rate" > "$OUT_DIR/summary.csv"

for b in "${BATCHES[@]}"; do
    echo "=== HOLO_GPU_BATCH=$b ==="
    out=$(HOLO_GPU_BATCH="$b" HOLO_K_ALPHA=0.008 "$BIN" "$CSV" --out-dir "$OUT_DIR" 2>&1)
    echo "$out" | grep -E "Elapsed|Signals executed|Sharpe|Terminal PnL|Push wait|Win Rate"

    elapsed=$(echo "$out"    | grep "Elapsed"          | grep -oE '[0-9]+\.[0-9]+' | head -1)
    push_wait=$(echo "$out"  | grep "Push wait"         | grep -oE '[0-9]+\.[0-9]+' | head -1)
    signals=$(echo "$out"    | grep "Signals executed"  | grep -oE '[0-9]+')
    sharpe=$(echo "$out"     | grep "Sharpe"             | grep -oE '[0-9.-]+' | tail -1)
    pnl=$(echo "$out"        | grep "Terminal PnL"       | grep -oE '[0-9.-]+' | head -1)
    winrate=$(echo "$out"    | grep "Win Rate"           | grep -oE '[0-9]+\.[0-9]+')

    echo "$b,${elapsed:-NA},${push_wait:-NA},${signals:-0},${sharpe:-NA},${pnl:-NA},${winrate:-NA}" >> "$OUT_DIR/summary.csv"

    # advanced_metrics.csv gets overwritten each run -- move it to a
    # batch-specific name before the next iteration clobbers it.
    if [ -f "$OUT_DIR/advanced_metrics.csv" ]; then
        mv "$OUT_DIR/advanced_metrics.csv" "$OUT_DIR/batch_${b}.csv"
    fi
done

echo ""
echo "Sweep complete. Summary: $OUT_DIR/summary.csv"
echo "Per-batch equity data: $OUT_DIR/batch_<N>.csv"
column -s, -t "$OUT_DIR/summary.csv"
