#!/usr/bin/env bash
# Quick threshold sweep with a real smoothness metric (R^2 of a straight-
# line fit to cumulative PnL vs trade index) instead of eyeballing PDFs.
# R^2 close to 1 = equity curve tracks a straight line closely (steady
# growth). R^2 close to 0 = choppy/non-monotonic.
#
# Honest caveat this script deliberately keeps printing: this is still
# ONE day's data. A high R^2 here is a description of what this one day
# looked like, not a prediction that the same threshold will look smooth
# on a different day -- same caveat as everything else this week.
#
# Usage: ./quick_smoothness_sweep.sh [csv_path] [threshold1 threshold2 ...]
# Default thresholds: 5 8 10 12 15 20 25
set -euo pipefail
cd ~/holographic_market

CSV="${1:-data/test_data.csv}"
shift || true
THRESHOLDS=("${@:-5 8 10 12 15 20 25}")
if [ "${#THRESHOLDS[@]}" -eq 1 ] && [[ "${THRESHOLDS[0]}" == *" "* ]]; then
    read -ra THRESHOLDS <<< "${THRESHOLDS[0]}"
fi
BIN="./engine/build/bin/holographic_backtest"

if [ ! -x "$BIN" ]; then
    echo "FATAL: $BIN not found/executable." >&2
    exit 1
fi

echo "threshold,signals,sharpe,pnl_bps,smoothness_r2"

for t in "${THRESHOLDS[@]}"; do
    out=$(HOLO_THRESHOLD="$t" "$BIN" "$CSV" --out-dir data 2>&1) || true
    signals=$(echo "$out" | grep "Signals executed" | grep -oE '[0-9]+' || echo "")
    sharpe=$(echo "$out"  | grep "Sharpe"           | grep -oE '[0-9.-]+' | tail -1 || echo "")
    pnl=$(echo "$out"     | grep "Terminal PnL"     | grep -oE '[0-9.-]+' | head -1 || echo "")

    r2=$(python3 - <<'PYEOF'
import sys
import pandas as pd
import numpy as np
try:
    df = pd.read_csv("data/advanced_metrics.csv")
    if len(df) < 3:
        print("NA")
    else:
        x = np.arange(len(df))
        y = df["cumulative_pnl"].to_numpy()
        # R^2 of a straight-line fit: 1 - SS_res/SS_tot
        coeffs = np.polyfit(x, y, 1)
        y_fit = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"{r2:.4f}")
except Exception as e:
    print("NA")
PYEOF
)

    echo "$t,${signals:-0},${sharpe:-NA},${pnl:-NA},${r2}"
done
