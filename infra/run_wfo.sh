#!/usr/bin/env bash
# *** DEPRECATED ***
# Replaced by the (day, alpha) grid + Python analysis pipeline:
#   infra/download_historical_range_v2.sh <start> <end>
#   infra/build_sharpe_grid_v2.sh
#   python3 research/analyze_wfo.py --grid data/sharpe_grid_v2.csv --window 10
#
# Why: this script (a) re-invokes the backtest binary once per (day, alpha)
# PER OVERLAPPING WINDOW instead of once each -- fine for 20 days, prohibitive
# for months of history -- and (b) calls the now-hard-deprecated
# infra/download_binance_aggtrades.py, which truncates BTCUSDT/ETHUSDT at
# 500,000 rows/day and meaningfully distorted an earlier WFO analysis (see
# git history). It has no --regime / --exclude-month / bootstrap-significance
# support either. There's no reason to keep using it.
#
# This script now hard-exits unless RUN_WFO_I_UNDERSTAND_DEPRECATED=1 is set,
# so it can't be run by old habit and silently regenerate results from the
# capped downloader.
if [ "${RUN_WFO_I_UNDERSTAND_DEPRECATED:-0}" != "1" ]; then
    echo "FATAL: infra/run_wfo.sh is deprecated. Use instead:" >&2
    echo "    infra/download_historical_range_v2.sh <start> <end>" >&2
    echo "    infra/build_sharpe_grid_v2.sh" >&2
    echo "    python3 research/analyze_wfo.py --grid data/sharpe_grid_v2.csv --window 10" >&2
    echo "If you specifically need this legacy script (e.g. a regression" >&2
    echo "comparison), set RUN_WFO_I_UNDERSTAND_DEPRECATED=1." >&2
    exit 1
fi

# Walk-forward optimization over the last 20 days.
# Days 1-10 of the window: calibration pool only (never scored as OOS).
# Days 11-20: for each, pick k_alpha by best Sharpe over the PRECEDING 10 days,
# then run that day out-of-sample with the chosen k_alpha. Nothing from day t
# or later is ever used to choose day t's parameter.
set -euo pipefail
cd ~/holographic_market

N_DAYS=20
WINDOW=10
END="${1:-$(date -I)}"                       # override: ./run_wfo.sh 2026-07-04
START=$(date -I -d "$END - $((N_DAYS-1)) days")
ALPHA_GRID=(0.0005 0.001 0.002 0.003 0.005 0.008 0.01 0.02 0.03)

DATA_DIR="data/wfo_cache"
RESULTS_FILE="wfo_results.csv"
mkdir -p "$DATA_DIR"
echo "date,best_alpha,is_sharpe,signals,win_rate,mean_bps,sharpe,pnl_bps,updates_dropped" > "$RESULTS_FILE"

# ---- build list of the N_DAYS dates ----
days=()
d="$START"
while [ "$(date -d "$d" +%s)" -le "$(date -d "$END" +%s)" ]; do
    days+=("$d")
    d=$(date -I -d "$d + 1 day")
done

# ---- ensure each day's CSV is downloaded once and cached ----
for d in "${days[@]}"; do
    csv="$DATA_DIR/$d.csv"
    if [ ! -f "$csv" ]; then
        echo "=== Downloading $d (aggTrades OFI-proxy) ==="
        rm -f data/test_data.csv
        if python3 infra/download_binance_aggtrades.py --date "$d"; then
            cp data/test_data.csv "$csv"
        else
            echo "  -> no archive for $d yet, will be skipped"
        fi
    fi
done

# ---- sanity check: make sure the binary actually honors HOLO_K_ALPHA ----
echo "=== Verifying HOLO_K_ALPHA is wired into the binary ==="
probe_csv="$DATA_DIR/${days[0]}.csv"
v1=$(HOLO_K_ALPHA=0.005 ./engine/build/bin/holographic_backtest "$probe_csv" 2>&1 | grep -oE "HOLO_K_ALPHA effective = [0-9.]+")
v2=$(HOLO_K_ALPHA=0.120 ./engine/build/bin/holographic_backtest "$probe_csv" 2>&1 | grep -oE "HOLO_K_ALPHA effective = [0-9.]+")
echo "  with HOLO_K_ALPHA=0.005 -> $v1"
echo "  with HOLO_K_ALPHA=0.120 -> $v2"
if [ "$v1" = "$v2" ] || [ -z "$v1" ]; then
    echo "FATAL: binary does not reflect HOLO_K_ALPHA changes. Rebuild engine/build before running WFO:"
    echo "    cd engine/build && cmake --build . -j\$(nproc)"
    exit 1
fi

# run backtest for one day+alpha, echo "sharpe signals win_rate mean_bps pnl dropped"
run_one() {
    local csv="$1" alpha="$2"
    if [ ! -f "$csv" ]; then echo "NAN NAN NAN NAN NAN NAN"; return; fi
    local out
    out=$(HOLO_K_ALPHA="$alpha" ./engine/build/bin/holographic_backtest "$csv" 2>&1) || { echo "NAN NAN NAN NAN NAN NAN"; return; }
    local signals winrate meanbps sharpe pnl dropped
    signals=$(echo "$out" | grep "Signals executed" | grep -oE '[0-9]+')
    winrate=$(echo "$out" | grep "Win Rate"          | grep -oE '[0-9]+\.[0-9]+')
    meanbps=$(echo "$out" | grep "Mean net"           | grep -oE '[0-9.-]+' | tail -1)
    sharpe=$(echo "$out"  | grep "Sharpe"              | grep -oE '[0-9.-]+' | tail -1)
    pnl=$(echo "$out"     | grep "Terminal PnL"        | grep -oE '[0-9.-]+' | head -1)
    dropped=$(echo "$out" | grep "Updates dropped"     | grep -oE '[0-9]+')
    echo "${sharpe:-NAN} ${signals:-NAN} ${winrate:-NAN} ${meanbps:-NAN} ${pnl:-NAN} ${dropped:-NAN}"
}

# ---- walk forward ----
for ((i=WINDOW; i<N_DAYS; i++)); do
    test_date="${days[$i]}"
    test_csv="$DATA_DIR/$test_date.csv"

    if [ ! -f "$test_csv" ]; then
        echo "=== Skipping $test_date: no data (archive not published yet) ==="
        echo "$test_date,SKIP,,,,,,,no_data" >> "$RESULTS_FILE"
        continue
    fi

    echo "=== Calibrating for $test_date on preceding $WINDOW days ==="
    best_alpha=""
    best_is_sharpe="-999"
    for alpha in "${ALPHA_GRID[@]}"; do
        sum=0; n=0
        for ((j=i-WINDOW; j<i; j++)); do
            train_csv="$DATA_DIR/${days[$j]}.csv"
            read -r sh _ <<< "$(run_one "$train_csv" "$alpha")"
            [ "$sh" = "NAN" ] && continue
            sum=$(echo "$sum + $sh" | bc -l)
            n=$((n+1))
        done
        [ "$n" -eq 0 ] && continue
        avg=$(echo "$sum / $n" | bc -l)
        echo "    alpha=$alpha  in-sample avg Sharpe=$avg"
        if (( $(echo "$avg > $best_is_sharpe" | bc -l) )); then
            best_is_sharpe="$avg"
            best_alpha="$alpha"
        fi
    done

    echo "=== OOS test $test_date with k_alpha=$best_alpha (in-sample Sharpe=$best_is_sharpe) ==="
    read -r sharpe signals winrate meanbps pnl dropped <<< "$(run_one "$test_csv" "$best_alpha")"
    echo "$test_date,$best_alpha,$best_is_sharpe,$signals,$winrate,$meanbps,$sharpe,$pnl,$dropped" >> "$RESULTS_FILE"
done

echo ""
echo "=== WFO OUT-OF-SAMPLE RESULTS ==="
column -t -s',' "$RESULTS_FILE"