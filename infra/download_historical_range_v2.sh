#!/usr/bin/env bash
# Bulk-download the aggTrades-proxy cache for a date range, with retry/backoff.
#
# This does NOT change how a single day is fetched (still shells out to the
# existing infra/download_binance_aggtrades_v2.py) — it just wraps that in a
# loop over many days so you can build up months of wfo_cache/*.csv instead
# of the last ~10-20 days run_wfo.sh grabs.
#
# Usage:
#   ./infra/download_historical_range.sh 2026-03-01 2026-07-03 [max_retries]
#
# Output:
#   data/wfo_cache/<date>.csv         one file per successfully fetched day
#   data/download_log.csv             date,status,attempts for every day tried
#
# NOTE: retries help with transient network/rate-limit failures. They will
# NOT rescue a date for which Binance genuinely never published an archive
# (delisted symbol, very recent date not yet published, etc). Check
# data/download_log.csv afterwards and eyeball the "failed" rows by hand —
# don't assume "failed" always means "just retry harder".
set -euo pipefail
cd ~/holographic_market

START="${1:?usage: $0 START_DATE END_DATE [max_retries]}"
END="${2:?usage: $0 START_DATE END_DATE [max_retries]}"
MAX_RETRIES="${3:-4}"

DATA_DIR="data/wfo_cache_v2"
LOG_FILE="data/download_log.csv"
mkdir -p "$DATA_DIR"
echo "date,status,attempts" > "$LOG_FILE"

d="$START"
n_ok=0
n_cached=0
n_failed=0

while [ "$(date -d "$d" +%s)" -le "$(date -d "$END" +%s)" ]; do
    csv="$DATA_DIR/$d.csv"

    if [ -f "$csv" ]; then
        echo "[$d] already cached, skipping"
        echo "$d,cached,0" >> "$LOG_FILE"
        n_cached=$((n_cached+1))
        d=$(date -I -d "$d + 1 day")
        continue
    fi

    attempt=1
    ok=0
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        echo "=== [$d] attempt $attempt/$MAX_RETRIES ==="
        rm -f data/test_data.csv
        if python3 infra/download_binance_aggtrades_v2.py --date "$d"; then
            cp data/test_data.csv "$csv"
            ok=1
            break
        fi
        sleep_s=$(( attempt * attempt * 5 ))   # 5s, 20s, 45s, 80s, ...
        echo "  -> failed, backing off ${sleep_s}s before retry"
        sleep "$sleep_s"
        attempt=$((attempt+1))
    done

    if [ "$ok" -eq 1 ]; then
        echo "$d,ok,$attempt" >> "$LOG_FILE"
        n_ok=$((n_ok+1))
    else
        echo "  -> giving up on $d after $MAX_RETRIES attempts"
        echo "$d,failed,$MAX_RETRIES" >> "$LOG_FILE"
        n_failed=$((n_failed+1))
    fi

    d=$(date -I -d "$d + 1 day")
done

echo ""
echo "=== Download pass complete ==="
echo "  newly downloaded : $n_ok"
echo "  already cached   : $n_cached"
echo "  failed           : $n_failed"
echo "  see $LOG_FILE for per-day detail"
