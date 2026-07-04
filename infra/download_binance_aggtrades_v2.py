#!/usr/bin/env python3
"""
Uncapped, chunked, checksum-verified replacement for
infra/download_binance_aggtrades.py.

Same CLI and same output (data/test_data.csv, same column order, no header)
so it's a drop-in replacement everywhere the old script was called --
including infra/run_wfo.sh and infra/download_historical_range.sh, which
don't need to change at all.

What's different, and why:

1. NO ROW CAP.
   The old script had `nrows=500_000` on the read. Binance's daily archive
   is time-ordered, so on any day a symbol crosses that count (BTCUSDT and
   ETHUSDT do on ~90% of days, per the regime check we ran), the file was
   silently truncated to roughly the FIRST part of the day. That's a
   systematic, instrument- and volume-dependent sampling bias, not a random
   one, and it was quietly present in every backtest run so far. This
   version reads the whole archive.

2. CHUNKED PARSING + VECTORIZED QUOTE SYNTHESIS.
   Reading a whole day unbounded means some days can be several times
   larger than before. Instead of loading the entire decompressed CSV into
   one DataFrame and building the synthetic bid/ask stream with a Python
   for-loop (which is what made the 500k cap tempting in the first place),
   this script:
     - parses the CSV in fixed-size chunks (pandas `chunksize`), and
     - synthesizes the running bid/ask quote stream with vectorized
       pandas/numpy ops (forward-fill) instead of a per-row loop.
   The last known bid/ask state is carried across chunk boundaries
   ("stitching") so the result is IDENTICAL to processing the whole day in
   one shot, just bounded in peak memory.

3. CHECKSUM VERIFICATION.
   Binance publishes a `.CHECKSUM` (sha256) file alongside every daily
   archive. This script verifies it before trusting the contents, and
   skips/retries on mismatch instead of silently caching corrupted data.

4. RETRIES WITH BACKOFF + RATE-LIMIT AWARENESS.
   Transient HTTP failures (timeouts, 5xx, 429) are retried with
   exponential backoff. A 429 response's `Retry-After` header, if present,
   is honored. A real 404 (archive not published, e.g. today's date) is
   NOT retried -- that's a permanent "no data" for that date, not a
   transient failure.

5. ATOMIC OUTPUT.
   The final CSV is written to a temp file in the same directory and then
   atomically renamed into place. A crashed/killed run can never leave a
   half-written data/test_data.csv that a caching wrapper mistakes for a
   complete, valid day.

Usage (identical to the old script):
    python3 infra/download_binance_aggtrades_v2.py --date 2026-06-15

Extra, optional flags:
    --chunksize 200000       rows per parse chunk (default 200000)
    --max-retries 5          per-symbol HTTP retry budget (default 5)
    --no-checksum            skip CHECKSUM verification (not recommended)
    --out PATH               output path (default data/test_data.csv)
"""
import argparse
import hashlib
import io
import logging
import os
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
INSTRUMENT_MAP = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2, "BNBUSDT": 3}
# futures um aggTrades daily csv columns (no header row in the raw archive):
RAW_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
            "transact_time", "is_buyer_maker"]
OUT_COLS = ["timestamp_ns", "instrument_id", "bid_price", "bid_qty", "ask_price", "ask_qty"]


@dataclass
class QuoteState:
    """Carries the last known synthetic bid/ask across chunk boundaries."""
    bid_price: float = None
    bid_qty: float = 0.0
    ask_price: float = None
    ask_qty: float = 0.0
    seeded: bool = False


def _get_with_retry(url: str, max_retries: int, timeout: int = 30):
    """GET a URL with exponential backoff. Returns the Response, or None for
    a genuine 404 (not retryable -- means 'no archive published'). Raises
    on exhausting retries for transient failures."""
    backoff = 3.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        except requests.RequestException as e:
            last_exc = e
            logging.warning(f"  request error ({e}); retry {attempt}/{max_retries} in {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 200:
            return resp
        if resp.status_code == 404:
            return None  # permanent: archive doesn't exist for this date/symbol
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", backoff))
            logging.warning(f"  rate-limited (429); waiting {retry_after:.0f}s "
                             f"(attempt {attempt}/{max_retries})")
            time.sleep(retry_after)
            backoff *= 2
            continue
        # other 4xx/5xx: retry a few times, they're often transient on Binance's CDN
        logging.warning(f"  HTTP {resp.status_code}; retry {attempt}/{max_retries} in {backoff:.0f}s")
        time.sleep(backoff)
        backoff *= 2

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Exhausted {max_retries} retries fetching {url}")


def _verify_checksum(zip_bytes: bytes, zip_url: str) -> bool:
    """Fetch Binance's companion .CHECKSUM file and verify sha256. Returns
    True if verified OK, True (with a warning) if no checksum file is
    available, and False on an actual mismatch."""
    checksum_url = zip_url + ".CHECKSUM"
    try:
        resp = requests.get(checksum_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    except requests.RequestException as e:
        logging.warning(f"  could not fetch checksum ({e}); proceeding unverified")
        return True
    if resp.status_code != 200:
        logging.warning(f"  no checksum available (HTTP {resp.status_code}); proceeding unverified")
        return True

    expected = resp.text.strip().split()[0].lower()
    actual = hashlib.sha256(zip_bytes).hexdigest().lower()
    if actual != expected:
        logging.error(f"  CHECKSUM MISMATCH: expected {expected}, got {actual}")
        return False
    return True


def _synthesize_chunk(chunk: pd.DataFrame, state: QuoteState) -> pd.DataFrame:
    """Vectorized version of the old per-row loop:
        if is_buyer_maker: bid_price, bid_qty = price, qty   (taker sold -> hit the bid)
        else:               ask_price, ask_qty = price, qty  (taker bought -> lifted the ask)
    carried across chunks via `state`.
    """
    chunk = chunk.sort_values("transact_time")
    is_sell = chunk["is_buyer_maker"].to_numpy()

    price = chunk["price"].to_numpy()
    qty = chunk["quantity"].to_numpy()

    bid_upd_price = np.where(is_sell, price, np.nan)
    bid_upd_qty = np.where(is_sell, qty, np.nan)
    ask_upd_price = np.where(~is_sell, price, np.nan)
    ask_upd_qty = np.where(~is_sell, qty, np.nan)

    if not state.seeded:
        # Seed exactly like the original script: both sides start at the
        # very first trade price in the day, with zero synthetic quantity.
        seed_price = float(price[0])
        state.bid_price, state.ask_price = seed_price, seed_price
        state.bid_qty = state.ask_qty = 0.0
        state.seeded = True

    bid_price_s = pd.Series(bid_upd_price)
    bid_price_s.iloc[0] = bid_price_s.iloc[0] if not np.isnan(bid_price_s.iloc[0]) else state.bid_price
    bid_price_filled = bid_price_s.ffill().to_numpy()

    bid_qty_s = pd.Series(bid_upd_qty)
    bid_qty_s.iloc[0] = bid_qty_s.iloc[0] if not np.isnan(bid_qty_s.iloc[0]) else state.bid_qty
    bid_qty_filled = bid_qty_s.ffill().to_numpy()

    ask_price_s = pd.Series(ask_upd_price)
    ask_price_s.iloc[0] = ask_price_s.iloc[0] if not np.isnan(ask_price_s.iloc[0]) else state.ask_price
    ask_price_filled = ask_price_s.ffill().to_numpy()

    ask_qty_s = pd.Series(ask_upd_qty)
    ask_qty_s.iloc[0] = ask_qty_s.iloc[0] if not np.isnan(ask_qty_s.iloc[0]) else state.ask_qty
    ask_qty_filled = ask_qty_s.ffill().to_numpy()

    out = pd.DataFrame({
        "timestamp_ns": (chunk["transact_time"].to_numpy() * 1_000_000).astype(np.int64),
        "bid_price": bid_price_filled,
        "bid_qty": bid_qty_filled,
        "ask_price": ask_price_filled,
        "ask_qty": ask_qty_filled,
    })

    # carry state forward for the next chunk
    state.bid_price = float(bid_price_filled[-1])
    state.bid_qty = float(bid_qty_filled[-1])
    state.ask_price = float(ask_price_filled[-1])
    state.ask_qty = float(ask_qty_filled[-1])

    return out


def fetch_symbol_day(symbol: str, instrument_id: int, day: date,
                      chunksize: int, max_retries: int, verify_checksum: bool) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{day}.zip"
    logging.info(f"[{symbol}] fetching {day}...")

    resp = _get_with_retry(url, max_retries=max_retries)
    if resp is None:
        logging.info(f"[{symbol}] no archive for {day} (404)")
        return None

    if verify_checksum and not _verify_checksum(resp.content, url):
        logging.error(f"[{symbol}] checksum failed for {day}, discarding this download")
        return None

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
    raw_bytes = zf.read(csv_name)

    buf = io.BytesIO(raw_bytes)
    first_line = buf.readline()
    buf.seek(0)
    has_header = first_line.strip().lower().startswith(b"agg_trade_id")

    state = QuoteState()
    out_chunks = []
    total_rows = 0

    reader = pd.read_csv(
        buf,
        header=0 if has_header else None,
        names=None if has_header else RAW_COLS,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk.columns = [c.lower() for c in chunk.columns]
        for col in ("price", "quantity", "transact_time"):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk["is_buyer_maker"] = (
            chunk["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
        )
        chunk = chunk.dropna(subset=["price", "quantity", "transact_time"])
        if chunk.empty:
            continue
        total_rows += len(chunk)
        out_chunks.append(_synthesize_chunk(chunk, state))

    if not out_chunks:
        logging.warning(f"[{symbol}] no usable rows for {day}")
        return None

    out = pd.concat(out_chunks, ignore_index=True)
    out["instrument_id"] = instrument_id
    logging.info(f"[{symbol}] done: {total_rows} raw rows -> {len(out)} quote rows for {day} "
                 f"(uncapped)")
    return out


def _atomic_write_csv(df: pd.DataFrame, out_path: str):
    tmp_path = out_path + f".tmp{os.getpid()}"
    df[OUT_COLS].to_csv(tmp_path, index=False, header=False, float_format="%.8g")
    os.replace(tmp_path, out_path)  # atomic on POSIX, same filesystem


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default="data/test_data.csv")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--no-checksum", action="store_true")
    ap.add_argument("--symbols", default=",".join(INSTRUMENT_MAP.keys()),
                     help="comma-separated symbol list, default all four")
    args = ap.parse_args()

    target_date = date.fromisoformat(args.date)
    logging.info(f"Target date: {target_date} (uncapped, chunked, checksum="
                 f"{'off' if args.no_checksum else 'on'})")

    symbols = args.symbols.split(",")
    frames = []
    for sym in symbols:
        if sym not in INSTRUMENT_MAP:
            logging.error(f"Unknown symbol {sym}, skipping")
            continue
        try:
            df = fetch_symbol_day(
                sym, INSTRUMENT_MAP[sym], target_date,
                chunksize=args.chunksize,
                max_retries=args.max_retries,
                verify_checksum=not args.no_checksum,
            )
        except Exception as e:
            logging.error(f"[{sym}] failed after retries: {e}")
            df = None
        if df is not None:
            frames.append(df)
        # small stagger between symbols -- polite to Binance's CDN, cheap insurance
        # against tripping rate limits when running many days back-to-back.
        time.sleep(0.5)

    if not frames:
        logging.error(f"No data downloaded for {target_date} — try an earlier --date.")
        sys.exit(1)

    master = pd.concat(frames, ignore_index=True)
    master.sort_values("timestamp_ns", inplace=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _atomic_write_csv(master, args.out)
    logging.info(f"SUCCESS! {len(master)} rows across {len(frames)} instruments for "
                 f"{target_date} -> {args.out} (uncapped)")


if __name__ == "__main__":
    main()
