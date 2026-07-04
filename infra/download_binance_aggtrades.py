#!/usr/bin/env python3
"""
Drop-in replacement for download_binance_data.py, but sourced from `aggTrades`
daily archives (still actively published by Binance) instead of `bookTicker`
(frozen since 2024, see https://dev.binance.vision/t/.../36122).

IMPORTANT — this is a proxy, not real L2 data:
  - There is no real bid/ask here. We synthesize a "quote" stream by treating
    each taker-buy trade as an ask-side update and each taker-sell trade as a
    bid-side update. bid_qty / ask_qty therefore represent recent AGGRESSIVE
    TRADE FLOW imbalance (a la Cont/Kukanov/Stoikov order-flow imbalance),
    NOT resting order-book depth imbalance. This changes what the engine's
    "OIB" signal actually measures.
  - bid_price and ask_price start equal (zero synthetic spread). The C++ LOB
    core (lob_core.hpp: spread()) already falls back to a constant 2bps
    synthetic spread whenever bid == ask, so cost filtering still applies,
    just with an assumed flat 2bps instead of the true market spread.

Usage: python3 infra/download_binance_aggtrades.py --date 2026-06-15
"""
import argparse, io, logging, os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
INSTRUMENT_MAP = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2, "BNBUSDT": 3}
# futures um aggTrades daily csv columns (no header row in the file):
RAW_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
            "transact_time", "is_buyer_maker"]


@dataclass
class DownloadTask:
    symbol: str
    instrument_id: int
    day: date

    @property
    def url(self):
        return f"{BASE_URL}/{self.symbol}/{self.symbol}-aggTrades-{self.day}.zip"


def _fetch_and_transform(task: DownloadTask):
    logging.info(f"[{task.symbol}] Downloading {task.day}...")
    try:
        resp = requests.get(task.url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            logging.error(f"[{task.symbol}] HTTP {resp.status_code} for {task.day}")
            return None
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]

        buf = io.BytesIO(zf.read(csv_name))
        first_line = buf.readline()
        buf.seek(0)
        has_header = first_line.strip().lower().startswith(b"agg_trade_id")
        df = pd.read_csv(
            buf,
            header=0 if has_header else None,
            names=None if has_header else RAW_COLS,
            dtype=str,
            nrows=500_000,
        )
        df.columns = [c.lower() for c in df.columns]

        for col in ("price", "quantity", "transact_time"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
        df.dropna(subset=["price", "quantity", "transact_time"], inplace=True)
        df.sort_values("transact_time", inplace=True)

        # --- synthesize a running bid/ask quote stream from trade prints ---
        bid_price = ask_price = float(df["price"].iloc[0])
        bid_qty = ask_qty = 0.0
        rows = []
        for price, qty, ts, buyer_is_maker in zip(
            df["price"].to_numpy(), df["quantity"].to_numpy(),
            df["transact_time"].to_numpy(), df["is_buyer_maker"].to_numpy()
        ):
            if buyer_is_maker:
                # taker sold -> hit the bid
                bid_price, bid_qty = price, qty
            else:
                # taker bought -> lifted the ask
                ask_price, ask_qty = price, qty
            rows.append((int(ts) * 1_000_000, bid_price, bid_qty, ask_price, ask_qty))

        out = pd.DataFrame(rows, columns=["timestamp_ns", "bid_price", "bid_qty", "ask_price", "ask_qty"])
        out["instrument_id"] = task.instrument_id
        logging.info(f"[{task.symbol}] Done! ({len(out)} rows, OFI-proxy quotes)")
        return out
    except Exception as e:
        logging.error(f"[{task.symbol}] Error: {e}")
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.date)
    logging.info(f"Target date: {target_date}")

    tasks = [DownloadTask(s, i, target_date) for s, i in INSTRUMENT_MAP.items()]

    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for df in pool.map(_fetch_and_transform, tasks):
            if df is not None:
                frames.append(df)

    if not frames:
        logging.error(f"No data downloaded for {target_date} — try an earlier --date.")
        sys.exit(1)

    logging.info("Sorting and saving to data/test_data.csv...")
    master = pd.concat(frames, ignore_index=True)
    master.sort_values("timestamp_ns", inplace=True)

    os.makedirs("data", exist_ok=True)
    master[["timestamp_ns", "instrument_id", "bid_price", "bid_qty", "ask_price", "ask_qty"]].to_csv(
        "data/test_data.csv", index=False, header=False, float_format="%.8g"
    )
    logging.info(f"SUCCESS! {len(master)} rows across {len(frames)} instruments for {target_date} -> data/test_data.csv")


if __name__ == "__main__":
    main()