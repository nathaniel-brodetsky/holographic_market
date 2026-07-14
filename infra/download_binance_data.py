#!/usr/bin/env python3
import argparse, io, logging, os, zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/futures/um/daily/bookTicker"
INSTRUMENT_MAP = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2, "BNBUSDT": 3}
COLS = ["update_id", "symbol", "bid_price", "bid_qty", "ask_price", "ask_qty", "transaction_time", "event_time", "message_time"]

@dataclass
class DownloadTask:
    symbol: str; instrument_id: int; day: date
    @property
    def url(self): return f"{BASE_URL}/{self.symbol}/{self.symbol}-bookTicker-{self.day}.zip"

def _fetch_and_transform(task, nrows):
    logging.info(f"[{task.symbol}] Downloading {task.day}...")
    try:
        resp = requests.get(task.url, timeout=30)
        if resp.status_code != 200:
            logging.error(f"[{task.symbol}] HTTP {resp.status_code} — archive not available yet for {task.day}?")
            return None
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]

        logging.info(f"[{task.symbol}] Parsing (limit {nrows} rows to save RAM)...")
        buf = io.BytesIO(zf.read(csv_name))
        first_line = buf.readline()
        buf.seek(0)
        n_cols = first_line.count(b",") + 1

        df = pd.read_csv(buf, header=None, names=COLS[:n_cols], dtype=str, nrows=nrows)

        for col in ("bid_price", "bid_qty", "ask_price", "ask_qty", "transaction_time"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["bid_price", "bid_qty", "ask_price", "ask_qty", "transaction_time"], inplace=True)
        df["timestamp_ns"] = (df["transaction_time"].astype("int64") * 1_000_000)

        out = df[["timestamp_ns", "bid_price", "bid_qty", "ask_price", "ask_qty"]].copy()
        out = out.astype({"timestamp_ns": "int64", "bid_price": "float64", "bid_qty": "float64", "ask_price": "float64", "ask_qty": "float64"})
        out["instrument_id"] = task.instrument_id
        logging.info(f"[{task.symbol}] Done! ({len(out)} rows)")
        return out
    except Exception as e:
        logging.error(f"[{task.symbol}] Error: {e}")
        return None

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Download a day of Binance Futures bookTicker data for backtesting.")
    parser.add_argument("--date", type=str, default=None,
                         help="Target date YYYY-MM-DD. Defaults to 2 days before today "
                              "(Binance's daily archive usually isn't published for today or yesterday yet).")
    parser.add_argument("--nrows", type=int, default=500_000,
                         help="Max rows to read per instrument (keep the total across all "
                              "instruments x2 under the C++ ring buffer capacity, currently 8'388'608).")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=2))
    logging.info(f"Target date: {target_date}")

    tasks = [DownloadTask(s, i, target_date) for s, i in INSTRUMENT_MAP.items()]

    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for df in pool.map(lambda t: _fetch_and_transform(t, args.nrows), tasks):
            if df is not None: frames.append(df)

    if not frames:
        logging.error(f"No data downloaded for {target_date} — try an earlier --date (archive may not be published yet).")
        return

    logging.info("Sorting and saving to data/test_data.csv...")
    master = pd.concat(frames, ignore_index=True)
    master.sort_values("timestamp_ns", inplace=True)

    os.makedirs("data", exist_ok=True)
    master[["timestamp_ns", "instrument_id", "bid_price", "bid_qty", "ask_price", "ask_qty"]].to_csv("data/test_data.csv", index=False, header=False, float_format="%.8g")
    logging.info(f"SUCCESS! {len(master)} rows across {len(frames)} instruments for {target_date} -> data/test_data.csv")

if __name__ == "__main__": main()