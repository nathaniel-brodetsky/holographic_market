#!/usr/bin/env python3
import io, logging, os, zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
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

def _fetch_and_transform(task):
    logging.info(f"[{task.symbol}] Downloading {task.day}...")
    try:
        resp = requests.get(task.url, timeout=30)
        if resp.status_code != 200: return None
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        
        logging.info(f"[{task.symbol}] Parsing (limit 500k rows to save RAM)...")
        buf = io.BytesIO(zf.read(csv_name))
        first_line = buf.readline()
        buf.seek(0)
        n_cols = first_line.count(b",") + 1
        
        # ЧИТАЕМ ТОЛЬКО 500,000 СТРОК
        df = pd.read_csv(buf, header=None, names=COLS[:n_cols], dtype=str, nrows=500000)
        
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
    target_date = date.fromisoformat("2024-01-01")
    tasks = [DownloadTask(s, i, target_date) for s, i in INSTRUMENT_MAP.items()]
    
    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for df in pool.map(_fetch_and_transform, tasks):
            if df is not None: frames.append(df)
            
    logging.info("Sorting and saving to data/test_data.csv...")
    master = pd.concat(frames, ignore_index=True)
    master.sort_values("timestamp_ns", inplace=True)
    
    os.makedirs("data", exist_ok=True)
    master[["timestamp_ns", "instrument_id", "bid_price", "bid_qty", "ask_price", "ask_qty"]].to_csv("data/test_data.csv", index=False, header=False, float_format="%.8g")
    logging.info("SUCCESS! File is ready for C++ Backtester.")

if __name__ == "__main__": main()
