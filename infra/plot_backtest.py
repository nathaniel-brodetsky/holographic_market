#!/usr/bin/env python3
import sys, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV  = Path("data/equity_curve.csv")
OUTPUT_PNG = Path("data/backtest_chart.png")

def make_chart():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        return
    
    try:
        df = pd.read_csv(INPUT_CSV)
        cols = [str(c).strip().lower() for c in df.columns]
        if "step" not in cols:
            # Если текстовой шапки нет — читаем как чистые данные
            df = pd.read_csv(INPUT_CSV, header=None)
            df.columns = ["step", "equity"]
        else:
            df.columns = cols
    except Exception as e:
        print(f"Failed to parse CSV: {e}")
        return

    if len(df) < 2:
        print("Not enough points to plot.")
        return

    step = pd.to_numeric(df["step"], errors="coerce").fillna(0).to_numpy()
    equity = pd.to_numeric(df["equity"], errors="coerce").fillna(0).to_numpy()
    
    rmax = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(rmax != 0, (equity - rmax) / np.abs(rmax) * 100.0, 0.0)
    drawdown = np.nan_to_num(drawdown)
    
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[2.2, 1.0], gridspec_kw={"hspace": 0.08})
    fig.patch.set_facecolor("#0d1117")

    ax_eq.set_facecolor("#0d1117")
    ax_eq.grid(True, color="#21262d", alpha=0.8)
    ax_eq.tick_params(colors="#c9d1d9")
    ax_eq.plot(step, equity, color="#58a6ff", linewidth=1.5, label="Cumulative PnL")
    ax_eq.fill_between(step, equity, equity.min(), color="#58a6ff22")
    ax_eq.set_ylabel("Cumulative PnL (bps)", color="#c9d1d9", fontsize=11)
    ax_eq.set_title(f"Holographic Market Architecture — Equity Curve (Final: {equity[-1]:+.2f} bps)", color="#c9d1d9", fontsize=14, fontweight="bold", pad=15)
    ax_eq.legend(loc="upper left", facecolor="#0d1117", labelcolor="#c9d1d9")

    ax_dd.set_facecolor("#0d1117")
    ax_dd.grid(True, color="#21262d", alpha=0.8)
    ax_dd.tick_params(colors="#c9d1d9")
    ax_dd.plot(step, drawdown, color="#f85149", linewidth=1.2)
    ax_dd.fill_between(step, drawdown, 0, color="#f8514944")
    ax_dd.set_ylabel("Drawdown (%)", color="#c9d1d9", fontsize=11)
    ax_dd.set_xlabel("Trade Step (Floer-gated signals)", color="#c9d1d9", fontsize=11)
    
    os.makedirs(OUTPUT_PNG.parent, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)
    print(f"SUCCESS! Chart saved to {OUTPUT_PNG}")

if __name__ == "__main__":
    make_chart()
