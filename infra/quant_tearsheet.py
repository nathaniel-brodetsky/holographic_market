#!/usr/bin/env python3
import warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings("ignore")

BG = "#0d1117"
PANEL_BG = "#161b22"
ACCENT = "#58a6ff"
TEXT_CLR = "#c9d1d9"
GRID_CLR = "#21262d"

plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL_BG, "axes.edgecolor": GRID_CLR, "axes.labelcolor": TEXT_CLR, "axes.titlecolor": TEXT_CLR, "xtick.color": TEXT_CLR, "ytick.color": TEXT_CLR, "text.color": TEXT_CLR, "grid.color": GRID_CLR, "grid.alpha": 0.6})

CSV_PATH = Path("data/advanced_metrics.csv")
PDF_PATH = Path("data/Institutional_Tear_Sheet.pdf")

def load_data():
    if not CSV_PATH.exists(): sys.exit(f"ERROR: {CSV_PATH} not found.")
    df = pd.read_csv(CSV_PATH)
    df["trade_idx"] = np.arange(len(df))
    rmax = np.maximum.accumulate(df["cumulative_pnl"])
    df["drawdown_pct"] = np.where(rmax != 0, (df["cumulative_pnl"] - rmax) / np.abs(rmax) * 100.0, 0.0)
    return df

def plot_equity(ax, df):
    ax.plot(df["trade_idx"], df["cumulative_pnl"], color=ACCENT, lw=1.5)
    ax.fill_between(df["trade_idx"], df["cumulative_pnl"], df["cumulative_pnl"].min(), color=ACCENT, alpha=0.2)
    ax.set_title("1. Cumulative PnL vs Trade Index", fontweight="bold")
    ax.grid(True)

def plot_drawdown(ax, df):
    ax.plot(df["trade_idx"], df["drawdown_pct"], color="#f85149", lw=1.2)
    ax.fill_between(df["trade_idx"], df["drawdown_pct"], 0, color="#f85149", alpha=0.3)
    ax.set_title("2. Underwater Plot (Drawdown %)", fontweight="bold")
    ax.grid(True)

def plot_pnl_dist(ax, df):
    ax.hist(df["pnl"], bins=50, color="#3fb950", alpha=0.7, edgecolor=PANEL_BG)
    ax.set_title("3. Returns Distribution (bps)", fontweight="bold")
    ax.grid(True)

def plot_ym_scatter(ax, df):
    sc = ax.scatter(df["yang_mills_action"], df["pnl"], c=df["pnl"], cmap="coolwarm", s=10, alpha=0.8)
    ax.set_title("4. Yang-Mills Action vs PnL", fontweight="bold")
    ax.grid(True)

def main():
    df = load_data()
    pdf = PdfPages(PDF_PATH)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    fig.suptitle("HOLOGRAPHIC MARKET V2 - QUANTITATIVE TEAR SHEET", fontsize=18, fontweight="bold", color=ACCENT, y=0.95)
    
    plot_equity(fig.add_subplot(gs[0, 0]), df)
    plot_drawdown(fig.add_subplot(gs[0, 1]), df)
    plot_pnl_dist(fig.add_subplot(gs[1, 0]), df)
    plot_ym_scatter(fig.add_subplot(gs[1, 1]), df)
    
    pdf.savefig(fig, facecolor=BG, bbox_inches="tight")
    pdf.close()
    print(f"\n[SUCCESS] PDF Tear Sheet saved to {PDF_PATH} !!")

if __name__ == "__main__":
    main()
