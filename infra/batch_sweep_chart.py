#!/usr/bin/env python3
"""
Reads data/batch_sweep/batch_<N>.csv (one per HOLO_GPU_BATCH value tested
by batch_sweep.sh) plus summary.csv, and produces an overlaid equity-curve
comparison chart -- same institutional dark theme as quant_tearsheet.py --
so the speed/signal-quality tradeoff across batch sizes is visible at a
glance instead of read off a table.

Usage: python3 batch_sweep_chart.py [sweep_dir]
Default sweep_dir: data/batch_sweep
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#c9d1d9"
GRID_CLR = "#21262d"
# One distinct color per batch size, ordered smallest -> largest.
COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#39c5cf"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL_BG, "axes.edgecolor": GRID_CLR,
    "axes.labelcolor": TEXT_CLR, "axes.titlecolor": TEXT_CLR, "xtick.color": TEXT_CLR,
    "ytick.color": TEXT_CLR, "text.color": TEXT_CLR, "grid.color": GRID_CLR, "grid.alpha": 0.6,
})


def main():
    sweep_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/batch_sweep")
    summary_path = sweep_dir / "summary.csv"
    if not summary_path.exists():
        sys.exit(f"ERROR: {summary_path} not found -- run batch_sweep.sh first.")

    summary = pd.read_csv(summary_path)
    pdf_path = sweep_dir / "batch_sweep_comparison.pdf"
    pdf = PdfPages(pdf_path)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35)
    ax_equity = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    fig.suptitle("HOLOGRAPHIC MARKET — GPU BATCH SIZE SWEEP (Equity Curve Comparison)",
                 fontsize=15, fontweight="bold", color="#58a6ff", y=0.97)

    table_rows = []
    for i, row in summary.iterrows():
        batch = int(row["batch"])
        csv_path = sweep_dir / f"batch_{batch}.csv"
        color = COLORS[i % len(COLORS)]

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "cumulative_pnl" in df.columns and len(df) > 0:
                trade_idx = np.arange(len(df))
                ax_equity.plot(trade_idx, df["cumulative_pnl"], color=color, lw=1.6,
                                label=f"batch={batch}  (Sharpe={row['sharpe']:.4f}, "
                                      f"n={row['signals']:.0f})")

        table_rows.append([
            str(batch),
            f"{row['elapsed_s']:.1f}s" if pd.notna(row["elapsed_s"]) else "N/A",
            f"{row['push_wait_s']:.1f}s" if pd.notna(row["push_wait_s"]) else "N/A",
            f"{row['signals']:.0f}",
            f"{row['sharpe']:.4f}" if pd.notna(row["sharpe"]) else "N/A",
            f"{row['pnl_bps']:.1f}" if pd.notna(row["pnl_bps"]) else "N/A",
            f"{row['win_rate']:.1f}%" if pd.notna(row["win_rate"]) else "N/A",
        ])

    ax_equity.axhline(0, color=GRID_CLR, lw=1, ls="--")
    ax_equity.set_xlabel("Trade index")
    ax_equity.set_ylabel("Cumulative PnL (bps)")
    ax_equity.set_title("Cumulative PnL vs Trade Index, by HOLO_GPU_BATCH", fontsize=11)
    ax_equity.legend(loc="upper left", fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_CLR)
    ax_equity.grid(True)

    col_labels = ["Batch", "Elapsed", "Push wait", "Signals", "Sharpe", "PnL (bps)", "Win rate"]
    tbl = ax_table.table(cellText=table_rows, colLabels=col_labels, loc="center",
                          cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_CLR)
        cell.set_facecolor(PANEL_BG if r > 0 else "#21262d")
        cell.set_text_props(color=TEXT_CLR)

    pdf.savefig(fig, facecolor=BG, bbox_inches="tight")
    pdf.close()
    print(f"[SUCCESS] Comparison chart saved to {pdf_path}")


if __name__ == "__main__":
    main()
