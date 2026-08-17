#!/usr/bin/env python3
"""
Walk-forward analysis over a precomputed (day, alpha, threshold) sharpe grid.

Same methodology as research/analyze_wfo.py, extended from a 1D (alpha-only)
calibration grid to a 2D (alpha, threshold) grid built by
infra/build_sharpe_grid_v3.sh. Every check that mattered for the 1D version
matters MORE here, not less: a 2D grid has more candidate configs per
in-sample window, which means more opportunities to fit in-sample noise
before the daily recalibration step, so the IS/OOS correlation check and
the fixed-config benchmark comparison are the load-bearing parts of this
script, not the headline mean-Sharpe number.

Usage:
    python3 research/analyze_wfo_2d.py --grid data/sharpe_grid_v3.csv --window 10
    python3 research/analyze_wfo_2d.py --grid data/sharpe_grid_v3.csv --window 10 \\
        --exclude-month 2024-05
    python3 research/analyze_wfo_2d.py --grid data/sharpe_grid_v3.csv --window 10 \\
        --regime --cache-dir data/wfo_cache_v2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_grid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # backtest binary failures, and "no signals passed filter" at high
    # thresholds, both show up as the literal string "NAN" written by
    # build_sharpe_grid_v3.sh -- coerce to actual NaN.
    for col in ("sharpe", "signals", "win_rate", "mean_bps", "pnl_bps"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def walk_forward(df: pd.DataFrame, window: int) -> pd.DataFrame:
    dates = sorted(df["date"].unique())
    # MultiIndex columns: each (alpha, threshold) pair is one candidate
    # config. idxmax() on a Series with MultiIndex columns returns the
    # winning (alpha, threshold) tuple directly -- no manual key-joining
    # needed.
    sharpe_pivot = df.pivot(index="date", columns=["alpha", "threshold"], values="sharpe")
    pnl_pivot = df.pivot(index="date", columns=["alpha", "threshold"], values="pnl_bps")

    rows = []
    for i in range(window, len(dates)):
        test_date = dates[i]
        train_dates = dates[i - window:i]

        train_block = sharpe_pivot.loc[train_dates]
        is_means = train_block.mean(axis=0, skipna=True)
        if is_means.isna().all():
            continue
        best_alpha, best_threshold = is_means.idxmax()
        best_is_sharpe = is_means.max()

        oos_sharpe = sharpe_pivot.loc[test_date, (best_alpha, best_threshold)]
        oos_pnl = pnl_pivot.loc[test_date, (best_alpha, best_threshold)]
        if pd.isna(oos_sharpe):
            continue

        rows.append({
            "date": test_date,
            "best_alpha": best_alpha,
            "best_threshold": best_threshold,
            "is_sharpe": best_is_sharpe,
            "oos_sharpe": oos_sharpe,
            "oos_pnl_bps": oos_pnl,
        })

    return pd.DataFrame(rows)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_means = np.array([
        rng.choice(x, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    p_le_zero = float((boot_means <= 0).mean())
    return boot_means.mean(), ci_lo, ci_hi, p_le_zero


def monthly_breakdown(wf: pd.DataFrame) -> pd.DataFrame:
    tmp = wf.copy()
    tmp["month"] = tmp["date"].dt.to_period("M")
    g = tmp.groupby("month").agg(
        n_days=("oos_sharpe", "count"),
        mean_oos_sharpe=("oos_sharpe", "mean"),
        sum_pnl_bps=("oos_pnl_bps", "sum"),
    )
    total_pnl = tmp["oos_pnl_bps"].sum()
    g["pct_of_total_pnl"] = (g["sum_pnl_bps"] / total_pnl * 100) if total_pnl != 0 else float("nan")
    return g


def concentration_check(wf: pd.DataFrame, top_n: int = 10):
    top_n = min(top_n, len(wf))
    total_pnl = wf["oos_pnl_bps"].sum()
    top = wf.nlargest(top_n, "oos_pnl_bps")
    top_sum = top["oos_pnl_bps"].sum()
    rest_sum = total_pnl - top_sum
    frac = (top_sum / total_pnl * 100) if total_pnl != 0 else float("nan")
    return total_pnl, top_sum, rest_sum, frac, top_n


def compute_regime_metrics(cache_dir: str, dates) -> pd.DataFrame:
    """Same rough per-day realized-vol/range proxy as analyze_wfo.py,
    unchanged -- it's independent of the alpha/threshold grid dimensionality,
    it just reads the raw cached tick data."""
    rows = []
    cache_dir = Path(cache_dir)
    col_names = ["timestamp_ns", "instrument_id", "bid_price", "bid_qty", "ask_price", "ask_qty"]
    for d in dates:
        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        f = cache_dir / f"{date_str}.csv"
        if not f.exists():
            continue
        try:
            raw = pd.read_csv(f, header=None, names=col_names,
                              usecols=["timestamp_ns", "instrument_id", "bid_price", "ask_price"])
        except Exception as e:
            print(f"  (regime) skipping {date_str}: {e}", file=sys.stderr)
            continue
        raw["mid"] = (raw["bid_price"] + raw["ask_price"]) / 2.0
        vols, ranges = [], []
        for instr, grp in raw.groupby("instrument_id"):
            mid = grp["mid"].to_numpy()
            if len(mid) < 10:
                continue
            log_ret = np.diff(np.log(mid))
            vols.append(np.std(log_ret))
            ranges.append((mid.max() - mid.min()) / mid.mean())
        if not vols:
            continue
        rows.append({
            "date": pd.Timestamp(d),
            "realized_vol": float(np.mean(vols)),
            "price_range_pct": float(np.mean(ranges)) * 100,
            "n_rows": len(raw),
        })
    return pd.DataFrame(rows)


def fixed_config_benchmark(df: pd.DataFrame, test_dates, top_k: int = 15) -> pd.DataFrame:
    """Benchmark against holding ONE (alpha, threshold) pair fixed for the
    whole period -- the 2D analogue of analyze_wfo.py's fixed_alpha_benchmark.
    With alpha x threshold this can be dozens of configs; top_k caps how many
    get printed (ranked by mean OOS Sharpe) so the table stays readable --
    it does not change the underlying computation, every config is still
    evaluated."""
    sharpe_pivot = df.pivot(index="date", columns=["alpha", "threshold"], values="sharpe")
    sub_dates = list(test_dates)
    sub = sharpe_pivot.loc[sub_dates] if sub_dates else sharpe_pivot
    out = []
    for alpha, threshold in sub.columns:
        series = sub[(alpha, threshold)].dropna().to_numpy()
        if len(series) == 0:
            continue
        mean, lo, hi, p = bootstrap_mean_ci(series)
        out.append({
            "alpha": alpha, "threshold": threshold, "n_days": len(series),
            "mean_oos_sharpe": mean, "ci_lo": lo, "ci_hi": hi,
            "p_mean_le_0": p,
        })
    # Sort by ci_lo, not mean_oos_sharpe: a high mean with a wide CI (few
    # signals, high variance -- e.g. a strict threshold that only fires a
    # handful of times) can outrank a lower but statistically solid mean.
    # ci_lo answers "how confident are we this isn't zero", which is the
    # actually decision-relevant question here -- see threshold=25 in past
    # runs of this script for a concrete example of the mean-only ranking
    # getting this backwards.
    return pd.DataFrame(out).sort_values("ci_lo", ascending=False).head(top_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/sharpe_grid_v3.csv")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--exclude-month", action="append", default=[],
                    help="YYYY-MM to exclude from OOS results, e.g. --exclude-month 2024-05. "
                         "Repeatable.")
    ap.add_argument("--regime", action="store_true",
                    help="Also compute rough daily volatility/range from the cached tick "
                         "data and correlate it with OOS Sharpe.")
    ap.add_argument("--cache-dir", default="data/wfo_cache_v2",
                    help="Where the per-day tick CSVs live (only used with --regime).")
    args = ap.parse_args()

    df = load_grid(args.grid)
    n_days = df["date"].nunique()
    n_alphas = df["alpha"].nunique()
    n_thresholds = df["threshold"].nunique()
    n_configs = n_alphas * n_thresholds
    print(f"Loaded grid: {n_days} days x {n_alphas} alphas x {n_thresholds} thresholds "
          f"({n_configs} configs) from {args.grid}\n")

    if n_days < args.window + 5:
        print(f"WARNING: only {n_days} days available with window={args.window}. "
              f"You'll get very few OOS points ({max(0, n_days - args.window)}). "
              f"With {n_configs} candidate configs being searched over an even smaller "
              f"in-sample window, overfitting risk here is HIGHER than in the 1D alpha-only "
              f"analysis -- treat any positive result with extra suspicion until you have "
              f"significantly more history cached.\n", file=sys.stderr)
    elif n_configs > n_days:
        print(f"WARNING: {n_configs} candidate (alpha, threshold) configs but only {n_days} "
              f"days of data. There are more knobs than data points -- the calibration step "
              f"has ample room to fit in-sample noise. Lean harder on the IS/OOS correlation "
              f"check and the fixed-config benchmark below than on the headline adaptive mean.\n",
              file=sys.stderr)

    wf = walk_forward(df, args.window)
    if wf.empty:
        print("No valid OOS days produced -- check the grid for NaNs / missing data.")
        return

    print("=== Walk-forward OOS results (adaptive per-day alpha+threshold) ===")
    print(wf.to_string(index=False, formatters={
        "is_sharpe": "{:.4f}".format,
        "oos_sharpe": "{:.4f}".format,
        "oos_pnl_bps": "{:.2f}".format,
    }))

    pearson_r = wf["is_sharpe"].corr(wf["oos_sharpe"], method="pearson")
    spearman_r = wf["is_sharpe"].corr(wf["oos_sharpe"], method="spearman")
    print(f"\nIS-Sharpe vs OOS-Sharpe correlation: pearson r={pearson_r:.3f}, "
          f"spearman rho={spearman_r:.3f}")
    if pearson_r <= 0:
        print("  -> Non-positive. The calibration step is not predictive: days where "
              "it was 'more confident' are not systematically better OOS. With a 2D grid "
              "this is an even stronger warning sign than in the 1D case -- there were "
              "more configs available to overfit to, and it still didn't help.")
    else:
        print("  -> Positive. Better in-sample fits do correspond to better OOS "
              "results -- worth digging further into whether this holds with more data, "
              "and whether it's driven by alpha, threshold, or their interaction.")

    mean, lo, hi, p = bootstrap_mean_ci(wf["oos_sharpe"].to_numpy())
    print(f"\nBootstrap (day-level resample, 5000 draws) on mean adaptive OOS Sharpe:")
    print(f"  mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  P(mean<=0)~{p:.3f}")
    if lo <= 0 <= hi:
        print("  -> Zero is inside the CI: cannot reject 'no real edge' yet.")

    # Which threshold/alpha actually got picked, and how often -- a
    # calibration step that's actually working should show some
    # consistency here, not a different winner every single day.
    print("\n=== How often each threshold / alpha was selected ===")
    print("threshold selection counts:")
    print(wf["best_threshold"].value_counts().sort_index().to_string())
    print("\nalpha selection counts:")
    print(wf["best_alpha"].value_counts().sort_index().to_string())

    print(f"\n=== Fixed-config benchmark (no daily recalibration), top {15} by mean OOS Sharpe ===")
    bench = fixed_config_benchmark(df, wf["date"])
    print(bench.to_string(index=False, formatters={
        "mean_oos_sharpe": "{:.4f}".format,
        "ci_lo": "{:.4f}".format,
        "ci_hi": "{:.4f}".format,
        "p_mean_le_0": "{:.3f}".format,
    }))

    best_fixed = bench.iloc[0]
    print(f"\nAdaptive mean OOS Sharpe: {mean:.4f}")
    print(f"Best fixed config (alpha={best_fixed['alpha']}, threshold={best_fixed['threshold']}) "
          f"mean OOS Sharpe: {best_fixed['mean_oos_sharpe']:.4f}")
    if best_fixed["mean_oos_sharpe"] >= mean:
        print("  -> A single fixed (alpha, threshold) config does at least as well as daily "
              "recalibration. The adaptive 2D search is not currently earning its complexity "
              "-- and given how many configs it searched over, that's the expected outcome "
              "unless there's a genuinely stable effect.")
    else:
        print("  -> Adaptive selection beats the best fixed config here -- but with this many "
              "candidate configs and this little data, verify this holds up as n_days grows "
              "before trusting it at all.")

    print("\n=== Monthly breakdown (adaptive OOS results) ===")
    mb = monthly_breakdown(wf)
    print(mb.to_string(formatters={
        "mean_oos_sharpe": "{:.4f}".format,
        "sum_pnl_bps": "{:.2f}".format,
        "pct_of_total_pnl": "{:.1f}".format,
    }))

    total_pnl, top_sum, rest_sum, frac, top_n_used = concentration_check(wf, top_n=10)
    print(f"\nConcentration check: top {top_n_used} of {len(wf)} days = {top_sum:.2f} bps "
          f"({frac:.1f}% of total {total_pnl:.2f} bps). "
          f"Remaining {len(wf) - top_n_used} days sum to {rest_sum:.2f} bps.")
    if rest_sum <= 0:
        print("  -> Excluding the 10 best days, the remainder is flat-or-negative. "
              "The average result is being carried by a small number of days/months, "
              "not a broad-based daily edge.")

    if args.exclude_month:
        excluded = set(args.exclude_month)
        wf2 = wf[~wf["date"].dt.to_period("M").astype(str).isin(excluded)].copy()
        print(f"\n=== Re-run excluding month(s) {sorted(excluded)}: "
              f"{len(wf)} -> {len(wf2)} OOS days ===")
        if wf2.empty:
            print("  Nothing left after exclusion.")
        else:
            mean2, lo2, hi2, p2 = bootstrap_mean_ci(wf2["oos_sharpe"].to_numpy())
            print(f"  mean OOS Sharpe={mean2:.4f}  95% CI=[{lo2:.4f}, {hi2:.4f}]  "
                  f"P(mean<=0)~{p2:.3f}")
            if lo2 <= 0 <= hi2:
                print("  -> With those month(s) excluded, zero is back inside the CI: "
                      "the significant result depended on the excluded period(s).")
            else:
                print("  -> Still significant without those month(s).")

    if args.regime:
        print(f"\n=== Regime check (volatility/range vs OOS Sharpe) ===")
        regime = compute_regime_metrics(args.cache_dir, wf["date"])
        if regime.empty:
            print("  Could not compute regime metrics -- check --cache-dir path.")
        else:
            merged = wf.merge(regime, on="date", how="inner")
            r_vol = merged["realized_vol"].corr(merged["oos_sharpe"])
            r_range = merged["price_range_pct"].corr(merged["oos_sharpe"])
            print(f"  n days with regime data: {len(merged)}")
            print(f"  corr(realized_vol, oos_sharpe)   = {r_vol:.3f}")
            print(f"  corr(price_range_pct, oos_sharpe) = {r_range:.3f}")
            if r_vol > 0.2 or r_range > 0.2:
                print("  -> Positive association: performance tends to be better on more "
                      "volatile/wide-range days.")
            else:
                print("  -> No strong association with this simple vol/range proxy.")


if __name__ == "__main__":
    main()