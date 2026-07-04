#!/usr/bin/env python3
"""
Walk-forward analysis over a precomputed (day, alpha) sharpe grid.

Reads data/sharpe_grid.csv (built by infra/build_sharpe_grid.sh) and, entirely
in Python with no C++ re-invocation:

  1. Replays the same walk-forward logic as run_wfo.sh (pick the alpha with
     the best mean in-sample Sharpe over the preceding `--window` days, then
     read off that day's out-of-sample Sharpe from the grid).
  2. Reports the correlation between the in-sample Sharpe used to pick alpha
     and the resulting out-of-sample Sharpe. If this is at or below zero,
     the calibration step is not adding predictive value -- it's fitting
     noise in the training window.
  3. Bootstraps the mean OOS Sharpe (resampling at the day level, since
     within-day signals are almost certainly autocorrelated and a naive
     per-signal t-test would overstate significance) to get a confidence
     interval and a rough p-value against "true mean <= 0".
  4. Benchmarks the adaptive per-day alpha selection against simply using a
     single fixed alpha for the whole period, so you can tell whether the
     daily calibration is worth its complexity at all.
  5. Breaks results down by calendar month, and optionally re-runs the whole
     analysis with specific months excluded (--exclude-month), so you can
     check whether the result depends on one unusually good/bad stretch.
  6. Optionally (--regime) computes a rough daily volatility/trend signature
     straight from the cached tick data and correlates it with OOS Sharpe,
     to check whether performance is regime-dependent (e.g. only works on
     trending/volatile days).

Usage:
    python3 research/analyze_wfo.py --grid data/sharpe_grid.csv --window 10
    python3 research/analyze_wfo.py --grid data/sharpe_grid.csv --window 10 \\
        --exclude-month 2026-06
    python3 research/analyze_wfo.py --grid data/sharpe_grid.csv --window 10 \\
        --regime --cache-dir data/wfo_cache
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
    # backtest binary failures show up as the literal string "NAN"
    for col in ("sharpe", "signals", "win_rate", "mean_bps", "pnl_bps"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def walk_forward(df: pd.DataFrame, window: int) -> pd.DataFrame:
    dates = sorted(df["date"].unique())
    sharpe_pivot = df.pivot(index="date", columns="alpha", values="sharpe")
    pnl_pivot = df.pivot(index="date", columns="alpha", values="pnl_bps")

    rows = []
    for i in range(window, len(dates)):
        test_date = dates[i]
        train_dates = dates[i - window:i]

        train_block = sharpe_pivot.loc[train_dates]
        is_means = train_block.mean(axis=0, skipna=True)
        if is_means.isna().all():
            continue
        best_alpha = is_means.idxmax()
        best_is_sharpe = is_means.max()

        oos_sharpe = sharpe_pivot.loc[test_date, best_alpha]
        oos_pnl = pnl_pivot.loc[test_date, best_alpha]
        if pd.isna(oos_sharpe):
            continue

        rows.append({
            "date": test_date,
            "best_alpha": best_alpha,
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
    total_pnl = wf["oos_pnl_bps"].sum()
    top = wf.nlargest(top_n, "oos_pnl_bps")
    top_sum = top["oos_pnl_bps"].sum()
    rest_sum = total_pnl - top_sum
    frac = (top_sum / total_pnl * 100) if total_pnl != 0 else float("nan")
    return total_pnl, top_sum, rest_sum, frac


def compute_regime_metrics(cache_dir: str, dates) -> pd.DataFrame:
    """Rough per-day realized volatility and price range, straight from the
    cached bid/ask tick data. This is NOT the strategy's own signal -- just
    a simple, independent yardstick for 'how volatile/eventful was this day'
    so we can check whether OOS performance correlates with regime."""
    rows = []
    cache_dir = Path(cache_dir)
    # NB: these files are written by infra/download_binance_aggtrades.py with
    # to_csv(..., header=False, ...) -- there is no header row, and the fixed
    # column order is timestamp_ns, instrument_id, bid_price, bid_qty,
    # ask_price, ask_qty. Don't rely on named columns here.
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
        cap_hits = []
        coverage_fracs = []
        day_span_ns = 24 * 3600 * 1_000_000_000
        for instr, grp in raw.groupby("instrument_id"):
            mid = grp["mid"].to_numpy()
            if len(mid) < 10:
                continue
            log_ret = np.diff(np.log(mid))
            vols.append(np.std(log_ret))
            ranges.append((mid.max() - mid.min()) / mid.mean())
            # infra/download_binance_aggtrades.py caps each symbol at
            # nrows=500_000 read from the raw (time-ordered) archive. If an
            # instrument hits that cap, its rows for that day are truncated
            # to roughly the FIRST part of the day only -- not a random
            # sample, and not aligned with the other (uncapped) instruments.
            if len(grp) >= 499_000:
                cap_hits.append(instr)
                ts = grp["timestamp_ns"].to_numpy()
                covered = (ts.max() - ts.min()) / day_span_ns
                coverage_fracs.append(covered)
        if not vols:
            continue
        rows.append({
            "date": pd.Timestamp(d),
            "realized_vol": float(np.mean(vols)),
            "price_range_pct": float(np.mean(ranges)) * 100,
            "n_rows": len(raw),
            "n_instruments_capped": len(cap_hits),
            "capped_day_coverage_pct": float(np.mean(coverage_fracs)) * 100 if coverage_fracs else np.nan,
        })
    return pd.DataFrame(rows)


def fixed_alpha_benchmark(df: pd.DataFrame, test_dates) -> pd.DataFrame:
    sharpe_pivot = df.pivot(index="date", columns="alpha", values="sharpe")
    sub = sharpe_pivot.loc[sub_dates] if (sub_dates := list(test_dates)) else sharpe_pivot
    out = []
    for alpha in sub.columns:
        series = sub[alpha].dropna().to_numpy()
        if len(series) == 0:
            continue
        mean, lo, hi, p = bootstrap_mean_ci(series)
        out.append({
            "alpha": alpha, "n_days": len(series),
            "mean_oos_sharpe": mean, "ci_lo": lo, "ci_hi": hi,
            "p_mean_le_0": p,
        })
    return pd.DataFrame(out).sort_values("mean_oos_sharpe", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/sharpe_grid.csv")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--exclude-month", action="append", default=[],
                     help="YYYY-MM to exclude from OOS results, e.g. --exclude-month 2026-06. "
                          "Repeatable.")
    ap.add_argument("--regime", action="store_true",
                     help="Also compute rough daily volatility/range from the cached tick "
                          "data and correlate it with OOS Sharpe.")
    ap.add_argument("--cache-dir", default="data/wfo_cache",
                     help="Where the per-day tick CSVs live (only used with --regime).")
    args = ap.parse_args()

    df = load_grid(args.grid)
    n_days = df["date"].nunique()
    n_alphas = df["alpha"].nunique()
    print(f"Loaded grid: {n_days} days x {n_alphas} alphas from {args.grid}\n")

    if n_days < args.window + 5:
        print(f"WARNING: only {n_days} days available with window={args.window}. "
              f"You'll get very few OOS points ({max(0, n_days - args.window)}). "
              f"Results below will not be statistically meaningful until you "
              f"have significantly more history cached.\n", file=sys.stderr)

    wf = walk_forward(df, args.window)
    if wf.empty:
        print("No valid OOS days produced -- check the grid for NaNs / missing data.")
        return

    print("=== Walk-forward OOS results (adaptive per-day alpha) ===")
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
              "it was 'more confident' are not systematically better OOS. This is "
              "the classic fingerprint of fitting in-sample noise, not signal.")
    else:
        print("  -> Positive. Better in-sample fits do correspond to better OOS "
              "results -- worth digging further into whether this holds with more data.")

    mean, lo, hi, p = bootstrap_mean_ci(wf["oos_sharpe"].to_numpy())
    print(f"\nBootstrap (day-level resample, 5000 draws) on mean adaptive OOS Sharpe:")
    print(f"  mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  P(mean<=0)~{p:.3f}")
    if lo <= 0 <= hi:
        print("  -> Zero is inside the CI: cannot reject 'no real edge' yet.")

    print("\n=== Fixed-alpha benchmark (no daily recalibration) over the same OOS dates ===")
    bench = fixed_alpha_benchmark(df, wf["date"])
    print(bench.to_string(index=False, formatters={
        "mean_oos_sharpe": "{:.4f}".format,
        "ci_lo": "{:.4f}".format,
        "ci_hi": "{:.4f}".format,
        "p_mean_le_0": "{:.3f}".format,
    }))

    best_fixed = bench.iloc[0]
    print(f"\nAdaptive mean OOS Sharpe: {mean:.4f}")
    print(f"Best fixed-alpha ({best_fixed['alpha']}) mean OOS Sharpe: "
          f"{best_fixed['mean_oos_sharpe']:.4f}")
    if best_fixed["mean_oos_sharpe"] >= mean:
        print("  -> A single fixed alpha does at least as well as daily recalibration. "
              "The adaptive WFO machinery is not currently earning its complexity.")
    else:
        print("  -> Adaptive selection beats the best fixed alpha here -- but check "
              "this holds up as n_days grows before trusting it.")

    # --- Monthly breakdown + concentration check -------------------------
    print("\n=== Monthly breakdown (adaptive OOS results) ===")
    mb = monthly_breakdown(wf)
    print(mb.to_string(formatters={
        "mean_oos_sharpe": "{:.4f}".format,
        "sum_pnl_bps": "{:.2f}".format,
        "pct_of_total_pnl": "{:.1f}".format,
    }))

    total_pnl, top_sum, rest_sum, frac = concentration_check(wf, top_n=10)
    print(f"\nConcentration check: top 10 of {len(wf)} days = {top_sum:.2f} bps "
          f"({frac:.1f}% of total {total_pnl:.2f} bps). "
          f"Remaining {len(wf) - 10} days sum to {rest_sum:.2f} bps.")
    if rest_sum <= 0:
        print("  -> Excluding the 10 best days, the remainder is flat-or-negative. "
              "The average result is being carried by a small number of days/months, "
              "not a broad-based daily edge. Treat the overall positive mean with caution "
              "until this concentration eases with more data or more careful regime analysis.")

    # --- Optional: re-run everything excluding specific months -----------
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
                print("  -> Still significant without those month(s) -- more encouraging, "
                      "the result isn't purely a single-period artifact.")

    # --- Optional: regime correlation ------------------------------------
    if args.regime:
        print(f"\n=== Regime check (volatility/range vs OOS Sharpe) ===")
        regime = compute_regime_metrics(args.cache_dir, wf["date"])
        if regime.empty:
            print("  Could not compute regime metrics -- check --cache-dir path and that "
                  "the CSVs have bid_price/ask_price/instrument_id columns.")
        else:
            merged = wf.merge(regime, on="date", how="inner")
            r_vol = merged["realized_vol"].corr(merged["oos_sharpe"])
            r_range = merged["price_range_pct"].corr(merged["oos_sharpe"])
            print(f"  n days with regime data: {len(merged)}")
            print(f"  corr(realized_vol, oos_sharpe)   = {r_vol:.3f}")
            print(f"  corr(price_range_pct, oos_sharpe) = {r_range:.3f}")
            n_capped_days = int((regime["n_instruments_capped"] > 0).sum())
            if n_capped_days > 0:
                print(f"\n  WARNING: {n_capped_days}/{len(regime)} days had at least one "
                      f"instrument hit the downloader's 500,000-row cap. On those days that "
                      f"instrument's data covers only roughly the FIRST part of the day "
                      f"(the raw archive is time-ordered and gets truncated), not the full "
                      f"24h, and not aligned with uncapped instruments on the same day. This "
                      f"is a real confound, independent of alpha tuning -- worth checking "
                      f"whether the strong months (e.g. June) have more or fewer capped days "
                      f"than the weak ones.")
                cov = regime.dropna(subset=["capped_day_coverage_pct"]).copy()
                if not cov.empty:
                    cov["month"] = cov["date"].dt.to_period("M")
                    cov_by_month = cov.groupby("month")["capped_day_coverage_pct"].mean()
                    print("\n  Average time-of-day coverage (%) for capped instruments, by month:")
                    print("  " + cov_by_month.to_string().replace("\n", "\n  "))
                    print("  (Lower % = the 500k-row cap is hit earlier in the day for that "
                          "month, i.e. that month's high-volume instruments are systematically "
                          "sampling an earlier, possibly different, part of the trading day.)")
            if r_vol > 0.2 or r_range > 0.2:
                print("  -> Positive association: performance tends to be better on more "
                      "volatile/wide-range days. Worth checking whether the strong months "
                      "(see monthly breakdown above) also had unusually high volatility -- "
                      "that would mean the 'edge' is really a volatility regime dependency, "
                      "not a stable daily effect.")
            else:
                print("  -> No strong association with this simple vol/range proxy. "
                      "The month-level concentration isn't obviously explained by volatility "
                      "alone -- worth looking at other regime descriptors (trend strength, "
                      "correlation across instruments, etc).")


if __name__ == "__main__":
    main()