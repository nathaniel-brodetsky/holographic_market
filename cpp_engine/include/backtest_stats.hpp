#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>

namespace holo
{

struct BacktestResult
{
    double sharpe_ratio{0.0};
    double max_drawdown{0.0};
    double win_rate{0.0};
    double total_return{0.0};
    double annualized_vol{0.0};
    std::uint64_t n_samples{0U};
    std::uint64_t n_wins{0U};
    std::uint64_t n_losses{0U};
};

class BacktestEvaluator final
{
public:
    BacktestEvaluator() = default;
    ~BacktestEvaluator() = default;

    BacktestEvaluator(const BacktestEvaluator&) = delete;
    BacktestEvaluator& operator=(const BacktestEvaluator&) = delete;
    BacktestEvaluator(BacktestEvaluator&&) = delete;
    BacktestEvaluator& operator=(BacktestEvaluator&&) = delete;

    // equity_curve: absolute equity values, length N.
    // returns_per_bar: pre-computed period returns, length N-1.
    // If returns_per_bar is empty, they are computed internally from equity_curve.
    // bars_per_year: annualisation factor (e.g. 252 * 390 for 1-min equity bars).
    [[nodiscard]] static BacktestResult evaluate(
        std::span<const double> equity_curve,
        std::span<const double> returns_per_bar,
        double                  bars_per_year = 252.0 * 390.0) noexcept
    {
        BacktestResult r{};

        const std::size_t n_eq = equity_curve.size();
        if (n_eq < 2U) return r;

        r.n_samples = static_cast<std::uint64_t>(n_eq);

        // Total return.
        const double e0 = equity_curve[0];
        const double eN = equity_curve[n_eq - 1U];
        r.total_return = (e0 > 0.0) ? (eN - e0) / e0 : 0.0;

        // Compute period returns if not supplied.
        // Stack-only scratch for small curves; heap fallback via VLA-style pointer.
        // Because zero-allocation constraint applies to hot-path at runtime,
        // and this evaluator runs once post-replay, we use alloca for small N
        // and tolerate the single allocation for large N only in evaluation.
        const bool use_supplied = !returns_per_bar.empty();
        const std::size_t n_ret = use_supplied
            ? returns_per_bar.size()
            : (n_eq - 1U);

        double* ret_buf = nullptr;
        double  stack_buf[1024];
        bool    heap_alloc = false;

        if (!use_supplied)
        {
            if (n_ret <= 1024U)
            {
                ret_buf = stack_buf;
            }
            else
            {
                ret_buf = new (std::nothrow) double[n_ret];
                heap_alloc = (ret_buf != nullptr);
                if (!ret_buf) return r;
            }
            for (std::size_t i = 0U; i < n_ret; ++i)
            {
                const double prev = equity_curve[i];
                ret_buf[i] = (prev > 0.0)
                    ? (equity_curve[i + 1U] - prev) / prev
                    : 0.0;
            }
        }
        else
        {
            ret_buf = const_cast<double*>(returns_per_bar.data());
        }

        // Mean return.
        double mean = 0.0;
        for (std::size_t i = 0U; i < n_ret; ++i) mean += ret_buf[i];
        mean /= static_cast<double>(n_ret);

        // Variance (population).
        double var = 0.0;
        for (std::size_t i = 0U; i < n_ret; ++i)
        {
            const double d = ret_buf[i] - mean;
            var += d * d;
        }
        var /= static_cast<double>(n_ret);

        r.annualized_vol = std::sqrt(var * bars_per_year);

        // Sharpe (rf = 0).
        const double sigma = std::sqrt(var);
        r.sharpe_ratio = (sigma > 0.0)
            ? (mean / sigma) * std::sqrt(bars_per_year)
            : 0.0;

        // Win rate.
        for (std::size_t i = 0U; i < n_ret; ++i)
        {
            if (ret_buf[i] > 0.0)       ++r.n_wins;
            else if (ret_buf[i] < 0.0)  ++r.n_losses;
        }
        r.win_rate = (n_ret > 0U)
            ? static_cast<double>(r.n_wins) / static_cast<double>(n_ret)
            : 0.0;

        if (heap_alloc) delete[] ret_buf;

        // Max drawdown: O(N) peak-tracking pass over equity curve.
        double peak = equity_curve[0];
        double max_dd = 0.0;
        for (std::size_t i = 1U; i < n_eq; ++i)
        {
            if (equity_curve[i] > peak) peak = equity_curve[i];
            const double dd = (peak > 0.0)
                ? (peak - equity_curve[i]) / peak
                : 0.0;
            if (dd > max_dd) max_dd = dd;
        }
        r.max_drawdown = max_dd;

        return r;
    }

    static void print_brutalist_summary(const BacktestResult& r) noexcept
    {
        std::printf("\n");
        std::printf("╔══════════════════════════════════════════════════╗\n");
        std::printf("║       HOLOGRAPHIC BACKTEST SUMMARY — PHASE VI    ║\n");
        std::printf("╠══════════════════════════════════════════════════╣\n");
        std::printf("║  Total Return      : %+10.4f %%                ║\n",
            r.total_return * 100.0);
        std::printf("║  Sharpe Ratio      : %+10.4f                   ║\n",
            r.sharpe_ratio);
        std::printf("║  Max Drawdown      : %10.4f %%                ║\n",
            r.max_drawdown * 100.0);
        std::printf("║  Annualised Vol    : %10.4f %%                ║\n",
            r.annualized_vol * 100.0);
        std::printf("║  Win Rate          : %10.4f %%                ║\n",
            r.win_rate * 100.0);
        std::printf("║  Wins / Losses     : %8llu / %-8llu          ║\n",
            static_cast<unsigned long long>(r.n_wins),
            static_cast<unsigned long long>(r.n_losses));
        std::printf("║  Equity Samples    : %10llu                   ║\n",
            static_cast<unsigned long long>(r.n_samples));
        std::printf("╚══════════════════════════════════════════════════╝\n");
        std::printf("\n");
    }
};

} // namespace holo