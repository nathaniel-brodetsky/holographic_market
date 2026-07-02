#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <algorithm>
#include <numeric>
#include <vector>

namespace holo::backtest
{

struct BacktestResult
{
    double sharpe_ratio{0.0};
    double max_drawdown{0.0};
    double win_rate{0.0};
    double annualized_volatility{0.0};
    double total_return{0.0};
    double mean_return{0.0};
    std::size_t n_trades{0U};
    std::size_t n_wins{0U};
};

class BacktestEvaluator
{
public:
    static BacktestResult evaluate(
        std::span<const double> equity_curve,
        std::size_t             n_wins,
        std::size_t             n_trades,
        double                  ticks_per_year = 252.0 * 6.5 * 3600.0) noexcept
    {
        BacktestResult r{};
        const std::size_t n = equity_curve.size();
        if (n < 2U) return r;

        r.n_trades = n_trades;
        r.n_wins   = n_wins;
        r.win_rate = n_trades > 0U
            ? static_cast<double>(n_wins) / static_cast<double>(n_trades)
            : 0.0;

        std::vector<double> returns;
        returns.reserve(n - 1U);
        for (std::size_t i = 1U; i < n; ++i)
        {
            const double prev = equity_curve[i - 1U];
            if (prev == 0.0) continue;
            returns.push_back((equity_curve[i] - prev) / prev);
        }

        if (returns.empty()) return r;

        const double mean = std::accumulate(returns.begin(), returns.end(), 0.0)
                          / static_cast<double>(returns.size());
        double var = 0.0;
        for (double x : returns) { const double d = x - mean; var += d * d; }
        var /= static_cast<double>(returns.size());
        const double stdv = std::sqrt(var);

        r.mean_return            = mean;
        r.annualized_volatility  = stdv * std::sqrt(ticks_per_year);
        r.sharpe_ratio           = (stdv > 1e-12)
            ? (mean / stdv) * std::sqrt(ticks_per_year)
            : 0.0;

        double peak    = equity_curve[0];
        double max_dd  = 0.0;
        for (std::size_t i = 1U; i < n; ++i)
        {
            if (equity_curve[i] > peak) peak = equity_curve[i];
            const double dd = (peak > 0.0)
                ? (peak - equity_curve[i]) / peak
                : 0.0;
            if (dd > max_dd) max_dd = dd;
        }
        r.max_drawdown = max_dd;
        r.total_return = (equity_curve[0] != 0.0)
            ? (equity_curve.back() - equity_curve[0]) / equity_curve[0]
            : 0.0;

        return r;
    }

    static void print(const BacktestResult& r) noexcept
    {
        std::puts("\n══════════════════════════════════════════════════════");
        std::puts("  BACKTEST STATISTICAL EVALUATION");
        std::puts("══════════════════════════════════════════════════════");
        std::printf("  Total return          : %+.4f%%\n", r.total_return * 100.0);
        std::printf("  Sharpe ratio (ann.)   : %.4f\n",   r.sharpe_ratio);
        std::printf("  Annualized volatility : %.4f%%\n", r.annualized_volatility * 100.0);
        std::printf("  Max drawdown          : %.4f%%\n", r.max_drawdown * 100.0);
        std::printf("  Win rate              : %.2f%%\n", r.win_rate * 100.0);
        std::printf("  Trades                : %zu\n",    r.n_trades);
        std::printf("  Wins                  : %zu\n",    r.n_wins);
        std::puts("══════════════════════════════════════════════════════\n");
    }
};

} // namespace holo::backtest