#include <common/memory_arena.hpp>
#include <common/lockfree_ring_buffer.hpp>
#include <common/types.hpp>
#include <data_feed/lob_core.hpp>
#include <data_feed/csv_replay.hpp>
#include <compute/cuda_pipeline.cuh>
#include <trading/alpha_model.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace holo;

struct CliArgs {
    std::string   csv_path;
    ReplayMode    mode{ReplayMode::Fastest};
    std::uint32_t throttle_rate{100'000U};
    int           gpu_device{0};
    std::size_t   arena_mb{512U};
};

static CliArgs parse_args(int /*argc*/, char **argv) {
    CliArgs a; a.csv_path = argv[1]; return a;
}

struct TickRecord {
    std::uint64_t timestamp_ns;
    double        pnl;
    double        cumulative_pnl;
    float         yang_mills_action;
    float         max_curl;
    float         floer_asd_residual;
    int           betti_1;
    float         spread_bps;
};

// ─────────────────────────────────────────────────────────────────────────────
// EWMA baseline per instrument-pair (i, j): tracks log(P_j / P_i) mean & var.
// Used to compute z-score dislocation — the ONLY meaningful cross-asset signal.
//
// z = (log(P_j/P_i) - mu_ij) / sigma_ij
//
// gross_ret_bps = |z| * sigma_ij * 1e4   [expected mean-reversion in bps]
// capped at k_max_gross_bps to prevent outlier contamination.
// ─────────────────────────────────────────────────────────────────────────────
struct EwmaPair {
    double mu{0.0};       // EWMA of log-ratio
    double m2{1e-8};      // EWMA of squared deviation (variance proxy)
    bool   warm{false};   // false until first update
    static constexpr double k_alpha         = 0.02;    // ~50-tick half-life
    static constexpr double k_min_sigma     = 1e-5;
    static constexpr float  k_max_gross_bps = 5.0F;

    double update(double log_ratio) noexcept {
        if (!warm) { mu = log_ratio; m2 = 1e-8; warm = true; return 0.0; }
        const double diff = log_ratio - mu;
        mu += k_alpha * diff;
        m2  = (1.0 - k_alpha) * m2 + k_alpha * diff * diff;
        const double sigma = std::sqrt(std::max(m2, k_min_sigma * k_min_sigma));
        return (log_ratio - mu) / sigma;
    }

    float sigma_bps() const noexcept {
        return static_cast<float>(std::sqrt(std::max(m2, EwmaPair::k_min_sigma
                                                       * EwmaPair::k_min_sigma)) * 1e4);
    }
};

struct PnlTracker {
    std::vector<double>     equity_curve{0.0};
    std::vector<TickRecord> ticks;
    double                  current_pnl{0.0};
    double                  sum_ret{0.0};
    double                  sum_ret2{0.0};
    std::uint64_t           n{0U};
    std::uint64_t           wins{0U};
    std::uint64_t           filtered{0U};
    std::uint64_t           prev_signal_ts_ns{0U};

    // Pair baselines: 4×4 grid, index = src*4 + dst
    EwmaPair baselines[16]{};

    void record(
        const RoutedEdge               &edge,
        float                           mid_src,
        float                           mid_dst,
        float                           spread_price,
        const holo::cuda::SignalRecord  &sig) noexcept
    {
        if (mid_src <= 0.0F || mid_dst <= 0.0F) return;

        const std::size_t src = static_cast<std::size_t>(edge.src_instrument);
        const std::size_t dst = static_cast<std::size_t>(edge.dst_instrument);
        if (src >= 4U || dst >= 4U) return;

        // 1. Log-ratio and z-score via EWMA baseline
        const double log_ratio = std::log(static_cast<double>(mid_dst)
                                        / static_cast<double>(mid_src));
        EwmaPair &bl = baselines[src * 4U + dst];
        const double z_score = bl.update(log_ratio);

        // Skip until baseline is warm (first tick just initialises mu)
        if (!bl.warm) return;

        // 2. Signal direction and strength
        const float signal_strength = std::tanh(edge.harmonic_flow);

        // 3. Expected mean-reversion in bps:
        //    if z >> 0: pair is stretched above mean → expect reversion → short log_ratio
        //    signal_strength from harmonic flow confirms / gates the trade
        const float sigma_bps   = bl.sigma_bps();
        const float z_f         = static_cast<float>(z_score);
        const float direction   = -std::copysign(1.0F, z_f) * std::copysign(1.0F, signal_strength);
        const float gross_ret   = std::min(
            std::abs(signal_strength) * std::abs(z_f) * sigma_bps,
            EwmaPair::k_max_gross_bps);

        // 4. Transaction cost
        const float spread_bps = (spread_price / mid_src) * 1e4F;

        // 5. Cost filter
        if (gross_ret <= spread_bps || std::abs(z_f) < 0.5F) {
            ++filtered; return;
        }

        // 6. Net PnL
        const float net_ret = direction * (gross_ret - spread_bps);

        sum_ret  += net_ret;
        sum_ret2 += static_cast<double>(net_ret) * net_ret;
        ++n;
        if (net_ret > 0.0F) ++wins;
        current_pnl += net_ret;
        equity_curve.push_back(current_pnl);

        ticks.push_back(TickRecord{
            .timestamp_ns       = sig.timestamp_ns,
            .pnl                = net_ret,
            .cumulative_pnl     = current_pnl,
            .yang_mills_action  = sig.yang_mills_action,
            .max_curl           = sig.max_curl,
            .floer_asd_residual = sig.floer_asd_residual,
            .betti_1            = sig.floer_HF1,
            .spread_bps         = spread_bps,
        });
        prev_signal_ts_ns = sig.timestamp_ns;
    }

    void print_report(double elapsed_s) const noexcept {
        std::puts("\n══════════════════════════════════════════════");
        std::puts("  BACKTEST REPORT (NET OF SPREAD COSTS)");
        std::puts("══════════════════════════════════════════════");
        if (n == 0U) {
            std::printf("  No signals passed filter. Filtered: %llu\n",
                        static_cast<unsigned long long>(filtered));
            return;
        }
        const double mean   = sum_ret / static_cast<double>(n);
        const double var    = (sum_ret2 / static_cast<double>(n)) - (mean * mean);
        const double stdv   = (var > 0.0) ? std::sqrt(var) : 1.0;
        const double sharpe = mean / stdv;
        const double wr     = static_cast<double>(wins) / static_cast<double>(n) * 100.0;

        std::printf("  Signals executed : %llu\n",  static_cast<unsigned long long>(n));
        std::printf("  Filtered (cost)  : %llu\n",  static_cast<unsigned long long>(filtered));
        std::printf("  Win Rate         : %.1f%%\n", wr);
        std::printf("  Mean net(bps)    : %.4f\n",   mean);
        std::printf("  Std dev  (bps)   : %.4f\n",   stdv);
        std::printf("  Sharpe (per-sig) : %.4f\n",   sharpe);
        std::printf("  Terminal PnL     : %.2f bps\n", current_pnl);
        std::printf("  Elapsed          : %.2f s\n",   elapsed_s);
        std::puts("══════════════════════════════════════════════\n");
    }

    void flush_csv(const std::string &dir) const {
        std::filesystem::create_directories(dir);
        std::ofstream f(dir + "/advanced_metrics.csv", std::ios::trunc);
        if (!f) { std::fputs("[ERROR] Cannot open advanced_metrics.csv\n", stderr); return; }
        f << "timestamp_ns,pnl,cumulative_pnl,yang_mills_action,"
             "max_curl,floer_asd_residual,betti_1,spread_bps\n";
        for (const auto &t : ticks) {
            f << t.timestamp_ns       << ','
              << t.pnl                << ','
              << t.cumulative_pnl     << ','
              << t.yang_mills_action  << ','
              << t.max_curl           << ','
              << t.floer_asd_residual << ','
              << t.betti_1            << ','
              << t.spread_bps         << '\n';
        }
        std::printf("  [SUCCESS] advanced_metrics.csv saved (%zu ticks)\n", ticks.size());
    }
};

int main(int argc, char **argv) {
    if (argc < 2) { std::puts("Usage: backtest <lob_data.csv>"); return 1; }

    const CliArgs cli = parse_args(argc, argv);
    MemoryArena   arena{cli.arena_mb * 1024UL * 1024UL};

    LobSoA lob_soa{arena, 4U, 1U};
    DynamicSpscRingBuffer<LobUpdate> ring{arena, 1'048'576U};
    CsvReplayHandler replay{ring, cli.csv_path,
                            ReplayConfig{cli.mode, cli.throttle_rate, 500U}};
    holo::cuda::CudaPipeline gpu_pipeline{lob_soa, arena, cli.gpu_device};

    // AlphaModel replaces the old SignalRouter::route() call — it owns its
    // own edge topology and top-K selection, so callers no longer need to
    // build edge_src/edge_dst arrays by hand.
    AlphaModel             alpha;
    PnlTracker              pnl;
    AlphaModel::TopKBuffer  top_edges{};

    const auto t_start = std::chrono::steady_clock::now();
    replay.start();
    std::puts("[backtest] Processing...");

    std::atomic<std::uint64_t> lob_update_count{0U};
    std::atomic<bool>          lob_done{false};

    std::thread lob_thread([&]() {
        std::uint64_t local_count = 0U;
        LobUpdate u{};
        while (!replay.finished() || !ring.empty()) {
            if (ring.try_pop(u)) {
                lob_soa.apply(u);
                ++local_count;
                if (local_count % 128U == 0U)
                    lob_update_count.store(local_count, std::memory_order_release);
            } else { __builtin_ia32_pause(); }
        }
        lob_update_count.store(local_count, std::memory_order_release);
        lob_done.store(true, std::memory_order_release);
    });

    std::thread gpu_thread([&]() {
        std::uint64_t next_gpu_target = 2048U;
        while (true) {
            const std::uint64_t cc = lob_update_count.load(std::memory_order_acquire);
            if (cc >= next_gpu_target) {
                gpu_pipeline.run_once();
                const auto sig = gpu_pipeline.last_signal();

                const std::size_t n_routed = alpha.evaluate(sig, lob_soa, top_edges);
                for (std::size_t r = 0U; r < n_routed; ++r) {
                    const auto &re = top_edges[r];
                    pnl.record(
                        re,
                        lob_soa.mid_price(re.src_instrument),
                        lob_soa.mid_price(re.dst_instrument),
                        lob_soa.spread(re.src_instrument),
                        sig);
                }
                next_gpu_target += 2048U;
            } else {
                if (lob_done.load(std::memory_order_acquire) &&
                    cc < next_gpu_target) break;
                __builtin_ia32_pause();
            }
        }
    });

    lob_thread.join();
    gpu_thread.join();

    const double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();

    pnl.print_report(elapsed);
    pnl.flush_csv("../data");
    return 0;
}
