#include <core/memory_arena.hpp>
#include <core/lockfree_ring_buffer.hpp>
#include <core/lob_core.hpp>
#include <math/cuda_pipeline.cuh>
#include <net/csv_replay.hpp>
#include <net/signal_router.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
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

struct EwmaPair {
    double mu{0.0};
    double m2{1e-8};
    bool   warm{false};
    static constexpr double k_alpha         = 0.02;
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

        const double log_ratio = std::log(static_cast<double>(mid_dst)
                                        / static_cast<double>(mid_src));
        EwmaPair &bl = baselines[src * 4U + dst];
        const double z_score = bl.update(log_ratio);

        if (!bl.warm) return;

        const float signal_strength = std::tanh(edge.harmonic_flow);
        const float sigma_bps   = bl.sigma_bps();
        const float z_f         = static_cast<float>(z_score);
        const float direction   = -std::copysign(1.0F, z_f) * std::copysign(1.0F, signal_strength);
        const float gross_ret   = std::min(
            std::abs(signal_strength) * std::abs(z_f) * sigma_bps,
            EwmaPair::k_max_gross_bps);

        const float spread_bps = (spread_price / mid_src) * 1e4F;

        if (gross_ret <= spread_bps || std::abs(z_f) < 0.5F) {
            ++filtered; return;
        }

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
    DynamicSpscRingBuffer<LobUpdate> ring{arena, 8'388'608U};
    CsvReplayHandler replay{ring, cli.csv_path,
                            ReplayConfig{cli.mode, cli.throttle_rate, 500U}};
    holo::cuda::CudaPipeline gpu_pipeline{lob_soa, arena, cli.gpu_device};

    SignalRouter             router{4U};
    PnlTracker               pnl;
    SignalRouter::TopKBuffer top_edges{};

    const auto t_start = std::chrono::steady_clock::now();
    replay.start();
    std::puts("[backtest] Processing...");

    // Deterministic single-consumer loop: the CSV replay thread (inside
    // CsvReplayHandler) is the only producer feeding the ring buffer, and
    // this loop is the only consumer. Every 2048th applied LOB update
    // triggers exactly one gpu_pipeline.run_once() call, in file order,
    // regardless of how fast/slow the GPU happens to run on this particular
    // execution. This removes the previous two-thread race where a slow
    // GPU pass (e.g. under compute-sanitizer, or just system jitter) caused
    // whole batches of market updates to be silently skipped, making the
    // backtest's signal count and win rate non-reproducible run to run.
    constexpr std::uint64_t k_gpu_batch = 2048U;

    std::uint64_t local_count = 0U;
    LobUpdate     u{};

    const auto process_signal = [&]() {
        gpu_pipeline.run_once();
        const auto sig = gpu_pipeline.last_signal();

        if (sig.yang_mills_action > 1e-4F) {
            const auto topo = gpu_pipeline.last_topology();

            if (topo.n_edges > 0 && topo.harmonic_flow != nullptr) {
                const std::size_t n_routed = router.route(
                    sig,
                    std::span<const float>{topo.harmonic_flow, static_cast<std::size_t>(topo.n_edges)},
                    std::span<const int>{topo.edge_src,        static_cast<std::size_t>(topo.n_edges)},
                    std::span<const int>{topo.edge_dst,        static_cast<std::size_t>(topo.n_edges)},
                    top_edges);

                for (std::size_t r = 0U; r < n_routed; ++r) {
                    const auto &re = top_edges[r];
                    pnl.record(
                        re,
                        lob_soa.mid_price(re.src_instrument),
                        lob_soa.mid_price(re.dst_instrument),
                        lob_soa.spread(re.src_instrument),
                        sig);
                }
            }
        }
    };

    while (!replay.finished() || !ring.empty()) {
        if (ring.try_pop(u)) {
            lob_soa.apply(u);
            ++local_count;
            if (local_count % k_gpu_batch == 0U)
                process_signal();
        } else {
            __builtin_ia32_pause();
        }
    }

    // Flush the trailing partial batch (< k_gpu_batch updates) instead of
    // silently dropping it, so the very end of the file is still analyzed.
    if (local_count % k_gpu_batch != 0U)
        process_signal();

    const double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();

    pnl.print_report(elapsed);
    const auto &rm = replay.metrics();
    std::printf("  Rows read        : %llu\n", static_cast<unsigned long long>(rm.rows_read.load()));
    std::printf("  Updates dropped  : %llu\n", static_cast<unsigned long long>(rm.updates_dropped.load()));
    pnl.flush_csv("../data");
    return 0;
}