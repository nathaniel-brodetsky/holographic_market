#include <core/memory_arena.hpp>
#include <core/lockfree_ring_buffer.hpp>
#include <core/lob_core.hpp>
#include <math/cuda_pipeline.cuh>
#include <net/csv_replay.hpp>
#include <net/signal_router.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <filesystem>

using namespace holo;

struct CliArgs {
    std::string csv_path;
    ReplayMode  mode{ReplayMode::Fastest};
    uint32_t    throttle_rate{100'000U};
    int         gpu_device{0};
    size_t      arena_mb{512U};
};

static CliArgs parse_args(int argc, char** argv) {
    CliArgs a; a.csv_path = argv[1]; return a;
}

struct TickRecord {
    uint64_t timestamp_ns;
    double   pnl;
    double   cumulative_pnl;
    float    yang_mills_action;
    float    max_curl;
    float    floer_asd_residual;
    int      betti_1;
    float    spread_bps;
};

struct PnlTracker {
    std::vector<double>     equity_curve{0.0};
    std::vector<TickRecord> ticks;
    double                  current_pnl{0.0};
    double                  sum_ret{0.0};
    double                  sum_ret2{0.0};
    uint64_t                n{0U};
    uint64_t                wins{0U};

    void record(const RoutedEdge& edge, float mid_src, float mid_dst, const holo::cuda::SignalRecord& sig) noexcept {
        if (mid_src <= 0.0F || mid_dst <= 0.0F) return;
        const float spread = (mid_dst - mid_src) / mid_src;
        const float ret = edge.harmonic_flow * spread * 1e4F;
        sum_ret += ret; sum_ret2 += ret * ret; ++n;
        if (ret > 0.0F) ++wins;
        current_pnl += ret;
        equity_curve.push_back(current_pnl);

        ticks.push_back(TickRecord{
            .timestamp_ns       = sig.timestamp_ns,
            .pnl                = ret,
            .cumulative_pnl     = current_pnl,
            .yang_mills_action  = sig.yang_mills_action,
            .max_curl           = sig.max_curl,
            .floer_asd_residual = sig.floer_asd_residual,
            .betti_1            = sig.floer_HF1,
            .spread_bps         = (mid_src > 0.0F) ? std::abs(spread) * 1e4F : 0.0F
        });
    }

    void print_report(double elapsed_s) const noexcept {
        const double mean = sum_ret / static_cast<double>(n);
        const double var  = (sum_ret2 / static_cast<double>(n)) - (mean * mean);
        const double stdv = (var > 0.0) ? std::sqrt(var) : 1.0;
        const double sharpe = mean / stdv;
        std::printf("\n  Signals: %llu | Win Rate: %.1f%% | Sharpe: %.4f | Elapsed: %.2fs\n", 
                    (unsigned long long)n, (double)wins / n * 100.0, sharpe, elapsed_s);
    }

    void flush_csv() const {
        std::filesystem::create_directories("../data");
        std::ofstream f("../data/advanced_metrics.csv", std::ios::trunc);
        if (f) {
            f << "timestamp_ns,pnl,cumulative_pnl,yang_mills_action,max_curl,floer_asd_residual,betti_1,spread_bps\n";
            for (const auto& t : ticks) {
                f << t.timestamp_ns << ',' << t.pnl << ',' << t.cumulative_pnl << ',' 
                  << t.yang_mills_action << ',' << t.max_curl << ',' << t.floer_asd_residual << ',' 
                  << t.betti_1 << ',' << t.spread_bps << '\n';
            }
            std::printf("  [SUCCESS] advanced_metrics.csv saved (%zu ticks)\n", ticks.size());
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    const CliArgs cli = parse_args(argc, argv);
    MemoryArena arena{cli.arena_mb * 1024UL * 1024UL};

    LobSoA lob_soa{arena, 4U, 1U};
    DynamicSpscRingBuffer<LobUpdate> ring{arena, 1'048'576U};
    CsvReplayHandler replay{ring, cli.csv_path, ReplayConfig{cli.mode, cli.throttle_rate, 500U}};
    holo::cuda::CudaPipeline gpu_pipeline{lob_soa, arena, cli.gpu_device};

    SignalRouter router{4U};
    PnlTracker pnl;
    SignalRouter::TopKBuffer top_edges{};

    std::vector<int> edge_src, edge_dst;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) { edge_src.push_back(i); edge_dst.push_back(j); }

    const auto t_start = std::chrono::steady_clock::now();
    replay.start();
    std::puts("[backtest] Processing...");

    std::atomic<uint64_t> lob_update_count{0U};
    std::atomic<bool> lob_done{false};

    std::thread lob_thread([&]() {
        uint64_t local_count = 0; LobUpdate u{};
        while (!replay.finished() || !ring.empty()) {
            if (ring.try_pop(u)) {
                lob_soa.apply(u); ++local_count;
                if (local_count % 128U == 0U) lob_update_count.store(local_count, std::memory_order_release);
            } else { __builtin_ia32_pause(); }
        }
        lob_update_count.store(local_count, std::memory_order_release);
        lob_done.store(true, std::memory_order_release);
    });

    std::thread gpu_thread([&]() {
        uint64_t next_gpu_target = 2048U;
        while (true) {
            const uint64_t current_count = lob_update_count.load(std::memory_order_acquire);
            if (current_count >= next_gpu_target) {
                gpu_pipeline.run_once();
                const auto sig = gpu_pipeline.last_signal();

                if (sig.yang_mills_action > 1e-4F) {
                    std::vector<float> h_curl_flow(12, sig.yang_mills_action / 12.0F);
                    size_t n_routed = router.route(sig, std::span<const float>{h_curl_flow}, 
                                                   std::span<const int>{edge_src}, std::span<const int>{edge_dst}, top_edges);
                    for (size_t r = 0U; r < n_routed; ++r) {
                        pnl.record(top_edges[r], lob_soa.mid_price(top_edges[r].src_instrument), lob_soa.mid_price(top_edges[r].dst_instrument), sig);
                    }
                }
                next_gpu_target += 2048U;
            } else {
                if (lob_done.load(std::memory_order_acquire) && current_count < next_gpu_target) break;
                __builtin_ia32_pause();
            }
        }
    });

    lob_thread.join(); gpu_thread.join();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    
    pnl.print_report(elapsed);
    pnl.flush_csv();
    return 0;
}
