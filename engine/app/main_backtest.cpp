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
#include <string_view>
#include <thread>
#include <vector>
#include <fstream>
#include <filesystem>

using namespace holo;

struct CliArgs {
    std::string   csv_path;
    ReplayMode    mode{ReplayMode::Fastest};
    std::uint32_t throttle_rate{100'000U};
    int           gpu_device{0};
    std::size_t   arena_mb{512U};
};

static void print_usage(const char* argv0) noexcept {
    std::printf("Usage: %s <lob_data.csv>\n", argv0);
}

static CliArgs parse_args(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); std::exit(1); }
    CliArgs a;
    a.csv_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string_view arg{argv[i]};
        if (arg == "--mode" && i + 1 < argc) {
            std::string_view m{argv[++i]};
            if      (m == "realtime")  a.mode = ReplayMode::Realtime;
            else if (m == "throttled") a.mode = ReplayMode::Throttled;
            else                       a.mode = ReplayMode::Fastest;
        }
        else if (arg == "--rate" && i + 1 < argc) a.throttle_rate = std::atoi(argv[++i]);
        else if (arg == "--gpu" && i + 1 < argc) a.gpu_device = std::atoi(argv[++i]);
        else if (arg == "--arena" && i + 1 < argc) a.arena_mb = std::atoi(argv[++i]);
    }
    return a;
}

struct PnlTracker {
    std::vector<double> equity_curve{0.0};
    double              current_pnl{0.0};
    double              sum_ret{0.0};
    double              sum_ret2{0.0};
    std::uint64_t       n{0U};
    std::uint64_t       wins{0U};

    void record(const RoutedEdge& edge, float mid_src, float mid_dst) noexcept {
        if (mid_src <= 0.0F || mid_dst <= 0.0F) return;

        const float spread = (mid_dst - mid_src) / mid_src;
        const float ret = edge.harmonic_flow * spread * 1e4F;

        sum_ret  += ret;
        sum_ret2 += ret * ret;
        ++n;

        if (ret > 0.0F) ++wins;

        current_pnl += ret;
        equity_curve.push_back(current_pnl);
    }

    void print_report(double elapsed_s) const noexcept {
        std::puts("\n══════════════════════════════════════════════");
        std::puts("  BACKTEST REPORT");
        std::puts("══════════════════════════════════════════════");
        if (n == 0U) { std::puts("  No signals generated."); return; }

        const double mean = sum_ret / static_cast<double>(n);
        const double var  = (sum_ret2 / static_cast<double>(n)) - (mean * mean);
        const double stdv = (var > 0.0) ? std::sqrt(var) : 1.0;
        const double sharpe = mean / stdv;
        const double win_rate = static_cast<double>(wins) / static_cast<double>(n) * 100.0;

        std::printf("  Signals generated : %llu\n",  static_cast<unsigned long long>(n));
        std::printf("  Win rate          : %.1f%%\n", win_rate);
        std::printf("  Mean return (bps) : %.4f\n",  mean);
        std::printf("  Std dev  (bps)    : %.4f\n",  stdv);
        std::printf("  Sharpe (per-sig)  : %.4f\n",  sharpe);
        std::printf("  Elapsed           : %.2f s\n", elapsed_s);
        std::puts("══════════════════════════════════════════════\n");
    }
};

int main(int argc, char** argv) {
    const CliArgs cli = parse_args(argc, argv);
    MemoryArena arena{cli.arena_mb * 1024UL * 1024UL};

    static constexpr std::size_t k_n_instruments = 4U;
    LobSoA lob_soa{arena, k_n_instruments, 1U};
    DynamicSpscRingBuffer<LobUpdate> ring{arena, 1'048'576U};

    CsvReplayHandler replay{ring, cli.csv_path, ReplayConfig{cli.mode, cli.throttle_rate, 500U}};
    holo::cuda::CudaPipeline gpu_pipeline{lob_soa, arena, cli.gpu_device};

    SignalRouter router{k_n_instruments};
    PnlTracker   pnl;
    SignalRouter::TopKBuffer top_edges{};

    std::vector<int> edge_src, edge_dst;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (i != j) { edge_src.push_back(i); edge_dst.push_back(j); }

    const auto t_start = std::chrono::steady_clock::now();
    replay.start();
    std::puts("[backtest] Replay started. Processing...");

    std::atomic<std::uint64_t> lob_update_count{0U};
    std::atomic<bool>          lob_done{false};

    std::thread lob_thread([&]() {
        std::uint64_t local_count = 0;
        LobUpdate u{};
        while (true) {
            if (ring.try_pop(u)) {
                lob_soa.apply(u);
                ++local_count;
                if (local_count % 128U == 0U) lob_update_count.store(local_count, std::memory_order_release);
            } else {
                if (replay.finished()) break;
                __builtin_ia32_pause();
            }
        }
        while (ring.try_pop(u)) { lob_soa.apply(u); ++local_count; }
        lob_update_count.store(local_count, std::memory_order_release);
        lob_done.store(true, std::memory_order_release);
    });

    std::thread gpu_thread([&]() {
        std::uint64_t next_gpu_target = 2048U;
        while (true) {
            const std::uint64_t current_count = lob_update_count.load(std::memory_order_acquire);
            if (current_count >= next_gpu_target) {
                gpu_pipeline.run_once();
                const holo::cuda::SignalRecord sig = gpu_pipeline.last_signal();

                if (sig.yang_mills_action > 1e-4F) {
                    std::vector<float> h_curl_flow(12, sig.yang_mills_action / 12.0F);
                    const std::size_t n_routed = router.route(
                        sig, std::span<const float>{h_curl_flow}, 
                        std::span<const int>{edge_src}, std::span<const int>{edge_dst}, top_edges);

                    for (std::size_t r = 0U; r < n_routed; ++r) {
                        const auto& re = top_edges[r];
                        pnl.record(re, lob_soa.mid_price(re.src_instrument), lob_soa.mid_price(re.dst_instrument));
                    }
                }
                next_gpu_target += 2048U;
            } else {
                if (lob_done.load(std::memory_order_acquire) && current_count < next_gpu_target) break;
                __builtin_ia32_pause();
            }
        }
    });

    lob_thread.join();
    gpu_thread.join();

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    pnl.print_report(elapsed);

    // =========================================================================
    // ТОТ САМЫЙ БЛОК СОХРАНЕНИЯ CSV (ТЕПЕРЬ ОН ТУТ 100%)
    // =========================================================================
    {
        std::filesystem::create_directories("../data");
        std::ofstream eq_file("../data/equity_curve.csv", std::ios::trunc);
        if (eq_file) {
            eq_file << "step,equity\n";
            for (std::size_t i = 0U; i < pnl.equity_curve.size(); ++i)
                eq_file << i << ',' << pnl.equity_curve[i] << '\n';
            std::printf("  [SUCCESS] Equity curve saved to data/equity_curve.csv (%zu points)\n", pnl.equity_curve.size());
        }
    }

    return 0;
}
