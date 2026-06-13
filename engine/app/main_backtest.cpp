// app/main_backtest.cpp
// Phase VI: historical LOB CSV replay → full GPU pipeline → PnL / Sharpe report.
//
// Usage:
//   holographic_backtest <lob_data.csv> [--mode fastest|realtime|throttled]
//                                       [--rate <msgs_per_sec>]
//                                       [--gpu <device_id>]
//                                       [--arena <mb>]

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

using namespace holo;

// ── CLI parsing ───────────────────────────────────────────────────────────────

struct CliArgs
{
    std::string   csv_path;
    ReplayMode    mode{ReplayMode::Fastest};
    std::uint32_t throttle_rate{100'000U};
    int           gpu_device{0};
    std::size_t   arena_mb{512U};
};

static void print_usage(const char* argv0) noexcept
{
    std::printf(
        "Usage: %s <lob_data.csv>\n"
        "       [--mode fastest|realtime|throttled]\n"
        "       [--rate <msgs_per_sec>]     (throttled mode only, default 100000)\n"
        "       [--gpu  <device_id>]        (default 0)\n"
        "       [--arena <mb>]              (memory arena size, default 512)\n",
        argv0);
}

static CliArgs parse_args(int argc, char** argv)
{
    if (argc < 2) { print_usage(argv[0]); std::exit(1); }

    CliArgs a;
    a.csv_path = argv[1];

    for (int i = 2; i < argc; ++i)
    {
        std::string_view arg{argv[i]};

        if (arg == "--mode" && i + 1 < argc)
        {
            std::string_view m{argv[++i]};
            if      (m == "realtime")  a.mode = ReplayMode::Realtime;
            else if (m == "throttled") a.mode = ReplayMode::Throttled;
            else                       a.mode = ReplayMode::Fastest;
        }
        else if (arg == "--rate" && i + 1 < argc)
            a.throttle_rate = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        else if (arg == "--gpu" && i + 1 < argc)
            a.gpu_device = std::atoi(argv[++i]);
        else if (arg == "--arena" && i + 1 < argc)
            a.arena_mb = static_cast<std::size_t>(std::atoi(argv[++i]));
        else
        {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]); std::exit(1);
        }
    }
    return a;
}

// ── PnL / Sharpe accumulator ──────────────────────────────────────────────────
// Simplified mark-to-market: each routed signal is treated as a
// notional trade. We track return per signal and compute Sharpe at the end.
// No actual order book position sizing — that belongs to Phase VII.

struct PnlTracker
{
    void record(const RoutedEdge& edge, float mid_src, float mid_dst) noexcept
    {
        if (mid_src <= 0.0F || mid_dst <= 0.0F) return;

        // Proxy return: normalised curl flow weighted by mid-price spread
        const float spread  = (mid_dst - mid_src) / mid_src;
        const float ret     = edge.harmonic_flow * spread * 1e4F;  // bps

        sum_ret  += ret;
        sum_ret2 += ret * ret;
        ++n;

        if (ret > 0.0F) ++wins;
    }

    void print_report(double elapsed_s) const noexcept
    {
        std::puts("\n══════════════════════════════════════════════");
        std::puts("  BACKTEST REPORT");
        std::puts("══════════════════════════════════════════════");

        if (n == 0U)
        {
            std::puts("  No signals generated.");
            return;
        }

        const double mean = sum_ret / static_cast<double>(n);
        const double var  = (sum_ret2 / static_cast<double>(n)) - (mean * mean);
        const double stdv = (var > 0.0) ? std::sqrt(var) : 1.0;
        const double sharpe = mean / stdv;         // per-signal, not annualised
        const double win_rate = static_cast<double>(wins) / static_cast<double>(n) * 100.0;

        std::printf("  Signals generated : %llu\n",  static_cast<unsigned long long>(n));
        std::printf("  Win rate          : %.1f%%\n", win_rate);
        std::printf("  Mean return (bps) : %.4f\n",  mean);
        std::printf("  Std dev  (bps)    : %.4f\n",  stdv);
        std::printf("  Sharpe (per-sig)  : %.4f\n",  sharpe);
        std::printf("  Elapsed           : %.2f s\n", elapsed_s);
        std::puts("══════════════════════════════════════════════\n");
    }

    double        sum_ret{0.0};
    double        sum_ret2{0.0};
    std::uint64_t n{0U};
    std::uint64_t wins{0U};
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    const CliArgs cli = parse_args(argc, argv);

    std::printf("\n[holographic_backtest] file   : %s\n", cli.csv_path.c_str());
    std::printf("[holographic_backtest] mode   : %s\n",
        cli.mode == ReplayMode::Realtime  ? "realtime"  :
        cli.mode == ReplayMode::Throttled ? "throttled" : "fastest");
    std::printf("[holographic_backtest] gpu    : %d\n", cli.gpu_device);
    std::printf("[holographic_backtest] arena  : %zu MB\n\n", cli.arena_mb);

    // ── Allocate arena ───────────────────────────────────────────────────────

    MemoryArena arena{cli.arena_mb * 1024UL * 1024UL};

    // ── LOB state ────────────────────────────────────────────────────────────

    static constexpr std::size_t k_n_instruments = 4U;
    static constexpr std::size_t k_depth         = 1U;   // bookTicker = best bid/ask only

    LobSoA lob_soa{arena, k_n_instruments, k_depth};

    // ── Ring buffer ──────────────────────────────────────────────────────────

    static constexpr std::size_t k_ring_capacity = 1'048'576U;
    DynamicSpscRingBuffer<LobUpdate> ring{arena, k_ring_capacity};

    // ── CSV replay handler ───────────────────────────────────────────────────

    ReplayConfig replay_cfg;
    replay_cfg.mode         = cli.mode;
    replay_cfg.msgs_per_sec = cli.throttle_rate;

    CsvReplayHandler replay{ring, cli.csv_path, replay_cfg};

    // ── GPU pipeline ─────────────────────────────────────────────────────────

    holo::cuda::CudaPipeline gpu_pipeline{lob_soa, arena, cli.gpu_device};

    // ── Signal router + PnL tracker ──────────────────────────────────────────

    SignalRouter router{k_n_instruments};
    PnlTracker   pnl;

    SignalRouter::TopKBuffer top_edges{};

    // ── Start replay ─────────────────────────────────────────────────────────

    const auto t_start = std::chrono::steady_clock::now();

    replay.start();
    std::puts("[backtest] Replay started. Processing...");

    // ── Two-thread architecture ───────────────────────────────────────────────
    
    std::atomic<std::uint64_t> lob_update_count{0U};
    std::atomic<bool>          lob_done{false};

    static constexpr std::uint64_t k_gpu_trigger = 2048U;
    static constexpr std::size_t k_n_edges = k_n_instruments * (k_n_instruments - 1U);
    std::vector<int> edge_src; edge_src.reserve(k_n_edges);
    std::vector<int> edge_dst; edge_dst.reserve(k_n_edges);
    for (int i = 0; i < static_cast<int>(k_n_instruments); ++i)
        for (int j = 0; j < static_cast<int>(k_n_instruments); ++j)
            if (i != j) { edge_src.push_back(i); edge_dst.push_back(j); }

    // Thread 1: Drain ring buffer into LOB SoA as fast as possible
    std::thread lob_thread([&]() {
        std::uint64_t local_count = 0;
        while (true)
        {
            LobUpdate u{};
            if (ring.try_pop(u))
            {
                lob_soa.apply(u);
                ++local_count;
                
                // Batch atomic updates to prevent cache-line bouncing with GPU thread
                if (local_count % 128U == 0U)
                    lob_update_count.store(local_count, std::memory_order_release);
            }
            else
            {
                if (replay.finished()) break;
                __builtin_ia32_pause(); // Spin yield
            }
        }
        
        // Drain any remaining updates
        LobUpdate u{};
        while (ring.try_pop(u)) { lob_soa.apply(u); ++local_count; }
        
        lob_update_count.store(local_count, std::memory_order_release);
        lob_done.store(true, std::memory_order_release);
    });

    // Thread 2: Run GPU Pipeline asynchronously 
    std::thread gpu_thread([&]() {
        std::uint64_t next_gpu_target = k_gpu_trigger;
        while (true)
        {
            const std::uint64_t current_count = lob_update_count.load(std::memory_order_acquire);
            if (current_count >= next_gpu_target)
            {
                gpu_pipeline.run_once();
                const holo::cuda::SignalRecord sig = gpu_pipeline.last_signal();

                std::vector<float> h_curl_flow(k_n_edges, sig.yang_mills_action / static_cast<float>(k_n_edges));

                const std::size_t n_routed = router.route(
                    sig, std::span<const float>{h_curl_flow}, 
                    std::span<const int>{edge_src}, std::span<const int>{edge_dst}, top_edges);

                for (std::size_t r = 0U; r < n_routed; ++r)
                {
                    const auto& re = top_edges[r];
                    pnl.record(re, lob_soa.mid_price(re.src_instrument), lob_soa.mid_price(re.dst_instrument));
                }
                next_gpu_target += k_gpu_trigger;
            }
            else
            {
                if (lob_done.load(std::memory_order_acquire) && current_count < next_gpu_target) break;
                __builtin_ia32_pause();
            }
        }
    });

    lob_thread.join();
    gpu_thread.join();

    // Drain any remaining updates after replay EOF
    {
        LobUpdate u{};
        while (ring.try_pop(u)) lob_soa.apply(u);
    }

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    // ── Print results ────────────────────────────────────────────────────────

    const auto& rm = replay.metrics();
    const auto& gm = gpu_pipeline.metrics();
    const auto& sm = router.metrics();

    std::puts("\n──────────────────────────────────────────────");
    std::puts("  REPLAY STATS");
    std::puts("──────────────────────────────────────────────");
    std::printf("  Rows read         : %llu\n",
        static_cast<unsigned long long>(rm.rows_read.load()));
    std::printf("  LOB updates pushed: %llu\n",
        static_cast<unsigned long long>(rm.updates_pushed.load()));
    std::printf("  LOB updates dropped: %llu\n",
        static_cast<unsigned long long>(rm.updates_dropped.load()));
    std::printf("  Parse errors      : %llu\n",
        static_cast<unsigned long long>(rm.parse_errors.load()));
    std::printf("  GPU cycles        : %llu\n",
        static_cast<unsigned long long>(gm.n_decompositions.load()));
    std::printf("  Signals routed    : %llu\n",
        static_cast<unsigned long long>(sm.edges_routed.load()));
    std::printf("  Signals suppressed: %llu\n",
        static_cast<unsigned long long>(sm.signals_suppressed.load()));

    pnl.print_report(elapsed);

    return 0;
}
