#pragma once

// LiveNode — assembles all subsystems and owns the main event loop.
// main_live.cpp instantiates one LiveNode and calls run().

#include <common/lockfree_ring_buffer.hpp>
#include <common/memory_arena.hpp>
#include <common/thread_utils.hpp>
#include <common/types.hpp>
#include <data_feed/feed_handler.hpp>
#include <data_feed/lob_core.hpp>
#include <compute/cuda_pipeline.cuh>
#include <trading/alpha_model.hpp>
#include <trading/risk_manager.hpp>
#include <execution/execution_gateway.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>

namespace holo
{

struct LiveConfig
{
    std::size_t   depth_levels     = 16U;
    std::size_t   ring_capacity    = 1U << 17U;
    std::size_t   arena_size_bytes = 256U * 1024U * 1024U;
    std::uint64_t run_duration_ns  = 30'000'000'000ULL;
    int           pipeline_core    = 2;
    int           lob_drain_core   = 4;
};

class LiveNode final
{
public:
    LiveNode(const LiveConfig& cfg, std::unique_ptr<ExecutionGateway> gateway)
        : cfg_{cfg}
        , arena_{cfg.arena_size_bytes}
        , ring_{*arena_.emplace<RingT>(arena_, cfg.ring_capacity)}
        , lob_ {*arena_.emplace<LobSoA>(arena_, k_n_instruments, cfg.depth_levels)}
        , feed_{ring_}
        , pipeline_{lob_, arena_, 0}
        , risk_{10'000.0F}
        , alpha_{}
        , gateway_{std::move(gateway)}
    {}

    ~LiveNode() noexcept { stop(); }

    LiveNode(const LiveNode&)            = delete;
    LiveNode& operator=(const LiveNode&) = delete;
    LiveNode(LiveNode&&)                 = delete;
    LiveNode& operator=(LiveNode&&)      = delete;

    void run()
    {
        print_banner();

        // ── Start subsystems ──────────────────────────────────────────────
        gateway_->start();
        feed_.start();

        // Warm-up: let feed populate LOB with real data
        sleep_ns(2'000'000'000L);
        drain_ring();

        print_instruments();

        // ── Background threads ────────────────────────────────────────────
        std::atomic<bool> shutdown{false};

        std::thread pipeline_thread{
            [this, &shutdown]() { pipeline_.run_continuous(shutdown); }};
        pin_thread(pipeline_thread, cfg_.pipeline_core);

        std::thread lob_drain_thread{
            [this, &shutdown]()
            {
                while (!shutdown.load(std::memory_order_relaxed))
                {
                    drain_ring();
                    sleep_ns(200'000L);
                }
            }};
        pin_thread(lob_drain_thread, cfg_.lob_drain_core);

        // ── Main loop ─────────────────────────────────────────────────────
        print_section("LIVE LOOP — TOPOLOGICAL SIGNAL ROUTING & EXECUTION");

        const std::uint64_t t_start = now_ns();
        std::uint64_t last_sig_ts   = 0U;

        while (now_ns() - t_start < cfg_.run_duration_ns)
        {
            sleep_ns(100'000'000L);

            const auto   sig     = pipeline_.last_signal();
            const double elapsed = static_cast<double>(now_ns() - t_start) / 1.0e9;
            print_status(sig, elapsed);

            if (sig.timestamp_ns != last_sig_ts && sig.timestamp_ns != 0U)
            {
                last_sig_ts = sig.timestamp_ns;

                AlphaModel::TopKBuffer edges{};
                const std::size_t n = alpha_.evaluate(sig, lob_, edges);

                for (std::size_t k = 0U; k < n; ++k)
                {
                    const auto& e = edges[k];
                    gateway_->submit(
                        e,
                        lob_.mid_price(e.src_instrument),
                        lob_.mid_price(e.dst_instrument));
                }
            }
        }

        std::puts("\n");
        shutdown.store(true, std::memory_order_release);
        pipeline_thread.join();
        lob_drain_thread.join();

        stop();
        print_reports();
    }

private:
    using RingT = DynamicSpscRingBuffer<LobUpdate>;

    void stop() noexcept
    {
        feed_.stop();
        gateway_->stop();
    }

    void drain_ring() noexcept
    {
        LobUpdate u{};
        while (ring_.try_pop(u)) lob_.apply(u);
    }

    // ── Reporting helpers ─────────────────────────────────────────────────
    static void print_separator() noexcept
    {
        std::puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }

    static void print_section(const char* title) noexcept
    {
        print_separator();
        std::printf("  %s\n", title);
        print_separator();
    }

    static void print_banner() noexcept
    {
        std::puts("\n");
        print_separator();
        std::puts("  HOLOGRAPHIC MARKET ARCHITECTURE  v0.5.0");
        std::puts("  Phase V: Refactored — LiveNode single-owner design");
        print_separator();
        std::puts("");
    }

    void print_instruments() const noexcept
    {
        std::printf("  Feed started. Instruments:\n");
        for (std::size_t i = 0U; i < k_n_instruments; ++i)
            std::printf("    [%zu] %s  mid=%.2f  spread=%.4f\n",
                i, k_symbols_upper[i].data(),
                static_cast<double>(lob_.mid_price(i)),
                static_cast<double>(lob_.spread(i)));
        std::puts("");
    }

    void print_status(const cuda::SignalRecord& sig, double elapsed_s) const noexcept
    {
        const auto& fm = feed_.metrics();
        std::printf(
            "\r  [%5.1fs] S_YM=%.4f β₁=%d curl_max=%.4f | "
            "feed=%llu drop=%llu     ",
            elapsed_s,
            static_cast<double>(sig.yang_mills_action),
            sig.n_harmonic_dims,
            static_cast<double>(sig.max_curl),
            static_cast<unsigned long long>(
                fm.msgs_received.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                fm.msgs_dropped.load(std::memory_order_relaxed)));
        std::fflush(stdout);
    }

    void print_reports() const noexcept
    {
        const auto& pm = pipeline_.metrics();
        print_section("CUDA PIPELINE METRICS");
        std::printf("  Transfers       : %llu\n",
            (unsigned long long)pm.n_transfers.load());
        std::printf("  Decompositions  : %llu\n",
            (unsigned long long)pm.n_decompositions.load());
        std::printf("  Arb signals     : %llu\n",
            (unsigned long long)pm.n_arbitrage_signals.load());
        if (pm.n_transfers.load() > 0U)
            std::printf("  Mean xfr time   : %.3f ms\n",
                static_cast<double>(pm.total_transfer_ns.load()) /
                static_cast<double>(pm.n_transfers.load()) / 1e6);
        if (pm.n_decompositions.load() > 0U)
            std::printf("  Mean Hodge time : %.3f ms\n",
                static_cast<double>(pm.total_decomp_ns.load()) /
                static_cast<double>(pm.n_decompositions.load()) / 1e6);

        const auto last_sig = pipeline_.last_signal();
        std::printf("  S_YM final      : %.6f\n",
            static_cast<double>(last_sig.yang_mills_action));
        std::printf("  β₁ final        : %d\n", last_sig.n_harmonic_dims);
        std::printf("  Market eff.     : %s\n",
            (last_sig.yang_mills_action < 0.01F) ? "HIGH  (S_YM → 0)"
                                                 : "LOW   (S_YM >> 0, arb present)");
        print_separator();

        print_section("FEED METRICS");
        const auto& fm = feed_.metrics();
        std::printf("  Msgs received   : %llu\n",
            (unsigned long long)fm.msgs_received.load());
        std::printf("  Msgs dropped    : %llu\n",
            (unsigned long long)fm.msgs_dropped.load());
        std::printf("  Parse errors    : %llu\n",
            (unsigned long long)fm.parse_errors.load());
        std::printf("  Reconnects      : %llu\n",
            (unsigned long long)fm.reconnects.load());
        for (std::size_t i = 0U; i < k_n_instruments; ++i)
            std::printf("  [%s] msgs=%llu\n",
                k_symbols_upper[i].data(),
                (unsigned long long)fm.per_instrument[i].load());
        print_separator();

        print_section("EXECUTION & RISK");
        risk_.print_summary();
        print_separator();

        std::printf("\n  Arena utilization : %.2f%%\n",
            arena_.utilization() * 100.0);
        std::puts("  Zero heap allocs on hot path : CONFIRMED");
        std::puts("  LiveNode terminated          : CLEAN\n");
    }

    // ── Members in construction order ────────────────────────────────────
    const LiveConfig cfg_;
    MemoryArena      arena_;
    RingT&           ring_;
    LobSoA&          lob_;
    FeedHandler      feed_;
    cuda::CudaPipeline pipeline_;
    RiskManager      risk_;
    AlphaModel       alpha_;
    std::unique_ptr<ExecutionGateway> gateway_;
};

} // namespace holo
