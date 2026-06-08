#include <backtest_stats.hpp>
#include <binance_feed.hpp>
#include <csv_replay.hpp>
#include <cuda_pipeline.cuh>
#include <execution_engine.hpp>
#include <lob_core.hpp>
#include <lockfree_ring_buffer.hpp>
#include <memory_arena.hpp>
#include <signal_router.hpp>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static constexpr std::size_t k_arena_bytes       = 256ULL * 1024ULL * 1024ULL;
static constexpr std::size_t k_ring_capacity     = 1U << 17U; // 131072
static constexpr std::size_t k_equity_reserve    = 1U << 20U;
static constexpr std::size_t k_snapshot_interval = 512U;
static constexpr double      k_initial_equity    = 1'000'000.0;
static constexpr double      k_bars_per_year     = 252.0 * 390.0;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr,
            "Usage: holographic_backtest <csv_path> [max_instruments]\n"
            "CSV format: timestamp_ns,instrument_id,bid_price,bid_qty,ask_price,ask_qty\n");
        return EXIT_FAILURE;
    }

    const char* csv_path = argv[1];
    const std::size_t n_instruments = (argc >= 3)
        ? static_cast<std::size_t>(std::atoi(argv[2]))
        : holo::k_feed_n_instruments;

    std::printf("[backtest] arena=%zu MB  ring=%zu  instruments=%zu\n",
        k_arena_bytes / (1024ULL * 1024ULL), k_ring_capacity, n_instruments);

    holo::MemoryArena arena{k_arena_bytes};

    holo::DynamicSpscRingBuffer<holo::LobUpdate> ring{arena, k_ring_capacity};

    holo::LobSoA lob_soa{arena, n_instruments, holo::k_max_depth};

    holo::CsvReplayHandler replay{ring};
    if (!replay.open(csv_path))
    {
        std::fprintf(stderr, "[backtest] FATAL: cannot mmap '%s': %s\n",
            csv_path, std::strerror(errno));
        return EXIT_FAILURE;
    }

    holo::cuda::CudaPipeline cuda_pipeline{lob_soa, arena, 0};
    holo::SignalRouter        signal_router{n_instruments};
    holo::ExecutionEngine     exec_engine{"", "", 100'000.0F, 500.0F};

    std::vector<double> equity_curve;
    equity_curve.reserve(k_equity_reserve);
    equity_curve.push_back(k_initial_equity);

    std::atomic<bool> pipeline_shutdown{false};

    // Consumer thread: drain ring → apply LOB → run CUDA → route → paper-trade.
    std::thread consumer([&]()
    {
        holo::LobUpdate upd{};
        std::uint64_t   tick_count = 0U;

        holo::SignalRouter::TopKBuffer top_k{};

        while (true)
        {
            const bool got = ring.try_pop(upd);

            if (!got)
            {
                if (replay.done() && ring.empty()) break;
                __builtin_ia32_pause();
                continue;
            }

            lob_soa.apply(upd);
            ++tick_count;

            if ((tick_count & (k_snapshot_interval - 1U)) == 0U)
            {
                cuda_pipeline.run_once();

                const holo::cuda::SignalRecord sig = cuda_pipeline.last_signal();

                // Retrieve host-side harmonic/curl arrays from pipeline.
                // The pipeline exposes them via the CUDA pipeline internals;
                // here we use the last signal metadata to drive paper PnL.
                // Full edge routing requires access to h_curl_out_/h_edge arrays
                // which are pipeline-internal — we derive a scalar proxy trade.
                const float ym = sig.yang_mills_action;

                if (ym > holo::k_signal_min_curl)
                {
                    // Scalar paper-trade: long instrument 0 on positive YM signal.
                    // A production wiring would call signal_router.route() with
                    // the full curl_flow/edge_src/edge_dst spans exposed by pipeline.
                    const float mid0 = lob_soa.mid_price(0U);
                    const float mid1 = (n_instruments > 1U) ? lob_soa.mid_price(1U) : mid0;

                    holo::RoutedEdge synthetic{};
                    synthetic.src_instrument    = 0U;
                    synthetic.dst_instrument    = static_cast<std::uint32_t>(
                        n_instruments > 1U ? 1U : 0U);
                    synthetic.harmonic_flow     = sig.max_curl;
                    synthetic.yang_mills_action = ym;
                    synthetic.signal_ts_ns      = sig.timestamp_ns;
                    synthetic.edge_index        = 0U;

                    exec_engine.submit(synthetic, mid0, mid1);
                }

                // Snapshot equity.
                float mids_buf[holo::k_max_instruments];
                for (std::size_t i = 0U; i < n_instruments && i < holo::k_max_instruments; ++i)
                    mids_buf[i] = lob_soa.mid_price(i);

                const std::span<const float> mids_span{mids_buf, n_instruments};
                const double eq = k_initial_equity +
                    static_cast<double>(exec_engine.total_unrealized_pnl(mids_span));
                equity_curve.push_back(eq);
            }
        }

        pipeline_shutdown.store(true, std::memory_order_release);
    });

    replay.start();
    consumer.join();

    exec_engine.stop();

    // Replay diagnostics.
    {
        const auto& rm = replay.metrics();
        std::printf("[replay]  lines_parsed=%llu  parse_errors=%llu  ring_drops=%llu\n",
            static_cast<unsigned long long>(
                rm.lines_parsed.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                rm.parse_errors.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                rm.ring_drops.load(std::memory_order_relaxed)));
    }

    {
        const auto& pm = cuda_pipeline.metrics();
        std::printf("[cuda]    transfers=%llu  decomps=%llu  signals=%llu  "
                    "last_ym=%.6f  harmonic_dims=%d\n",
            static_cast<unsigned long long>(
                pm.n_transfers.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                pm.n_decompositions.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                pm.n_arbitrage_signals.load(std::memory_order_relaxed)),
            static_cast<double>(
                pm.last_ym_action.load(std::memory_order_relaxed)),
            pm.last_harmonic_dims.load(std::memory_order_relaxed));
    }

    {
        const auto& sm = signal_router.metrics();
        std::printf("[router]  processed=%llu  routed=%llu  suppressed=%llu\n",
            static_cast<unsigned long long>(
                sm.signals_processed.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                sm.edges_routed.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                sm.signals_suppressed.load(std::memory_order_relaxed)));
    }

    exec_engine.print_risk_summary();

    if (equity_curve.size() < 2U)
    {
        std::fprintf(stderr,
            "[backtest] Insufficient equity samples (%zu) — no trades generated.\n",
            equity_curve.size());
        return EXIT_SUCCESS;
    }

    const holo::BacktestResult result = holo::BacktestEvaluator::evaluate(
        std::span<const double>{equity_curve.data(), equity_curve.size()},
        std::span<const double>{},
        k_bars_per_year);

    holo::BacktestEvaluator::print_brutalist_summary(result);

    return EXIT_SUCCESS;
}