#pragma once

#include <math/cuda_pipeline.cuh>
#include <math/hodge_kernel.cuh>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace holo::net
{

static constexpr std::size_t k_signal_router_top_k = 3U;

struct alignas(32) RoutedEdge
{
    std::uint32_t src_instrument{0U};
    std::uint32_t dst_instrument{0U};
    float         harmonic_flow{0.0F};
    float         yang_mills_action{0.0F};
    std::uint64_t signal_ts_ns{0U};
    std::uint32_t edge_index{0U};
    std::uint32_t _pad{0U};
};
static_assert(sizeof(RoutedEdge) == 32U);

struct alignas(64) RouterMetrics
{
    std::atomic<std::uint64_t> signals_processed{0U};
    std::atomic<std::uint64_t> edges_routed{0U};
    std::atomic<std::uint64_t> signals_suppressed{0U};
    std::byte _pad[64U - 3U * sizeof(std::atomic<std::uint64_t>)]{};
};
static_assert(sizeof(RouterMetrics) == 64U);

static float active_threshold() noexcept { if (const char* env = std::getenv("HOLO_THRESHOLD")) return std::strtof(env, nullptr); return 10.0f; }

class SignalRouter final
{
public:
    using TopKBuffer = std::array<RoutedEdge, k_signal_router_top_k>;

    explicit SignalRouter(std::size_t n_instruments) noexcept
        : n_instruments_{n_instruments} {}

    SignalRouter(const SignalRouter&)            = delete;
    SignalRouter& operator=(const SignalRouter&) = delete;
    SignalRouter(SignalRouter&&)                 = delete;
    SignalRouter& operator=(SignalRouter&&)      = delete;

    [[nodiscard]] std::size_t route(
        const cuda::SignalRecord& sig,
        std::span<const float>   h_curl_flow,
        std::span<const int>     h_edge_src,
        std::span<const int>     h_edge_dst,
        TopKBuffer&              out_buf) noexcept
    {
        metrics_.signals_processed.fetch_add(1U, std::memory_order_relaxed);

        if (sig.yang_mills_action < active_threshold())
        {
            metrics_.signals_suppressed.fetch_add(1U, std::memory_order_relaxed);
            return 0U;
        }

        const std::size_t n_edges = h_curl_flow.size();
        if (n_edges == 0U) return 0U;

        // Instrument-disjoint top-K. Picking the raw top-K edges by
        // strength (the old approach) can select several edges that all
        // share one instrument (e.g. BTC is in 3 of the 6 edges of a
        // 4-instrument complete graph) -- those then race each other to
        // the exchange, all priced off the same local book snapshot for
        // that shared instrument. By the time the second/third order
        // arrives, the first may have already moved the real book, so
        // they're no longer valid post-only prices -- this is what was
        // producing repeated "could not be executed as maker" rejects
        // and tripping the circuit breaker, even after adding a
        // one-tick pricing safety margin (see main_live.cpp
        // safe_maker_price()) -- the margin doesn't help when the
        // *same* instrument gets re-priced by a second in-flight order
        // a couple milliseconds later.
        //
        // Fix: greedily fill each of the k_signal_router_top_k output
        // slots with the strongest remaining edge whose BOTH endpoints
        // are not yet claimed by an edge already selected in this same
        // call. Guarantees every edge in one routed batch touches
        // disjoint instruments. O(top_k * n_edges), no allocation --
        // n_edges is small (6 for today's 4-instrument complete graph),
        // so this is cheap even though it's less clever than the old
        // running-min-slot scan.
        static constexpr std::uint32_t kNoInstrument = 0xFFFFFFFFU;
        std::array<std::uint32_t, k_signal_router_top_k * 2U> used_instruments{};
        for (auto& u : used_instruments) u = kNoInstrument;
        std::size_t n_used = 0U;

        const auto is_used = [&](std::uint32_t instr) noexcept {
            for (std::size_t i = 0U; i < n_used; ++i)
                if (used_instruments[i] == instr) return true;
            return false;
        };

        std::size_t written = 0U;
        for (std::size_t slot = 0U; slot < k_signal_router_top_k; ++slot)
        {
            float       best_af  = 0.0F;
            std::size_t best_idx = n_edges;
            bool        found    = false;

            for (std::size_t e = 0U; e < n_edges; ++e)
            {
                const float af = (h_curl_flow[e] < 0.0F) ? -h_curl_flow[e] : h_curl_flow[e];
                if (af < active_threshold()) continue;
                if (e >= h_edge_src.size() || e >= h_edge_dst.size()) continue;

                const auto src = static_cast<std::uint32_t>(h_edge_src[e]);
                const auto dst = static_cast<std::uint32_t>(h_edge_dst[e]);
                if (is_used(src) || is_used(dst)) continue;

                if (!found || af > best_af)
                {
                    best_af  = af;
                    best_idx = e;
                    found    = true;
                }
            }

            if (!found) break;

            const std::size_t e = best_idx;
            auto& re = out_buf[written];
            re.src_instrument    = static_cast<std::uint32_t>(h_edge_src[e]);
            re.dst_instrument    = static_cast<std::uint32_t>(h_edge_dst[e]);
            re.harmonic_flow     = h_curl_flow[e];
            re.yang_mills_action = sig.yang_mills_action;
            re.signal_ts_ns      = sig.timestamp_ns;
            re.edge_index        = static_cast<std::uint32_t>(e);
            re._pad              = 0U;
            ++written;

            used_instruments[n_used++] = re.src_instrument;
            used_instruments[n_used++] = re.dst_instrument;
        }

        metrics_.edges_routed.fetch_add(written, std::memory_order_relaxed);
        return written;
    }

    [[nodiscard]] const RouterMetrics& metrics() const noexcept { return metrics_; }

private:
    const std::size_t n_instruments_;
    RouterMetrics     metrics_;
};

} // namespace holo::net