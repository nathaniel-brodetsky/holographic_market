#pragma once

#include <cuda_pipeline.cuh>
#include <hodge_kernel.cuh>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace holo
{

static constexpr std::size_t k_signal_router_top_k = 3U;
static constexpr float       k_signal_min_curl      = 1e-4F;

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

class SignalRouter final
{
public:
    using TopKBuffer = std::array<RoutedEdge, k_signal_router_top_k>;

    explicit SignalRouter(std::size_t n_instruments) noexcept
        : n_instruments_{n_instruments}
    {}

    SignalRouter(const SignalRouter&) = delete;
    SignalRouter& operator=(const SignalRouter&) = delete;
    SignalRouter(SignalRouter&&) = delete;
    SignalRouter& operator=(SignalRouter&&) = delete;

    // Returns number of valid edges written into out_buf.
    [[nodiscard]] std::size_t route(
        const cuda::SignalRecord& sig,
        std::span<const float>   h_curl_flow,
        std::span<const int>     h_edge_src,
        std::span<const int>     h_edge_dst,
        TopKBuffer&              out_buf) noexcept
    {
        metrics_.signals_processed.fetch_add(1U, std::memory_order_relaxed);

        if (sig.yang_mills_action < k_signal_min_curl)
        {
            metrics_.signals_suppressed.fetch_add(1U, std::memory_order_relaxed);
            return 0U;
        }

        const std::size_t n_edges = h_curl_flow.size();
        if (n_edges == 0U) return 0U;

        // Partial sort: find top-K by |harmonic_flow|.
        // Zero-allocation: use fixed-size scratch on stack.
        struct Candidate
        {
            float        abs_flow;
            std::size_t  idx;
        };

        std::array<Candidate, k_signal_router_top_k> top{};
        for (auto& c : top) { c.abs_flow = -1.0F; c.idx = 0U; }

        for (std::size_t e = 0U; e < n_edges; ++e)
        {
            const float af = (h_curl_flow[e] < 0.0F)
                ? -h_curl_flow[e] : h_curl_flow[e];

            // Insert into top-K min-heap (tiny, k<=3 → linear scan).
            std::size_t min_pos = 0U;
            for (std::size_t t = 1U; t < k_signal_router_top_k; ++t)
                if (top[t].abs_flow < top[min_pos].abs_flow)
                    min_pos = t;

            if (af > top[min_pos].abs_flow)
            {
                top[min_pos] = {af, e};
            }
        }

        // Sort top descending.
        std::sort(top.begin(), top.end(),
            [](const Candidate& a, const Candidate& b) noexcept
            { return a.abs_flow > b.abs_flow; });

        std::size_t written = 0U;
        for (std::size_t t = 0U; t < k_signal_router_top_k; ++t)
        {
            if (top[t].abs_flow < k_signal_min_curl) break;

            const std::size_t e = top[t].idx;
            if (e >= h_edge_src.size()) break;

            RoutedEdge& re = out_buf[written];
            re.src_instrument    = static_cast<std::uint32_t>(h_edge_src[e]);
            re.dst_instrument    = static_cast<std::uint32_t>(h_edge_dst[e]);
            re.harmonic_flow     = h_curl_flow[e];
            re.yang_mills_action = sig.yang_mills_action;
            re.signal_ts_ns      = sig.timestamp_ns;
            re.edge_index        = static_cast<std::uint32_t>(e);
            re._pad              = 0U;
            ++written;
        }

        metrics_.edges_routed.fetch_add(written, std::memory_order_relaxed);
        return written;
    }

    [[nodiscard]] const RouterMetrics& metrics() const noexcept { return metrics_; }

private:
    const std::size_t n_instruments_;
    RouterMetrics     metrics_;
};

} // namespace holo