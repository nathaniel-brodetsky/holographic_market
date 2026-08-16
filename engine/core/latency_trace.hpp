#pragma once

// Stage-by-stage latency measurement for the live tick-to-order path.
//
// Design constraints, deliberately:
//   - record() must be cheap enough to call on every tick without becoming
//     part of the latency problem it's measuring: no locks, no heap, no
//     syscalls other than the clock read itself.
//   - Log2-bucketed histogram, not a running mean/stddev: latency
//     distributions here are heavy-tailed (network jitter, GC-less but
//     scheduler-driven CPU jitter, occasional GPU stalls), and a mean
//     hides exactly the tail behavior we're trying to find. p50/p99/max
//     from real buckets is honest; a mean is not.
//   - Same clock source as CudaPipeline's pipeline_now_ns() so every stage
//     boundary's timestamps are directly comparable/subtractable.
//
// Usage:
//   const std::uint64_t t0 = holo::core::latency_now_ns();
//   ... do the thing ...
//   holo::core::g_latency.record(holo::core::Stage::WsRecvToRingPush,
//                                 holo::core::latency_now_ns() - t0);
//
// Periodically (e.g. once every few seconds from one thread), call
// g_latency.dump(std::cerr) to print p50/p90/p99/max per stage and reset.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <array>
#include <string>
#include <string_view>

#if defined(__linux__)
#include <time.h>
#endif

namespace holo::core
{

[[nodiscard]] inline std::uint64_t latency_now_ns() noexcept
{
#if defined(__linux__)
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
         + static_cast<std::uint64_t>(ts.tv_nsec);
#else
    return 0U;
#endif
}

// Add new stages here as more of the pipeline gets instrumented. Keep
// COUNT last -- it sizes the histogram's stage array.
enum class Stage : std::uint32_t
{
    WsRecvToRingPush = 0U,   // socket read complete -> LobUpdate pushed to ring
    RingPopToLobApply,       // ring pop -> LobSoA updated (drain_thread)
    GpuRunOnce,               // CudaPipeline::run_once() wall time
    SignalToOrderSend,        // SignalRecord.timestamp_ns -> co_spawn(on_signal)
    COUNT
};

[[nodiscard]] constexpr std::string_view stage_name(Stage s) noexcept
{
    switch (s)
    {
        case Stage::WsRecvToRingPush:  return "ws_recv_to_ring_push";
        case Stage::RingPopToLobApply: return "ring_pop_to_lob_apply";
        case Stage::GpuRunOnce:        return "gpu_run_once";
        case Stage::SignalToOrderSend: return "signal_to_order_send";
        default:                       return "unknown";
    }
}

class LatencyHistogram final
{
public:
    // Buckets are log2 of the nanosecond value: bucket 0 = [1ns,2ns),
    // bucket 10 = [1024ns,2048ns) i.e. ~1-2us, bucket 20 = ~1-2ms,
    // bucket 30 = ~1-2s. 40 buckets covers 1ns..~1100s, comfortably wide
    // for anything this pipeline should ever see; anything that would
    // overflow it is already a bug worth seeing in the max, not a
    // histogram sizing problem.
    static constexpr std::size_t k_buckets = 40U;

    void record(Stage stage, std::uint64_t ns) noexcept
    {
        const std::size_t s = static_cast<std::size_t>(stage);
        if (s >= static_cast<std::size_t>(Stage::COUNT)) return;

        const std::size_t bucket = bucket_for(ns);
        buckets_[s][bucket].fetch_add(1U, std::memory_order_relaxed);
        count_[s].fetch_add(1U, std::memory_order_relaxed);

        // max_ is not perfectly race-free under concurrent writers (a
        // classic read-compare-CAS-retry loop would be), but a lost
        // update here only means we occasionally under-report the max
        // by one sample out of however many arrive in that instant --
        // acceptable for a monitoring histogram, not for correctness-
        // critical state.
        std::uint64_t prev = max_[s].load(std::memory_order_relaxed);
        while (ns > prev &&
               !max_[s].compare_exchange_weak(prev, ns, std::memory_order_relaxed))
        {
        }
    }

    // Prints p50/p90/p99/max per stage (only stages with samples since
    // the last dump), then resets counters. Call this from exactly one
    // thread, periodically -- it's not meant to be called concurrently
    // with itself.
    void dump(std::FILE* out) noexcept
    {
        std::fprintf(out, "[latency] ---- stage    p50        p90        p99        max        n ----\n");
        for (std::size_t s = 0U; s < static_cast<std::size_t>(Stage::COUNT); ++s)
        {
            const std::uint64_t n = count_[s].load(std::memory_order_relaxed);
            if (n == 0U) continue;

            const std::uint64_t p50 = percentile_ns(s, n, 0.50);
            const std::uint64_t p90 = percentile_ns(s, n, 0.90);
            const std::uint64_t p99 = percentile_ns(s, n, 0.99);
            const std::uint64_t mx  = max_[s].load(std::memory_order_relaxed);

            std::fprintf(out, "[latency] %-22s %-10s %-10s %-10s %-10s %llu\n",
                std::string(stage_name(static_cast<Stage>(s))).c_str(),
                format_ns(p50).c_str(), format_ns(p90).c_str(),
                format_ns(p99).c_str(), format_ns(mx).c_str(),
                static_cast<unsigned long long>(n));

            for (auto& b : buckets_[s]) b.store(0U, std::memory_order_relaxed);
            count_[s].store(0U, std::memory_order_relaxed);
            max_[s].store(0U, std::memory_order_relaxed);
        }
    }

private:
    [[nodiscard]] static std::size_t bucket_for(std::uint64_t ns) noexcept
    {
        if (ns < 1U) return 0U;
        std::size_t b = 0U;
        // ns bit-length, clamped to k_buckets-1. Cheap, branch-predictable
        // for the common case (small latencies -> small bit-lengths).
        while (ns > 1U && b < k_buckets - 1U) { ns >>= 1U; ++b; }
        return b;
    }

    [[nodiscard]] std::uint64_t percentile_ns(std::size_t stage, std::uint64_t n, double p) const noexcept
    {
        const std::uint64_t target = static_cast<std::uint64_t>(static_cast<double>(n) * p);
        std::uint64_t cum = 0U;
        for (std::size_t b = 0U; b < k_buckets; ++b)
        {
            cum += buckets_[stage][b].load(std::memory_order_relaxed);
            if (cum > target) return (1ULL << b);
        }
        return (1ULL << (k_buckets - 1U));
    }

    [[nodiscard]] static std::string format_ns(std::uint64_t ns)
    {
        char buf[24];
        if (ns < 1'000ULL)                std::snprintf(buf, sizeof(buf), "%lluns", (unsigned long long)ns);
        else if (ns < 1'000'000ULL)       std::snprintf(buf, sizeof(buf), "%.1fus", ns / 1'000.0);
        else if (ns < 1'000'000'000ULL)   std::snprintf(buf, sizeof(buf), "%.2fms", ns / 1'000'000.0);
        else                              std::snprintf(buf, sizeof(buf), "%.2fs", ns / 1'000'000'000.0);
        return std::string(buf);
    }

    std::array<std::array<std::atomic<std::uint64_t>, k_buckets>,
               static_cast<std::size_t>(Stage::COUNT)> buckets_{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Stage::COUNT)> count_{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Stage::COUNT)> max_{};
};

// One process-wide instance. `inline` (C++17) gives this a single
// definition across translation units without a separate .cpp -- same
// pattern as header-only usage elsewhere in core/.
inline LatencyHistogram g_latency;

} // namespace holo::core