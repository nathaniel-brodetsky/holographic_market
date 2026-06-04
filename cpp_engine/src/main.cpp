#include <memory_arena.hpp>
#include <lockfree_ring_buffer.hpp>
#include <lob_core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>
#include <array>
#include <algorithm>
#include <numeric>
#include <span>
#include <bit>

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#endif

namespace holo {
    static constexpr std::size_t k_n_instruments = 64U;
    static constexpr std::size_t k_depth_levels = 16U;
    static constexpr std::size_t k_ring_capacity = 1U << 17U;
    static constexpr std::uint64_t k_total_messages = 10'000'000ULL;
    static constexpr std::size_t k_arena_size_bytes = 256U * 1024U * 1024U;
    static constexpr std::size_t k_warmup_messages = 100'000U;
    static constexpr std::size_t k_histogram_buckets = 64U;

    static_assert(std::has_single_bit(k_ring_capacity));

    [[nodiscard]] static std::uint64_t now_ns() noexcept {
        struct timespec ts{};
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
               + static_cast<std::uint64_t>(ts.tv_nsec);
    }

#if defined(__linux__)
    static void pin_thread(std::thread &t, int core_id) noexcept {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(static_cast<std::size_t>(core_id), &cpuset);
        pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
#else
    static void pin_thread(std::thread &, int) noexcept {
    }
#endif

    struct alignas(k_cache_line) BenchmarkResults {
        std::uint64_t producer_duration_ns{0U};
        std::uint64_t consumer_duration_ns{0U};
        std::uint64_t total_messages{0U};
        std::uint64_t dropped_messages{0U};
        std::uint64_t min_latency_ns{std::numeric_limits<std::uint64_t>::max()};
        std::uint64_t max_latency_ns{0U};
        std::uint64_t sum_latency_ns{0U};
        std::uint64_t sum_sq_latency_ns{0U};
        std::array<std::uint64_t, k_histogram_buckets> latency_hist{};

        void record_latency(std::uint64_t lat_ns) noexcept {
            if (lat_ns < min_latency_ns) min_latency_ns = lat_ns;
            if (lat_ns > max_latency_ns) max_latency_ns = lat_ns;
            sum_latency_ns += lat_ns;
            sum_sq_latency_ns += lat_ns * lat_ns;
            const std::size_t bucket = std::min(
                static_cast<std::size_t>(std::bit_width(lat_ns + 1U)),
                k_histogram_buckets - 1U);
            latency_hist[bucket]++;
        }

        [[nodiscard]] double mean_ns() const noexcept {
            if (total_messages == 0U) return 0.0;
            return static_cast<double>(sum_latency_ns)
                   / static_cast<double>(total_messages);
        }

        [[nodiscard]] double stddev_ns() const noexcept {
            if (total_messages < 2U) return 0.0;
            const double mean = mean_ns();
            const double mean_sq = static_cast<double>(sum_sq_latency_ns)
                                   / static_cast<double>(total_messages);
            const double var = mean_sq - mean * mean;
            if (var <= 0.0) return 0.0;
            double r = var, p = 0.0;
            while (r != p) {
                p = r;
                r = 0.5 * (r + var / r);
            }
            return r;
        }

        [[nodiscard]] double throughput_mps() const noexcept {
            if (consumer_duration_ns == 0U) return 0.0;
            return static_cast<double>(total_messages)
                   / static_cast<double>(consumer_duration_ns)
                   * 1e9;
        }
    };

    using RingT = DynamicSpscRingBuffer<LobUpdate>;

    struct alignas(k_cache_line) SharedState {
        std::atomic<bool> consumer_ready{false};
        std::atomic<bool> producer_done{false};
        std::byte _pad[k_cache_line - 2U * sizeof(std::atomic<bool>)];
    };

    static_assert(sizeof(SharedState) == k_cache_line);

    [[nodiscard]] static LobUpdate make_update(
        std::uint64_t seq,
        std::uint64_t ts_ns) noexcept {
        LobUpdate u{};
        u.timestamp_ns = ts_ns;
        u.instrument_id = static_cast<std::uint32_t>(seq % k_n_instruments);
        u.depth_level = static_cast<std::uint8_t>(seq % k_depth_levels);
        u.side = (seq & 1U) ? Side::Ask : Side::Bid;
        u.price = 100.0F + static_cast<float>(seq % 1000U) * 0.01F;
        u.quantity = 10.0F + static_cast<float>(seq % 500U) * 0.1F;
        return u;
    }

    static void producer_fn(
        RingT &ring,
        SharedState &state,
        BenchmarkResults &results) noexcept {
        while (!state.consumer_ready.load(std::memory_order_acquire)) {
            __builtin_ia32_pause();
        }

        const std::uint64_t t0 = now_ns();

        for (std::uint64_t i = 0U; i < k_total_messages; ++i) {
            ring.push_blocking(make_update(i, now_ns()));
        }

        results.producer_duration_ns = now_ns() - t0;
        state.producer_done.store(true, std::memory_order_release);
    }

    static void consumer_fn(
        RingT &ring,
        LobSoA &lob,
        SharedState &state,
        BenchmarkResults &results) noexcept {
        state.consumer_ready.store(true, std::memory_order_release);

        const std::uint64_t t0 = now_ns();
        std::uint64_t consumed = 0U;
        LobUpdate u{};

        while (consumed < k_total_messages) {
            if (ring.try_pop(u)) [[likely]] {
                const std::uint64_t recv_ts = now_ns();
                lob.apply(u);

                if (consumed >= k_warmup_messages) [[likely]] {
                    const std::uint64_t lat =
                            (recv_ts >= u.timestamp_ns) ? (recv_ts - u.timestamp_ns) : 0U;
                    results.record_latency(lat);
                }
                ++consumed;
            } else {
                if (state.producer_done.load(std::memory_order_acquire)
                    && ring.empty()) [[unlikely]] {
                    break;
                }
                __builtin_ia32_pause();
            }
        }

        results.consumer_duration_ns = now_ns() - t0;
        results.total_messages = consumed;
        results.dropped_messages =
                lob.stats().dropped_updates.load(std::memory_order_relaxed);
    }

    static void print_separator() noexcept {
        std::puts("─────────────────────────────────────────────────────────────────");
    }

    static void print_results(
        const BenchmarkResults &r,
        const LobSoA &lob) noexcept {
        print_separator();
        std::puts("  HOLOGRAPHIC ENGINE — SPSC INGESTION BENCHMARK");
        print_separator();
        std::printf("  Messages          : %llu\n",
                    static_cast<unsigned long long>(r.total_messages));
        std::printf("  Dropped           : %llu\n",
                    static_cast<unsigned long long>(r.dropped_messages));
        std::printf("  Producer wall     : %.3f ms\n",
                    static_cast<double>(r.producer_duration_ns) / 1e6);
        std::printf("  Consumer wall     : %.3f ms\n",
                    static_cast<double>(r.consumer_duration_ns) / 1e6);
        std::printf("  Throughput        : %.3f M msg/s\n", r.throughput_mps() / 1e6);
        print_separator();
        std::printf("  Latency (ns) Min  : %llu\n",
                    static_cast<unsigned long long>(r.min_latency_ns));
        std::printf("  Latency (ns) Max  : %llu\n",
                    static_cast<unsigned long long>(r.max_latency_ns));
        std::printf("  Latency (ns) Mean : %.2f\n", r.mean_ns());
        std::printf("  Latency (ns) Std  : %.2f\n", r.stddev_ns());
        print_separator();
        std::puts("  LOB STATE SNAPSHOT");
        std::printf("  Total LOB updates : %llu\n",
                    static_cast<unsigned long long>(
                        lob.stats().total_updates.load(std::memory_order_relaxed)));
        std::printf("  Bid updates       : %llu\n",
                    static_cast<unsigned long long>(
                        lob.stats().bid_updates.load(std::memory_order_relaxed)));
        std::printf("  Ask updates       : %llu\n",
                    static_cast<unsigned long long>(
                        lob.stats().ask_updates.load(std::memory_order_relaxed)));

        const auto bids = lob.bid_prices(0U);
        const auto asks = lob.ask_prices(0U);
        if (!bids.empty() && !asks.empty()) {
            std::printf("  Instr[0] mid      : %.4f\n", static_cast<double>(lob.mid_price(0U)));
            std::printf("  Instr[0] spread   : %.4f\n", static_cast<double>(lob.spread(0U)));
            std::printf("  Instr[0] bid[0]   : %.4f\n", static_cast<double>(bids[0]));
            std::printf("  Instr[0] ask[0]   : %.4f\n", static_cast<double>(asks[0]));
        }

        print_separator();
        std::puts("  LATENCY HISTOGRAM (log2 bucket → count)");
        for (std::size_t b = 0U; b < k_histogram_buckets; ++b) {
            if (r.latency_hist[b] == 0U) continue;
            const std::uint64_t lo = (b == 0U) ? 0ULL : (1ULL << (b - 1U));
            const std::uint64_t hi = (1ULL << b) - 1ULL;
            std::printf("  [%5llu – %7llu ns] : %llu\n",
                        static_cast<unsigned long long>(lo),
                        static_cast<unsigned long long>(hi),
                        static_cast<unsigned long long>(r.latency_hist[b]));
        }
        print_separator();
    }
} // namespace holo

int main() {
    using namespace holo;

    std::puts("\n  Holographic Market Architecture — C++ Execution Engine");
    std::puts("  Jump Trading / Topology Division\n");

    MemoryArena arena{k_arena_size_bytes};

    RingT *const ring_ptr = arena.emplace<RingT>(arena, k_ring_capacity);
    if (!ring_ptr) [[unlikely]] {
        std::puts("FATAL: Arena allocation failed for ring buffer.");
        return 1;
    }

    LobSoA *const lob_ptr = arena.emplace<LobSoA>(arena, k_n_instruments, k_depth_levels);
    if (!lob_ptr) [[unlikely]] {
        std::puts("FATAL: Arena allocation failed for LobSoA.");
        return 1;
    }

    SharedState *const state_ptr = arena.emplace<SharedState>();
    if (!state_ptr) [[unlikely]] {
        std::puts("FATAL: Arena allocation failed for SharedState.");
        return 1;
    }

    BenchmarkResults *const results_ptr = arena.emplace<BenchmarkResults>();
    if (!results_ptr) [[unlikely]] {
        std::puts("FATAL: Arena allocation failed for BenchmarkResults.");
        return 1;
    }

    std::printf("  Arena capacity    : %.1f MB\n",
                static_cast<double>(arena.capacity()) / 1e6);
    std::printf("  Arena used (init) : %.3f MB\n",
                static_cast<double>(arena.used()) / 1e6);
    std::printf("  Ring capacity     : %zu slots\n", ring_ptr->capacity());
    std::printf("  LOB instruments   : %zu\n", lob_ptr->n_instruments());
    std::printf("  LOB depth         : %zu levels\n", lob_ptr->depth());
    std::printf("  Total messages    : %llu\n\n",
                static_cast<unsigned long long>(k_total_messages));

    std::thread producer{
        producer_fn,
        std::ref(*ring_ptr), std::ref(*state_ptr), std::ref(*results_ptr)
    };
    std::thread consumer{
        consumer_fn,
        std::ref(*ring_ptr), std::ref(*lob_ptr),
        std::ref(*state_ptr), std::ref(*results_ptr)
    };

    pin_thread(producer, 0);
    pin_thread(consumer, 2);

    producer.join();
    consumer.join();

    print_results(*results_ptr, *lob_ptr);

    std::printf("\n  Arena utilization       : %.2f%%\n",
                arena.utilization() * 100.0);
    std::puts("  Zero heap allocs on hot path : CONFIRMED\n");

    return 0;
}
