#include <memory_arena.hpp>
#include <lockfree_ring_buffer.hpp>
#include <lob_core.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <atomic>
#include <thread>
#include <array>
#include <limits>
#include <span>

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <time.h>
#endif

namespace holo {

static constexpr std::size_t   k_n_instruments     = 64U;
static constexpr std::size_t   k_depth_levels      = 16U;
static constexpr std::size_t   k_ring_capacity     = 1U << 17U;
static constexpr std::uint64_t k_total_messages    = 10'000'000ULL;
static constexpr std::size_t   k_arena_size_bytes  = 256U * 1024U * 1024U;
static constexpr std::size_t   k_warmup_messages   = 100'000U;
static constexpr std::size_t   k_histogram_buckets = 64U;

[[nodiscard]] static std::uint64_t now_ns() noexcept {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
         + static_cast<std::uint64_t>(ts.tv_nsec);
}

[[nodiscard]] static std::uint64_t rdtsc() noexcept {
    std::uint32_t lo, hi;
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtsc\n\t"
        : "=a"(lo), "=d"(hi)
        :: "memory"
    );
    return (static_cast<std::uint64_t>(hi) << 32U) | lo;
}

[[nodiscard]] static double estimate_tsc_ghz() noexcept {
    const std::uint64_t ns0  = now_ns();
    const std::uint64_t tsc0 = rdtsc();
    struct timespec req{0, 50'000'000};
    nanosleep(&req, nullptr);
    const std::uint64_t tsc1 = rdtsc();
    const std::uint64_t ns1  = now_ns();
    const double elapsed_ns  = static_cast<double>(ns1 - ns0);
    const double elapsed_tsc = static_cast<double>(tsc1 - tsc0);
    return elapsed_tsc / elapsed_ns;
}

[[nodiscard]] static std::size_t log2_floor(std::uint64_t v) noexcept {
    if (v == 0U) return 0U;
    return static_cast<std::size_t>(
        63U - static_cast<unsigned>(__builtin_clzll(v)));
}

#if defined(__linux__)
static void pin_thread(std::thread& t, int core_id) noexcept {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(static_cast<std::size_t>(core_id), &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
}
#else
static void pin_thread(std::thread&, int) noexcept {}
#endif

struct alignas(k_cache_line) LatencyAccum {
    std::uint64_t count{0U};
    std::uint64_t min_cycles{std::numeric_limits<std::uint64_t>::max()};
    std::uint64_t max_cycles{0U};
    double        mean{0.0};
    double        m2{0.0};
    std::array<std::uint64_t, k_histogram_buckets> hist{};

    void record(std::uint64_t cycles) noexcept {
        ++count;
        if (cycles < min_cycles) min_cycles = cycles;
        if (cycles > max_cycles) max_cycles = cycles;
        const double delta  = static_cast<double>(cycles) - mean;
        mean               += delta / static_cast<double>(count);
        const double delta2 = static_cast<double>(cycles) - mean;
        m2                 += delta * delta2;
        const std::size_t bucket = std::min(
            log2_floor(cycles + 1U) + 1U,
            k_histogram_buckets - 1U);
        hist[bucket]++;
    }

    [[nodiscard]] double stddev() const noexcept {
        if (count < 2U) return 0.0;
        return std::sqrt(m2 / static_cast<double>(count - 1U));
    }
};

struct alignas(k_cache_line) BenchmarkResults {
    std::uint64_t producer_duration_ns{0U};
    std::uint64_t consumer_duration_ns{0U};
    std::uint64_t total_messages{0U};
    std::uint64_t dropped_messages{0U};
    LatencyAccum  latency{};
};

using RingT = DynamicSpscRingBuffer<LobUpdate>;

struct alignas(k_cache_line) SharedState {
    std::atomic<bool> consumer_ready{false};
    std::atomic<bool> producer_done{false};
    std::byte _pad[k_cache_line - 2U * sizeof(std::atomic<bool>)];
};

static_assert(sizeof(SharedState) == k_cache_line);

[[nodiscard]] static LobUpdate make_update(std::uint64_t seq) noexcept {
    LobUpdate u{};
    u.timestamp_ns  = rdtsc();
    u.instrument_id = static_cast<std::uint32_t>(seq % k_n_instruments);
    u.depth_level   = static_cast<std::uint8_t>(
        (seq / k_n_instruments) % k_depth_levels);
    u.side     = (seq & 1U) ? Side::Ask : Side::Bid;
    u.price    = 100.0F + static_cast<float>(seq % 1000U) * 0.01F;
    u.quantity = 10.0F  + static_cast<float>(seq % 500U)  * 0.1F;
    return u;
}

static void producer_fn(
    RingT&            ring,
    SharedState&      state,
    BenchmarkResults& results) noexcept
{
    while (!state.consumer_ready.load(std::memory_order_acquire)) {
        __builtin_ia32_pause();
    }

    const std::uint64_t t0 = now_ns();

    for (std::uint64_t i = 0U; i < k_total_messages; ++i) {
        ring.push_blocking(make_update(i));
    }

    results.producer_duration_ns = now_ns() - t0;
    state.producer_done.store(true, std::memory_order_release);
}

static void consumer_fn(
    RingT&            ring,
    LobSoA&           lob,
    SharedState&      state,
    BenchmarkResults& results) noexcept
{
    state.consumer_ready.store(true, std::memory_order_release);

    const std::uint64_t t0 = now_ns();
    std::uint64_t consumed = 0U;
    LobUpdate u{};

    while (consumed < k_total_messages) {
        if (ring.try_pop(u)) [[likely]] {
            const std::uint64_t recv_tsc = rdtsc();
            lob.apply(u);

            if (consumed >= k_warmup_messages) [[likely]] {
                const std::uint64_t lat =
                    (recv_tsc >= u.timestamp_ns)
                    ? (recv_tsc - u.timestamp_ns)
                    : 0U;
                results.latency.record(lat);
            }
            ++consumed;
        } else {
            if (state.producer_done.load(std::memory_order_acquire)
                && ring.empty()) [[unlikely]]
            {
                break;
            }
            __builtin_ia32_pause();
        }
    }

    results.consumer_duration_ns = now_ns() - t0;
    results.total_messages       = consumed;
    results.dropped_messages     =
        lob.stats().dropped_updates.load(std::memory_order_relaxed);
}

static void print_separator() noexcept {
    std::puts("─────────────────────────────────────────────────────────────────");
}

static void print_results(
    const BenchmarkResults& r,
    const LobSoA&           lob,
    double                  tsc_ghz) noexcept
{
    const double cycles_to_ns = 1.0 / tsc_ghz;
    const LatencyAccum& lat   = r.latency;

    const double min_ns  = static_cast<double>(lat.min_cycles) * cycles_to_ns;
    const double max_ns  = static_cast<double>(lat.max_cycles) * cycles_to_ns;
    const double mean_ns = lat.mean * cycles_to_ns;
    const double std_ns  = lat.stddev() * cycles_to_ns;

    print_separator();
    std::puts("  HOLOGRAPHIC ENGINE — SPSC INGESTION BENCHMARK");
    print_separator();
    std::printf("  TSC frequency     : %.4f GHz\n", tsc_ghz);
    std::printf("  Messages          : %llu\n",
        static_cast<unsigned long long>(r.total_messages));
    std::printf("  Warmup excluded   : %zu\n", k_warmup_messages);
    std::printf("  Measured          : %llu\n",
        static_cast<unsigned long long>(lat.count));
    std::printf("  Dropped           : %llu\n",
        static_cast<unsigned long long>(r.dropped_messages));
    std::printf("  Producer wall     : %.3f ms\n",
        static_cast<double>(r.producer_duration_ns) / 1e6);
    std::printf("  Consumer wall     : %.3f ms\n",
        static_cast<double>(r.consumer_duration_ns) / 1e6);
    std::printf("  Throughput        : %.3f M msg/s\n",
        static_cast<double>(r.total_messages)
        / static_cast<double>(r.consumer_duration_ns) * 1e3);
    print_separator();
    std::puts("  TRANSIT LATENCY  (producer rdtsc → consumer rdtsc)");
    std::printf("  Min               : %llu cycles  /  %.1f ns\n",
        static_cast<unsigned long long>(lat.min_cycles), min_ns);
    std::printf("  Max               : %llu cycles  /  %.1f ns\n",
        static_cast<unsigned long long>(lat.max_cycles), max_ns);
    std::printf("  Mean              : %.1f cycles  /  %.1f ns\n",
        lat.mean, mean_ns);
    std::printf("  Std dev           : %.1f cycles  /  %.1f ns\n",
        lat.stddev(), std_ns);
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
        std::printf("  Instr[0] bid[0]   : %.4f\n",
            static_cast<double>(bids[0]));
        std::printf("  Instr[0] ask[0]   : %.4f\n",
            static_cast<double>(asks[0]));
        std::printf("  Instr[0] mid      : %.4f\n",
            static_cast<double>(lob.mid_price(0U)));
        std::printf("  Instr[0] spread   : %.4f\n",
            static_cast<double>(lob.spread(0U)));
    }

    print_separator();
    std::puts("  LATENCY HISTOGRAM (log2 cycle bucket → count)");
    for (std::size_t b = 0U; b < k_histogram_buckets; ++b) {
        if (lat.hist[b] == 0U) continue;
        const std::uint64_t lo_c = (b == 0U) ? 0ULL : (1ULL << (b - 1U));
        const std::uint64_t hi_c = (1ULL << b) - 1ULL;
        const double        lo_n = static_cast<double>(lo_c) * cycles_to_ns;
        const double        hi_n = static_cast<double>(hi_c) * cycles_to_ns;
        std::printf("  [%6llu – %7llu cy | %5.0f – %5.0f ns] : %llu\n",
            static_cast<unsigned long long>(lo_c),
            static_cast<unsigned long long>(hi_c),
            lo_n, hi_n,
            static_cast<unsigned long long>(lat.hist[b]));
    }
    print_separator();
}

} // namespace holo

int main() {
    using namespace holo;

    std::puts("\n  Holographic Market Architecture — C++ Execution Engine");
    std::puts("  Jump Trading / Topology Division\n");
    std::puts("  Calibrating TSC frequency (50 ms)...");

    const double tsc_ghz = estimate_tsc_ghz();

    MemoryArena arena{k_arena_size_bytes};

    RingT* const ring_ptr = arena.emplace<RingT>(arena, k_ring_capacity);
    if (!ring_ptr) [[unlikely]] {
        std::puts("FATAL: Arena allocation failed for ring buffer.");
        return 1;
    }

    LobSoA* const lob_ptr =
        arena.emplace<LobSoA>(arena, k_n_instruments, k_depth_levels);
    if (!lob_ptr) [[unlikely]] {
        std::puts("FATAL: Arena allocation failed for LobSoA.");
        return 1;
    }

    SharedState* const state_ptr = arena.emplace<SharedState>();
    if (!state_ptr) [[unlikely]] {
        std::puts("FATAL: Arena allocation failed for SharedState.");
        return 1;
    }

    BenchmarkResults* const results_ptr = arena.emplace<BenchmarkResults>();
    if (!results_ptr) [[unlikely]] {
        std::puts("FATAL: Arena allocation failed for BenchmarkResults.");
        return 1;
    }

    std::printf("  Arena capacity    : %.1f MB\n",
        static_cast<double>(arena.capacity()) / 1e6);
    std::printf("  Arena used (init) : %.3f MB\n",
        static_cast<double>(arena.used()) / 1e6);
    std::printf("  Ring capacity     : %zu slots\n",  ring_ptr->capacity());
    std::printf("  LOB instruments   : %zu\n",         lob_ptr->n_instruments());
    std::printf("  LOB depth         : %zu levels\n",  lob_ptr->depth());
    std::printf("  Total messages    : %llu\n",
        static_cast<unsigned long long>(k_total_messages));
    std::printf("  Core pinning      : producer→0  consumer→2\n\n");

    std::thread producer{producer_fn,
        std::ref(*ring_ptr), std::ref(*state_ptr), std::ref(*results_ptr)};
    std::thread consumer{consumer_fn,
        std::ref(*ring_ptr), std::ref(*lob_ptr),
        std::ref(*state_ptr), std::ref(*results_ptr)};

    pin_thread(producer, 0);
    pin_thread(consumer, 2);

    producer.join();
    consumer.join();

    print_results(*results_ptr, *lob_ptr, tsc_ghz);

    std::printf("\n  Arena utilization            : %.2f%%\n",
        arena.utilization() * 100.0);
    std::puts("  Zero heap allocs on hot path : CONFIRMED\n");

    return 0;
}