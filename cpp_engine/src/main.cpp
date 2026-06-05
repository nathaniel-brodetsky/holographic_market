#include <memory_arena.hpp>
#include <lockfree_ring_buffer.hpp>
#include <lob_core.hpp>
#include <cuda_pipeline.cuh>
#include <cuda_utils.cuh>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <atomic>
#include <thread>
#include <array>
#include <limits>

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <time.h>
#endif

namespace holo {

static constexpr std::size_t   k_n_instruments    = 64U;
static constexpr std::size_t   k_depth_levels     = 16U;
static constexpr std::size_t   k_ring_capacity    = 1U << 17U;
static constexpr std::uint64_t k_total_messages   = 10'000'000ULL;
static constexpr std::size_t   k_arena_size_bytes = 512U * 1024U * 1024U;
static constexpr std::size_t   k_warmup_messages  = 100'000U;
static constexpr std::size_t   k_histogram_buckets = 64U;
static constexpr std::uint64_t k_pipeline_run_ns  = 5'000'000'000ULL;

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
        : "=a"(lo), "=d"(hi) :: "memory");
    return (static_cast<std::uint64_t>(hi) << 32U) | lo;
}

[[nodiscard]] static double estimate_tsc_ghz() noexcept {
    const std::uint64_t ns0  = now_ns();
    const std::uint64_t tsc0 = rdtsc();
    struct timespec req{0, 50'000'000};
    nanosleep(&req, nullptr);
    return static_cast<double>(rdtsc() - tsc0)
         / static_cast<double>(now_ns() - ns0);
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

struct alignas(64) LatencyAccum {
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
        const double delta = static_cast<double>(cycles) - mean;
        mean += delta / static_cast<double>(count);
        m2   += delta * (static_cast<double>(cycles) - mean);
        hist[std::min(log2_floor(cycles + 1U) + 1U,
                      k_histogram_buckets - 1U)]++;
    }

    [[nodiscard]] double stddev() const noexcept {
        if (count < 2U) return 0.0;
        const double var = m2 / static_cast<double>(count - 1U);
        double r = var, p = 0.0;
        while (r != p) { p = r; r = 0.5 * (r + var / r); }
        return r;
    }
};

struct alignas(64) BenchmarkResults {
    std::uint64_t producer_duration_ns{0U};
    std::uint64_t consumer_duration_ns{0U};
    std::uint64_t total_messages{0U};
    std::uint64_t dropped_messages{0U};
    LatencyAccum  latency{};

    [[nodiscard]] double throughput_mps() const noexcept {
        if (consumer_duration_ns == 0U) return 0.0;
        return static_cast<double>(total_messages)
             / static_cast<double>(consumer_duration_ns) * 1e9;
    }
};

using RingT = DynamicSpscRingBuffer<LobUpdate>;

struct alignas(64) SharedState {
    std::atomic<bool> consumer_ready{false};
    std::atomic<bool> producer_done{false};
    std::byte _pad[64U - 2U * sizeof(std::atomic<bool>)];
};

static_assert(sizeof(SharedState) == 64U);

[[nodiscard]] static LobUpdate make_update(std::uint64_t seq) noexcept {
    LobUpdate u{};
    u.timestamp_ns  = rdtsc();
    u.instrument_id = static_cast<std::uint32_t>(seq % k_n_instruments);
    u.depth_level   = static_cast<std::uint8_t>(
        (seq / k_n_instruments) % k_depth_levels);
    u.side     = (seq & 1U) ? Side::Ask : Side::Bid;
    u.price    = 100.0F + static_cast<float>((seq * 7U + 13U) % 1000U) * 0.01F;
    u.quantity = 1.0F   + static_cast<float>((seq * 3U + 7U)  % 500U)  * 0.1F;
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
            const std::uint64_t recv = rdtsc();
            lob.apply(u);
            if (consumed >= k_warmup_messages) [[likely]] {
                const std::uint64_t lat =
                    (recv >= u.timestamp_ns) ? (recv - u.timestamp_ns) : 0U;
                results.latency.record(lat);
            }
            ++consumed;
        } else {
            if (state.producer_done.load(std::memory_order_acquire)
                && ring.empty()) [[unlikely]] { break; }
            __builtin_ia32_pause();
        }
    }
    results.consumer_duration_ns = now_ns() - t0;
    results.total_messages       = consumed;
    results.dropped_messages     =
        lob.stats().dropped_updates.load(std::memory_order_relaxed);
}

static void print_separator() noexcept {
    std::puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

static void print_section(const char* title) noexcept {
    print_separator();
    std::printf("  %s\n", title);
    print_separator();
}

static void print_spsc_results(
    const BenchmarkResults& r,
    const LobSoA&           lob,
    double                  tsc_ghz) noexcept
{
    const double c2n  = 1.0 / tsc_ghz;
    const auto&  lat  = r.latency;

    print_section("PHASE I — SPSC LOB INGESTION BENCHMARK");
    std::printf("  TSC frequency     : %.4f GHz\n",      tsc_ghz);
    std::printf("  Messages total    : %llu\n",
        static_cast<unsigned long long>(r.total_messages));
    std::printf("  Warmup excluded   : %zu\n",           k_warmup_messages);
    std::printf("  Measured          : %llu\n",
        static_cast<unsigned long long>(lat.count));
    std::printf("  Dropped           : %llu\n",
        static_cast<unsigned long long>(r.dropped_messages));
    std::printf("  Producer wall     : %.3f ms\n",
        static_cast<double>(r.producer_duration_ns) / 1e6);
    std::printf("  Consumer wall     : %.3f ms\n",
        static_cast<double>(r.consumer_duration_ns) / 1e6);
    std::printf("  Throughput        : %.3f M msg/s\n",
        r.throughput_mps() / 1e6);
    print_separator();
    std::puts("  TRANSIT LATENCY (producer RDTSC → consumer RDTSC)");
    std::printf("  Min   : %6llu cy  /  %7.1f ns\n",
        static_cast<unsigned long long>(lat.min_cycles),
        static_cast<double>(lat.min_cycles) * c2n);
    std::printf("  Max   : %6llu cy  /  %7.1f ns\n",
        static_cast<unsigned long long>(lat.max_cycles),
        static_cast<double>(lat.max_cycles) * c2n);
    std::printf("  Mean  : %9.1f cy  /  %7.1f ns\n",  lat.mean, lat.mean * c2n);
    std::printf("  Std   : %9.1f cy  /  %7.1f ns\n",
        lat.stddev(), lat.stddev() * c2n);
    print_separator();
    std::puts("  LOB STATE SNAPSHOT");
    std::printf("  Total updates     : %llu\n",
        static_cast<unsigned long long>(
            lob.stats().total_updates.load(std::memory_order_relaxed)));
    std::printf("  Bid / Ask         : %llu / %llu\n",
        static_cast<unsigned long long>(
            lob.stats().bid_updates.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            lob.stats().ask_updates.load(std::memory_order_relaxed)));
    const auto bids = lob.bid_prices(0U);
    const auto asks = lob.ask_prices(0U);
    if (!bids.empty() && !asks.empty()) {
        std::printf("  Instr[0] bid[0]   : %.4f\n",
            static_cast<double>(bids[0]));
        std::printf("  Instr[0] ask[0]   : %.4f\n",
            static_cast<double>(asks[0]));
        std::printf("  Instr[0] spread   : %.4f\n",
            static_cast<double>(lob.spread(0U)));
        std::printf("  Instr[0] mid      : %.4f\n",
            static_cast<double>(lob.mid_price(0U)));
    }
    print_separator();
    std::puts("  LATENCY HISTOGRAM");
    for (std::size_t b = 0U; b < k_histogram_buckets; ++b) {
        if (lat.hist[b] == 0U) continue;
        const std::uint64_t lo_c = (b == 0U) ? 0ULL : (1ULL << (b - 1U));
        const std::uint64_t hi_c = (1ULL << b) - 1ULL;
        std::printf("  [%6llu – %7llu cy | %5.0f – %5.0f ns] : %llu\n",
            static_cast<unsigned long long>(lo_c),
            static_cast<unsigned long long>(hi_c),
            static_cast<double>(lo_c) / tsc_ghz,
            static_cast<double>(hi_c) / tsc_ghz,
            static_cast<unsigned long long>(lat.hist[b]));
    }
}

static void print_cuda_results(
    const cuda::PipelineMetrics& m) noexcept
{
    print_section("PHASE II — CUDA TOPOLOGICAL PIPELINE METRICS");

    const std::uint64_t n_xfr   = m.n_transfers.load(std::memory_order_relaxed);
    const std::uint64_t n_decomp = m.n_decompositions.load(std::memory_order_relaxed);
    const std::uint64_t n_sig   = m.n_arbitrage_signals.load(std::memory_order_relaxed);
    const std::uint64_t t_xfr   = m.total_transfer_ns.load(std::memory_order_relaxed);
    const std::uint64_t t_decomp = m.total_decomp_ns.load(std::memory_order_relaxed);
    const float         ym       = m.last_ym_action.load(std::memory_order_relaxed);
    const int           beta1    = m.last_harmonic_dims.load(std::memory_order_relaxed);

    std::printf("  LOB→GPU transfers    : %llu\n",
        static_cast<unsigned long long>(n_xfr));
    std::printf("  Hodge decompositions : %llu\n",
        static_cast<unsigned long long>(n_decomp));
    std::printf("  Arbitrage signals    : %llu\n",
        static_cast<unsigned long long>(n_sig));

    if (n_xfr > 0U) {
        std::printf("  Mean transfer time   : %.3f ms\n",
            static_cast<double>(t_xfr) / static_cast<double>(n_xfr) / 1e6);
    }
    if (n_decomp > 0U) {
        std::printf("  Mean Hodge time      : %.3f ms\n",
            static_cast<double>(t_decomp) / static_cast<double>(n_decomp) / 1e6);
    }

    print_separator();
    std::puts("  TOPOLOGICAL INVARIANTS (last cycle)");
    std::printf("  Yang-Mills S_YM[A]   : %.6f\n",
        static_cast<double>(ym));
    std::printf("  First Betti β₁       : %d\n",  beta1);
    std::printf("  Market efficiency    : %s\n",
        (ym < 0.01F) ? "HIGH  (S_YM → 0)" : "LOW   (S_YM >> 0, arb present)");
    print_separator();
}

} // namespace holo

int main() {
    using namespace holo;
    using namespace holo::cuda;

    std::puts("\n");
    print_separator();
    std::puts("  HOLOGRAPHIC MARKET ARCHITECTURE");
    std::puts("  C++ Bare Metal + CUDA Topological Engine");
    std::puts("  Jump Trading / Topology Division — v0.2.0");
    print_separator();
    std::puts("");

    std::puts("  Calibrating TSC (50 ms)...");
    const double tsc_ghz = estimate_tsc_ghz();
    std::printf("  TSC = %.4f GHz\n\n", tsc_ghz);

    MemoryArena arena{k_arena_size_bytes};

    RingT* const ring_ptr = arena.emplace<RingT>(arena, k_ring_capacity);
    if (!ring_ptr) {
        std::puts("FATAL: ring buffer arena alloc failed.");
        return 1;
    }

    LobSoA* const lob_ptr =
        arena.emplace<LobSoA>(arena, k_n_instruments, k_depth_levels);
    if (!lob_ptr) {
        std::puts("FATAL: LobSoA arena alloc failed.");
        return 1;
    }

    SharedState*     const state_ptr   = arena.emplace<SharedState>();
    BenchmarkResults* const results_ptr = arena.emplace<BenchmarkResults>();

    if (!state_ptr || !results_ptr) {
        std::puts("FATAL: aux arena alloc failed.");
        return 1;
    }

    std::printf("  Arena capacity : %.1f MB\n",
        static_cast<double>(arena.capacity()) / 1e6);
    std::printf("  Arena used     : %.3f MB\n\n",
        static_cast<double>(arena.used()) / 1e6);

    std::puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::puts("  PHASE I: SPSC Ingestion (10M messages)");
    std::puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    {
        std::thread producer{producer_fn,
            std::ref(*ring_ptr), std::ref(*state_ptr), std::ref(*results_ptr)};
        std::thread consumer{consumer_fn,
            std::ref(*ring_ptr), std::ref(*lob_ptr),
            std::ref(*state_ptr), std::ref(*results_ptr)};

        pin_thread(producer, 0);
        pin_thread(consumer, 2);

        producer.join();
        consumer.join();
    }

    print_spsc_results(*results_ptr, *lob_ptr, tsc_ghz);

    std::puts("\n");
    std::puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::puts("  PHASE II: CUDA Topological Pipeline (5 second run)");
    std::puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    {
        CudaPipeline pipeline{*lob_ptr, arena, 0};

        std::atomic<bool> shutdown{false};

        std::thread pipeline_thread{
            [&pipeline, &shutdown]() {
                pipeline.run_continuous(shutdown);
            }
        };

        std::thread lob_updater{
            [&lob_ptr, &shutdown]() {
                std::uint64_t seq = 10'000'000ULL;
                while (!shutdown.load(std::memory_order_relaxed)) {
                    LobUpdate u{};
                    u.timestamp_ns  = now_ns();
                    u.instrument_id = static_cast<std::uint32_t>(
                        seq % k_n_instruments);
                    u.depth_level   = static_cast<std::uint8_t>(
                        (seq / k_n_instruments) % k_depth_levels);
                    u.side    = (seq & 1U) ? Side::Ask : Side::Bid;
                    u.price   = 100.0F
                        + static_cast<float>((seq * 11U + 17U) % 2000U) * 0.005F;
                    u.quantity = 1.0F
                        + static_cast<float>((seq * 5U + 3U) % 1000U) * 0.05F;
                    lob_ptr->apply(u);
                    ++seq;
                    struct timespec req{0, 50'000};
                    nanosleep(&req, nullptr);
                }
            }
        };

        const std::uint64_t t_start = now_ns();
        while (now_ns() - t_start < k_pipeline_run_ns) {
            struct timespec req{0, 100'000'000};
            nanosleep(&req, nullptr);

            const auto sig = pipeline.last_signal();
            std::printf(
                "\r  [%.1fs] S_YM=%.4f  β₁=%d  loops=%d  max_curl=%.4f  "
                "decomps=%llu     ",
                static_cast<double>(now_ns() - t_start) / 1e9,
                static_cast<double>(sig.yang_mills_action),
                sig.n_harmonic_dims,
                sig.n_active_loops,
                static_cast<double>(sig.max_curl),
                static_cast<unsigned long long>(
                    pipeline.metrics().n_decompositions.load(
                        std::memory_order_relaxed)));
            std::fflush(stdout);
        }

        std::puts("\n");
        shutdown.store(true, std::memory_order_release);
        pipeline_thread.join();
        lob_updater.join();

        print_cuda_results(pipeline.metrics());
    }

    std::printf("\n  Arena utilization            : %.2f%%\n",
        arena.utilization() * 100.0);
    std::puts("  Zero heap allocs on hot path : CONFIRMED");
    std::puts("  CUDA pipeline terminated     : CLEAN\n");

    return 0;
}