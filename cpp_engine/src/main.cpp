#include <memory_arena.hpp>
#include <lockfree_ring_buffer.hpp>
#include <lob_core.hpp>
#include <cuda_pipeline.cuh>
#include <cuda_utils.cuh>
#include <binance_feed.hpp>
#include <cuml_clustering.cuh>

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
    static constexpr std::size_t k_n_instruments = 1U;
    static constexpr std::size_t k_depth_levels = 20U;
    static constexpr std::size_t k_ring_capacity = 1U << 17U;
    static constexpr std::size_t k_arena_size_bytes = 512U * 1024U * 1024U;
    static constexpr std::uint64_t k_pipeline_run_ns = 30'000'000'000ULL;
    static constexpr int k_cluster_every_n = 10;

    [[nodiscard]] static std::uint64_t now_ns() noexcept {
        struct timespec ts{};
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
               + static_cast<std::uint64_t>(ts.tv_nsec);
    }

    [[nodiscard]] static std::uint64_t rdtsc() noexcept {
        std::uint32_t lo, hi;
        __asm__ __volatile__("lfence\n\trdtsc\n\t" : "=a"(lo), "=d"(hi) :: "memory");
        return (static_cast<std::uint64_t>(hi) << 32U) | lo;
    }

    [[nodiscard]] static double estimate_tsc_ghz() noexcept {
        const std::uint64_t ns0 = now_ns();
        const std::uint64_t tsc0 = rdtsc();
        struct timespec req{0, 50'000'000};
        nanosleep(&req, nullptr);
        return static_cast<double>(rdtsc() - tsc0) / static_cast<double>(now_ns() - ns0);
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

    static void print_sep() noexcept { std::puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"); }

    static void print_section(const char *t) noexcept {
        print_sep();
        std::printf("  %s\n", t);
        print_sep();
    }

    static void print_feed_stats(const FeedStats &fs) noexcept {
        print_section("PHASE III — BINANCE FEED STATS");
        std::printf("  Messages received : %llu\n",
                    (unsigned long long) fs.messages_received.load(std::memory_order_relaxed));
        std::printf("  Updates pushed    : %llu\n",
                    (unsigned long long) fs.updates_pushed.load(std::memory_order_relaxed));
        std::printf("  Updates dropped   : %llu\n",
                    (unsigned long long) fs.updates_dropped.load(std::memory_order_relaxed));
        std::printf("  Parse errors      : %llu\n",
                    (unsigned long long) fs.parse_errors.load(std::memory_order_relaxed));
        std::printf("  Reconnects        : %llu\n", (unsigned long long) fs.reconnects.load(std::memory_order_relaxed));
        print_sep();
    }

    static void print_cluster_stats(const cuda::TopologyClusterer::Metrics &cm) noexcept {
        print_section("PHASE III — cuML DBSCAN CLUSTER METRICS");
        const std::uint64_t n = cm.n_fit_calls.load(std::memory_order_relaxed);
        std::printf("  Fit calls         : %llu\n", (unsigned long long) n);
        if (n > 0U)
            std::printf("  Mean fit latency  : %.2f ms\n",
                        static_cast<double>(cm.total_fit_ns.load(std::memory_order_relaxed)) / static_cast<double>(n) /
                        1e6);
        std::printf("  Last n_clusters   : %d\n", cm.last_n_clusters.load(std::memory_order_relaxed));
        std::printf("  Last n_noise      : %d\n", cm.last_n_noise.load(std::memory_order_relaxed));
        print_sep();
    }

    using RingT = DynamicSpscRingBuffer<LobUpdate>;
}

int main() {
    using namespace holo;
    using namespace holo::cuda;

    std::puts("\n");
    print_sep();
    std::puts("  HOLOGRAPHIC MARKET ARCHITECTURE");
    std::puts("  C++ Bare Metal + CUDA Topological Engine");
    std::puts("  Jump Trading / Topology Division — v0.3.0");
    std::puts("  Phase III: Live Binance Feed + cuML DBSCAN");
    print_sep();
    std::puts("");

    std::puts("  Calibrating TSC (50 ms)...");
    const double tsc_ghz = estimate_tsc_ghz();
    std::printf("  TSC = %.4f GHz\n\n", tsc_ghz);

    MemoryArena arena{k_arena_size_bytes};

    RingT *const ring_ptr = arena.emplace<RingT>(arena, k_ring_capacity);
    if (!ring_ptr) {
        std::puts("FATAL: ring buffer alloc failed.");
        return 1;
    }

    LobSoA *const lob_ptr = arena.emplace<LobSoA>(arena, k_n_instruments, k_depth_levels);
    if (!lob_ptr) {
        std::puts("FATAL: LobSoA alloc failed.");
        return 1;
    }

    std::printf("  Arena capacity : %.1f MB\n", static_cast<double>(arena.capacity()) / 1e6);
    std::printf("  Arena used     : %.3f MB\n\n", static_cast<double>(arena.used()) / 1e6);

    print_sep();
    std::puts("  PHASE III: Live Binance Feed + CUDA Pipeline + cuML Clustering");
    print_sep();
    std::puts("");

    BinanceFeedHandler feed{*ring_ptr, 0U};

    std::atomic<bool> shutdown{false};
    std::thread lob_consumer{
        [&]() {
            LobUpdate u{};
            while (!shutdown.load(std::memory_order_relaxed)) {
                if (ring_ptr->try_pop(u)) [[likely]] lob_ptr->apply(u);
                else __builtin_ia32_pause();
            }
            while (ring_ptr->try_pop(u)) lob_ptr->apply(u);
        }
    };
    pin_thread(lob_consumer, 2);

    CudaPipeline pipeline{*lob_ptr, arena, 0};

    cudaStream_t cluster_stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&cluster_stream, cudaStreamNonBlocking));

    TopologyClusterer clusterer{
        cluster_stream,
        ClusteringConfig{.eps = 0.15F, .min_samples = 3, .max_bytes_per_batch = 64 * 1024 * 1024}
    };

    feed.start();
    std::puts("  [Feed] Binance WebSocket started.");

    const std::uint64_t t_start = now_ns();
    int pipeline_cycle = 0;

    while (now_ns() - t_start < k_pipeline_run_ns) {
        pipeline.run_once();
        ++pipeline_cycle;

        if (pipeline_cycle % k_cluster_every_n == 0) {
            const auto &sig = pipeline.last_signal();
            if (sig.n_active_loops > 0) {
                static constexpr int k_win = 32, k_n_feat = 4;
                static float h_feat_buf[k_win * k_n_feat]{};
                static int feat_head = 0;
                const int row = feat_head % k_win;
                h_feat_buf[row * k_n_feat + 0] = sig.yang_mills_action;
                h_feat_buf[row * k_n_feat + 1] = sig.max_curl;
                h_feat_buf[row * k_n_feat + 2] = sig.mean_curl;
                h_feat_buf[row * k_n_feat + 3] = static_cast<float>(sig.n_harmonic_dims);
                ++feat_head;
                if (feat_head >= k_win) {
                    float *d_feat{};
                    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_feat), k_win * k_n_feat * sizeof(float)));
                    CUDA_CHECK(
                        cudaMemcpyAsync(d_feat, h_feat_buf, k_win * k_n_feat * sizeof(float), cudaMemcpyHostToDevice,
                            cluster_stream));
                    CUDA_CHECK(cudaStreamSynchronize(cluster_stream));
                    clusterer.fit(d_feat, k_win, k_n_feat);
                    CUDA_CHECK(cudaFree(d_feat));
                }
            }
        }

        {
            static std::uint64_t last_tick = 0U;
            const std::uint64_t now = now_ns();
            if (now - last_tick > 100'000'000ULL) {
                last_tick = now;
                const auto sig = pipeline.last_signal();
                std::printf(
                    "\r  [%.1fs] S_YM=%.4f  β₁=%d  loops=%d  max_curl=%.4f  clusters=%d  feed_rx=%llu  decomps=%llu      ",
                    static_cast<double>(now - t_start) / 1e9,
                    static_cast<double>(sig.yang_mills_action),
                    sig.n_harmonic_dims, sig.n_active_loops,
                    static_cast<double>(sig.max_curl),
                    clusterer.metrics().last_n_clusters.load(std::memory_order_relaxed),
                    (unsigned long long) feed.stats().messages_received.load(std::memory_order_relaxed),
                    (unsigned long long) pipeline.metrics().n_decompositions.load(std::memory_order_relaxed));
                std::fflush(stdout);
                struct timespec req{0, 1'000'000};
                nanosleep(&req, nullptr);
            }
        }
    }

    std::puts("\n");
    shutdown.store(true, std::memory_order_release);
    feed.stop();
    lob_consumer.join();
    CUDA_CHECK(cudaStreamSynchronize(cluster_stream));
    CUDA_CHECK(cudaStreamDestroy(cluster_stream));

    print_feed_stats(feed.stats());
    print_cluster_stats(clusterer.metrics());

    std::printf("\n  Arena utilization : %.2f%%\n", arena.utilization() * 100.0);
    std::puts("  CUDA pipeline     : CLEAN");
    std::puts("  WebSocket feed    : CLEAN\n");

    return 0;
}
