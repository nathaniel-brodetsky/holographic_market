#!/usr/bin/env bash
# infra/inject_phase3.sh
# Создаёт все Phase III файлы прямо на VM и пересобирает проект.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${REPO_ROOT}/cpp_engine"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log() { echo -e "${CYAN}[INJECT]${NC} $*"; }
ok()  { echo -e "${GREEN}[OK]${NC} $*"; }

log "Repo root: ${REPO_ROOT}"

# ─── binance_feed.hpp ────────────────────────────────────────────────────────
log "Writing binance_feed.hpp"
cat > "${ENGINE}/include/binance_feed.hpp" << 'EOF'
#pragma once

#include <lob_core.hpp>
#include <lockfree_ring_buffer.hpp>

#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>

#include <simdjson.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <thread>

namespace holo
{

namespace net  = boost::asio;
namespace ssl  = boost::asio::ssl;
namespace beast = boost::beast;
namespace ws   = boost::beast::websocket;
namespace http = boost::beast::http;

using tcp      = net::ip::tcp;
using WsStream = ws::stream<beast::ssl_stream<beast::tcp_stream>>;

static constexpr std::string_view k_binance_host   = "fstream.binance.com";
static constexpr std::string_view k_binance_port   = "443";
static constexpr std::string_view k_binance_target = "/stream?streams=btcusdt@depth20@100ms";
static constexpr std::size_t k_read_buffer_bytes   = 1U << 17U;
static constexpr std::size_t k_simdjson_padding    = simdjson::SIMDJSON_PADDING;

struct FeedStats
{
    std::atomic<std::uint64_t> messages_received{0U};
    std::atomic<std::uint64_t> updates_pushed{0U};
    std::atomic<std::uint64_t> updates_dropped{0U};
    std::atomic<std::uint64_t> parse_errors{0U};
    std::atomic<std::uint64_t> reconnects{0U};
};

class BinanceFeedHandler final
{
public:
    explicit BinanceFeedHandler(
        DynamicSpscRingBuffer<LobUpdate> &ring,
        std::uint32_t instrument_id = 0U) noexcept
        : ring_{ring}, instrument_id_{instrument_id}
    {}

    ~BinanceFeedHandler() noexcept { stop(); }

    BinanceFeedHandler(const BinanceFeedHandler &)            = delete;
    BinanceFeedHandler &operator=(const BinanceFeedHandler &) = delete;
    BinanceFeedHandler(BinanceFeedHandler &&)                 = delete;
    BinanceFeedHandler &operator=(BinanceFeedHandler &&)      = delete;

    void start()
    {
        shutdown_.store(false, std::memory_order_release);
        thread_ = std::thread([this]() { run_loop(); });
    }

    void stop() noexcept
    {
        shutdown_.store(true, std::memory_order_release);
        ioc_.stop();
        if (thread_.joinable()) thread_.join();
    }

    [[nodiscard]] const FeedStats &stats() const noexcept { return stats_; }

private:
    void run_loop() noexcept
    {
        while (!shutdown_.load(std::memory_order_acquire))
        {
            try { ioc_.restart(); connect_and_consume(); }
            catch (const std::exception &e)
            {
                std::fprintf(stderr, "[BinanceFeed] %s — reconnecting\n", e.what());
                stats_.reconnects.fetch_add(1U, std::memory_order_relaxed);
                struct timespec req{1, 0};
                nanosleep(&req, nullptr);
            }
        }
    }

    void connect_and_consume()
    {
        ssl::context ssl_ctx{ssl::context::tlsv13_client};
        ssl_ctx.set_default_verify_paths();
        ssl_ctx.set_verify_mode(ssl::verify_peer);

        WsStream stream{ioc_, ssl_ctx};

        tcp::resolver resolver{ioc_};
        const auto endpoints = resolver.resolve(
            std::string(k_binance_host), std::string(k_binance_port));

        beast::get_lowest_layer(stream).connect(endpoints);
        beast::get_lowest_layer(stream).expires_never();

        if (!SSL_set_tlsext_host_name(
                stream.next_layer().native_handle(),
                k_binance_host.data()))
            throw boost::system::system_error{
                static_cast<int>(::ERR_get_error()),
                boost::asio::error::get_ssl_category()};

        stream.next_layer().handshake(ssl::stream_base::client);

        stream.set_option(ws::stream_base::decorator(
            [](ws::request_type &req) {
                req.set(http::field::user_agent, "holographic-engine/0.3.0");
            }));
        stream.handshake(std::string(k_binance_host), std::string(k_binance_target));

        std::fprintf(stdout, "[BinanceFeed] connected to %s%s\n",
                     k_binance_host.data(), k_binance_target.data());

        beast::flat_buffer buf;
        buf.reserve(k_read_buffer_bytes);

        simdjson::ondemand::parser parser;
        std::string padded;
        padded.reserve(k_read_buffer_bytes + k_simdjson_padding);

        while (!shutdown_.load(std::memory_order_acquire))
        {
            buf.clear();
            stream.read(buf);
            stats_.messages_received.fetch_add(1U, std::memory_order_relaxed);
            const auto data = buf.cdata();
            dispatch_message(
                {static_cast<const char *>(data.data()), data.size()},
                parser, padded);
        }

        beast::error_code ec;
        stream.close(ws::close_code::normal, ec);
    }

    void dispatch_message(
        std::string_view raw,
        simdjson::ondemand::parser &parser,
        std::string &padded) noexcept
    {
        padded.assign(raw);
        padded.resize(raw.size() + k_simdjson_padding, '\0');

        simdjson::ondemand::document doc;
        if (parser.iterate(padded.data(), raw.size(), padded.size()).get(doc) != simdjson::SUCCESS)
        { stats_.parse_errors.fetch_add(1U, std::memory_order_relaxed); return; }

        simdjson::ondemand::object data_obj;
        if (doc["data"].get(data_obj) != simdjson::SUCCESS) return;

        simdjson::ondemand::array bids_arr, asks_arr;
        if (data_obj["b"].get(bids_arr) == simdjson::SUCCESS) push_levels(bids_arr, Side::Bid);
        if (data_obj["a"].get(asks_arr) == simdjson::SUCCESS) push_levels(asks_arr, Side::Ask);
    }

    void push_levels(simdjson::ondemand::array &levels, Side side) noexcept
    {
        std::uint8_t depth_idx = 0U;
        for (auto entry : levels)
        {
            simdjson::ondemand::array pair;
            if (entry.get(pair) != simdjson::SUCCESS) break;
            auto it = pair.begin();
            if (it == pair.end()) break;
            std::string_view price_sv;
            if ((*it).get_string().get(price_sv) != simdjson::SUCCESS) break;
            ++it;
            if (it == pair.end()) break;
            std::string_view qty_sv;
            if ((*it).get_string().get(qty_sv) != simdjson::SUCCESS) break;

            LobUpdate u{};
            u.timestamp_ns  = static_cast<std::uint64_t>(__builtin_ia32_rdtsc());
            u.instrument_id = instrument_id_;
            u.depth_level   = depth_idx;
            u.side          = side;
            std::sscanf(price_sv.data(), "%f", &u.price);
            std::sscanf(qty_sv.data(),   "%f", &u.quantity);

            if (ring_.try_push(u)) [[likely]]
                stats_.updates_pushed.fetch_add(1U, std::memory_order_relaxed);
            else
                stats_.updates_dropped.fetch_add(1U, std::memory_order_relaxed);

            if (++depth_idx >= static_cast<std::uint8_t>(k_max_depth)) break;
        }
    }

    DynamicSpscRingBuffer<LobUpdate> &ring_;
    std::uint32_t                     instrument_id_;
    std::atomic<bool>                 shutdown_{true};
    net::io_context                   ioc_{1};
    std::thread                       thread_;
    FeedStats                         stats_;
};

} // namespace holo
EOF
ok "binance_feed.hpp written"

# ─── cuml_clustering.cuh ─────────────────────────────────────────────────────
log "Writing cuml_clustering.cuh"
cat > "${ENGINE}/include/cuml_clustering.cuh" << 'EOF'
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <atomic>

namespace holo::cuda
{

struct ClusteringResult
{
    int  *d_labels{nullptr};
    int   n_clusters{0};
    int   n_noise{0};
    int   n_samples{0};
};

struct ClusteringConfig
{
    float eps{0.15F};
    int   min_samples{3};
    int   max_bytes_per_batch{64 * 1024 * 1024};
};

class TopologyClusterer final
{
public:
    explicit TopologyClusterer(cudaStream_t stream, ClusteringConfig cfg = {}) noexcept;
    ~TopologyClusterer() noexcept;

    TopologyClusterer(const TopologyClusterer &)            = delete;
    TopologyClusterer &operator=(const TopologyClusterer &) = delete;
    TopologyClusterer(TopologyClusterer &&)                 = delete;
    TopologyClusterer &operator=(TopologyClusterer &&)      = delete;

    void fit(const float *d_features, int n_samples, int n_features);

    [[nodiscard]] const ClusteringResult &result() const noexcept { return result_; }
    [[nodiscard]] const ClusteringConfig &config() const noexcept { return cfg_; }

    struct Metrics
    {
        std::atomic<std::uint64_t> n_fit_calls{0U};
        std::atomic<std::uint64_t> total_fit_ns{0U};
        std::atomic<int>           last_n_clusters{0};
        std::atomic<int>           last_n_noise{0};
    };

    [[nodiscard]] const Metrics &metrics() const noexcept { return metrics_; }

private:
    void ensure_label_buffer(int n_samples);

    cudaStream_t     stream_;
    ClusteringConfig cfg_;
    ClusteringResult result_;
    Metrics          metrics_;
    int              label_capacity_{0};
};

} // namespace holo::cuda
EOF
ok "cuml_clustering.cuh written"

# ─── cuml_clustering.cu ──────────────────────────────────────────────────────
log "Writing cuml_clustering.cu"
cat > "${ENGINE}/src/cuml_clustering.cu" << 'EOF'
#include <cuml_clustering.cuh>
#include <cuda_utils.cuh>

#include <cuml/cluster/dbscan.hpp>
#include <raft/core/handle.hpp>

#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
#include <vector>

namespace holo::cuda
{

namespace {
[[nodiscard]] static std::uint64_t now_ns_cu() noexcept
{
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
         + static_cast<std::uint64_t>(ts.tv_nsec);
}
}

TopologyClusterer::TopologyClusterer(cudaStream_t stream, ClusteringConfig cfg) noexcept
    : stream_{stream}, cfg_{cfg} {}

TopologyClusterer::~TopologyClusterer() noexcept
{
    if (result_.d_labels) { cudaFree(result_.d_labels); result_.d_labels = nullptr; }
}

void TopologyClusterer::ensure_label_buffer(int n_samples)
{
    if (n_samples <= label_capacity_) return;
    if (result_.d_labels) CUDA_CHECK(cudaFree(result_.d_labels));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&result_.d_labels),
        static_cast<std::size_t>(n_samples) * sizeof(int)));
    label_capacity_ = n_samples;
}

void TopologyClusterer::fit(const float *d_features, int n_samples, int n_features)
{
    if (n_samples <= 0 || n_features <= 0 || !d_features) return;

    ensure_label_buffer(n_samples);

    raft::handle_t handle{stream_};

    const std::uint64_t t0 = now_ns_cu();

    ML::dbscanFit(
        handle,
        d_features,
        static_cast<std::size_t>(n_samples),
        static_cast<std::size_t>(n_features),
        cfg_.eps,
        cfg_.min_samples,
        raft::distance::DistanceType::L2SqrtUnexpanded,
        result_.d_labels,
        static_cast<int *>(nullptr),
        static_cast<std::size_t>(cfg_.max_bytes_per_batch),
        false);

    handle.sync_stream(stream_);

    const std::uint64_t elapsed = now_ns_cu() - t0;

    std::vector<int> h_labels(static_cast<std::size_t>(n_samples));
    CUDA_CHECK(cudaMemcpy(h_labels.data(), result_.d_labels,
        static_cast<std::size_t>(n_samples) * sizeof(int), cudaMemcpyDeviceToHost));

    int max_label = -1, noise_cnt = 0;
    for (int l : h_labels) { if (l == -1) ++noise_cnt; else if (l > max_label) max_label = l; }

    result_.n_samples  = n_samples;
    result_.n_clusters = max_label + 1;
    result_.n_noise    = noise_cnt;

    metrics_.n_fit_calls.fetch_add(1U, std::memory_order_relaxed);
    metrics_.total_fit_ns.fetch_add(elapsed, std::memory_order_relaxed);
    metrics_.last_n_clusters.store(result_.n_clusters, std::memory_order_release);
    metrics_.last_n_noise.store(result_.n_noise, std::memory_order_release);

    std::fprintf(stdout,
        "[TopologyClusterer] fit: n=%d  clusters=%d  noise=%d  t=%.2f ms\n",
        n_samples, result_.n_clusters, result_.n_noise,
        static_cast<double>(elapsed) / 1e6);
}

} // namespace holo::cuda
EOF
ok "cuml_clustering.cu written"

# ─── main.cpp ────────────────────────────────────────────────────────────────
log "Writing main.cpp (Phase III)"
cat > "${ENGINE}/src/main.cpp" << 'EOF'
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

namespace holo
{
    static constexpr std::size_t   k_n_instruments    = 1U;
    static constexpr std::size_t   k_depth_levels     = 20U;
    static constexpr std::size_t   k_ring_capacity    = 1U << 17U;
    static constexpr std::size_t   k_arena_size_bytes = 512U * 1024U * 1024U;
    static constexpr std::uint64_t k_pipeline_run_ns  = 30'000'000'000ULL;
    static constexpr int           k_cluster_every_n  = 10;

    [[nodiscard]] static std::uint64_t now_ns() noexcept
    {
        struct timespec ts{};
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL
             + static_cast<std::uint64_t>(ts.tv_nsec);
    }

    [[nodiscard]] static std::uint64_t rdtsc() noexcept
    {
        std::uint32_t lo, hi;
        __asm__ __volatile__("lfence\n\trdtsc\n\t" : "=a"(lo), "=d"(hi) :: "memory");
        return (static_cast<std::uint64_t>(hi) << 32U) | lo;
    }

    [[nodiscard]] static double estimate_tsc_ghz() noexcept
    {
        const std::uint64_t ns0  = now_ns();
        const std::uint64_t tsc0 = rdtsc();
        struct timespec req{0, 50'000'000};
        nanosleep(&req, nullptr);
        return static_cast<double>(rdtsc() - tsc0) / static_cast<double>(now_ns() - ns0);
    }

#if defined(__linux__)
    static void pin_thread(std::thread &t, int core_id) noexcept
    {
        cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(static_cast<std::size_t>(core_id), &cpuset);
        pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    }
#else
    static void pin_thread(std::thread &, int) noexcept {}
#endif

    static void print_sep() noexcept { std::puts("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"); }
    static void print_section(const char *t) noexcept { print_sep(); std::printf("  %s\n", t); print_sep(); }

    static void print_feed_stats(const FeedStats &fs) noexcept
    {
        print_section("PHASE III — BINANCE FEED STATS");
        std::printf("  Messages received : %llu\n", (unsigned long long)fs.messages_received.load(std::memory_order_relaxed));
        std::printf("  Updates pushed    : %llu\n", (unsigned long long)fs.updates_pushed.load(std::memory_order_relaxed));
        std::printf("  Updates dropped   : %llu\n", (unsigned long long)fs.updates_dropped.load(std::memory_order_relaxed));
        std::printf("  Parse errors      : %llu\n", (unsigned long long)fs.parse_errors.load(std::memory_order_relaxed));
        std::printf("  Reconnects        : %llu\n", (unsigned long long)fs.reconnects.load(std::memory_order_relaxed));
        print_sep();
    }

    static void print_cluster_stats(const cuda::TopologyClusterer::Metrics &cm) noexcept
    {
        print_section("PHASE III — cuML DBSCAN CLUSTER METRICS");
        const std::uint64_t n = cm.n_fit_calls.load(std::memory_order_relaxed);
        std::printf("  Fit calls         : %llu\n", (unsigned long long)n);
        if (n > 0U)
            std::printf("  Mean fit latency  : %.2f ms\n",
                static_cast<double>(cm.total_fit_ns.load(std::memory_order_relaxed)) / static_cast<double>(n) / 1e6);
        std::printf("  Last n_clusters   : %d\n", cm.last_n_clusters.load(std::memory_order_relaxed));
        std::printf("  Last n_noise      : %d\n", cm.last_n_noise.load(std::memory_order_relaxed));
        print_sep();
    }

    using RingT = DynamicSpscRingBuffer<LobUpdate>;
}

int main()
{
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
    if (!ring_ptr) { std::puts("FATAL: ring buffer alloc failed."); return 1; }

    LobSoA *const lob_ptr = arena.emplace<LobSoA>(arena, k_n_instruments, k_depth_levels);
    if (!lob_ptr) { std::puts("FATAL: LobSoA alloc failed."); return 1; }

    std::printf("  Arena capacity : %.1f MB\n", static_cast<double>(arena.capacity()) / 1e6);
    std::printf("  Arena used     : %.3f MB\n\n", static_cast<double>(arena.used()) / 1e6);

    print_sep();
    std::puts("  PHASE III: Live Binance Feed + CUDA Pipeline + cuML Clustering");
    print_sep();
    std::puts("");

    BinanceFeedHandler feed{*ring_ptr, 0U};

    std::atomic<bool> shutdown{false};
    std::thread lob_consumer{[&]() {
        LobUpdate u{};
        while (!shutdown.load(std::memory_order_relaxed))
        { if (ring_ptr->try_pop(u)) [[likely]] lob_ptr->apply(u); else __builtin_ia32_pause(); }
        while (ring_ptr->try_pop(u)) lob_ptr->apply(u);
    }};
    pin_thread(lob_consumer, 2);

    CudaPipeline pipeline{*lob_ptr, arena, 0};

    cudaStream_t cluster_stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&cluster_stream, cudaStreamNonBlocking));

    TopologyClusterer clusterer{
        cluster_stream,
        ClusteringConfig{.eps = 0.15F, .min_samples = 3, .max_bytes_per_batch = 64 * 1024 * 1024}};

    feed.start();
    std::puts("  [Feed] Binance WebSocket started.");

    const std::uint64_t t_start = now_ns();
    int pipeline_cycle = 0;

    while (now_ns() - t_start < k_pipeline_run_ns)
    {
        pipeline.run_once();
        ++pipeline_cycle;

        if (pipeline_cycle % k_cluster_every_n == 0)
        {
            const auto &sig = pipeline.last_signal();
            if (sig.n_active_loops > 0)
            {
                static constexpr int k_win = 32, k_n_feat = 4;
                static float h_feat_buf[k_win * k_n_feat]{};
                static int feat_head = 0;
                const int row = feat_head % k_win;
                h_feat_buf[row * k_n_feat + 0] = sig.yang_mills_action;
                h_feat_buf[row * k_n_feat + 1] = sig.max_curl;
                h_feat_buf[row * k_n_feat + 2] = sig.mean_curl;
                h_feat_buf[row * k_n_feat + 3] = static_cast<float>(sig.n_harmonic_dims);
                ++feat_head;
                if (feat_head >= k_win)
                {
                    float *d_feat{};
                    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_feat), k_win * k_n_feat * sizeof(float)));
                    CUDA_CHECK(cudaMemcpyAsync(d_feat, h_feat_buf, k_win * k_n_feat * sizeof(float), cudaMemcpyHostToDevice, cluster_stream));
                    CUDA_CHECK(cudaStreamSynchronize(cluster_stream));
                    clusterer.fit(d_feat, k_win, k_n_feat);
                    CUDA_CHECK(cudaFree(d_feat));
                }
            }
        }

        {
            static std::uint64_t last_tick = 0U;
            const std::uint64_t  now       = now_ns();
            if (now - last_tick > 100'000'000ULL)
            {
                last_tick = now;
                const auto sig = pipeline.last_signal();
                std::printf(
                    "\r  [%.1fs] S_YM=%.4f  β₁=%d  loops=%d  max_curl=%.4f  clusters=%d  feed_rx=%llu  decomps=%llu      ",
                    static_cast<double>(now - t_start) / 1e9,
                    static_cast<double>(sig.yang_mills_action),
                    sig.n_harmonic_dims, sig.n_active_loops,
                    static_cast<double>(sig.max_curl),
                    clusterer.metrics().last_n_clusters.load(std::memory_order_relaxed),
                    (unsigned long long)feed.stats().messages_received.load(std::memory_order_relaxed),
                    (unsigned long long)pipeline.metrics().n_decompositions.load(std::memory_order_relaxed));
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
EOF
ok "main.cpp written"

# ─── rebuild ─────────────────────────────────────────────────────────────────
log "Rebuilding..."
cd "${ENGINE}"

CUML_ROOT_FLAG=""
if [[ -d "/opt/miniforge3/envs/rapids-holographic" ]]; then
    CUML_ROOT_FLAG="-DCUML_ROOT=/opt/miniforge3/envs/rapids-holographic"
fi

source /etc/profile.d/cuda.sh 2>/dev/null || true
[[ -f /etc/profile.d/cuml.sh ]] && source /etc/profile.d/cuml.sh || true

rm -rf build/
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_C_COMPILER=gcc-12 \
    -DCMAKE_CXX_COMPILER=g++-12 \
    ${CUML_ROOT_FLAG} \
    2>&1 | tee build/cmake_configure.log

cmake --build build -j"$(nproc)" 2>&1 | tee build/cmake_build.log

ok "Binary: ${ENGINE}/build/bin/holographic_bench"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  PHASE III INJECT COMPLETE${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Run: cd ${ENGINE} && ./build/bin/holographic_bench"
echo ""
