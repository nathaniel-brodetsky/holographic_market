#include <ai/cuml_clustering.cuh>
#include <math/cuda_utils.cuh>

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
