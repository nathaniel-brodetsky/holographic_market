#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <atomic>

namespace holo::cuda {
    struct ClusteringResult {
        int *d_labels{nullptr};
        int n_clusters{0};
        int n_noise{0};
        int n_samples{0};
    };

    struct ClusteringConfig {
        float eps{0.15F};
        int min_samples{3};
        int max_bytes_per_batch{64 * 1024 * 1024};
    };

    class TopologyClusterer final {
    public:
        explicit TopologyClusterer(cudaStream_t stream, ClusteringConfig cfg = {}) noexcept;

        ~TopologyClusterer() noexcept;

        TopologyClusterer(const TopologyClusterer &) = delete;

        TopologyClusterer &operator=(const TopologyClusterer &) = delete;

        TopologyClusterer(TopologyClusterer &&) = delete;

        TopologyClusterer &operator=(TopologyClusterer &&) = delete;

        void fit(const float *d_features, int n_samples, int n_features);

        [[nodiscard]] const ClusteringResult &result() const noexcept { return result_; }
        [[nodiscard]] const ClusteringConfig &config() const noexcept { return cfg_; }

        struct Metrics {
            std::atomic<std::uint64_t> n_fit_calls{0U};
            std::atomic<std::uint64_t> total_fit_ns{0U};
            std::atomic<int> last_n_clusters{0};
            std::atomic<int> last_n_noise{0};
        };

        [[nodiscard]] const Metrics &metrics() const noexcept { return metrics_; }

    private:
        void ensure_label_buffer(int n_samples);

        cudaStream_t stream_;
        ClusteringConfig cfg_;
        ClusteringResult result_;
        Metrics metrics_;
        int label_capacity_{0};
    };
} // namespace holo::cuda

namespace holo::ai { using TopologyClusterer = holo::cuda::TopologyClusterer; }
