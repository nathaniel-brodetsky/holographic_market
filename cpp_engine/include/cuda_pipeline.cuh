#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cuda_runtime.h>
#include <cuda_utils.cuh>
#include <gpu_lob_mirror.cuh>
#include <lobpcg_solver.cuh>
#include <hodge_kernel.cuh>
#include <lob_core.hpp>
#include <memory_arena.hpp>

namespace holo::cuda {
    static constexpr uint64_t k_pipeline_poll_ns = 1'000'000ULL;
    static constexpr int k_n_cuda_streams = 3;
    static constexpr size_t k_signal_history_len = 1024U;

    struct alignas(64) PipelineMetrics {
        std::atomic<uint64_t> n_transfers{0U};
        std::atomic<uint64_t> n_decompositions{0U};
        std::atomic<uint64_t> n_arbitrage_signals{0U};
        std::atomic<uint64_t> total_transfer_ns{0U};
        std::atomic<uint64_t> total_decomp_ns{0U};
        std::atomic<float> last_ym_action{0.0F};
        std::atomic<int> last_harmonic_dims{0};
        std::byte _pad[64U
                       - 6U * sizeof(std::atomic<uint64_t>)
                       - 1U * sizeof(std::atomic<float>)
                       - 1U * sizeof(std::atomic<int>)
                       - sizeof(std::byte *)
        ];
    };

    struct SignalRecord {
        uint64_t timestamp_ns;
        float yang_mills_action;
        int n_active_loops;
        int n_harmonic_dims;
        float max_curl;
        float mean_curl;
    };

    class CudaPipeline final {
    public:
        CudaPipeline(
            LobSoA &lob_soa,
            MemoryArena &arena,
            int gpu_device = 0);

        ~CudaPipeline() noexcept;

        CudaPipeline(const CudaPipeline &) = delete;

        CudaPipeline &operator=(const CudaPipeline &) = delete;

        CudaPipeline(CudaPipeline &&) = delete;

        CudaPipeline &operator=(CudaPipeline &&) = delete;

        void run_once();

        void run_continuous(std::atomic<bool> &shutdown_flag);

        [[nodiscard]] const PipelineMetrics &metrics() const noexcept {
            return metrics_;
        }

        [[nodiscard]] SignalRecord last_signal() const noexcept {
            return signal_history_[
                (signal_write_idx_.load(std::memory_order_acquire)
                 + k_signal_history_len - 1U)
                % k_signal_history_len];
        }

    private:
        void init_gpu(int device);

        void transfer_lob_to_gpu();

        void run_spectral_pruning();

        void run_hodge_decomposition();

        void record_signal(const ArbitrageSignal &sig);

        LobSoA &lob_soa_;

        CudaHandles handles_;
        StreamGuard transfer_stream_;
        StreamGuard compute_stream_;
        StreamGuard signal_stream_;

        EventGuard ev_transfer_start_;
        EventGuard ev_transfer_done_;
        EventGuard ev_compute_start_;
        EventGuard ev_compute_done_;

        GpuLobMirror *gpu_mirror_{nullptr};
        SparseLaplacian laplacian_;
        LobpcgWorkspace lobpcg_ws_;
        HodgeWorkspace hodge_ws_;

        float *h_harmonic_out_{nullptr};
        float *h_curl_out_{nullptr};

        PipelineMetrics metrics_;

        SignalRecord signal_history_[k_signal_history_len]{};
        std::atomic<uint64_t> signal_write_idx_{0U};

        uint64_t last_lob_generation_{0U};
        int sm_count_{0};
    };
} // namespace holo::cuda
