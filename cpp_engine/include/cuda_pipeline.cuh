#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>

#include <cuda_runtime.h>

#include <cuda_utils.cuh>
#include <lobpcg_solver.cuh>
#include <hodge_kernel.cuh>
#include <lob_core.hpp>
#include <memory_arena.hpp>

namespace holo::cuda {

static constexpr std::uint64_t k_pipeline_poll_ns  = 1'000'000ULL;
static constexpr std::size_t   k_signal_history_len = 1024U;

struct alignas(16) GpuLobSnapshot {
    float*        d_bid_prices  {nullptr};
    float*        d_ask_prices  {nullptr};
    float*        d_bid_qtys    {nullptr};
    float*        d_ask_qtys    {nullptr};
    std::uint32_t n_instruments {0U};
    std::uint32_t depth         {0U};
    std::uint64_t generation    {0U};
    std::uint64_t transfer_ts_ns{0U};
};

class GpuLobMirror final {
public:
    GpuLobMirror(
        const LobSoA& lob_soa,
        cudaStream_t  transfer_stream);

    ~GpuLobMirror() noexcept;

    GpuLobMirror(const GpuLobMirror&)            = delete;
    GpuLobMirror& operator=(const GpuLobMirror&) = delete;
    GpuLobMirror(GpuLobMirror&&)                 = delete;
    GpuLobMirror& operator=(GpuLobMirror&&)      = delete;

    void async_push(std::uint64_t current_generation) noexcept;

    [[nodiscard]] GpuLobSnapshot snapshot() const noexcept;

    [[nodiscard]] std::size_t n_floats()      const noexcept { return n_floats_;      }
    [[nodiscard]] std::size_t n_instruments() const noexcept { return n_instruments_; }
    [[nodiscard]] std::size_t depth()         const noexcept { return depth_;         }

private:
    std::size_t    n_instruments_;
    std::size_t    depth_;
    std::size_t    n_floats_;
    cudaStream_t   stream_;
    std::uint64_t  generation_;

    const float*   h_bid_prices_;
    const float*   h_ask_prices_;
    const float*   h_bid_qtys_;
    const float*   h_ask_qtys_;

    float*         d_bid_prices_{nullptr};
    float*         d_ask_prices_{nullptr};
    float*         d_bid_qtys_  {nullptr};
    float*         d_ask_qtys_  {nullptr};
};

struct alignas(64) PipelineMetrics {
    std::atomic<std::uint64_t> n_transfers          {0U};
    std::atomic<std::uint64_t> n_decompositions     {0U};
    std::atomic<std::uint64_t> n_arbitrage_signals  {0U};
    std::atomic<std::uint64_t> total_transfer_ns    {0U};
    std::atomic<std::uint64_t> total_decomp_ns      {0U};
    std::atomic<float>         last_ym_action        {0.0F};
    std::atomic<int>           last_harmonic_dims    {0};

    static constexpr std::size_t k_used =
          6U * sizeof(std::atomic<std::uint64_t>)
        + 1U * sizeof(std::atomic<float>)
        + 1U * sizeof(std::atomic<int>);

    std::byte _pad[64U - k_used % 64U]{};
};

struct SignalRecord {
    std::uint64_t timestamp_ns      {0U};
    float         yang_mills_action {0.0F};
    int           n_active_loops    {0};
    int           n_harmonic_dims   {0};
    float         max_curl          {0.0F};
    float         mean_curl         {0.0F};
};

class CudaPipeline final {
public:
    CudaPipeline(
        LobSoA&      lob_soa,
        MemoryArena& arena,
        int          gpu_device = 0);

    ~CudaPipeline() noexcept;

    CudaPipeline(const CudaPipeline&)            = delete;
    CudaPipeline& operator=(const CudaPipeline&) = delete;
    CudaPipeline(CudaPipeline&&)                 = delete;
    CudaPipeline& operator=(CudaPipeline&&)      = delete;

    void run_once();

    void run_continuous(std::atomic<bool>& shutdown_flag);

    [[nodiscard]] const PipelineMetrics& metrics() const noexcept {
        return metrics_;
    }

    [[nodiscard]] SignalRecord last_signal() const noexcept {
        const std::uint64_t write_idx =
            signal_write_idx_.load(std::memory_order_acquire);
        const std::uint64_t read_idx =
            (write_idx + k_signal_history_len - 1U) % k_signal_history_len;
        return signal_history_[read_idx];
    }

private:
    void init_gpu(int device);
    void transfer_lob_to_gpu();
    void run_spectral_pruning();
    void run_hodge_decomposition();
    void record_signal(const ArbitrageSignal& sig);

    LobSoA&       lob_soa_;

    CudaHandles   handles_;
    StreamGuard   transfer_stream_;
    StreamGuard   compute_stream_;
    StreamGuard   signal_stream_;

    EventGuard    ev_transfer_start_;
    EventGuard    ev_transfer_done_;
    EventGuard    ev_compute_start_;
    EventGuard    ev_compute_done_;

    GpuLobMirror*    gpu_mirror_   {nullptr};
    SparseLaplacian  laplacian_;
    LobpcgWorkspace  lobpcg_ws_;
    HodgeWorkspace   hodge_ws_;

    float*           h_harmonic_out_{nullptr};
    float*           h_curl_out_    {nullptr};

    PipelineMetrics  metrics_;

    SignalRecord     signal_history_[k_signal_history_len]{};
    std::atomic<std::uint64_t> signal_write_idx_{0U};

    std::uint64_t    last_lob_generation_{0U};
    int              sm_count_           {0};
};

} // namespace holo::cuda