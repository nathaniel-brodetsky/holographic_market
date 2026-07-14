#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>

#include <cuda_runtime.h>

#include <math/cuda_utils.cuh>
#include <math/lobpcg_solver.cuh>
#include <math/hodge_kernel.cuh>
#include <math/floer_homology.cuh>
#include <core/lob_core.hpp>
#include <core/memory_arena.hpp>

namespace holo::cuda
{

static constexpr std::uint64_t k_pipeline_poll_ns  = 1'000'000ULL;
static constexpr std::size_t   k_signal_history_len = 1024U;

struct alignas(16) GpuLobSnapshot
{
    float        *d_bid_prices{nullptr};
    float        *d_ask_prices{nullptr};
    float        *d_bid_qtys{nullptr};
    float        *d_ask_qtys{nullptr};
    std::uint32_t n_instruments{0U};
    std::uint32_t depth{0U};
    std::uint64_t generation{0U};
    std::uint64_t transfer_ts_ns{0U};
};

// Diagnostics for the seqlock-guarded staging copy in GpuLobMirror.
// snapshots_taken / retries let us confirm in production that the writer
// never actually starves the reader (retries should stay ~0..low single
// digits); forced_after_max_retries should stay at 0 forever — if it
// isn't, the bounded-retry fallback below is firing and a torn snapshot
// may have been pushed to the GPU.
struct alignas(holo::core::k_cache_line) MirrorMetrics
{
    std::atomic<std::uint64_t> snapshots_taken{0U};
    std::atomic<std::uint64_t> retries{0U};
    std::atomic<std::uint64_t> forced_after_max_retries{0U};
    std::byte _pad[holo::core::k_cache_line - 3U * sizeof(std::atomic<std::uint64_t>)];
};

static_assert(sizeof(MirrorMetrics) == holo::core::k_cache_line);

class GpuLobMirror final
{
public:
    GpuLobMirror(const holo::core::LobSoA &lob_soa, cudaStream_t transfer_stream);
    ~GpuLobMirror() noexcept;

    GpuLobMirror(const GpuLobMirror &)            = delete;
    GpuLobMirror &operator=(const GpuLobMirror &) = delete;
    GpuLobMirror(GpuLobMirror &&)                 = delete;
    GpuLobMirror &operator=(GpuLobMirror &&)      = delete;

    void async_push(std::uint64_t current_generation) noexcept;

    [[nodiscard]] GpuLobSnapshot snapshot() const noexcept;
    [[nodiscard]] std::size_t n_floats()      const noexcept { return n_floats_; }
    [[nodiscard]] std::size_t n_instruments() const noexcept { return n_instruments_; }
    [[nodiscard]] std::size_t depth()         const noexcept { return depth_; }
    [[nodiscard]] const MirrorMetrics &metrics() const noexcept { return metrics_; }

private:
    // Seqlock-guarded host-to-host copy: reads a consistent snapshot of
    // LobSoA's raw arrays into the pinned staging buffers below. Bounded
    // retry loop -- see .cu for rationale.
    void stage_snapshot() noexcept;

    std::size_t  n_instruments_;
    std::size_t  depth_;
    std::size_t  n_floats_;
    cudaStream_t stream_;
    std::uint64_t generation_;

    const holo::core::LobSoA *lob_ref_;

    const float *h_bid_prices_;
    const float *h_ask_prices_;
    const float *h_bid_qtys_;
    const float *h_ask_qtys_;

    // Pinned staging (shadow) buffers. stage_snapshot() copies a
    // seqlock-verified consistent snapshot from h_*_ into these; only
    // THESE are ever the source of cudaMemcpyAsync, so the async DMA
    // never races against LobSoA's writer thread.
    float *stage_bid_prices_{nullptr};
    float *stage_ask_prices_{nullptr};
    float *stage_bid_qtys_{nullptr};
    float *stage_ask_qtys_{nullptr};

    float *d_bid_prices_{nullptr};
    float *d_ask_prices_{nullptr};
    float *d_bid_qtys_{nullptr};
    float *d_ask_qtys_{nullptr};

    MirrorMetrics metrics_{};
};

struct alignas(64) PipelineMetrics
{
    std::atomic<std::uint64_t> n_transfers{0U};
    std::atomic<std::uint64_t> n_decompositions{0U};
    std::atomic<std::uint64_t> n_arbitrage_signals{0U};
    std::atomic<std::uint64_t> total_transfer_ns{0U};
    std::atomic<std::uint64_t> total_decomp_ns{0U};
    std::atomic<float>         last_ym_action{0.0F};
    std::atomic<int>           last_harmonic_dims{0};

    static constexpr std::size_t k_used =
        5U * sizeof(std::atomic<std::uint64_t>) +
        1U * sizeof(std::atomic<float>) +
        1U * sizeof(std::atomic<int>);
    std::byte _pad[64U - k_used % 64U]{};
};

struct SignalRecord
{
    std::uint64_t timestamp_ns{0U};
    float         yang_mills_action{0.0F};
    int           n_active_loops{0};
    int           n_harmonic_dims{0};
    float         max_curl{0.0F};
    float         mean_curl{0.0F};
    int           n_edges{0};
    int           floer_HF0{0};
    int           floer_HF1{0};
    int           floer_HF2{0};
    int           floer_euler{0};
    int           floer_n_instantons{0};
    float         floer_entry_action{0.0F};
    float         floer_exit_action{0.0F};
    float         floer_instanton_len{0.0F};
    float         floer_asd_residual{0.0F};
    std::uint8_t  floer_signal{1U};
    bool          floer_instanton_found{false};
};

    struct TopologySnapshot {
        const float* harmonic_flow{nullptr};
        const int*   edge_src{nullptr};
        const int*   edge_dst{nullptr};
        int          n_edges{0};
    };

class CudaPipeline final
{
public:
    CudaPipeline(
        holo::core::LobSoA  &lob_soa,
        holo::core::MemoryArena &arena,
        int                  gpu_device = 0);

    ~CudaPipeline() noexcept;

    CudaPipeline(const CudaPipeline &)            = delete;
    CudaPipeline &operator=(const CudaPipeline &) = delete;
    CudaPipeline(CudaPipeline &&)                 = delete;
    CudaPipeline &operator=(CudaPipeline &&)      = delete;

    void run_once();
    void run_continuous(std::atomic<bool> &shutdown_flag);

    [[nodiscard]] TopologySnapshot last_topology() const noexcept
    {
        return TopologySnapshot{
            .harmonic_flow = h_harmonic_out_,
            .edge_src      = h_edge_src_,
            .edge_dst      = h_edge_dst_,
            .n_edges       = last_signal().n_edges
        };
    }

    [[nodiscard]] const PipelineMetrics &metrics() const noexcept { return metrics_; }

    [[nodiscard]] SignalRecord last_signal() const noexcept
    {
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
    void record_signal(const ArbitrageSignal &sig);
    void run_floer_pass(SignalRecord &rec);

    holo::core::LobSoA &lob_soa_;

    CudaHandles   handles_;
    StreamGuard   transfer_stream_;
    StreamGuard   compute_stream_;
    StreamGuard   signal_stream_;

    EventGuard    ev_transfer_start_;
    EventGuard    ev_transfer_done_;
    EventGuard    ev_compute_start_;
    EventGuard    ev_compute_done_;

    GpuLobMirror   *gpu_mirror_{nullptr};
    SparseLaplacian  laplacian_;
    LobpcgWorkspace  lobpcg_ws_;
    HodgeWorkspace   hodge_ws_;
    FloerWorkspace   floer_ws_;

    float *h_harmonic_out_{nullptr};
    float *h_curl_out_{nullptr};
    int   *h_edge_src_{nullptr};
    int   *h_edge_dst_{nullptr};

    PipelineMetrics  metrics_;

    SignalRecord              signal_history_[k_signal_history_len]{};
    std::atomic<std::uint64_t> signal_write_idx_{0U};

    std::uint64_t last_lob_generation_{0U};
    int           sm_count_{0};
};

} // namespace holo::cuda

namespace holo::math { namespace cuda = holo::cuda; }
