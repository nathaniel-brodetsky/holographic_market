#include <cuda_pipeline.cuh>
#include <cuda_utils.cuh>
#include <hodge_kernel.cuh>
#include <lobpcg_solver.cuh>

#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>

#if defined(__linux__)
#include <time.h>
#endif

namespace holo::cuda {

static uint64_t pipeline_now_ns() noexcept {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
}

CudaPipeline::CudaPipeline(
    LobSoA&      lob_soa,
    MemoryArena& arena,
    int          gpu_device)
    : lob_soa_         {lob_soa}
    , ev_transfer_start_{cudaEventDisableTiming}
    , ev_transfer_done_ {cudaEventDefault}
    , ev_compute_start_ {cudaEventDisableTiming}
    , ev_compute_done_  {cudaEventDefault}
{
    (void)arena;
    init_gpu(gpu_device);

    gpu_mirror_ = new GpuLobMirror(lob_soa_, transfer_stream_);

    const int n = static_cast<int>(lob_soa_.n_instruments());

    hodge_workspace_init(
        hodge_ws_,
        n,
        k_max_edges,
        k_max_triangles,
        compute_stream_);

    h_harmonic_out_ = pinned_alloc<float>(static_cast<size_t>(k_max_edges));
    h_curl_out_     = pinned_alloc<float>(static_cast<size_t>(k_max_edges));

    std::memset(h_harmonic_out_, 0, static_cast<size_t>(k_max_edges) * sizeof(float));
    std::memset(h_curl_out_,     0, static_cast<size_t>(k_max_edges) * sizeof(float));

    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

CudaPipeline::~CudaPipeline() noexcept {
    cudaStreamSynchronize(transfer_stream_);
    cudaStreamSynchronize(compute_stream_);
    cudaStreamSynchronize(signal_stream_);

    delete gpu_mirror_;

    if (h_harmonic_out_) cudaFreeHost(h_harmonic_out_);
    if (h_curl_out_)     cudaFreeHost(h_curl_out_);
}

void CudaPipeline::init_gpu(int device) {
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    sm_count_ = prop.multiProcessorCount;

    std::printf(
        "  [GPU] %-30s | SM: %d | VRAM: %.1f GB | "
        "PCIe Gen%d x%d | CC: %d.%d\n",
        prop.name,
        prop.multiProcessorCount,
        static_cast<double>(prop.totalGlobalMem) / 1e9,
        prop.pciBusID,
        prop.pciDeviceID,
        prop.major, prop.minor);
}

void CudaPipeline::transfer_lob_to_gpu() {
    const uint64_t gen =
        lob_soa_.stats().total_updates.load(std::memory_order_acquire);

    if (gen == last_lob_generation_) return;

    ev_transfer_start_.record(transfer_stream_);
    gpu_mirror_->async_push(gen);
    ev_transfer_done_.record(transfer_stream_);

    CUDA_CHECK(cudaStreamWaitEvent(
        compute_stream_, ev_transfer_done_.event, 0));

    last_lob_generation_ = gen;

    const uint64_t t0 = pipeline_now_ns();
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
    const uint64_t t1 = pipeline_now_ns();

    metrics_.total_transfer_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
    metrics_.n_transfers.fetch_add(1U, std::memory_order_relaxed);
}

void CudaPipeline::run_spectral_pruning() {
    const GpuLobSnapshot snap = gpu_mirror_->snapshot();
    if (snap.n_instruments == 0U) return;

    build_normalized_laplacian(
        snap.d_bid_prices,
        snap.d_ask_prices,
        snap.n_instruments,
        snap.depth,
        laplacian_,
        handles_.cusparse,
        compute_stream_);

    if (laplacian_.n_rows < 4) return;

    lobpcg_workspace_init(
        lobpcg_ws_,
        laplacian_.n_rows,
        k_lobpcg_block_size,
        laplacian_,
        handles_.cusparse,
        compute_stream_);

    const FiedlerResult fiedler = lobpcg_solve(
        lobpcg_ws_,
        laplacian_,
        handles_.cublas,
        handles_.cusparse,
        compute_stream_);

    fiedler_prune_mask(
        fiedler.d_fiedler_vec,
        laplacian_.n_rows,
        99.0F,
        hodge_ws_.d_prune_mask,
        compute_stream_);
}

void CudaPipeline::run_hodge_decomposition() {
    if (laplacian_.n_rows < 4 || laplacian_.nnz == 0) return;

    const GpuLobSnapshot snap = gpu_mirror_->snapshot();

    build_incidence_matrices(
        hodge_ws_,
        laplacian_,
        hodge_ws_.d_prune_mask,
        handles_.cusparse,
        compute_stream_);

    if (hodge_ws_.n_edges <= 0) return;

    const int blk = k_blk;
    const int grd = (hodge_ws_.n_edges + blk - 1) / blk;

    kernel_build_flow_vector<<<grd, blk, 0, compute_stream_>>>(
        snap.d_bid_prices,
        snap.d_ask_prices,
        snap.d_bid_qtys,
        snap.d_ask_qtys,
        hodge_ws_.d_edge_src,
        hodge_ws_.d_edge_dst,
        hodge_ws_.n_edges,
        snap.depth,
        hodge_ws_.d_omega);

    const uint64_t t0 = pipeline_now_ns();

    compute_hodge_decomposition(
        hodge_ws_,
        handles_.cublas,
        handles_.cusparse,
        compute_stream_);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    const uint64_t t1 = pipeline_now_ns();

    metrics_.total_decomp_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
    metrics_.n_decompositions.fetch_add(1U, std::memory_order_relaxed);
}

void CudaPipeline::record_signal(const ArbitrageSignal& sig) {
    if (sig.n_edges <= 0) return;

    copy_signal_to_host(
        sig,
        h_harmonic_out_,
        h_curl_out_,
        sig.n_edges,
        signal_stream_);

    CUDA_CHECK(cudaStreamSynchronize(signal_stream_));

    float max_curl  = 0.0F;
    float mean_curl = 0.0F;
    for (int e = 0; e < sig.n_edges; ++e) {
        if (h_curl_out_[e] > max_curl) max_curl = h_curl_out_[e];
        mean_curl += h_curl_out_[e];
    }
    mean_curl /= static_cast<float>(sig.n_edges);

    const uint64_t idx =
        signal_write_idx_.fetch_add(1U, std::memory_order_acq_rel)
        % k_signal_history_len;

    signal_history_[idx] = SignalRecord{
        .timestamp_ns      = sig.signal_ts_ns,
        .yang_mills_action = sig.yang_mills_action,
        .n_active_loops    = sig.n_active_loops,
        .n_harmonic_dims   = hodge_ws_.n_harmonic_dims,
        .max_curl          = max_curl,
        .mean_curl         = mean_curl,
    };

    metrics_.n_arbitrage_signals.fetch_add(1U, std::memory_order_relaxed);
    metrics_.last_ym_action.store(sig.yang_mills_action, std::memory_order_relaxed);
    metrics_.last_harmonic_dims.store(
        hodge_ws_.n_harmonic_dims, std::memory_order_relaxed);
}

void CudaPipeline::run_once() {
    transfer_lob_to_gpu();
    run_spectral_pruning();
    run_hodge_decomposition();

    if (hodge_ws_.n_edges > 0) {
        ArbitrageSignal sig = extract_arbitrage_signal(hodge_ws_, compute_stream_);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        record_signal(sig);
    }
}

void CudaPipeline::run_continuous(std::atomic<bool>& shutdown_flag) {
    while (!shutdown_flag.load(std::memory_order_acquire)) {
        run_once();

        const uint64_t t0 = pipeline_now_ns();
        while (pipeline_now_ns() - t0 < k_pipeline_poll_ns
            && !shutdown_flag.load(std::memory_order_relaxed))
        {
            __builtin_ia32_pause();
        }
    }
}

} // namespace holo::cuda