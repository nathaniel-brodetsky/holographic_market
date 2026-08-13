// Phase III, stage ⑥: on-device DBSCAN regime clustering over a rolling
// window of (S_YM, max|γ|, mean|γ|, β₁) feature vectors — see README.md
// §III.5 for the design this implements.
//
// Gated on HOLO_HAS_CUML, which CMakeLists.txt only defines when a RAPIDS
// cuML install is found (see DEPENDENCIES.md "RAPIDS cuML Installation").
// Without it, this compiles to the same no-op behavior the stub had before,
// except fit() now logs once so a missing cuML install is visible instead
// of silently producing zero clusters forever — matches the "gracefully
// bypass Phase VI" language in DEPENDENCIES.md "Lightweight Builds".
//
// SCOPE OF THIS CHANGE — what this file does and does not do:
//   DOES:    makes TopologyClusterer::fit() and ensure_label_buffer()
//            functionally correct, calling the real ML::Dbscan::fit()
//            from cuML against a raft::handle_t bound to this object's
//            stream. Smoke-tested standalone (../smoke_test_clustering.cu)
//            against a real cuML install: 450 synthetic samples in 3
//            well-separated blobs -> 3 clusters, 0 noise, no crash.
//   DOES:    is now instantiated and driven from CudaPipeline
//            (cuda_pipeline.cu, CudaPipeline::build_cluster_window_and_fit())
//            over the rolling (S_YM, max|γ|, mean|γ|, β₁) window built from
//            signal_history_. See cuda_pipeline.cuh/.cu for the wiring.
//   DOES NOT: get exercised end-to-end against a live/replayed feed yet —
//            only against the synthetic smoke test and (via the pipeline
//            wiring) still needs a real run of main_live.cpp /
//            main_backtest.cpp to confirm behavior on real
//            market-derived feature vectors, not just well-separated
//            synthetic blobs.
//
// Built and smoke-tested against a real RAPIDS cuML install (see above),
// but only the standalone TopologyClusterer::fit() path, with cfg.eps
// chosen for that synthetic data. The production ClusteringConfig{}
// default (eps=0.15F) is tuned for real normalized signal data and has
// NOT itself been validated yet — re-check it once live/replayed feature
// vectors are actually flowing through build_cluster_window_and_fit().
// Also double-check that this branch's cuML/RAFT version keeps matching
// the ML::Dbscan::fit header signature below if you ever bump CUML_ROOT —
// that argument order has changed across cuML releases before.

#include "cuml_clustering.cuh"
#include <math/cuda_utils.cuh>   // CUDA_CHECK

#include <cstdio>

#if defined(HOLO_HAS_CUML)
#include <algorithm>
#include <chrono>
#include <vector>

#include <cuml/cluster/dbscan.hpp>
#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>
#endif

namespace holo::cuda {

TopologyClusterer::TopologyClusterer(cudaStream_t stream, ClusteringConfig cfg) noexcept
    : stream_(stream), cfg_(cfg) {}

TopologyClusterer::~TopologyClusterer() noexcept {
    // Best-effort: a destructor must not throw, so this intentionally does
    // not go through CUDA_CHECK (which aborts on failure). A failed free
    // here means the CUDA context is already in a bad state on shutdown;
    // there's nothing more useful to do than leak and move on.
    if (result_.d_labels != nullptr) {
        cudaFreeAsync(result_.d_labels, stream_);
    }
}

void TopologyClusterer::ensure_label_buffer(int n_samples) {
    if (n_samples <= label_capacity_) return;

    // Grows once and stays put — this is the "zero allocations after
    // warmup" contract. If this fires on every call, the caller's window
    // size isn't actually fixed, which defeats that contract; that's a
    // caller bug, not something to paper over here.
    if (result_.d_labels != nullptr) {
        CUDA_CHECK(cudaFreeAsync(result_.d_labels, stream_));
    }
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&result_.d_labels),
                                static_cast<std::size_t>(n_samples) * sizeof(int),
                                stream_));
    label_capacity_ = n_samples;
}

void TopologyClusterer::fit(const float* d_features, int n_samples, int n_features) {
#if defined(HOLO_HAS_CUML)
    if (d_features == nullptr || n_samples <= 0 || n_features <= 0) return;

    const auto t0 = std::chrono::steady_clock::now();

    ensure_label_buffer(n_samples);

    // raft::handle_t derives its cuBLAS/cuSOLVER/cuSPARSE sub-handles from
    // this stream_, so DBSCAN's internal work stays ordered with the rest
    // of the pipeline without an explicit sync — right up until the
    // cudaStreamSynchronize below, which IS required because we read the
    // labels back on the host afterward.
    //
    // Constructing a raft::handle_t isn't free (it allocates a small
    // amount of host-side state). At the ~5-20ms async DBSCAN budget from
    // README §III.6 this is noise, but if fit() ever moves onto a tighter
    // budget, hoist this into a member constructed once in the ctor
    // instead of once per call.
    const raft::handle_t handle{stream_};

    // ML::Dbscan::fit takes a non-const float* (some cuML code paths reuse
    // `input` as scratch for the pairwise-distance matrix). d_features is
    // caller-owned device memory — if the caller needs it unmodified after
    // this call, it must pass a copy, not the live signal buffer.
    ML::Dbscan::fit(handle,
                     const_cast<float*>(d_features),
                     n_samples,
                     n_features,
                     cfg_.eps,
                     cfg_.min_samples,
                     raft::distance::DistanceType::L2SqrtUnexpanded,
                     result_.d_labels,
                     /*core_sample_indices=*/nullptr,
                     /*sample_weight=*/nullptr,
                     static_cast<std::size_t>(cfg_.max_bytes_per_batch));

    // cuML's fit() doesn't return cluster/noise counts directly — derive
    // them from the labels. This copy is small (n_samples ints) and only
    // happens on the async/every-10-cycles DBSCAN path, not the <1ms
    // per-tick hot path, so it isn't competing with the latency budget in
    // README §III.6.
    std::vector<int> h_labels(static_cast<std::size_t>(n_samples));
    CUDA_CHECK(cudaMemcpyAsync(h_labels.data(), result_.d_labels,
                                h_labels.size() * sizeof(int),
                                cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // cuML convention: cluster ids are 0..K-1 contiguous, noise is -1.
    int max_label = -1;
    int noise = 0;
    for (const int lbl : h_labels) {
        if (lbl < 0) ++noise; else max_label = std::max(max_label, lbl);
    }
    result_.n_clusters = max_label + 1;
    result_.n_noise    = noise;
    result_.n_samples  = n_samples;

    const auto elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());

    metrics_.n_fit_calls.fetch_add(1, std::memory_order_relaxed);
    metrics_.total_fit_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    metrics_.last_n_clusters.store(result_.n_clusters, std::memory_order_relaxed);
    metrics_.last_n_noise.store(result_.n_noise, std::memory_order_relaxed);
#else
    // No cuML at build time (see DEPENDENCIES.md "Lightweight Builds (No
    // cuML)"). Loud once, not silent forever: a missing install should be
    // visible in the logs, not indistinguishable from "zero clusters found
    // all the time."
    static bool warned = false;
    if (!warned) {
        std::fprintf(stderr,
            "[TopologyClusterer] built without HOLO_HAS_CUML -- fit() is a "
            "no-op and result()/metrics() will stay at their defaults. "
            "Install RAPIDS cuML and reconfigure CMake to enable stage 6 "
            "regime clustering (see DEPENDENCIES.md).\n");
        warned = true;
    }
    (void)d_features;
    (void)n_samples;
    (void)n_features;
#endif
}

} // namespace holo::cuda
