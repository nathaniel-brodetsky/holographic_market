// STUB — NOT IMPLEMENTED. Every method below is an empty body; this class
// is also not instantiated anywhere in main_live.cpp / main_backtest.cpp /
// cuda_pipeline.cu. See README.md section III.5 for the intended design.
// Do not assume regime-clustering output influences trading decisions —
// it currently cannot, because fit() never runs.
#include "cuml_clustering.cuh"
namespace holo::cuda {
    TopologyClusterer::TopologyClusterer(cudaStream_t /*stream*/, ClusteringConfig /*cfg*/) noexcept {}
    TopologyClusterer::~TopologyClusterer() noexcept {}
    void TopologyClusterer::fit(const float*, int, int) {}
    void TopologyClusterer::ensure_label_buffer(int) {}
}
