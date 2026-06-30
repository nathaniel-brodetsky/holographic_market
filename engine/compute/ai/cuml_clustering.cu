#include "cuml_clustering.cuh"
namespace holo::cuda {
    TopologyClusterer::TopologyClusterer(cudaStream_t stream, ClusteringConfig cfg) noexcept {}
    TopologyClusterer::~TopologyClusterer() noexcept {}
    void TopologyClusterer::fit(const float*, int, int) {}
    void TopologyClusterer::ensure_label_buffer(int) {}
}
