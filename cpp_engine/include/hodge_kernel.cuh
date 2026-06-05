#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_utils.cuh>
#include <lobpcg_solver.cuh>

namespace holo::cuda {
    static constexpr int k_hodge_warp_size = 32;
    static constexpr int k_hodge_block_dim = 256;
    static constexpr float k_harmonic_threshold = 1e-4F;
    static constexpr float k_arbitrage_threshold = 1e-3F;
    static constexpr int k_max_edges = 65536;
    static constexpr int k_max_triangles = 131072;

    struct HodgeWorkspace {
        float *d_B1{nullptr};
        float *d_B2{nullptr};
        float *d_L1{nullptr};
        float *d_omega{nullptr};
        float *d_eigenvalues_L1{nullptr};
        float *d_eigenvectors_L1{nullptr};
        float *d_gamma{nullptr};
        float *d_exact{nullptr};
        float *d_coexact{nullptr};
        float *d_curl_magnitude{nullptr};
        float *d_arb_signal{nullptr};
        uint8_t *d_prune_mask{nullptr};
        int *d_edge_src{nullptr};
        int *d_edge_dst{nullptr};
        int *d_triangle_edges{nullptr};
        int n_nodes{0};
        int n_edges{0};
        int n_triangles{0};
        int n_harmonic_dims{0};
        float yang_mills_action{0.0F};

        ~HodgeWorkspace() noexcept {
            auto f = [](float *p) { if (p) cudaFree(p); };
            auto fi = [](int *p) { if (p) cudaFree(p); };
            auto fu = [](uint8_t *p) { if (p) cudaFree(p); };
            f(d_B1);
            f(d_B2);
            f(d_L1);
            f(d_omega);
            f(d_eigenvalues_L1);
            f(d_eigenvectors_L1);
            f(d_gamma);
            f(d_exact);
            f(d_coexact);
            f(d_curl_magnitude);
            f(d_arb_signal);
            fu(d_prune_mask);
            fi(d_edge_src);
            fi(d_edge_dst);
            fi(d_triangle_edges);
        }

        HodgeWorkspace(const HodgeWorkspace &) = delete;

        HodgeWorkspace &operator=(const HodgeWorkspace &) = delete;

        HodgeWorkspace() = default;
    };

    struct ArbitrageSignal {
        float *d_harmonic_flow;
        float *d_curl_magnitude;
        float yang_mills_action;
        int n_active_loops;
        int n_edges;
        uint64_t signal_ts_ns;
    };

    void hodge_workspace_init(
        HodgeWorkspace &ws,
        int n_nodes,
        int max_edges,
        int max_triangles,
        cudaStream_t stream);

    void build_incidence_matrices(
        HodgeWorkspace &ws,
        const SparseLaplacian &pruned_laplacian,
        const uint8_t *d_prune_mask,
        cusparseHandle_t cusparse,
        cudaStream_t stream);

    void compute_hodge_decomposition(
        HodgeWorkspace &ws,
        cublasHandle_t cublas,
        cusparseHandle_t cusparse,
        cudaStream_t stream);

    ArbitrageSignal extract_arbitrage_signal(
        HodgeWorkspace &ws,
        cudaStream_t stream);

    void copy_signal_to_host(
        const ArbitrageSignal &signal,
        float *h_harmonic_out,
        float *h_curl_out,
        int n_edges,
        cudaStream_t stream);
} // namespace holo::cuda
