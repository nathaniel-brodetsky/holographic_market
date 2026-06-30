#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <compute/cuda_utils.cuh>
#include <compute/lobpcg_solver.cuh>

namespace holo::cuda
{

static constexpr int   k_hodge_block_dim      = 256;
static constexpr float k_harmonic_threshold   = 1e-4F;
static constexpr float k_arbitrage_threshold  = 1e-3F;

struct HodgeWorkspace final
{
    float   *d_B1{nullptr};
    float   *d_B2{nullptr};
    float   *d_L1{nullptr};
    float   *d_omega{nullptr};
    float   *d_eigenvalues_L1{nullptr};
    float   *d_eigenvectors_L1{nullptr};
    float   *d_gamma{nullptr};
    float   *d_exact{nullptr};
    float   *d_coexact{nullptr};
    float   *d_curl_magnitude{nullptr};
    float   *d_arb_signal{nullptr};
    uint8_t *d_prune_mask{nullptr};
    int     *d_edge_src{nullptr};
    int     *d_edge_dst{nullptr};
    int     *d_triangle_edges{nullptr};

    int   n_nodes{0};
    int   n_edges{0};
    int   n_triangles{0};
    int   n_harmonic_dims{0};
    float yang_mills_action{0.0F};

    HodgeWorkspace() = default;

    ~HodgeWorkspace() noexcept
    {
        auto ff = [](float   *p) noexcept { if (p) cudaFree(p); };
        auto fi = [](int     *p) noexcept { if (p) cudaFree(p); };
        auto fu = [](uint8_t *p) noexcept { if (p) cudaFree(p); };
        ff(d_B1); ff(d_B2); ff(d_L1); ff(d_omega);
        ff(d_eigenvalues_L1); ff(d_eigenvectors_L1);
        ff(d_gamma); ff(d_exact); ff(d_coexact);
        ff(d_curl_magnitude); ff(d_arb_signal);
        fu(d_prune_mask);
        fi(d_edge_src); fi(d_edge_dst); fi(d_triangle_edges);
    }

    HodgeWorkspace(const HodgeWorkspace &)            = delete;
    HodgeWorkspace &operator=(const HodgeWorkspace &) = delete;
    HodgeWorkspace(HodgeWorkspace &&)                 = delete;
    HodgeWorkspace &operator=(HodgeWorkspace &&)      = delete;
};

struct ArbitrageSignal
{
    float        *d_harmonic_flow{nullptr};
    float        *d_curl_magnitude{nullptr};
    float         yang_mills_action{0.0F};
    int           n_active_loops{0};
    int           n_edges{0};
    float         max_curl{0.0F};
    float         mean_curl{0.0F};
    int           n_harmonic_dims{0};
    std::uint64_t signal_ts_ns{0U};
    std::uint64_t timestamp_ns{0U};
};

// ── Kernel declarations (implemented in math/hodge_kernel.cu) ────────────────

__global__ void kernel_build_incidence_B1(
    const int *__restrict__ d_edge_src,
    const int *__restrict__ d_edge_dst,
    int n_edges, int n_nodes,
    float *d_B1);

__global__ void kernel_build_incidence_B2(
    const int *__restrict__ d_triangle_edges,
    int n_triangles, int n_edges,
    float *d_B2);

__global__ void kernel_hodge_decompose(
    const float *__restrict__ d_eigenvectors,
    const float *__restrict__ d_eigenvalues,
    int n_edges, int n_eigs,
    float *d_gamma, float *d_exact, float *d_coexact,
    float k_harmonic_threshold_val);

__global__ void kernel_yang_mills_action(
    const float *__restrict__ d_gamma,
    const float *__restrict__ d_exact,
    const float *__restrict__ d_coexact,
    int n_edges, float coupling,
    float *d_ym_action);

__global__ void kernel_count_active_loops(
    const float *__restrict__ d_arb_signal,
    int n_edges, int *d_count);

__global__ void kernel_build_edge_list_from_csr(
    const int *__restrict__ d_row_ptr,
    const int *__restrict__ d_col_idx,
    int n_nodes,
    int *d_edge_src, int *d_edge_dst, int *d_edge_counter);

__global__ void kernel_find_triangles(
    const int *__restrict__ d_row_ptr,
    const int *__restrict__ d_col_idx,
    const int *__restrict__ d_edge_src,
    const int *__restrict__ d_edge_dst,
    int n_edges, int n_nodes,
    int *d_triangle_edges, int *d_triangle_counter);

// ── Host API ─────────────────────────────────────────────────────────────────

void hodge_workspace_init(
    HodgeWorkspace &ws,
    int             n_nodes,
    int             max_edges,
    int             max_triangles,
    cudaStream_t    stream);

void build_incidence_matrices(
    HodgeWorkspace        &ws,
    const SparseLaplacian &laplacian,
    const uint8_t         *d_prune_mask,
    cusparseHandle_t       cusparse,
    cudaStream_t           stream);

void compute_hodge_decomposition(
    HodgeWorkspace   &ws,
    cublasHandle_t    cublas,
    cusparseHandle_t  cusparse,
    cudaStream_t      stream);

ArbitrageSignal extract_arbitrage_signal(
    HodgeWorkspace &ws,
    cudaStream_t    stream);

void copy_signal_to_host(
    const ArbitrageSignal &signal,
    float                 *h_harmonic_out,
    float                 *h_curl_out,
    int                    n_edges,
    cudaStream_t           stream);

    __global__ void kernel_build_flow_vector(
        const float *__restrict__ d_bid_prices,
        const float *__restrict__ d_ask_prices,
        const float *__restrict__ d_bid_qtys,
        const float *__restrict__ d_ask_qtys,
        const int *__restrict__ d_edge_src,
        const int *__restrict__ d_edge_dst,
        int n_edges,
        std::uint32_t depth,
        float *d_omega);

} // namespace holo::cuda

namespace holo::math { namespace cuda = holo::cuda; }

