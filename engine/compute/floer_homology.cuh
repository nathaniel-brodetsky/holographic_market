#pragma once

#include <cuda_runtime.h>
#include <compute/cuda_utils.cuh>

#include <cstdint>

namespace holo::cuda
{

static constexpr int   k_floer_block_dim      = 128;
static constexpr int   k_floer_max_criticals  = 256;
static constexpr float k_instanton_threshold  = 1e-3F;
static constexpr float k_self_dual_tolerance  = 0.15F;

enum class MorseIndex : std::uint8_t
{
    Minimum = 0,
    Saddle  = 1,
    Maximum = 2,
};

struct CriticalPoint
{
    int        edge_idx{-1};
    float      action{0.0F};
    float      curl_magnitude{0.0F};
    MorseIndex morse_index{MorseIndex::Saddle};
    bool       self_dual{false};
};

struct InstantonPath
{
    int   src_critical{-1};
    int   dst_critical{-1};
    float path_action{0.0F};
    float self_dual_residual{0.0F};
    bool  valid{false};
};

struct alignas(64) FloerRecord
{
    MorseIndex  entry_signal{MorseIndex::Saddle};
    bool        instanton_found{false};
    float       entry_action{0.0F};
    float       exit_action{0.0F};
    float       instanton_length{0.0F};
    int         rank_HF0{0};
    int         rank_HF1{0};
    int         rank_HF2{0};
    int         euler_characteristic{0};
    int         n_criticals{0};
    int         n_instantons{0};
    float       mean_self_dual_residual{0.0F};
};

struct FloerWorkspace final
{
    CriticalPoint  *d_criticals{nullptr};
    InstantonPath  *d_instantons{nullptr};
    float          *d_hessian{nullptr};
    int            *d_boundary_matrix{nullptr};
    int            *d_n_criticals{nullptr};
    int            *d_n_instantons{nullptr};

    void allocate(int max_criticals)
    {
        CUDA_CHECK(cudaMalloc(&d_criticals,
            max_criticals * sizeof(CriticalPoint)));
        CUDA_CHECK(cudaMalloc(&d_instantons,
            max_criticals * max_criticals * sizeof(InstantonPath)));
        CUDA_CHECK(cudaMalloc(&d_hessian,
            max_criticals * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_boundary_matrix,
            max_criticals * max_criticals * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_n_criticals, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_n_instantons, sizeof(int)));
    }

    void free_all() noexcept
    {
        auto ff = [](void *p) noexcept { if (p) cudaFree(p); };
        ff(d_criticals); ff(d_instantons); ff(d_hessian);
        ff(d_boundary_matrix); ff(d_n_criticals); ff(d_n_instantons);
        d_criticals = nullptr; d_instantons = nullptr;
    }

    ~FloerWorkspace() noexcept { free_all(); }
};

__global__ void kernel_find_critical_points(
    const float *__restrict__ d_curl_magnitude,
    const float *__restrict__ d_curl_coexact,
    const int   *__restrict__ d_edge_src,
    const int   *__restrict__ d_edge_dst,
    int          n_edges,
    float        ym_action,
    CriticalPoint *d_criticals,
    int           *d_n_criticals);

__global__ void kernel_find_instantons(
    const CriticalPoint *__restrict__ d_criticals,
    int                              n_criticals,
    const float *__restrict__        d_curl_magnitude,
    const float *__restrict__        d_curl_coexact,
    int                              n_edges,
    float                            ym_action,
    InstantonPath                   *d_instantons,
    int                             *d_n_instantons);

__global__ void kernel_build_floer_boundary(
    const CriticalPoint *__restrict__ d_criticals,
    const InstantonPath *__restrict__ d_instantons,
    int   n_criticals,
    int   n_instantons,
    int  *d_boundary_matrix);

FloerRecord compute_floer_homology(
    const CriticalPoint *h_criticals,
    const InstantonPath *h_instantons,
    const int           *h_boundary_matrix,
    int                  n_criticals,
    int                  n_instantons);

FloerRecord run_floer_analysis(
    FloerWorkspace      &floer_ws,
    const float         *d_curl_magnitude,
    const float         *d_curl_coexact,
    const int           *d_edge_src,
    const int           *d_edge_dst,
    int                  n_edges,
    float                ym_action,
    cudaStream_t         stream);

} // namespace holo::cuda