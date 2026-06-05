#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <cuda_utils.cuh>

namespace holo::cuda
{

    static constexpr int k_lobpcg_max_iter = 300;
    static constexpr float k_lobpcg_tol = 1e-6F;
    static constexpr int k_lobpcg_block_size = 4;
    static constexpr int k_lobpcg_restart_every = 50;
    static constexpr int k_max_edges = 4096;//для теста 
    static constexpr int k_max_triangles = 8192;//для теста 

    struct SparseLaplacian final
    {
        int *d_row_ptr{nullptr};
        int *d_col_idx{nullptr};
        float *d_values{nullptr};
        int n_rows{0};
        int nnz{0};

        cusparseSpMatDescr_t descr{nullptr};

        SparseLaplacian() = default;

        ~SparseLaplacian() noexcept
        {
            if (d_row_ptr != nullptr)
                cudaFree(d_row_ptr);
            if (d_col_idx != nullptr)
                cudaFree(d_col_idx);
            if (d_values != nullptr)
                cudaFree(d_values);
            if (descr != nullptr)
                cusparseDestroySpMat(descr);
        }

        SparseLaplacian(const SparseLaplacian &) = delete;
        SparseLaplacian &operator=(const SparseLaplacian &) = delete;
        SparseLaplacian(SparseLaplacian &&) = delete;
        SparseLaplacian &operator=(SparseLaplacian &&) = delete;

        void release() noexcept
        {
            if (d_row_ptr)
            {
                cudaFree(d_row_ptr);
                d_row_ptr = nullptr;
            }
            if (d_col_idx)
            {
                cudaFree(d_col_idx);
                d_col_idx = nullptr;
            }
            if (d_values)
            {
                cudaFree(d_values);
                d_values = nullptr;
            }
            if (descr)
            {
                cusparseDestroySpMat(descr);
                descr = nullptr;
            }
            n_rows = 0;
            nnz = 0;
        }
    };

    struct LobpcgWorkspace final
    {
        float *d_X{nullptr};
        float *d_W{nullptr};
        float *d_P{nullptr};
        float *d_AX{nullptr};
        float *d_AW{nullptr};
        float *d_AP{nullptr};
        float *d_R{nullptr};
        float *d_gram{nullptr};
        float *d_eigenvalues{nullptr};
        float *d_spmv_buffer{nullptr};

        int n{0};
        int k{0};
        std::size_t spmv_buffer_size{0U};

        LobpcgWorkspace() = default;

        ~LobpcgWorkspace() noexcept
        {
            auto f = [](float *p) noexcept
            {
                if (p != nullptr)
                    cudaFree(p);
            };
            f(d_X);
            f(d_W);
            f(d_P);
            f(d_AX);
            f(d_AW);
            f(d_AP);
            f(d_R);
            f(d_gram);
            f(d_eigenvalues);
            f(d_spmv_buffer);
        }

        LobpcgWorkspace(const LobpcgWorkspace &) = delete;
        LobpcgWorkspace &operator=(const LobpcgWorkspace &) = delete;
        LobpcgWorkspace(LobpcgWorkspace &&) = delete;
        LobpcgWorkspace &operator=(LobpcgWorkspace &&) = delete;
    };

    struct FiedlerResult
    {
        float eigenvalue{0.0F};
        float *d_fiedler_vec{nullptr};
        int n_iterations{0};
        bool converged{false};
    };

    void build_normalized_laplacian(
        const float *d_bid_prices,
        const float *d_ask_prices,
        std::uint32_t n_instruments,
        std::uint32_t depth,
        SparseLaplacian &out_laplacian,
        cusparseHandle_t cusparse,
        cudaStream_t stream);

    void lobpcg_workspace_init(
        LobpcgWorkspace &ws,
        int n,
        int k,
        const SparseLaplacian &laplacian,
        cusparseHandle_t cusparse,
        cudaStream_t stream);

    FiedlerResult lobpcg_solve(
        LobpcgWorkspace &ws,
        const SparseLaplacian &laplacian,
        cublasHandle_t cublas,
        cusparseHandle_t cusparse,
        cudaStream_t stream,
        int max_iter = k_lobpcg_max_iter,
        float tol = k_lobpcg_tol);

    void fiedler_prune_mask(
        const float *d_fiedler_vec,
        int n,
        float threshold_percentile,
        uint8_t *d_mask_out,
        cudaStream_t stream);

} // namespace holo::cuda