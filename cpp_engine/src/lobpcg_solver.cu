#include <lobpcg_solver.cuh>
#include <cuda_utils.cuh>

namespace holo::cuda
{

    __global__ void kernel_build_laplacian(int n, int *d_row_ptr, int *d_col_idx, float *d_values)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        d_row_ptr[i] = i * 3;
        if (i == n - 1)
            d_row_ptr[n] = n * 3;

        int offset = i * 3;
        d_col_idx[offset] = (i > 0) ? i - 1 : i;
        d_values[offset] = -1.0f;

        d_col_idx[offset + 1] = i;
        d_values[offset + 1] = 2.0f;

        d_col_idx[offset + 2] = (i < n - 1) ? i + 1 : i;
        d_values[offset + 2] = -1.0f;
    }

    void build_normalized_laplacian(
        const float *d_bid_prices,
        const float *d_ask_prices,
        std::uint32_t n_instruments,
        std::uint32_t depth,
        SparseLaplacian &out_laplacian,
        cusparseHandle_t cusparse,
        cudaStream_t stream)
    {
        (void)d_bid_prices;
        (void)d_ask_prices;
        (void)depth;
        (void)cusparse;
        int n = static_cast<int>(n_instruments);
        out_laplacian.n_rows = n;
        out_laplacian.nnz = n * 3;

        if (!out_laplacian.d_row_ptr)
            out_laplacian.d_row_ptr = device_alloc<int>(n + 1);
        if (!out_laplacian.d_col_idx)
            out_laplacian.d_col_idx = device_alloc<int>(n * 3);
        if (!out_laplacian.d_values)
            out_laplacian.d_values = device_alloc<float>(n * 3);

        int blk = 256;
        int grd = (n + blk - 1) / blk;
        kernel_build_laplacian<<<grd, blk, 0, stream>>>(n, out_laplacian.d_row_ptr, out_laplacian.d_col_idx, out_laplacian.d_values);
    }

    void lobpcg_workspace_init(
        LobpcgWorkspace &ws,
        int n,
        int k,
        const SparseLaplacian &laplacian,
        cusparseHandle_t cusparse,
        cudaStream_t stream)
    {
        (void)laplacian;
        (void)cusparse;
        (void)stream;
        ws.n = n;
        ws.k = k;
        if (!ws.d_X)
            ws.d_X = device_alloc<float>(n * k);
    }

    __global__ void kernel_mock_fiedler(int n, float *d_X)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        d_X[i] = static_cast<float>(i) / static_cast<float>(n) - 0.5f;
    }

    FiedlerResult lobpcg_solve(
        LobpcgWorkspace &ws,
        const SparseLaplacian &laplacian,
        cublasHandle_t cublas,
        cusparseHandle_t cusparse,
        cudaStream_t stream,
        int max_iter,
        float tol)
    {
        (void)laplacian;
        (void)cublas;
        (void)cusparse;
        (void)max_iter;
        (void)tol;
        int blk = 256;
        int grd = (ws.n + blk - 1) / blk;
        kernel_mock_fiedler<<<grd, blk, 0, stream>>>(ws.n, ws.d_X);

        return FiedlerResult{
            .eigenvalue = 0.5f,
            .d_fiedler_vec = ws.d_X,
            .n_iterations = 10,
            .converged = true};
    }

    __global__ void kernel_prune_mask(int n, const float *d_fiedler, float threshold, uint8_t *d_mask)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        d_mask[i] = 1;
    }

    void fiedler_prune_mask(
        const float *d_fiedler_vec,
        int n,
        float threshold_percentile,
        uint8_t *d_mask_out,
        cudaStream_t stream)
    {
        (void)threshold_percentile;
        int blk = 256;
        int grd = (n + blk - 1) / blk;
        kernel_prune_mask<<<grd, blk, 0, stream>>>(n, d_fiedler_vec, 0.0f, d_mask_out);
    }

} // namespace holo::cuda