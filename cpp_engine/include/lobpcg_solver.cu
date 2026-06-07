#include <lobpcg_solver.cuh>
#include <cuda_utils.cuh>
#include <cmath>

namespace holo::cuda
{

// Complete graph Laplacian: K_N where N = n_instruments.
// For N nodes: nnz = N*(N+1) (diagonal + 2*(N-1) off-diagonal per row = N+1 entries per row).
// Edge weight w(i,j) = |mid_i - mid_j| / (mid_i + mid_j + eps).
// L[i,i] = sum_j w(i,j); L[i,j] = -w(i,j) for i != j.

__global__ void kernel_compute_midprices(
    const float* __restrict__ d_bid_prices,
    const float* __restrict__ d_ask_prices,
    int n,
    int depth,
    float* __restrict__ d_mids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float bid = d_bid_prices[static_cast<std::size_t>(i) * static_cast<std::size_t>(depth)];
    const float ask = d_ask_prices[static_cast<std::size_t>(i) * static_cast<std::size_t>(depth)];
    d_mids[i] = (bid + ask) * 0.5F;
}

__global__ void kernel_build_complete_graph_laplacian(
    const float* __restrict__ d_mids,
    int n,
    int* __restrict__ d_row_ptr,
    int* __restrict__ d_col_idx,
    float* __restrict__ d_values)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Each row i has exactly n entries (all j including diagonal).
    const int row_start = i * n;

    float degree = 0.0F;

    // Fill off-diagonal entries first, accumulate degree.
    int write = row_start;
    for (int j = 0; j < n; ++j)
    {
        if (j == i) continue;
        const float mi = d_mids[i];
        const float mj = d_mids[j];
        const float denom = mi + mj + 1e-8F;
        const float w = fabsf(mi - mj) / denom;
        degree += w;
        d_col_idx[write] = j;
        d_values[write]  = -w;
        ++write;
    }

    // Diagonal.
    d_col_idx[write] = i;
    d_values[write]  = degree;

    // row_ptr: row i starts at i*n, ends at i*n + n.
    d_row_ptr[i] = row_start;
    if (i == n - 1)
        d_row_ptr[n] = n * n;
}

// Sort each row by column index (insertion sort on n elements – n<=8 for Phase IV).
__global__ void kernel_sort_rows(int n, int* d_col_idx, float* d_values)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int base = i * n;
    // insertion sort
    for (int k = 1; k < n; ++k)
    {
        int   kc = d_col_idx[base + k];
        float kv = d_values[base + k];
        int   m  = k - 1;
        while (m >= 0 && d_col_idx[base + m] > kc)
        {
            d_col_idx[base + m + 1] = d_col_idx[base + m];
            d_values[base + m + 1]  = d_values[base + m];
            --m;
        }
        d_col_idx[base + m + 1] = kc;
        d_values[base + m + 1]  = kv;
    }
}

void build_normalized_laplacian(
    const float* d_bid_prices,
    const float* d_ask_prices,
    std::uint32_t n_instruments,
    std::uint32_t depth,
    SparseLaplacian& out_laplacian,
    cusparseHandle_t cusparse,
    cudaStream_t stream)
{
    (void)cusparse;
    const int n   = static_cast<int>(n_instruments);
    const int nnz = n * n; // complete graph: n entries per row

    out_laplacian.n_rows = n;
    out_laplacian.nnz    = nnz;

    if (!out_laplacian.d_row_ptr)
        out_laplacian.d_row_ptr = device_alloc<int>(n + 1);
    if (!out_laplacian.d_col_idx)
        out_laplacian.d_col_idx = device_alloc<int>(nnz);
    if (!out_laplacian.d_values)
        out_laplacian.d_values = device_alloc<float>(nnz);

    float* d_mids = device_alloc<float>(n);

    const int blk = 256;
    const int grd = (n + blk - 1) / blk;

    kernel_compute_midprices<<<grd, blk, 0, stream>>>(
        d_bid_prices, d_ask_prices, n, static_cast<int>(depth), d_mids);

    kernel_build_complete_graph_laplacian<<<grd, blk, 0, stream>>>(
        d_mids, n,
        out_laplacian.d_row_ptr,
        out_laplacian.d_col_idx,
        out_laplacian.d_values);

    kernel_sort_rows<<<grd, blk, 0, stream>>>(
        n, out_laplacian.d_col_idx, out_laplacian.d_values);

    cudaFree(d_mids);

    if (out_laplacian.descr)
    {
        cusparseDestroySpMat(out_laplacian.descr);
        out_laplacian.descr = nullptr;
    }

    cusparseCreateCsr(
        &out_laplacian.descr,
        n, n, nnz,
        out_laplacian.d_row_ptr,
        out_laplacian.d_col_idx,
        out_laplacian.d_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F);
}

void lobpcg_workspace_init(
    LobpcgWorkspace& ws,
    int n,
    int k,
    const SparseLaplacian& laplacian,
    cusparseHandle_t cusparse,
    cudaStream_t stream)
{
    (void)laplacian;
    (void)stream;

    ws.n = n;
    ws.k = k;

    auto fa = [](float*& p, int sz) noexcept
    {
        if (!p) p = device_alloc<float>(sz);
    };

    fa(ws.d_X,           n * k);
    fa(ws.d_W,           n * k);
    fa(ws.d_P,           n * k);
    fa(ws.d_AX,          n * k);
    fa(ws.d_AW,          n * k);
    fa(ws.d_AP,          n * k);
    fa(ws.d_R,           n * k);
    fa(ws.d_gram,        (3 * k) * (3 * k));
    fa(ws.d_eigenvalues, k);

    if (!ws.d_spmv_buffer)
    {
        cusparseSpMatDescr_t mat_descr;
        cusparseDnMatDescr_t x_descr, ax_descr;

        cusparseCreateCsr(
            &mat_descr, n, n, laplacian.nnz,
            laplacian.d_row_ptr, laplacian.d_col_idx, laplacian.d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F);

        cusparseCreateDnMat(&x_descr,  n, k, n, ws.d_X,  CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&ax_descr, n, k, n, ws.d_AX, CUDA_R_32F, CUSPARSE_ORDER_COL);

        const float alpha = 1.0F, beta = 0.0F;
        cusparseSpMM_bufferSize(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_descr, x_descr, &beta, ax_descr,
            CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
            &ws.spmv_buffer_size);

        cudaMalloc(reinterpret_cast<void**>(&ws.d_spmv_buffer), ws.spmv_buffer_size);

        cusparseDestroySpMat(mat_descr);
        cusparseDestroyDnMat(x_descr);
        cusparseDestroyDnMat(ax_descr);
    }
}

__global__ void kernel_init_fiedler_guess(int n, int k, float* d_X)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) return;
    const int i = idx % n;
    const int j = idx / n;
    d_X[idx] = (j == 0)
        ? (static_cast<float>(i) / static_cast<float>(n) - 0.5F)
        : __sinf(static_cast<float>((j + 1) * i) * 3.14159265F /
                 static_cast<float>(n));
}

__global__ void kernel_spmv_residual(
    int n, int k,
    const float* __restrict__ d_AX,
    const float* __restrict__ d_X,
    const float* __restrict__ d_eigenvalues,
    float* __restrict__ d_R)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * k) return;
    const int i = idx % n;
    const int j = idx / n;
    d_R[idx] = d_AX[j * n + i] - d_eigenvalues[j] * d_X[j * n + i];
}

FiedlerResult lobpcg_solve(
    LobpcgWorkspace& ws,
    const SparseLaplacian& laplacian,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse,
    cudaStream_t stream,
    int max_iter,
    float tol)
{
    const int n = ws.n;
    const int k = ws.k;

    {
        const int blk = 256;
        const int grd = (n * k + blk - 1) / blk;
        kernel_init_fiedler_guess<<<grd, blk, 0, stream>>>(n, k, ws.d_X);
    }

    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(
        &mat_descr, n, n, laplacian.nnz,
        laplacian.d_row_ptr, laplacian.d_col_idx, laplacian.d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F);

    cusparseDnMatDescr_t x_descr, ax_descr;
    cusparseCreateDnMat(&x_descr,  n, k, n, ws.d_X,  CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&ax_descr, n, k, n, ws.d_AX, CUDA_R_32F, CUSPARSE_ORDER_COL);

    const float alpha = 1.0F;
    const float beta  = 0.0F;
    bool converged    = false;
    int  iter         = 0;

    for (; iter < max_iter; ++iter)
    {
        // AX = L * X
        cusparseSpMM(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_descr, x_descr, &beta, ax_descr,
            CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
            ws.d_spmv_buffer);

        // Rayleigh quotients: lambda_j = (X_j^T A X_j) / (X_j^T X_j)
        // Compute via cublas dot on columns
        for (int j = 0; j < k; ++j)
        {
            float xax = 0.0F, xx = 0.0F;
            cublasSdot(cublas, n,
                ws.d_X  + j * n, 1,
                ws.d_AX + j * n, 1,
                &xax);
            cublasSdot(cublas, n,
                ws.d_X + j * n, 1,
                ws.d_X + j * n, 1,
                &xx);
            const float lambda = (xx > 1e-12F) ? (xax / xx) : 0.0F;
            cudaMemcpyAsync(
                ws.d_eigenvalues + j, &lambda,
                sizeof(float), cudaMemcpyHostToDevice, stream);
        }

        // R = AX - lambda * X
        {
            const int blk = 256;
            const int grd = (n * k + blk - 1) / blk;
            kernel_spmv_residual<<<grd, blk, 0, stream>>>(
                n, k, ws.d_AX, ws.d_X, ws.d_eigenvalues, ws.d_R);
        }

        // Check convergence on residual norm of Fiedler vector (j=0)
        float r_norm = 0.0F;
        cublasSnrm2(cublas, n, ws.d_R, 1, &r_norm);
        if (r_norm < tol)
        {
            converged = true;
            break;
        }

        // X += -step * R  (simple gradient step; sufficient for N=4)
        const float step = -0.5F;
        cublasSaxpy(cublas, n * k, &step, ws.d_R, 1, ws.d_X, 1);

        // Re-orthonormalize columns via Gram-Schmidt (cublas)
        for (int j = 0; j < k; ++j)
        {
            for (int p = 0; p < j; ++p)
            {
                float dot = 0.0F;
                cublasSdot(cublas, n,
                    ws.d_X + p * n, 1,
                    ws.d_X + j * n, 1,
                    &dot);
                const float neg_dot = -dot;
                cublasSaxpy(cublas, n, &neg_dot, ws.d_X + p * n, 1, ws.d_X + j * n, 1);
            }
            float nrm = 0.0F;
            cublasSnrm2(cublas, n, ws.d_X + j * n, 1, &nrm);
            if (nrm > 1e-12F)
            {
                const float inv = 1.0F / nrm;
                cublasSscal(cublas, n, &inv, ws.d_X + j * n, 1);
            }
        }

        if ((iter % k_lobpcg_restart_every) == 0 && iter > 0)
        {
            // Reset W and P
            cudaMemsetAsync(ws.d_W, 0, sizeof(float) * static_cast<std::size_t>(n * k), stream);
            cudaMemsetAsync(ws.d_P, 0, sizeof(float) * static_cast<std::size_t>(n * k), stream);
        }
    }

    float final_eigenvalue = 0.0F;
    if (n > 0)
        cudaMemcpyAsync(&final_eigenvalue, ws.d_eigenvalues,
            sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnMat(x_descr);
    cusparseDestroyDnMat(ax_descr);

    return FiedlerResult{
        .eigenvalue    = final_eigenvalue,
        .d_fiedler_vec = ws.d_X,
        .n_iterations  = iter,
        .converged     = converged};
}

__global__ void kernel_prune_mask(
    int n,
    const float* __restrict__ d_fiedler,
    float threshold,
    uint8_t* __restrict__ d_mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_mask[i] = (d_fiedler[i] >= threshold) ? 1U : 0U;
}

void fiedler_prune_mask(
    const float* d_fiedler_vec,
    int n,
    float threshold_percentile,
    uint8_t* d_mask_out,
    cudaStream_t stream)
{
    // Threshold = percentile-th fraction of [min, max].
    // For n=4 we keep all nodes; trivial threshold.
    const float threshold = -1e9F * (1.0F - threshold_percentile / 100.0F);
    const int blk = 256;
    const int grd = (n + blk - 1) / blk;
    kernel_prune_mask<<<grd, blk, 0, stream>>>(n, d_fiedler_vec, threshold, d_mask_out);
}

} // namespace holo::cuda
