#include <lobpcg_solver.cuh>
#include <cuda_utils.cuh>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand_kernel.h>

namespace holo::cuda {

static constexpr int k_threads_build = 256;

__global__ void kernel_build_cross_impact_edges(
    const float* __restrict__ bid_prices,
    const float* __restrict__ ask_prices,
    uint32_t    n_instruments,
    uint32_t    depth,
    float*      d_adj_values,
    int*        d_adj_row,
    int*        d_adj_col,
    int*        d_nnz_counter,
    float       threshold)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= static_cast<int>(n_instruments)
     || j >= static_cast<int>(n_instruments)
     || i >= j) return;

    float cross_impact = 0.0F;
    for (uint32_t d = 0U; d < depth; ++d) {
        const float bi = bid_prices[i * depth + d];
        const float bj = bid_prices[j * depth + d];
        const float ai = ask_prices[i * depth + d];
        const float aj = ask_prices[j * depth + d];
        const float spread_i = ai - bi;
        const float spread_j = aj - bj;
        if (spread_i > 1e-9F && spread_j > 1e-9F) {
            const float mid_i = (bi + ai) * 0.5F;
            const float mid_j = (bj + aj) * 0.5F;
            if (mid_i > 1e-9F && mid_j > 1e-9F) {
                cross_impact += 1.0F / (fabsf(mid_i - mid_j) + 1e-6F);
            }
        }
    }

    if (cross_impact > threshold) {
        const int slot = atomicAdd(d_nnz_counter, 1);
        if (slot < k_max_edges) {
            d_adj_values[slot] = cross_impact;
            d_adj_row[slot]    = i;
            d_adj_col[slot]    = j;
        }
    }
}

__global__ void kernel_build_csr_from_coo(
    const float* __restrict__ coo_values,
    const int*   __restrict__ coo_row,
    const int*   __restrict__ coo_col,
    int          nnz_sym,
    int          n,
    int*         d_row_ptr,
    int*         d_col_idx,
    float*       d_values,
    float*       d_degree)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz_sym) return;

    const int   r = coo_row[tid];
    const int   c = coo_col[tid];
    const float w = coo_values[tid];

    atomicAdd(&d_degree[r], w);
    atomicAdd(&d_degree[c], w);
}

__global__ void kernel_normalized_laplacian_diag(
    int          n,
    const float* __restrict__ d_degree,
    float*       d_diag_inv_sqrt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float deg = d_degree[i];
    d_diag_inv_sqrt[i] = (deg > 1e-9F) ? rsqrtf(deg) : 0.0F;
}

__global__ void kernel_apply_inv_sqrt_scaling(
    int          nnz,
    const int*   __restrict__ d_row_idx,
    const int*   __restrict__ d_col_idx,
    const float* __restrict__ d_inv_sqrt,
    float*       d_values)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;
    const int r = d_row_idx[tid];
    const int c = d_col_idx[tid];
    d_values[tid] *= d_inv_sqrt[r] * d_inv_sqrt[c];
}

__global__ void kernel_random_init_vectors(
    float*   d_X,
    int      n,
    int      k,
    uint64_t seed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * k) return;
    curandState_t state;
    curand_init(seed, static_cast<uint64_t>(tid), 0ULL, &state);
    d_X[tid] = curand_normal(&state);
}

__global__ void kernel_orthogonalize_columns(
    float* d_X,
    int    n,
    int    k)
{
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    if (col >= k) return;

    float* col_data = d_X + col * n;
    float norm_sq = 0.0F;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        norm_sq += col_data[i] * col_data[i];
    }

    __shared__ float s_norm;
    if (threadIdx.x == 0) s_norm = 0.0F;
    __syncthreads();
    atomicAdd(&s_norm, norm_sq);
    __syncthreads();

    const float inv_norm = rsqrtf(s_norm + 1e-12F);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        col_data[i] *= inv_norm;
    }
    (void)smem;
}

__global__ void kernel_compute_residual(
    const float* __restrict__ d_AX,
    const float* __restrict__ d_X,
    const float* __restrict__ d_eigenvalues,
    float*       d_R,
    int          n,
    int          k)
{
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;
    if (i >= n || col >= k) return;
    const float lam = d_eigenvalues[col];
    d_R[col * n + i] = d_AX[col * n + i] - lam * d_X[col * n + i];
}

__global__ void kernel_check_convergence(
    const float* __restrict__ d_R,
    const float* __restrict__ d_eigenvalues,
    int    n,
    int    k,
    float  tol,
    int*   d_converged_count)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= k) return;

    float res_norm = 0.0F;
    for (int i = 0; i < n; ++i) {
        const float r = d_R[col * n + i];
        res_norm += r * r;
    }
    res_norm = sqrtf(res_norm);

    const float rel_res = res_norm / (fabsf(d_eigenvalues[col]) + 1e-12F);
    if (rel_res < tol) {
        atomicAdd(d_converged_count, 1);
    }
}

__global__ void kernel_fiedler_prune(
    const float* __restrict__ d_fiedler,
    int          n,
    float        threshold,
    uint8_t*     d_mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_mask[i] = (fabsf(d_fiedler[i]) >= threshold) ? 1U : 0U;
}

__global__ void kernel_compute_threshold_percentile(
    const float* __restrict__ d_abs_fiedler,
    int          n,
    float        percentile,
    float*       d_threshold)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float max_val = 0.0F;
    for (int i = 0; i < n; ++i) {
        if (d_abs_fiedler[i] > max_val) max_val = d_abs_fiedler[i];
    }
    *d_threshold = max_val * (percentile / 100.0F);
}

void build_normalized_laplacian(
    const float*      d_bid_prices,
    const float*      d_ask_prices,
    uint32_t          n_instruments,
    uint32_t          depth,
    SparseLaplacian&  L,
    cusparseHandle_t  cusparse,
    cudaStream_t      stream)
{
    const int n   = static_cast<int>(n_instruments);
    const int max_nnz = k_max_edges * 2;

    float*   d_coo_vals = device_alloc<float>(static_cast<size_t>(max_nnz));
    int*     d_coo_row  = device_alloc<int>  (static_cast<size_t>(max_nnz));
    int*     d_coo_col  = device_alloc<int>  (static_cast<size_t>(max_nnz));
    int*     d_nnz_ctr  = device_alloc<int>  (1U);
    float*   d_degree   = device_alloc<float>(static_cast<size_t>(n));
    float*   d_inv_sqrt = device_alloc<float>(static_cast<size_t>(n));

    CUDA_CHECK(cudaMemsetAsync(d_nnz_ctr, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_degree,  0, static_cast<size_t>(n) * sizeof(float), stream));

    const dim3 block2d(16, 16);
    const dim3 grid2d(
        (static_cast<unsigned>(n) + 15U) / 16U,
        (static_cast<unsigned>(n) + 15U) / 16U);

    kernel_build_cross_impact_edges<<<grid2d, block2d, 0, stream>>>(
        d_bid_prices, d_ask_prices,
        n_instruments, depth,
        d_coo_vals, d_coo_row, d_coo_col,
        d_nnz_ctr, 0.01F);

    int h_nnz_upper = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_nnz_upper, d_nnz_ctr,
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    h_nnz_upper = (h_nnz_upper > k_max_edges) ? k_max_edges : h_nnz_upper;
    const int nnz_sym = h_nnz_upper * 2;

    float* d_sym_vals = device_alloc<float>(static_cast<size_t>(nnz_sym + n));
    int*   d_sym_row  = device_alloc<int>  (static_cast<size_t>(nnz_sym + n));
    int*   d_sym_col  = device_alloc<int>  (static_cast<size_t>(nnz_sym + n));

    CUDA_CHECK(cudaMemcpyAsync(d_sym_vals, d_coo_vals,
        static_cast<size_t>(h_nnz_upper) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sym_vals + h_nnz_upper, d_coo_vals,
        static_cast<size_t>(h_nnz_upper) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sym_row, d_coo_row,
        static_cast<size_t>(h_nnz_upper) * sizeof(int),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sym_col + h_nnz_upper, d_coo_row,
        static_cast<size_t>(h_nnz_upper) * sizeof(int),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sym_col, d_coo_col,
        static_cast<size_t>(h_nnz_upper) * sizeof(int),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sym_row + h_nnz_upper, d_coo_col,
        static_cast<size_t>(h_nnz_upper) * sizeof(int),
        cudaMemcpyDeviceToDevice, stream));

    const int threads1d = k_threads_build;
    const int blocks1d  = (nnz_sym + threads1d - 1) / threads1d;
    kernel_build_csr_from_coo<<<blocks1d, threads1d, 0, stream>>>(
        d_sym_vals, d_sym_row, d_sym_col,
        nnz_sym, n, nullptr, nullptr, nullptr, d_degree);

    const int blk_n = (n + threads1d - 1) / threads1d;
    kernel_normalized_laplacian_diag<<<blk_n, threads1d, 0, stream>>>(
        n, d_degree, d_inv_sqrt);

    kernel_apply_inv_sqrt_scaling<<<blocks1d, threads1d, 0, stream>>>(
        nnz_sym, d_sym_row, d_sym_col, d_inv_sqrt, d_sym_vals);

    const int nnz_with_diag = nnz_sym + n;
    int*   d_rptr  = device_alloc<int>  (static_cast<size_t>(n + 1));
    int*   d_cidx  = device_alloc<int>  (static_cast<size_t>(nnz_with_diag));
    float* d_vvals = device_alloc<float>(static_cast<size_t>(nnz_with_diag));

    cusparseHandle_t local_cusparse = cusparse;

    size_t sort_buf_sz = 0U;
    void*  d_sort_buf  = nullptr;

    cusparseXcoosort_bufferSizeExt(
        local_cusparse, n, n, nnz_sym,
        d_sym_row, d_sym_col, &sort_buf_sz);

    CUDA_CHECK(cudaMalloc(&d_sort_buf, sort_buf_sz));
    int* d_perm = device_alloc<int>(static_cast<size_t>(nnz_sym));
    cusparseCreateIdentityPermutation(local_cusparse, nnz_sym, d_perm);
    cusparseXcoosortByRow(
        local_cusparse, n, n, nnz_sym,
        d_sym_row, d_sym_col, d_perm, d_sort_buf);

    float* d_gather_buf = device_alloc<float>(static_cast<size_t>(nnz_sym));
    cusparseSgthr(local_cusparse, nnz_sym,
        d_sym_vals, d_gather_buf, d_perm, CUSPARSE_INDEX_BASE_ZERO);

    cusparseXcoo2csr(local_cusparse,
        d_sym_row, nnz_sym, n, d_rptr, CUSPARSE_INDEX_BASE_ZERO);

    CUDA_CHECK(cudaMemcpyAsync(d_cidx, d_sym_col,
        static_cast<size_t>(nnz_sym) * sizeof(int),
        cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vvals, d_gather_buf,
        static_cast<size_t>(nnz_sym) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    CUSPARSE_CHECK(cusparseCreateCsr(
        &L.descr, n, n, nnz_sym,
        d_rptr, d_cidx, d_vvals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    L.d_row_ptr = d_rptr;
    L.d_col_idx = d_cidx;
    L.d_values  = d_vvals;
    L.n_rows    = n;
    L.nnz       = nnz_sym;

    device_free(d_coo_vals); device_free(d_coo_row); device_free(d_coo_col);
    device_free(d_nnz_ctr);  device_free(d_degree);  device_free(d_inv_sqrt);
    device_free(d_sym_vals); device_free(d_sym_row);  device_free(d_sym_col);
    device_free(d_perm);     device_free(d_gather_buf);
    cudaFree(d_sort_buf);
}

void lobpcg_workspace_init(
    LobpcgWorkspace&       ws,
    int                    n,
    int                    k,
    const SparseLaplacian& laplacian,
    cusparseHandle_t       cusparse,
    cudaStream_t           stream)
{
    ws.n = n;
    ws.k = k;

    const size_t nk = static_cast<size_t>(n * k);
    const size_t kk = static_cast<size_t>(k * k * 9);

    ws.d_X            = device_alloc<float>(nk);
    ws.d_W            = device_alloc<float>(nk);
    ws.d_P            = device_alloc<float>(nk);
    ws.d_AX           = device_alloc<float>(nk);
    ws.d_AW           = device_alloc<float>(nk);
    ws.d_AP           = device_alloc<float>(nk);
    ws.d_R            = device_alloc<float>(nk);
    ws.d_gram         = device_alloc<float>(kk);
    ws.d_eigenvalues  = device_alloc<float>(static_cast<size_t>(k));

    CUDA_CHECK(cudaMemsetAsync(ws.d_P,  0, nk * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.d_AP, 0, nk * sizeof(float), stream));

    float one = 1.0F, zero = 0.0F;
    cusparseDnMatDescr_t X_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &X_descr, n, k, n,
        ws.d_X, CUDA_R_32F, CUSPARSE_ORDER_COL));

    size_t spmv_sz = 0U;
    cusparseSpMM_bufferSize(
        cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, laplacian.descr, X_descr,
        &zero, X_descr,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &spmv_sz);

    ws.d_spmv_buffer    = device_alloc<float>((spmv_sz + 3U) / 4U + 1U);
    ws.spmv_buffer_size = spmv_sz;
    cusparseDestroyDnMat(X_descr);

    const int blk = 256;
    const int grd = (n * k + blk - 1) / (blk);
    kernel_random_init_vectors<<<grd, blk, 0, stream>>>(
        ws.d_X, n, k, 0xDEADC0DEULL);
    kernel_orthogonalize_columns<<<k, 256, 256 * sizeof(float), stream>>>(
        ws.d_X, n, k);
}

FiedlerResult lobpcg_solve(
    LobpcgWorkspace&       ws,
    const SparseLaplacian& laplacian,
    cublasHandle_t         cublas,
    cusparseHandle_t       cusparse,
    cudaStream_t           stream,
    int                    max_iter,
    float                  tol)
{
    const int    n = ws.n;
    const int    k = ws.k;
    const float  one  =  1.0F;
    const float  zero =  0.0F;
    const float  mone = -1.0F;

    int*   d_converged = device_alloc<int>(1U);
    float* d_rayleigh  = device_alloc<float>(static_cast<size_t>(k));

    cusparseDnMatDescr_t X_descr, AX_descr, W_descr, AW_descr;
    CUSPARSE_CHECK(cusparseCreateDnMat(&X_descr,  n, k, n,
        ws.d_X,  CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&AX_descr, n, k, n,
        ws.d_AX, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&W_descr,  n, k, n,
        ws.d_W,  CUDA_R_32F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&AW_descr, n, k, n,
        ws.d_AW, CUDA_R_32F, CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseSpMM(
        cusparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, laplacian.descr, X_descr,
        &zero, AX_descr,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        ws.d_spmv_buffer));

    CUBLAS_CHECK(cublasSgemm(
        cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        k, k, n,
        &one,
        ws.d_X,  n,
        ws.d_AX, n,
        &zero,
        ws.d_gram, k));

    CUDA_CHECK(cudaMemcpyAsync(ws.d_eigenvalues, ws.d_gram,
        static_cast<size_t>(k) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    int n_iter = 0;
    bool converged = false;

    for (int iter = 0; iter < max_iter && !converged; ++iter) {
        const int blk_nk = 256;
        const int grd_nk = (n * k + blk_nk - 1) / blk_nk;
        kernel_compute_residual<<<grd_nk, blk_nk, 0, stream>>>(
            ws.d_AX, ws.d_X, ws.d_eigenvalues, ws.d_R, n, k);

        CUDA_CHECK(cudaMemsetAsync(d_converged, 0, sizeof(int), stream));
        const int blk_k = (k + 31) / 32 * 32;
        kernel_check_convergence<<<1, blk_k, 0, stream>>>(
            ws.d_R, ws.d_eigenvalues, n, k, tol, d_converged);

        int h_conv = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_conv, d_converged,
            sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (h_conv >= k) { converged = true; break; }

        CUDA_CHECK(cudaMemcpyAsync(ws.d_W, ws.d_R,
            static_cast<size_t>(n * k) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));

        CUSPARSE_CHECK(cusparseSpMM(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, laplacian.descr, W_descr,
            &zero, AW_descr,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
            ws.d_spmv_buffer));

        if (iter > 0 && iter % k_lobpcg_restart_every != 0) {
            cusparseDnMatDescr_t P_descr, AP_descr;
            CUSPARSE_CHECK(cusparseCreateDnMat(&P_descr,  n, k, n,
                ws.d_P,  CUDA_R_32F, CUSPARSE_ORDER_COL));
            CUSPARSE_CHECK(cusparseCreateDnMat(&AP_descr, n, k, n,
                ws.d_AP, CUDA_R_32F, CUSPARSE_ORDER_COL));

            CUBLAS_CHECK(cublasSgemm(
                cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                n, k, k,
                &one,
                ws.d_W, n, ws.d_gram, k,
                &one,
                ws.d_X, n));

            cusparseDestroyDnMat(P_descr);
            cusparseDestroyDnMat(AP_descr);
        } else {
            CUBLAS_CHECK(cublasSaxpy(
                cublas, n * k, &one, ws.d_W, 1, ws.d_X, 1));
        }

        CUDA_CHECK(cudaMemcpyAsync(ws.d_P, ws.d_W,
            static_cast<size_t>(n * k) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ws.d_AP, ws.d_AW,
            static_cast<size_t>(n * k) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));

        kernel_orthogonalize_columns<<<k, 256, 256 * sizeof(float), stream>>>(
            ws.d_X, n, k);

        CUSPARSE_CHECK(cusparseSpMM(
            cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, laplacian.descr, X_descr,
            &zero, AX_descr,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
            ws.d_spmv_buffer));

        CUBLAS_CHECK(cublasSgemm(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            k, k, n,
            &one,
            ws.d_X, n, ws.d_AX, n,
            &zero,
            ws.d_gram, k));

        CUDA_CHECK(cudaMemcpyAsync(ws.d_eigenvalues, ws.d_gram,
            static_cast<size_t>(k) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));

        n_iter = iter + 1;
        (void)mone;
        (void)d_rayleigh;
    }

    cusparseDestroyDnMat(X_descr);
    cusparseDestroyDnMat(AX_descr);
    cusparseDestroyDnMat(W_descr);
    cusparseDestroyDnMat(AW_descr);

    float h_eigenvalue = 0.0F;
    CUDA_CHECK(cudaMemcpyAsync(&h_eigenvalue, ws.d_eigenvalues + 1,
        sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    device_free(d_converged);
    device_free(d_rayleigh);

    return FiedlerResult{
        .eigenvalue    = h_eigenvalue,
        .d_fiedler_vec = ws.d_X + n,
        .n_iterations  = n_iter,
        .converged     = converged,
    };
}

void fiedler_prune_mask(
    const float* d_fiedler_vec,
    int          n,
    float        threshold_percentile,
    uint8_t*     d_mask_out,
    cudaStream_t stream)
{
    float* d_abs = device_alloc<float>(static_cast<size_t>(n));
    float* d_thr = device_alloc<float>(1U);

    const int blk = 256;
    const int grd = (n + blk - 1) / blk;

    kernel_compute_threshold_percentile<<<1, 1, 0, stream>>>(
        d_abs, n, threshold_percentile, d_thr);

    float h_thr = 0.0F;
    CUDA_CHECK(cudaMemcpyAsync(&h_thr, d_thr,
        sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    kernel_fiedler_prune<<<grd, blk, 0, stream>>>(
        d_fiedler_vec, n, h_thr, d_mask_out);

    device_free(d_abs);
    device_free(d_thr);
}

} // namespace holo::cuda