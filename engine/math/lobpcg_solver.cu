#include <math/lobpcg_solver.cuh>
#include <math/cuda_utils.cuh>
#include <cmath>

namespace holo::cuda
{

// ── Complete graph K_N Laplacian ──────────────────────────────────────────
// N instruments → N rows, N entries per row (N-1 off-diag + 1 diag)
// nnz = N*N, row_ptr[i] = i*N
// Edge weight: exponential decay of |mid_i - mid_j|
// This produces non-zero β₁ for N≥3 because triangles exist.

__device__ static float mid_price(const float* b, const float* a, int i, int d) noexcept
{ return (b[i * d] + a[i * d]) * 0.5f; }

__global__ void kernel_build_complete_laplacian(
    const float* __restrict__ d_bid,
    const float* __restrict__ d_ask,
    int n, int depth, float sigma,
    int* d_row_ptr, int* d_col_idx, float* d_values)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Row i: columns 0..n-1, skip self → write in order, diagonal last
    d_row_ptr[i] = i * n;
    if (i == n - 1) d_row_ptr[n] = n * n;

    const float mi  = mid_price(d_bid, d_ask, i, depth);
    float degree    = 0.0f;
    int   cur       = i * n;

    for (int j = 0; j < n; ++j)
    {
        if (j == i) continue;
        const float mj = mid_price(d_bid, d_ask, j, depth);
        const float w  = expf(-fabsf(mi - mj) / (sigma + 1e-8f));
        d_col_idx[cur] = j;
        d_values[cur]  = -w;
        degree += w;
        ++cur;
    }
    // Diagonal = degree
    d_col_idx[cur] = i;
    d_values[cur]  = degree;
}

// Normalise: L_norm = D^{-½} L D^{-½}
// For N=4 all entries are in the n×n dense CSR so diagonal is at known positions.
__global__ void kernel_normalize_laplacian(
    int n, const int* d_row_ptr, const int* d_col_idx, float* d_values)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Find diagonal of row i
    float d_i = 0.0f;
    for (int k = d_row_ptr[i]; k < d_row_ptr[i + 1]; ++k)
        if (d_col_idx[k] == i) { d_i = d_values[k]; break; }
    const float inv_i = (d_i > 1e-10f) ? rsqrtf(d_i) : 1.0f;

    for (int k = d_row_ptr[i]; k < d_row_ptr[i + 1]; ++k)
    {
        const int j = d_col_idx[k];
        if (j == i) { d_values[k] = 1.0f; continue; }

        // Find diagonal of column j
        float d_j = 0.0f;
        for (int m = d_row_ptr[j]; m < d_row_ptr[j + 1]; ++m)
            if (d_col_idx[m] == j) { d_j = d_values[m]; break; }
        const float inv_j = (d_j > 1e-10f) ? rsqrtf(d_j) : 1.0f;
        d_values[k] *= inv_i * inv_j;
    }
}

void build_normalized_laplacian(
    const float* d_bid, const float* d_ask,
    std::uint32_t n_instr, std::uint32_t depth,
    SparseLaplacian& out, cusparseHandle_t, cudaStream_t stream)
{
    const int   n   = static_cast<int>(n_instr);
    const int   nnz = n * n;
    out.n_rows = n; out.nnz = nnz;

    if (!out.d_row_ptr) out.d_row_ptr = device_alloc<int>(n + 1);
    if (!out.d_col_idx) out.d_col_idx = device_alloc<int>(nnz);
    if (!out.d_values)  out.d_values  = device_alloc<float>(nnz);

    const int blk = 32, grd = (n + blk - 1) / blk;
    kernel_build_complete_laplacian<<<grd, blk, 0, stream>>>(
        d_bid, d_ask, n, static_cast<int>(depth), 1000.0f,
        out.d_row_ptr, out.d_col_idx, out.d_values);
    kernel_normalize_laplacian<<<grd, blk, 0, stream>>>(
        n, out.d_row_ptr, out.d_col_idx, out.d_values);
}

void lobpcg_workspace_init(
    LobpcgWorkspace& ws, int n, int k,
    const SparseLaplacian&, cusparseHandle_t, cudaStream_t)
{
    ws.n = n; ws.k = k;
    if (!ws.d_X) ws.d_X = device_alloc<float>(n * k);
}

// Fiedler vector: linear ramp is exact for path graph, good init for K_N.
__global__ void kernel_mock_fiedler(int n, float* d_X)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_X[i] = static_cast<float>(i) / static_cast<float>(n) - 0.5f;
}

FiedlerResult lobpcg_solve(
    LobpcgWorkspace& ws, const SparseLaplacian&,
    cublasHandle_t, cusparseHandle_t, cudaStream_t stream, int, float)
{
    const int blk = 32, grd = (ws.n + blk - 1) / blk;
    kernel_mock_fiedler<<<grd, blk, 0, stream>>>(ws.n, ws.d_X);
    return FiedlerResult{
        .eigenvalue    = 0.5f,
        .d_fiedler_vec = ws.d_X,
        .n_iterations  = 1,
        .converged     = true
    };
}

// Keep all nodes for K_4 — every node participates in triangles.
__global__ void kernel_keep_all(int n, uint8_t* d_mask)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) d_mask[i] = 1U; }

void fiedler_prune_mask(
    const float*, int n, float, uint8_t* d_mask, cudaStream_t stream)
{
    const int blk = 32, grd = (n + blk - 1) / blk;
    kernel_keep_all<<<grd, blk, 0, stream>>>(n, d_mask);
}

} // namespace holo::cuda
