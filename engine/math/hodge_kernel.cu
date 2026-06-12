#include <math/gpu_lob_mirror.cuh>
#include <math/hodge_kernel.cuh>
#include <math/cuda_utils.cuh>

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>

namespace holo::cuda
{

    static constexpr int k_blk = k_hodge_block_dim;

    __global__ void kernel_build_flow_vector(
        const float *__restrict__ d_bid_prices,
        const float *__restrict__ d_ask_prices,
        const float *__restrict__ d_bid_qtys,
        const float *__restrict__ d_ask_qtys,
        const int *__restrict__ d_edge_src,
        const int *__restrict__ d_edge_dst,
        int n_edges,
        uint32_t depth,
        float *d_omega)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= n_edges)
            return;

        const int i = d_edge_src[e];
        const int j = d_edge_dst[e];

        float flow = 0.0F;
        for (uint32_t d = 0U; d < depth; ++d)
        {
            const float bi = d_bid_prices[static_cast<size_t>(i) * depth + d];
            const float bj = d_bid_prices[static_cast<size_t>(j) * depth + d];
            const float ai = d_ask_prices[static_cast<size_t>(i) * depth + d];
            const float aj = d_ask_prices[static_cast<size_t>(j) * depth + d];
            const float qbi = d_bid_qtys[static_cast<size_t>(i) * depth + d];
            const float qbj = d_bid_qtys[static_cast<size_t>(j) * depth + d];
            const float qai = d_ask_qtys[static_cast<size_t>(i) * depth + d];
            const float qaj = d_ask_qtys[static_cast<size_t>(j) * depth + d];

            const float mid_i = (bi + ai) * 0.5F;
            const float mid_j = (bj + aj) * 0.5F;
            const float vol_i = qbi + qai;
            const float vol_j = qbj + qaj;
            const float imb_i = (vol_i > 1e-9F) ? (qbi - qai) / vol_i : 0.0F;
            const float imb_j = (vol_j > 1e-9F) ? (qbj - qaj) / vol_j : 0.0F;

            flow += (mid_j - mid_i) * (imb_i - imb_j);
        }

        d_omega[e] = flow;
    }

    __global__ void kernel_build_signed_incidence(
        const int *__restrict__ d_edge_src,
        const int *__restrict__ d_edge_dst,
        int n_nodes,
        int n_edges,
        float *d_B1)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= n_edges)
            return;

        const int src = d_edge_src[e];
        const int dst = d_edge_dst[e];

        d_B1[static_cast<size_t>(src) * static_cast<size_t>(n_edges) + static_cast<size_t>(e)] = -1.0F;
        d_B1[static_cast<size_t>(dst) * static_cast<size_t>(n_edges) + static_cast<size_t>(e)] = 1.0F;
    }

    __global__ void kernel_build_triangle_incidence(
        const int *__restrict__ d_edge_src,
        const int *__restrict__ d_edge_dst,
        const int *__restrict__ d_triangle_edges,
        int n_triangles,
        int n_edges,
        float *d_B2)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= n_triangles)
            return;

        const int e0 = d_triangle_edges[t * 3 + 0];
        const int e1 = d_triangle_edges[t * 3 + 1];
        const int e2 = d_triangle_edges[t * 3 + 2];

        if (e0 >= 0 && e0 < n_edges)
            d_B2[static_cast<size_t>(e0) * static_cast<size_t>(n_triangles) + static_cast<size_t>(t)] = 1.0F;
        if (e1 >= 0 && e1 < n_edges)
            d_B2[static_cast<size_t>(e1) * static_cast<size_t>(n_triangles) + static_cast<size_t>(t)] = 1.0F;
        if (e2 >= 0 && e2 < n_edges)
            d_B2[static_cast<size_t>(e2) * static_cast<size_t>(n_triangles) + static_cast<size_t>(t)] = -1.0F;
    }

    __global__ void kernel_hodge1_laplacian(
        const float *__restrict__ d_B1,
        const float *__restrict__ d_B2,
        int n_edges,
        int n_nodes,
        int n_triangles,
        float *d_L1)
    {
        const int row = blockIdx.x * blockDim.x + threadIdx.x;
        const int col = blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= n_edges || col >= n_edges)
            return;

        float l1_lower = 0.0F;
        for (int v = 0; v < n_nodes; ++v)
        {
            l1_lower += d_B1[static_cast<size_t>(v) * static_cast<size_t>(n_edges) + static_cast<size_t>(row)] * d_B1[static_cast<size_t>(v) * static_cast<size_t>(n_edges) + static_cast<size_t>(col)];
        }

        float l1_upper = 0.0F;
        for (int tri = 0; tri < n_triangles; ++tri)
        {
            l1_upper += d_B2[static_cast<size_t>(row) * static_cast<size_t>(n_triangles) + static_cast<size_t>(tri)] * d_B2[static_cast<size_t>(col) * static_cast<size_t>(n_triangles) + static_cast<size_t>(tri)];
        }

        d_L1[static_cast<size_t>(row) * static_cast<size_t>(n_edges) + static_cast<size_t>(col)] = l1_lower + l1_upper;
    }

    __global__ void kernel_harmonic_projection(
        const float *__restrict__ d_eigenvectors,
        const float *__restrict__ d_eigenvalues,
        const float *__restrict__ d_omega,
        int n_edges,
        int n_eigs,
        float threshold,
        float *d_gamma)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= n_edges)
            return;

        float gamma_e = 0.0F;
        for (int k = 0; k < n_eigs; ++k)
        {
            if (fabsf(d_eigenvalues[k]) < threshold)
            {
                float dot = 0.0F;
                for (int f = 0; f < n_edges; ++f)
                {
                    dot += d_eigenvectors[static_cast<size_t>(k) * static_cast<size_t>(n_edges) + static_cast<size_t>(f)] * d_omega[f];
                }
                gamma_e += d_eigenvectors[static_cast<size_t>(k) * static_cast<size_t>(n_edges) + static_cast<size_t>(e)] * dot;
            }
        }
        d_gamma[e] = gamma_e;
    }

    __global__ void kernel_exact_component(
        const float *__restrict__ d_B1,
        const float *__restrict__ d_omega,
        const float *__restrict__ d_gamma,
        int n_edges,
        int n_nodes,
        float *d_node_potential,
        float *d_exact)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= n_edges)
            return;

        const float remainder_e = d_omega[e] - d_gamma[e];

        float node_pot = 0.0F;
        for (int v = 0; v < n_nodes; ++v)
        {
            float col_sum = 0.0F;
            for (int f = 0; f < n_edges; ++f)
            {
                col_sum += d_B1[static_cast<size_t>(v) * static_cast<size_t>(n_edges) + static_cast<size_t>(f)] * (d_omega[f] - d_gamma[f]);
            }
            node_pot += d_B1[static_cast<size_t>(v) * static_cast<size_t>(n_edges) + static_cast<size_t>(e)] * col_sum;
        }
        d_exact[e] = node_pot;
        d_node_potential[e] = remainder_e - node_pot;
    }

    __global__ void kernel_coexact_component(
        const float *__restrict__ d_omega,
        const float *__restrict__ d_gamma,
        const float *__restrict__ d_exact,
        int n_edges,
        float *d_coexact)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= n_edges)
            return;
        d_coexact[e] = d_omega[e] - d_gamma[e] - d_exact[e];
    }

    __global__ void kernel_curl_magnitude(
        const float *__restrict__ d_gamma,
        int n_edges,
        float *d_curl_mag,
        float *d_arb_signal,
        float arb_threshold)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= n_edges)
            return;

        const float mag = fabsf(d_gamma[e]);
        d_curl_mag[e] = mag;
        d_arb_signal[e] = (mag > arb_threshold) ? d_gamma[e] : 0.0F;
    }

    __global__ void kernel_yang_mills_action(
        const float *__restrict__ d_gamma,
        const float *__restrict__ d_coexact,
        int n_edges,
        float coupling,
        float *d_ym_action)
    {
        extern __shared__ float smem[];
        const int tid = threadIdx.x;
        const int e = blockIdx.x * blockDim.x + tid;

        float local_sum = 0.0F;
        if (e < n_edges)
        {
            const float A_mu = d_gamma[e];
            const float A_nu = d_coexact[e];
            const float dA = A_mu - A_nu;
            const float comm = A_mu * A_nu - A_nu * A_mu;
            const float F_munu = dA + comm;
            local_sum = F_munu * F_munu;
        }

        smem[tid] = local_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
                smem[tid] += smem[tid + stride];
            __syncthreads();
        }

        if (tid == 0)
        {
            atomicAdd(d_ym_action, smem[0] / (4.0F * coupling * coupling));
        }
    }

    __global__ void kernel_count_active_loops(
        const float *__restrict__ d_arb_signal,
        int n_edges,
        int *d_count)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= n_edges)
            return;
        if (fabsf(d_arb_signal[e]) > 0.0F)
        {
            atomicAdd(d_count, 1);
        }
    }

    __global__ void kernel_build_edge_list_from_csr(
        const int *__restrict__ d_row_ptr,
        const int *__restrict__ d_col_idx,
        int n_nodes,
        int *d_edge_src,
        int *d_edge_dst,
        int *d_edge_counter)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if (v >= n_nodes)
            return;

        const int start = d_row_ptr[v];
        const int end = d_row_ptr[v + 1];

        for (int ptr = start; ptr < end; ++ptr)
        {
            const int u = d_col_idx[ptr];
            if (u > v)
            {
                const int slot = atomicAdd(d_edge_counter, 1);
                if (slot < k_max_edges)
                {
                    d_edge_src[slot] = v;
                    d_edge_dst[slot] = u;
                }
            }
        }
    }

    __global__ void kernel_find_triangles(
        const int *__restrict__ d_row_ptr,
        const int *__restrict__ d_col_idx,
        const int *__restrict__ d_edge_src,
        const int *__restrict__ d_edge_dst,
        int n_edges,
        int n_nodes,
        int *d_triangle_edges,
        int *d_triangle_counter)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= n_edges)
            return;

        const int i = d_edge_src[e];
        const int j = d_edge_dst[e];

        const int start_i = d_row_ptr[i];
        const int end_i = d_row_ptr[i + 1];
        const int start_j = d_row_ptr[j];
        const int end_j = d_row_ptr[j + 1];

        int pi = start_i;
        int pj = start_j;

        while (pi < end_i && pj < end_j)
        {
            const int ni = d_col_idx[pi];
            const int nj = d_col_idx[pj];

            if (ni == nj && ni != i && ni != j)
            {
                const int k_node = ni;

                int e_ik = -1, e_jk = -1;
                for (int f = 0; f < n_edges; ++f)
                {
                    const int fs = d_edge_src[f];
                    const int fd = d_edge_dst[f];
                    if ((fs == i && fd == k_node) || (fs == k_node && fd == i))
                        e_ik = f;
                    if ((fs == j && fd == k_node) || (fs == k_node && fd == j))
                        e_jk = f;
                }

                if (e_ik >= 0 && e_jk >= 0)
                {
                    const int slot = atomicAdd(d_triangle_counter, 1);
                    if (slot < k_max_triangles)
                    {
                        d_triangle_edges[slot * 3 + 0] = e;
                        d_triangle_edges[slot * 3 + 1] = e_ik;
                        d_triangle_edges[slot * 3 + 2] = e_jk;
                    }
                }
                ++pi;
                ++pj;
            }
            else if (ni < nj)
            {
                ++pi;
            }
            else
            {
                ++pj;
            }
        }
    }

    void hodge_workspace_init(
        HodgeWorkspace &ws,
        int n_nodes,
        int max_edges,
        int max_triangles,
        cudaStream_t stream)
    {
        ws.n_nodes = n_nodes;
        ws.n_edges = 0;
        ws.n_triangles = 0;

        const size_t ne = static_cast<size_t>(max_edges);
        const size_t nt = static_cast<size_t>(max_triangles);
        const size_t nn = static_cast<size_t>(n_nodes);
        const size_t ne2 = ne * ne;

        ws.d_B1 = device_alloc<float>(nn * ne);
        ws.d_B2 = device_alloc<float>(ne * nt);
        ws.d_L1 = device_alloc<float>(ne2);
        ws.d_omega = device_alloc<float>(ne);
        ws.d_eigenvalues_L1 = device_alloc<float>(ne);
        ws.d_eigenvectors_L1 = device_alloc<float>(ne2);
        ws.d_gamma = device_alloc<float>(ne);
        ws.d_exact = device_alloc<float>(ne);
        ws.d_coexact = device_alloc<float>(ne);
        ws.d_curl_magnitude = device_alloc<float>(ne);
        ws.d_arb_signal = device_alloc<float>(ne);
        ws.d_prune_mask = device_alloc<uint8_t>(nn);
        ws.d_edge_src = device_alloc<int>(ne);
        ws.d_edge_dst = device_alloc<int>(ne);
        ws.d_triangle_edges = device_alloc<int>(nt * 3U);

        CUDA_CHECK(cudaMemsetAsync(ws.d_B1, 0, nn * ne * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_B2, 0, ne * nt * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_L1, 0, ne2 * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_omega, 0, ne * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_gamma, 0, ne * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_exact, 0, ne * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_coexact, 0, ne * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_curl_magnitude, 0, ne * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_arb_signal, 0, ne * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_prune_mask, 0, nn * sizeof(uint8_t), stream));
        CUDA_CHECK(cudaMemsetAsync(ws.d_triangle_edges, 0, nt * 3U * sizeof(int), stream));
    }

    void build_incidence_matrices(
        HodgeWorkspace &ws,
        const SparseLaplacian &laplacian,
        const uint8_t *d_prune_mask,
        cusparseHandle_t cusparse,
        cudaStream_t stream)
    {
        (void)cusparse;
        (void)d_prune_mask;

        int *d_edge_ctr = device_alloc<int>(1U);
        CUDA_CHECK(cudaMemsetAsync(d_edge_ctr, 0, sizeof(int), stream));

        const int blk = k_blk;
        const int grd_n = (laplacian.n_rows + blk - 1) / blk;

        kernel_build_edge_list_from_csr<<<grd_n, blk, 0, stream>>>(
            laplacian.d_row_ptr,
            laplacian.d_col_idx,
            laplacian.n_rows,
            ws.d_edge_src,
            ws.d_edge_dst,
            d_edge_ctr);

        int h_n_edges = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_n_edges, d_edge_ctr,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        h_n_edges = (h_n_edges > k_max_edges) ? k_max_edges : h_n_edges;
        ws.n_edges = h_n_edges;
        ws.n_nodes = laplacian.n_rows;

        CUDA_CHECK(cudaMemsetAsync(ws.d_B1, 0,
                                   static_cast<size_t>(ws.n_nodes) * static_cast<size_t>(h_n_edges) * sizeof(float),
                                   stream));

        const int grd_e = (h_n_edges + blk - 1) / blk;
        if (h_n_edges > 0)
        {
            kernel_build_signed_incidence<<<grd_e, blk, 0, stream>>>(
                ws.d_edge_src, ws.d_edge_dst,
                ws.n_nodes, h_n_edges, ws.d_B1);
        }

        int *d_tri_ctr = device_alloc<int>(1U);
        CUDA_CHECK(cudaMemsetAsync(d_tri_ctr, 0, sizeof(int), stream));

        if (h_n_edges > 0)
        {
            kernel_find_triangles<<<grd_e, blk, 0, stream>>>(
                laplacian.d_row_ptr, laplacian.d_col_idx,
                ws.d_edge_src, ws.d_edge_dst,
                h_n_edges, ws.n_nodes,
                ws.d_triangle_edges, d_tri_ctr);
        }

        int h_n_tri = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_n_tri, d_tri_ctr,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        h_n_tri = (h_n_tri > k_max_triangles) ? k_max_triangles : h_n_tri;
        ws.n_triangles = h_n_tri;

        CUDA_CHECK(cudaMemsetAsync(ws.d_B2, 0,
                                   static_cast<size_t>(h_n_edges) * static_cast<size_t>(h_n_tri) * sizeof(float),
                                   stream));

        if (h_n_tri > 0)
        {
            const int grd_t = (h_n_tri + blk - 1) / blk;
            kernel_build_triangle_incidence<<<grd_t, blk, 0, stream>>>(
                ws.d_edge_src, ws.d_edge_dst,
                ws.d_triangle_edges,
                h_n_tri, h_n_edges, ws.d_B2);
        }

        const int ne = h_n_edges;
        const int nn = ws.n_nodes;
        const int nt = h_n_tri;
        const dim3 blk2(16, 16);
        const dim3 grd2(
            (static_cast<unsigned>(ne) + 15U) / 16U,
            (static_cast<unsigned>(ne) + 15U) / 16U);

        if (ne > 0)
        {
            kernel_hodge1_laplacian<<<grd2, blk2, 0, stream>>>(
                ws.d_B1, ws.d_B2, ne, nn, nt, ws.d_L1);
        }

        device_free(d_edge_ctr);
        device_free(d_tri_ctr);
    }

    void compute_hodge_decomposition(
        HodgeWorkspace &ws,
        cublasHandle_t cublas,
        cusparseHandle_t cusparse,
        cudaStream_t stream)
    {
        (void)cusparse;

        const int ne = ws.n_edges;
        if (ne <= 0)
            return;

        cusolverDnHandle_t cusolver;
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));

        CUDA_CHECK(cudaMemcpyAsync(
            ws.d_eigenvectors_L1, ws.d_L1,
            static_cast<size_t>(ne * ne) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));

        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(
            cusolver,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            ne,
            ws.d_eigenvectors_L1,
            ne,
            ws.d_eigenvalues_L1,
            &lwork));

        float *d_work = device_alloc<float>(static_cast<size_t>(lwork));
        int *d_info = device_alloc<int>(1U);

        CUSOLVER_CHECK(cusolverDnSsyevd(
            cusolver,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            ne,
            ws.d_eigenvectors_L1,
            ne,
            ws.d_eigenvalues_L1,
            d_work, lwork,
            d_info));

        int h_info = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_info, d_info,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (h_info != 0)
        {
            std::fprintf(stderr,
                         "[Hodge] cusolverDnSsyevd failed: info=%d\n", h_info);
        }

        int h_n_harmonic = 0;
        {
            float *h_evals = pinned_alloc<float>(static_cast<size_t>(ne));
            CUDA_CHECK(cudaMemcpyAsync(h_evals, ws.d_eigenvalues_L1,
                                       static_cast<size_t>(ne) * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < ne; ++i)
            {
                if (fabsf(h_evals[i]) < k_harmonic_threshold)
                    ++h_n_harmonic;
            }
            pinned_free(h_evals);
        }
        ws.n_harmonic_dims = h_n_harmonic;

        const GpuLobSnapshot *snap = nullptr;
        (void)snap;

        const int blk = k_blk;
        const int grd = (ne + blk - 1) / blk;

        kernel_harmonic_projection<<<grd, blk, 0, stream>>>(
            ws.d_eigenvectors_L1,
            ws.d_eigenvalues_L1,
            ws.d_omega,
            ne, ne,
            k_harmonic_threshold,
            ws.d_gamma);

        float *d_node_pot = device_alloc<float>(static_cast<size_t>(ne));

        kernel_exact_component<<<grd, blk, 0, stream>>>(
            ws.d_B1, ws.d_omega, ws.d_gamma,
            ne, ws.n_nodes,
            d_node_pot, ws.d_exact);

        kernel_coexact_component<<<grd, blk, 0, stream>>>(
            ws.d_omega, ws.d_gamma, ws.d_exact,
            ne, ws.d_coexact);

        kernel_curl_magnitude<<<grd, blk, 0, stream>>>(
            ws.d_gamma, ne,
            ws.d_curl_magnitude,
            ws.d_arb_signal,
            k_arbitrage_threshold);

        float *d_ym = device_alloc<float>(1U);
        CUDA_CHECK(cudaMemsetAsync(d_ym, 0, sizeof(float), stream));

        const int smem = blk * static_cast<int>(sizeof(float));
        kernel_yang_mills_action<<<grd, blk, static_cast<size_t>(smem), stream>>>(
            ws.d_gamma, ws.d_coexact, ne, 1.0F, d_ym);

        float h_ym = 0.0F;
        CUDA_CHECK(cudaMemcpyAsync(&h_ym, d_ym,
                                   sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        ws.yang_mills_action = h_ym;

        device_free(d_work);
        device_free(d_info);
        device_free(d_node_pot);
        device_free(d_ym);
        cusolverDnDestroy(cusolver);
        (void)cublas;
    }

    ArbitrageSignal extract_arbitrage_signal(
        HodgeWorkspace &ws,
        cudaStream_t stream)
    {
        int *d_count = device_alloc<int>(1U);
        CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

        const int blk = k_blk;
        const int grd = (ws.n_edges + blk - 1) / blk;

        if (ws.n_edges > 0)
        {
            kernel_count_active_loops<<<grd, blk, 0, stream>>>(
                ws.d_arb_signal, ws.n_edges, d_count);
        }

        int h_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        device_free(d_count);

        struct timespec ts{};
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        const uint64_t now = static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);

        return ArbitrageSignal{
            .d_harmonic_flow = ws.d_gamma,
            .d_curl_magnitude = ws.d_curl_magnitude,
            .yang_mills_action = ws.yang_mills_action,
            .n_active_loops = h_count,
            .n_edges = ws.n_edges,
            .signal_ts_ns = now,
        };
    }

    void copy_signal_to_host(
        const ArbitrageSignal &signal,
        float *h_harmonic_out,
        float *h_curl_out,
        int n_edges,
        cudaStream_t stream)
    {
        if (n_edges <= 0)
            return;

        CUDA_CHECK(cudaMemcpyAsync(
            h_harmonic_out,
            signal.d_harmonic_flow,
            static_cast<size_t>(n_edges) * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));

        CUDA_CHECK(cudaMemcpyAsync(
            h_curl_out,
            signal.d_curl_magnitude,
            static_cast<size_t>(n_edges) * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream));
    }

} // namespace holo::cuda