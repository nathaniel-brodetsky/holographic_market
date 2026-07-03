#include <math/floer_homology.cuh>
#include <math/cuda_utils.cuh>

#include <cstring>
#include <cstdio>
#include <cmath>

namespace holo::cuda
{

__global__ void kernel_find_critical_points(
    const float *__restrict__ d_curl_magnitude,
    const float *__restrict__ d_curl_coexact,
    const int   *__restrict__ d_edge_src,
    const int   *__restrict__ d_edge_dst,
    int          n_edges,
    float        ym_action,
    CriticalPoint *d_criticals,
    int           *d_n_criticals)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    const float curl_e    = d_curl_magnitude[e];
    const float coexact_e = d_curl_coexact[e];

    if (curl_e < k_instanton_threshold) return;

    const int src = d_edge_src[e];
    const int dst = d_edge_dst[e];
    float neighbour_sum   = 0.0F;
    int   neighbour_count = 0;

    for (int f = 0; f < n_edges; ++f)
    {
        if (f == e) continue;
        if (d_edge_src[f] == src || d_edge_dst[f] == dst ||
            d_edge_src[f] == dst || d_edge_dst[f] == src)
        {
            neighbour_sum += d_curl_magnitude[f];
            ++neighbour_count;
        }
    }

    if (neighbour_count == 0) return;

    const float avg_neighbour = neighbour_sum / static_cast<float>(neighbour_count);
    const float hessian       = curl_e - avg_neighbour;

    MorseIndex morse;
    constexpr float k_saddle_band = 0.05F;
    if      (hessian >  k_saddle_band) morse = MorseIndex::Maximum;
    else if (hessian < -k_saddle_band) morse = MorseIndex::Minimum;
    else                               morse = MorseIndex::Saddle;

    const float self_dual_dev = fabsf(coexact_e - curl_e) / (curl_e + 1e-8F);
    const bool  self_dual     = (self_dual_dev < k_self_dual_tolerance);

    const float local_action = (curl_e * curl_e) /
        (2.0F * 3.14159265F * (ym_action + 1e-8F));

    const int slot = atomicAdd(d_n_criticals, 1);
    if (slot < k_floer_max_criticals)
    {
        d_criticals[slot] = CriticalPoint{
            .edge_idx       = e,
            .action         = local_action,
            .curl_magnitude = curl_e,
            .morse_index    = morse,
            .self_dual      = self_dual,
        };
    }
}

__global__ void kernel_find_instantons(
    const CriticalPoint *__restrict__ d_criticals,
    int                              n_criticals,
    const float *__restrict__        d_curl_magnitude,
    const float *__restrict__        d_curl_coexact,
    int                              n_edges,
    float                            ym_action,
    InstantonPath                   *d_instantons,
    int                             *d_n_instantons)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = n_criticals;
    if (tid >= n * n) return;

    const int i = tid / n;
    const int j = tid % n;
    if (i >= j) return;

    const CriticalPoint &cp_i = d_criticals[i];
    const CriticalPoint &cp_j = d_criticals[j];

    const CriticalPoint *src = nullptr;
    const CriticalPoint *dst = nullptr;
    int src_idx = -1, dst_idx = -1;

    if (static_cast<int>(cp_i.morse_index) < static_cast<int>(cp_j.morse_index))
    { src = &cp_i; src_idx = i; dst = &cp_j; dst_idx = j; }
    else if (static_cast<int>(cp_j.morse_index) < static_cast<int>(cp_i.morse_index))
    { src = &cp_j; src_idx = j; dst = &cp_i; dst_idx = i; }
    else return;

    const float curl_src    = d_curl_magnitude[src->edge_idx];
    const float curl_dst    = d_curl_magnitude[dst->edge_idx];
    const float coexact_src = d_curl_coexact[src->edge_idx];
    const float coexact_dst = d_curl_coexact[dst->edge_idx];

    const float asd_residual = fabsf(coexact_src + coexact_dst) /
                               (curl_src + curl_dst + 1e-8F);
    const float path_action  = sqrtf(src->action * dst->action + 1e-10F);
    const int slot = atomicAdd(d_n_instantons, 1);

    if (slot < k_floer_max_criticals * k_floer_max_criticals)
    {
        d_instantons[slot] = InstantonPath{
            .src_critical       = src_idx,
            .dst_critical       = dst_idx,
            .path_action        = path_action,
            .self_dual_residual = asd_residual,
            .valid              = true,
        };
    }
}

__global__ void kernel_build_floer_boundary(
    const CriticalPoint *__restrict__ d_criticals,
    const InstantonPath *__restrict__ d_instantons,
    int   n_criticals,
    int   n_instantons,
    int  *d_boundary_matrix)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_instantons) return;

    const InstantonPath &inst = d_instantons[tid];
    if (!inst.valid) return;
    if (inst.src_critical < 0 || inst.dst_critical < 0) return;
    if (inst.src_critical >= n_criticals || inst.dst_critical >= n_criticals) return;

    const int src_morse = static_cast<int>(d_criticals[inst.src_critical].morse_index);
    const int dst_morse = static_cast<int>(d_criticals[inst.dst_critical].morse_index);
    if (abs(dst_morse - src_morse) != 1) return;

    atomicXor(&d_boundary_matrix[inst.dst_critical * n_criticals + inst.src_critical], 1);
}

static int gaussian_rank_z2(int *mat, int rows, int cols)
{
    int rank = 0, pivot_row = 0;
    for (int col = 0; col < cols && pivot_row < rows; ++col)
    {
        int pivot = -1;
        for (int r = pivot_row; r < rows; ++r)
            if (mat[r * cols + col] & 1) { pivot = r; break; }
        if (pivot < 0) continue;
        for (int c = 0; c < cols; ++c)
        {
            int t = mat[pivot_row * cols + c];
            mat[pivot_row * cols + c] = mat[pivot * cols + c];
            mat[pivot * cols + c] = t;
        }
        for (int r = 0; r < rows; ++r)
            if (r != pivot_row && (mat[r * cols + col] & 1))
                for (int c = 0; c < cols; ++c)
                    mat[r * cols + c] ^= mat[pivot_row * cols + c];
        ++rank; ++pivot_row;
    }
    return rank;
}

FloerRecord compute_floer_homology(
    const CriticalPoint *h_criticals,
    const InstantonPath *h_instantons,
    const int           *h_boundary_matrix,
    int                  n_criticals,
    int                  n_instantons)
{
    FloerRecord rec{};
    rec.n_criticals  = n_criticals;
    rec.n_instantons = n_instantons;
    if (n_criticals == 0) { rec.entry_signal = MorseIndex::Saddle; return rec; }

    int n_grade[3] = {0, 0, 0};
    for (int i = 0; i < n_criticals; ++i)
        ++n_grade[static_cast<int>(h_criticals[i].morse_index)];

    int rank_d1 = 0, rank_d2 = 0;

    if (n_grade[0] > 0 && n_grade[1] > 0)
    {
        const int r = n_grade[0], c = n_grade[1];
        int *sub = new int[r * c]();
        int ri = 0;
        for (int i = 0; i < n_criticals && ri < r; ++i)
        {
            if (h_criticals[i].morse_index != MorseIndex::Minimum) continue;
            int ci = 0;
            for (int j = 0; j < n_criticals && ci < c; ++j)
            {
                if (h_criticals[j].morse_index != MorseIndex::Saddle) continue;
                sub[ri * c + ci] = h_boundary_matrix[i * n_criticals + j];
                ++ci;
            }
            ++ri;
        }
        rank_d1 = gaussian_rank_z2(sub, r, c);
        delete[] sub;
    }

    if (n_grade[1] > 0 && n_grade[2] > 0)
    {
        const int r = n_grade[1], c = n_grade[2];
        int *sub = new int[r * c]();
        int ri = 0;
        for (int i = 0; i < n_criticals && ri < r; ++i)
        {
            if (h_criticals[i].morse_index != MorseIndex::Saddle) continue;
            int ci = 0;
            for (int j = 0; j < n_criticals && ci < c; ++j)
            {
                if (h_criticals[j].morse_index != MorseIndex::Maximum) continue;
                sub[ri * c + ci] = h_boundary_matrix[i * n_criticals + j];
                ++ci;
            }
            ++ri;
        }
        rank_d2 = gaussian_rank_z2(sub, r, c);
        delete[] sub;
    }

    rec.rank_HF0 = n_grade[0] - rank_d1;
    rec.rank_HF1 = n_grade[1] - rank_d1 - rank_d2;
    rec.rank_HF2 = n_grade[2] - rank_d2;
    if (rec.rank_HF0 < 0) rec.rank_HF0 = 0;
    if (rec.rank_HF1 < 0) rec.rank_HF1 = 0;
    if (rec.rank_HF2 < 0) rec.rank_HF2 = 0;
    rec.euler_characteristic = rec.rank_HF0 - rec.rank_HF1 + rec.rank_HF2;

    float best_entry = 1e30F, best_exit = 0.0F, best_len = 0.0F;
    float total_asd = 0.0F;
    bool  any = false;

    for (int k = 0; k < n_instantons; ++k)
    {
        const InstantonPath &inst = h_instantons[k];
        if (!inst.valid) continue;
        if (inst.src_critical < 0 || inst.src_critical >= n_criticals) continue;
        if (inst.dst_critical < 0 || inst.dst_critical >= n_criticals) continue;
        const CriticalPoint &s = h_criticals[inst.src_critical];
        const CriticalPoint &d = h_criticals[inst.dst_critical];
        if (s.morse_index == MorseIndex::Minimum && s.action < best_entry)
        { best_entry = s.action; best_len = inst.path_action; any = true; }
        if (d.morse_index == MorseIndex::Maximum && d.action > best_exit)
            best_exit = d.action;
        total_asd += inst.self_dual_residual;
    }

    rec.instanton_found         = any;
    rec.entry_action            = best_entry < 1e29F ? best_entry : 0.0F;
    rec.exit_action             = best_exit;
    rec.instanton_length        = best_len;
    rec.mean_self_dual_residual = n_instantons > 0
        ? total_asd / static_cast<float>(n_instantons) : 0.0F;

    if      (rec.rank_HF0 > rec.rank_HF2 && rec.rank_HF0 > rec.rank_HF1)
        rec.entry_signal = MorseIndex::Minimum;
    else if (rec.rank_HF2 > rec.rank_HF0 && rec.rank_HF2 > rec.rank_HF1)
        rec.entry_signal = MorseIndex::Maximum;
    else
        rec.entry_signal = MorseIndex::Saddle;

    return rec;
}

FloerRecord run_floer_analysis(
    FloerWorkspace      &ws,
    const float         *d_curl_magnitude,
    const float         *d_curl_coexact,
    const int           *d_edge_src,
    const int           *d_edge_dst,
    int                  n_edges,
    float                ym_action,
    cudaStream_t         stream)
{
    if (n_edges <= 0) return FloerRecord{};

    CUDA_CHECK(cudaMemsetAsync(ws.d_n_criticals,  0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.d_n_instantons, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.d_criticals, 0,
        k_floer_max_criticals * sizeof(CriticalPoint), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.d_boundary_matrix, 0,
        k_floer_max_criticals * k_floer_max_criticals * sizeof(int), stream));

    {
        const int blocks = (n_edges + k_floer_block_dim - 1) / k_floer_block_dim;
        kernel_find_critical_points<<<blocks, k_floer_block_dim, 0, stream>>>(
            d_curl_magnitude, d_curl_coexact,
            d_edge_src, d_edge_dst,
            n_edges, ym_action,
            ws.d_criticals, ws.d_n_criticals);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_n_criticals = 0;
    CUDA_CHECK(cudaMemcpy(&h_n_criticals, ws.d_n_criticals,
        sizeof(int), cudaMemcpyDeviceToHost));
    h_n_criticals = (h_n_criticals < k_floer_max_criticals)
        ? h_n_criticals : k_floer_max_criticals;

    if (h_n_criticals < 2) return FloerRecord{.n_criticals = h_n_criticals};

    {
        const int n_pairs = h_n_criticals * h_n_criticals;
        const int blocks  = (n_pairs + k_floer_block_dim - 1) / k_floer_block_dim;
        kernel_find_instantons<<<blocks, k_floer_block_dim, 0, stream>>>(
            ws.d_criticals, h_n_criticals,
            d_curl_magnitude, d_curl_coexact,
            n_edges, ym_action,
            ws.d_instantons, ws.d_n_instantons);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_n_instantons = 0;
    CUDA_CHECK(cudaMemcpy(&h_n_instantons, ws.d_n_instantons,
        sizeof(int), cudaMemcpyDeviceToHost));
    const int max_inst = k_floer_max_criticals * k_floer_max_criticals;
    h_n_instantons = (h_n_instantons < max_inst) ? h_n_instantons : max_inst;

    if (h_n_instantons > 0)
    {
        const int blocks = (h_n_instantons + k_floer_block_dim - 1) / k_floer_block_dim;
        kernel_build_floer_boundary<<<blocks, k_floer_block_dim, 0, stream>>>(
            ws.d_criticals, ws.d_instantons,
            h_n_criticals, h_n_instantons,
            ws.d_boundary_matrix);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    CriticalPoint h_criticals[k_floer_max_criticals]{};
    InstantonPath h_instantons[k_floer_max_criticals * 4]{};
    int           h_boundary[k_floer_max_criticals * k_floer_max_criticals]{};

    CUDA_CHECK(cudaMemcpy(h_criticals, ws.d_criticals,
        h_n_criticals * sizeof(CriticalPoint), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_instantons, ws.d_instantons,
        h_n_instantons * sizeof(InstantonPath), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_boundary, ws.d_boundary_matrix,
        h_n_criticals * h_n_criticals * sizeof(int), cudaMemcpyDeviceToHost));

    return compute_floer_homology(
        h_criticals, h_instantons, h_boundary,
        h_n_criticals, h_n_instantons);
}

} // namespace holo::cuda