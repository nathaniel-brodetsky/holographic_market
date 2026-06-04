import sys
import os
import time

sys.path.insert(0, os.path.abspath('src'))

import numpy as np
import jax.numpy as jnp
import scipy.sparse as sp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tensor_compression import LOBTensorTrain, LOBCompressionConfig
from spectral_pruning import FiedlerPruner, SpectralConfig
from gauge_theory_alpha import HodgeLaplacianJAX, HodgeConfig


def main():
    print("=" * 60)
    print(" HOLOGRAPHIC MARKET ARCHITECTURE - TQFT SIMULATION")
    print("=" * 60)

    print("\n[1/4] Tensor Network Compression (MPS)...")
    t0 = time.time()
    config_mps = LOBCompressionConfig(n_instruments=256, depth_levels=10, bond_dimension=16)
    tt_engine = LOBTensorTrain(config_mps)
    mps_state = tt_engine.compress()
    print(f"      Original Size: {mps_state.original_bytes / 1e6:.2f} MB")
    print(f"      Compressed:    {mps_state.compressed_bytes / 1e6:.4f} MB")
    print(f"      Ratio:         {mps_state.compression_ratio:.1f}x")
    print(f"      Time:          {time.time() - t0:.3f}s")

    print("\n[2/4] CPU-bound Spectral Pruning (Fiedler Vector)...")
    t0 = time.time()
    rng = np.random.default_rng(42)
    raw_adj = rng.uniform(0, 0.01, (500, 500))
    raw_adj = raw_adj + raw_adj.T
    np.fill_diagonal(raw_adj, 0)
    adj_sparse = sp.csr_matrix(raw_adj)

    pruner = FiedlerPruner(SpectralConfig(n_threads=32, retention_threshold=1e-3))
    pruned_graph = pruner.prune(adj_sparse)
    print(f"      Original Nodes: {pruned_graph.original_size}")
    print(f"      Retained (Alpha) Nodes: {pruned_graph.pruned_size}")
    print(f"      Time: {time.time() - t0:.3f}s")

    print("\n[3/4] JAX Hodge-De Rham Curvature Extraction...")
    t0 = time.time()
    n_active = pruned_graph.pruned_size
    edges = [(i, (i + 1) % n_active) for i in range(n_active)] + [(i, (i + 2) % n_active) for i in range(n_active)]
    faces = [(i, (i + 1) % n_active, (i + 2) % n_active) for i in range(n_active)]

    flow = rng.standard_normal(len(edges))

    hodge_engine = HodgeLaplacianJAX(HodgeConfig(use_bfloat16_topology=False))
    alpha_result = hodge_engine.extract_arbitrage_manifold(
        n_nodes=n_active, edges=edges, faces=faces, observed_flow=flow
    )
    print(f"      Yang-Mills Arbitrage Energy: {alpha_result.total_arbitrage_energy:.6f}")
    print(f"      Time: {time.time() - t0:.3f}s")

    print("\n[4/4] Rendering Holographic LOB Vector Field...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8), facecolor='#0D1117')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0D1117')

    theta = np.linspace(0, 2 * np.pi, len(edges))
    z = np.sin(3 * theta)
    x = np.cos(theta) * (1 + 0.5 * np.cos(5 * theta))
    y = np.sin(theta) * (1 + 0.5 * np.cos(5 * theta))

    curl_vectors = np.array(alpha_result.curl_flow)
    u, v, w = -y * curl_vectors, x * curl_vectors, z * curl_vectors

    quiver = ax.quiver(x, y, z, u, v, w, length=0.15, normalize=True,
                       cmap='magma', array=np.abs(curl_vectors), linewidths=1.5)

    ax.set_title(r"$\delta\beta$: Cohomological Arbitrage Vector Field", color='white', pad=20, fontsize=14)
    ax.axis('off')

    cbar = plt.colorbar(quiver, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(r'Yang-Mills Curvature $|F_{\mu\nu}|$', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    plt.savefig('holographic_curl_flow.png', dpi=200, bbox_inches='tight')
    print("      -> Saved to 'holographic_curl_flow.png'")
    print("=" * 60)
    print(" MISSION ACCOMPLISHED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
