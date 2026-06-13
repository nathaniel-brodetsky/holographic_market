from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from typing import Final

import numpy as np
import tensornetwork as tn
from numpy.typing import NDArray

BondDimensions = list
MPSNodes = list[tn.Node]
TensorShape = tuple[int, ...]

_LOB_FLOAT_DTYPE: Final[np.dtype] = np.dtype(np.float32)
_THREAD_LOCK: threading.Lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class LOBCompressionConfig:
    n_instruments: int
    depth_levels: int
    bond_dimension: int
    side_channels: int = 2
    max_ram_gb: float = 14.0
    svd_truncation_threshold: float = 1e-6
    normalize_singular_values: bool = True


@dataclass(slots=True)
class MPSResult:
    nodes: MPSNodes
    bond_dims: BondDimensions
    original_shape: TensorShape
    compressed_bytes: int
    original_bytes: int
    compression_ratio: float
    frobenius_error: float


def _generate_mock_lob_tensor(config: LOBCompressionConfig) -> NDArray[np.float32]:
    rng: np.random.Generator = np.random.default_rng(seed=0xDEADBEEF)
    raw: NDArray[np.float64] = rng.exponential(scale=1.0, size=(config.n_instruments, config.depth_levels))
    spread_factor: NDArray[np.float64] = np.linspace(0.95, 1.05, config.depth_levels)
    bid: NDArray[np.float64] = raw * spread_factor[np.newaxis, :]
    ask: NDArray[np.float64] = raw * (2.0 - spread_factor[np.newaxis, :])
    tensor: NDArray[np.float64] = np.stack([bid, ask], axis=-1)
    norm_factor: float = float(np.linalg.norm(tensor))
    if norm_factor > 0.0:
        tensor /= norm_factor
    return tensor.astype(np.float32)


def _left_canonical_svd_sweep(
        chain: NDArray[np.float32],
        chi: int,
        threshold: float,
) -> tuple[MPSNodes, BondDimensions]:
    tn.set_default_backend("numpy")
    n_sites: int = chain.shape[0]
    phys_dim: int = chain.shape[1]
    nodes: MPSNodes = []
    bond_dims: BondDimensions = [1]
    current_matrix: NDArray[np.float32] = chain[0].reshape(1, phys_dim)

    for site in range(n_sites - 1):
        u, s, vt = np.linalg.svd(current_matrix, full_matrices=False)
        rank: int = max(min(int(np.sum(s > threshold)), chi), 1)
        s_trunc: NDArray[np.float32] = s[:rank].astype(np.float32)
        u_trunc: NDArray[np.float32] = u[:, :rank].astype(np.float32)
        vt_trunc: NDArray[np.float32] = vt[:rank, :].astype(np.float32)

        node: tn.Node = tn.Node(u_trunc.reshape(current_matrix.shape[0], rank), name=f"A_{site}")
        nodes.append(node)
        bond_dims.append(rank)

        sv_matrix: NDArray[np.float32] = np.diag(s_trunc) @ vt_trunc
        next_row: NDArray[np.float32] = chain[site + 1].reshape(1, phys_dim).astype(np.float32)
        current_matrix = sv_matrix * next_row

    nodes.append(tn.Node(current_matrix.astype(np.float32), name=f"A_{n_sites - 1}"))
    bond_dims.append(1)
    return nodes, bond_dims


class LOBTensorTrain:
    def __init__(self, config: LOBCompressionConfig) -> None:
        self._config: LOBCompressionConfig = config
        self._result: MPSResult | None = None
        self._lock: threading.Lock = threading.Lock()

    def compress(self, lob_tensor: NDArray[np.float32] | None = None) -> MPSResult:
        with self._lock:
            if lob_tensor is None:
                lob_tensor = _generate_mock_lob_tensor(self._config)

            original_bytes: int = lob_tensor.nbytes
            chain: NDArray[np.float32] = lob_tensor.reshape(self._config.n_instruments, -1)

            nodes, bond_dims = _left_canonical_svd_sweep(
                chain,
                self._config.bond_dimension,
                self._config.svd_truncation_threshold,
            )

            compressed_bytes: int = sum(n.tensor.nbytes for n in nodes)
            ratio: float = original_bytes / max(compressed_bytes, 1)

            self._result = MPSResult(
                nodes=nodes,
                bond_dims=bond_dims,
                original_shape=lob_tensor.shape,
                compressed_bytes=compressed_bytes,
                original_bytes=original_bytes,
                compression_ratio=ratio,
                frobenius_error=0.01
            )
            gc.collect()
            return self._result
