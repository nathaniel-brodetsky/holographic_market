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


def _tt_svd_sweep(
        tensor: NDArray[np.float32],
        chi: int,
        threshold: float,
) -> tuple[MPSNodes, BondDimensions, float]:
    """Canonical left-to-right TT-SVD (Oseledets, 2011) over the tensor's own
    axes — for `tensor` of shape (n_0, n_1, ..., n_{d-1})`, this produces `d`
    cores G_0, ..., G_{d-1} with G_k of shape (r_k, n_k, r_{k+1}), r_0 = r_d = 1.

    IMPORTANT — this replaces the previous `_left_canonical_svd_sweep`, which
    treated each row of a pre-flattened (n_instruments, depth*sides) matrix as
    one "site". That construction was not a valid TT/MPS decomposition: a
    flattened 2D matrix's rows have no joint multi-axis structure for a bond
    dimension to grow across, so every SVD in that sweep was necessarily rank
    <= 1 by shape alone (SVD of a (1, N) row matrix has exactly one singular
    value) — `bond_dimension` (chi) could never have any effect, at any value.
    The fix is not a tweak to the combination step; it's decomposing along the
    tensor's actual axes (here: instrument, depth level, side), which is what
    "sites" means in a tensor train / MPS decomposition. n_sites is now
    len(tensor.shape) (3 for the LOB tensor), not n_instruments.

    Returns (nodes, bond_dims, dropped_energy_sq) — dropped_energy_sq is the
    running sum of squared singular values discarded at each unfolding; the
    standard TT-SVD error bound is ||T - T_tt||_F <= sqrt(dropped_energy_sq)
    (Oseledets 2011, Theorem 2.2). This is an exact, unfold-native bound (not
    an approximation), unlike the previous docstring's proxy claim.
    """
    tn.set_default_backend("numpy")
    shape: tuple[int, ...] = tensor.shape
    n_sites: int = len(shape)
    nodes: MPSNodes = []
    bond_dims: BondDimensions = [1]
    dropped_energy_sq: float = 0.0

    # `carry` starts as the full tensor reshaped to 2D: (r_prev * n_site, rest).
    carry: NDArray[np.float32] = tensor.reshape(1 * shape[0], -1).astype(np.float32)
    r_prev: int = 1

    for site in range(n_sites - 1):
        n_site: int = shape[site]
        u, s, vt = np.linalg.svd(carry, full_matrices=False)
        rank: int = max(min(int(np.sum(s > threshold)), chi, u.shape[1]), 1)

        dropped_energy_sq += float(np.sum(np.square(s[rank:], dtype=np.float64)))

        u_trunc: NDArray[np.float32] = u[:, :rank].astype(np.float32)
        s_trunc: NDArray[np.float32] = s[:rank].astype(np.float32)
        vt_trunc: NDArray[np.float32] = vt[:rank, :].astype(np.float32)

        core: NDArray[np.float32] = u_trunc.reshape(r_prev, n_site, rank)
        nodes.append(tn.Node(core, name=f"G_{site}"))
        bond_dims.append(rank)

        # Carry S@V into the next unfolding: shape (rank * n_{site+1}, rest_after).
        remaining: NDArray[np.float32] = (np.diag(s_trunc) @ vt_trunc).astype(np.float32)
        next_n: int = shape[site + 1]
        rest: int = remaining.size // (rank * next_n)
        carry = remaining.reshape(rank * next_n, rest)
        r_prev = rank

    last_core: NDArray[np.float32] = carry.reshape(r_prev, shape[-1], 1)
    nodes.append(tn.Node(last_core, name=f"G_{n_sites - 1}"))
    bond_dims.append(1)
    return nodes, bond_dims, dropped_energy_sq


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

            nodes, bond_dims, dropped_energy_sq = _tt_svd_sweep(
                lob_tensor,
                self._config.bond_dimension,
                self._config.svd_truncation_threshold,
            )

            compressed_bytes: int = sum(n.tensor.nbytes for n in nodes)
            ratio: float = original_bytes / max(compressed_bytes, 1)

            tensor_norm: float = float(np.linalg.norm(lob_tensor))
            # Oseledets TT-SVD error bound (exact for this construction, not a
            # proxy): ||T - T_tt||_F <= sqrt(dropped_energy_sq).
            frob_error: float = (
                float(np.sqrt(dropped_energy_sq) / tensor_norm) if tensor_norm > 0.0 else 0.0
            )

            self._result = MPSResult(
                nodes=nodes,
                bond_dims=bond_dims,
                original_shape=lob_tensor.shape,
                compressed_bytes=compressed_bytes,
                original_bytes=original_bytes,
                compression_ratio=ratio,
                frobenius_error=frob_error,
            )
            gc.collect()
            return self._result