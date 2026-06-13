from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Final, Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

NodeIndices = NDArray[np.int64]
AdjacencyMatrix = sp.csr_matrix
LaplacianMatrix = sp.csr_matrix

_PRUNING_TOLERANCE: Final[float] = 1e-7
_DEFAULT_THREAD_COUNT: Final[int] = 32


@dataclass(frozen=True, slots=True)
class SpectralConfig:
    retention_threshold: float = 1e-4
    max_eigen_iter: int = 5000
    n_threads: int = _DEFAULT_THREAD_COUNT
    solver_mode: Literal["ARPACK", "LOBPCG"] = "ARPACK"


@dataclass(slots=True)
class PruningResult:
    original_size: int
    pruned_size: int
    fiedler_eigenvalue: float
    retained_indices: NodeIndices
    pruned_adjacency: AdjacencyMatrix
    fiedler_vector: NDArray[np.float64]


def _build_laplacian(adj: AdjacencyMatrix) -> LaplacianMatrix:
    degrees: NDArray[np.float64] = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt: NDArray[np.float64] = np.zeros_like(degrees)
    valid: NDArray[np.bool_] = degrees > 1e-12
    d_inv_sqrt[valid] = 1.0 / np.sqrt(degrees[valid])
    d_mat: sp.diags = sp.diags(d_inv_sqrt, format="csr")
    identity: sp.csr_matrix = sp.eye(adj.shape[0], format="csr")
    normalized_laplacian: sp.csr_matrix = identity - d_mat @ adj @ d_mat
    return normalized_laplacian


def _compute_fiedler_arpack(
        laplacian: LaplacianMatrix,
        config: SpectralConfig,
) -> tuple[float, NDArray[np.float64]]:
    eigenvalues, eigenvectors = spla.eigsh(
        laplacian,
        k=2,
        which="SM",
        tol=_PRUNING_TOLERANCE,
        maxiter=config.max_eigen_iter,
        return_eigenvectors=True,
    )
    return float(eigenvalues[1]), eigenvectors[:, 1]


def _compute_fiedler_lobpcg(
        laplacian: LaplacianMatrix,
        config: SpectralConfig,
) -> tuple[float, NDArray[np.float64]]:
    n: int = laplacian.shape[0]
    x0: NDArray[np.float64] = np.random.default_rng(0xCAFEBABE).standard_normal((n, 2))
    eigenvalues, eigenvectors = spla.lobpcg(
        laplacian,
        x0,
        tol=_PRUNING_TOLERANCE,
        maxiter=config.max_eigen_iter,
        largest=False,
    )
    idx: NDArray[np.intp] = np.argsort(eigenvalues)
    return float(eigenvalues[idx[1]]), eigenvectors[:, idx[1]]


def _parallel_fiedler_extraction(
        laplacian: LaplacianMatrix,
        config: SpectralConfig,
) -> tuple[float, NDArray[np.float64]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.n_threads) as executor:
        if config.solver_mode == "ARPACK":
            future: concurrent.futures.Future[tuple[float, NDArray[np.float64]]] = executor.submit(
                _compute_fiedler_arpack, laplacian, config
            )
        else:
            future = executor.submit(
                _compute_fiedler_lobpcg, laplacian, config
            )
        return future.result()


class FiedlerPruner:
    def __init__(self, config: SpectralConfig) -> None:
        self._config: SpectralConfig = config

    @property
    def config(self) -> SpectralConfig:
        return self._config

    def prune(self, adjacency: AdjacencyMatrix) -> PruningResult:
        n_orig: int = adjacency.shape[0]
        lap: LaplacianMatrix = _build_laplacian(adjacency)
        f_val, f_vec = _parallel_fiedler_extraction(lap, self._config)

        retained_mask: NDArray[np.bool_] = np.abs(f_vec) >= self._config.retention_threshold
        retained_indices: NodeIndices = np.where(retained_mask)[0]

        pruned_adj: AdjacencyMatrix = adjacency[retained_indices, :][:, retained_indices]

        return PruningResult(
            original_size=n_orig,
            pruned_size=int(len(retained_indices)),
            fiedler_eigenvalue=f_val,
            retained_indices=retained_indices,
            pruned_adjacency=pruned_adj,
            fiedler_vector=f_vec,
        )

    def reconstruct_graph(self, result: PruningResult) -> nx.Graph:
        return nx.from_scipy_sparse_array(result.pruned_adjacency)
