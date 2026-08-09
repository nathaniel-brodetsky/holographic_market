from __future__ import annotations

import jax
import jax.numpy as jnp
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Final, Sequence
from jax import config as jax_config
from jax.experimental import sparse as jsparse

jax_config.update("jax_enable_x64", True)

JAXArray = jax.Array
JAXSparse = jsparse.BCOO
EdgeIndices = tuple[int, int]
FaceIndices = tuple[int, int, int]

_TOLERANCE_64: Final[float] = 1e-12
_ALPHA_LEARNING_RATE: Final[float] = 1e-3


@dataclass(frozen=True, slots=True)
class HodgeConfig:
    # NOTE: harmonic_flow = flow - gradient_flow - curl_flow is a *residual*
    # of nearly-equal terms by construction (it's the near-kernel component
    # of Delta_1) — this is exactly the computation bfloat16's ~3 significant
    # decimal digits are worst at. Casting up to float64 *after* computing in
    # bf16 (see self._dtype_exec below) does not recover precision already
    # lost during the subtraction itself. Default is float32 for the
    # decomposition; opt into bf16 explicitly only if you've verified the
    # resulting harmonic_flow signal-to-noise for your graph size/scale.
    use_bfloat16_topology: bool = False
    ricci_flow_steps: int = 100
    ricci_dt: float = 0.01


@dataclass(slots=True)
class CoExactFlowResult:
    gradient_flow: JAXArray
    harmonic_flow: JAXArray
    curl_flow: JAXArray
    ricci_evolved_flow: JAXArray
    total_arbitrage_energy: float


def _build_incidence_matrices(
        n_nodes: int,
        edges: Sequence[EdgeIndices],
        faces: Sequence[FaceIndices],
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    n_edges: int = len(edges)
    n_faces: int = len(faces)

    row_b1, col_b1, data_b1 = [], [], []
    for e_idx, (u, v) in enumerate(edges):
        row_b1.extend([e_idx, e_idx])
        col_b1.extend([u, v])
        data_b1.extend([-1.0, 1.0])

    b1: sp.csr_matrix = sp.csr_matrix((data_b1, (row_b1, col_b1)), shape=(n_edges, n_nodes))

    row_b2, col_b2, data_b2 = [], [], []
    edge_map: dict[tuple[int, int], int] = {}
    for i, (u, v) in enumerate(edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = -i - 1

    def _edge_lookup(a: int, b: int) -> int:
        """Both orientations of every real edge are already keys in edge_map
        (see loop above), so a single lookup on (a, b) suffices. If it's
        missing, the face references an edge that doesn't exist in `edges`
        at all — that's inconsistent input topology, not something to paper
        over with a silent default (a silent default here would corrupt B2,
        and therefore every downstream Hodge quantity, without any signal)."""
        if (a, b) not in edge_map:
            raise ValueError(
                f"face references edge ({a}, {b}) that is not present in "
                f"`edges` (in either orientation) — inconsistent topology input"
            )
        return edge_map[(a, b)]

    for f_idx, (u, v, w) in enumerate(faces):
        e1: int = _edge_lookup(u, v)
        e2: int = _edge_lookup(v, w)
        e3: int = _edge_lookup(w, u)

        s1: float = 1.0 if e1 >= 0 else -1.0
        s2: float = 1.0 if e2 >= 0 else -1.0
        s3: float = 1.0 if e3 >= 0 else -1.0

        idx1: int = e1 if e1 >= 0 else -e1 - 1
        idx2: int = e2 if e2 >= 0 else -e2 - 1
        idx3: int = e3 if e3 >= 0 else -e3 - 1

        row_b2.extend([f_idx, f_idx, f_idx])
        col_b2.extend([idx1, idx2, idx3])
        data_b2.extend([s1, s2, s3])

    b2: sp.csr_matrix = sp.csr_matrix((data_b2, (row_b2, col_b2)), shape=(n_faces, n_edges))
    return b1, b2


@jax.jit
def _hodge_decomposition(
        b1: JAXSparse,
        b2: JAXSparse,
        flow: JAXArray,
) -> tuple[JAXArray, JAXArray, JAXArray]:
    b1_dense: JAXArray = b1.todense()
    b2_dense: JAXArray = b2.todense()

    delta_0: JAXArray = b1_dense.T @ b1_dense
    delta_1: JAXArray = (b1_dense @ b1_dense.T) + (b2_dense.T @ b2_dense)

    alpha_potential: JAXArray = jnp.linalg.pinv(delta_0) @ b1_dense.T @ flow
    gradient_flow: JAXArray = b1_dense @ alpha_potential

    beta_potential: JAXArray = jnp.linalg.pinv(b2_dense @ b2_dense.T) @ b2_dense @ flow
    curl_flow: JAXArray = b2_dense.T @ beta_potential

    harmonic_flow: JAXArray = flow - gradient_flow - curl_flow
    return gradient_flow, harmonic_flow, curl_flow


@jax.jit
def _ricci_flow_step(
        metric_flow: JAXArray,
        curvature: JAXArray,
        dt: float,
) -> JAXArray:
    return metric_flow - 2.0 * curvature * dt


@jax.jit
def _evolve_ricci(
        initial_flow: JAXArray,
        curl_component: JAXArray,
        steps: int,
        dt: float,
) -> JAXArray:
    def body_fn(i: int, val: JAXArray) -> JAXArray:
        curvature_energy: JAXArray = jnp.tanh(val * curl_component)
        return _ricci_flow_step(val, curvature_energy, dt)

    return jax.lax.fori_loop(0, steps, body_fn, initial_flow)


class HodgeLaplacianJAX:
    def __init__(self, config: HodgeConfig) -> None:
        self._config: HodgeConfig = config
        self._dtype_topo: jnp.dtype = jnp.bfloat16 if config.use_bfloat16_topology else jnp.float32
        self._dtype_exec: jnp.dtype = jnp.float64

    def extract_arbitrage_manifold(
            self,
            n_nodes: int,
            edges: Sequence[EdgeIndices],
            faces: Sequence[FaceIndices],
            observed_flow: jnp.ndarray,
    ) -> CoExactFlowResult:
        b1_sp, b2_sp = _build_incidence_matrices(n_nodes, edges, faces)

        b1_jax: JAXSparse = jsparse.BCOO.from_scipy_sparse(b1_sp).astype(self._dtype_topo)
        b2_jax: JAXSparse = jsparse.BCOO.from_scipy_sparse(b2_sp).astype(self._dtype_topo)
        flow_jax: JAXArray = jnp.array(observed_flow, dtype=self._dtype_topo)

        grad_f, harm_f, curl_f = _hodge_decomposition(b1_jax, b2_jax, flow_jax)

        grad_64: JAXArray = grad_f.astype(self._dtype_exec)
        harm_64: JAXArray = harm_f.astype(self._dtype_exec)
        curl_64: JAXArray = curl_f.astype(self._dtype_exec)
        flow_64: JAXArray = flow_jax.astype(self._dtype_exec)

        evolved_flow: JAXArray = _evolve_ricci(
            flow_64,
            curl_64,
            self._config.ricci_flow_steps,
            self._config.ricci_dt,
        )

        energy: float = float(jnp.sum(jnp.square(curl_64)))

        return CoExactFlowResult(
            gradient_flow=grad_64,
            harmonic_flow=harm_64,
            curl_flow=curl_64,
            ricci_evolved_flow=evolved_flow,
            total_arbitrage_energy=energy,
        )