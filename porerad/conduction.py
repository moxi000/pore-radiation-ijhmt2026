"""Cell-centred finite-volume heat conduction on a digital microstructure.

Every cell carries the conductivity of its phase.  Neighbouring cells exchange
heat through the harmonic-mean face conductance k_f = 2 k_1 k_2 / (k_1 + k_2)
times dx^2 / dx, the lateral faces y and z are periodic, and a cell next to a
wall is connected to the fixed wall temperature through the half-cell
conductance 2 k dx.  The steady balance of the cells reads

    A T = b(T_h, T_c) + Q,

with Q the net heat (W) delivered to each cell by other mechanisms, here the
net radiative power of the surface patches the cell owns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, cg


@dataclass
class ConductionSystem:
    """Conduction matrix A and the wall responses of an x-oriented cell grid."""

    matrix: csr_matrix
    hot_rhs: np.ndarray
    cold_rhs: np.ndarray
    conductivity: np.ndarray
    H_m: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(map(int, self.conductivity.shape))

    def boundary_rhs(self, T_hot_K: float, T_cold_K: float) -> np.ndarray:
        """Right-hand side b contributed by the two fixed wall temperatures."""

        return self.hot_rhs * float(T_hot_K) + self.cold_rhs * float(T_cold_K)

    def wall_fluxes(self, temperature_K: np.ndarray, T_hot_K: float, T_cold_K: float) -> tuple[float, float]:
        """Conductive heat flux (W m^-2) into the slab at the hot wall and out at the cold wall."""

        k, T = self.conductivity, np.asarray(temperature_K, dtype=np.float64).reshape(self.shape)
        dx = self.H_m / k.shape[0]
        q_hot = np.mean(2.0 * k[0, :, :] * (float(T_hot_K) - T[0, :, :]) / dx)
        q_cold = np.mean(2.0 * k[-1, :, :] * (T[-1, :, :] - float(T_cold_K)) / dx)
        return float(q_hot), float(q_cold)


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b + 1.0e-30)


def assemble_conduction(conductivity_x: np.ndarray, H_m: float) -> ConductionSystem:
    """Assemble A and the wall responses for a conductivity field oriented along x."""

    k = np.ascontiguousarray(conductivity_x, dtype=np.float64)
    nx, ny, nz = map(int, k.shape)
    dx = float(H_m) / nx
    area = dx * dx
    cell = np.arange(nx * ny * nz, dtype=np.int32).reshape((nx, ny, nz))
    diagonal = np.zeros(cell.size, dtype=np.float64)
    rows, columns, values = [], [], []

    def add_faces(low_cells, high_cells, low_k, high_k) -> None:
        low, high = low_cells.reshape(-1), high_cells.reshape(-1)
        conductance = np.ascontiguousarray(_harmonic(low_k, high_k) * area / dx).reshape(-1)
        diagonal[low] += conductance
        diagonal[high] += conductance
        rows.extend((low, high))
        columns.extend((high, low))
        values.extend((-conductance, -conductance))

    add_faces(cell[:-1], cell[1:], k[:-1], k[1:])
    add_faces(cell[:, :-1], cell[:, 1:], k[:, :-1], k[:, 1:])
    add_faces(cell[:, -1], cell[:, 0], k[:, -1], k[:, 0])
    add_faces(cell[:, :, :-1], cell[:, :, 1:], k[:, :, :-1], k[:, :, 1:])
    add_faces(cell[:, :, -1], cell[:, :, 0], k[:, :, -1], k[:, :, 0])

    face_scale = dx * dx / dx
    hot = np.zeros(cell.size, dtype=np.float64)
    cold = np.zeros_like(hot)
    hot.reshape(k.shape)[0] = 2.0 * k[0] * face_scale
    cold.reshape(k.shape)[-1] = 2.0 * k[-1] * face_scale
    diagonal += hot + cold
    rows.append(cell.reshape(-1))
    columns.append(cell.reshape(-1))
    values.append(diagonal)
    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(cell.size, cell.size)
    ).tocsc().tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return ConductionSystem(matrix, hot, cold, k, float(H_m))


def solve_conduction(
    system: ConductionSystem, T_hot_K: float, T_cold_K: float, *, rtol: float = 1.0e-11
) -> np.ndarray:
    """Pure-conduction field by Jacobi-preconditioned conjugate gradients.

    The iteration starts from the linear profile between the wall
    temperatures and stops at a relative residual of ``rtol``.
    """

    A = system.matrix
    rhs = system.boundary_rhs(T_hot_K, T_cold_K)
    nx, ny, nz = system.shape
    line = float(T_hot_K) + (float(T_cold_K) - float(T_hot_K)) * ((np.arange(nx, dtype=np.float64) + 0.5) / nx)
    x0 = np.broadcast_to(line[:, None, None], (nx, ny, nz)).reshape(-1).copy()
    inverse_diagonal = 1.0 / A.diagonal()
    jacobi = LinearOperator(A.shape, matvec=lambda v: inverse_diagonal * v, dtype=np.float64)
    T, info = cg(A, rhs, x0=x0, rtol=float(rtol), atol=0.0, maxiter=max(50000, 20 * nx), M=jacobi)
    relative = float(np.linalg.norm(A @ T - rhs) / np.linalg.norm(rhs))
    if info != 0 or relative > 1.0e-9:
        raise RuntimeError(f"conduction solve did not converge: info={info}, residual={relative:.3e}")
    return T.reshape(system.shape)
