"""Projected view-factor operator and the gray diffuse radiosity reflection series.

With raw view factors F_ij = n_ij / R_i and patch areas A, the area-weighted
exchange matrix is replaced by its symmetric part plus a small diagonal
support term,

    S = 0.5 (diag(A) F + F^T diag(A)) + delta diag(A),    delta = 1e-15,

which imposes reciprocity exactly.  One positive diagonal factor d, applied
on both sides, then restores row closure, d * (S d) = A, and the projected
view factors are F_p = diag(1/A) diag(d) S diag(d).  Rows of F_p sum to one
and A_i F_p,ij = d_i S_ij d_j is symmetric.  F_p is never stored: it is
applied through the integer counts as F_p x = (d / A) S(d x) and
F_p^T x = d S(d x / A).

Diffuse reflections between gray surfaces are summed by the reflection series
of the radiosity balance, J = E + (1 - eps) F_p^T J, so that no reflection
cutoff is imposed beyond the stated relative tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import NoConvergence, anderson

from .constants import SIGMA

SUPPORT_DIAGONAL_FRACTION = 1.0e-15
CLOSURE_TOLERANCE = 1.0e-10
_TINY = np.finfo(np.float64).tiny
_LOG_MAX = float(np.log(np.finfo(np.float64).max))
_LOG_MIN = float(np.log(np.nextafter(0.0, 1.0)))


class ExchangeOperator:
    """Projected view-factor matrix F_p built from integer first-hit counts.

    ``origin``, ``receiver`` and ``count`` hold the counts n_ij and
    ``rays_per_origin`` the rays R_i; every row must account for all of its
    rays.  The diagonal factor d is found by Anderson acceleration on
    y = log d, with a damped multiplicative iteration as fallback, and the
    projected rows must close to within 1e-10.
    """

    def __init__(
        self,
        origin: np.ndarray,
        receiver: np.ndarray,
        count: np.ndarray,
        rays_per_origin: np.ndarray,
        patch_area_m2: np.ndarray,
        *,
        support_diagonal_fraction: float = SUPPORT_DIAGONAL_FRACTION,
        balance_tolerance: float = 1.0e-11,
        event_chunk_size: int = 4_000_000,
    ) -> None:
        rays = np.asarray(rays_per_origin, dtype=np.int64)
        n = int(rays.size)
        pair = np.asarray(origin, dtype=np.int64) * n + np.asarray(receiver, dtype=np.int64)
        order = np.argsort(pair, kind="stable")
        pair = pair[order]
        start = np.r_[0, np.flatnonzero(pair[1:] != pair[:-1]) + 1]
        count = np.add.reduceat(np.asarray(count, dtype=np.int64)[order], start)
        self.origin = (pair[start] // n).astype(np.int32)
        self.receiver = (pair[start] % n).astype(np.int32)
        observed = np.zeros(n, dtype=np.int64)
        np.add.at(observed, self.origin, count)
        if not np.array_equal(observed, rays):
            raise ValueError("first-hit counts do not account for every emitted ray")
        self.weight = count.astype(np.float64) / rays[self.origin]
        self.patch_area_m2 = np.asarray(patch_area_m2, dtype=np.float64).copy()
        self.support_diagonal_fraction = float(support_diagonal_fraction)
        self.event_chunk_size = int(event_chunk_size)
        self.n_patches = n
        self.scale, self.balance_method, self.balance_iterations = self._balance(float(balance_tolerance))
        self.row_sum_error = float(np.max(np.abs(self.matvec(np.ones(n)) - 1.0)))
        if self.row_sum_error > CLOSURE_TOLERANCE:
            raise RuntimeError(f"projected rows do not close: {self.row_sum_error:.3e}")

    def raw_matvec(self, x: np.ndarray) -> np.ndarray:
        """Raw row-normalised view factors applied to x, F x."""

        result = np.zeros(self.n_patches, dtype=np.float64)
        for start in range(0, self.weight.size, self.event_chunk_size):
            stop = min(start + self.event_chunk_size, self.weight.size)
            result += np.bincount(
                self.origin[start:stop],
                weights=self.weight[start:stop] * x[self.receiver[start:stop]],
                minlength=self.n_patches,
            )
        return result

    def raw_rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Transpose of the raw view factors applied to x, F^T x."""

        result = np.zeros(self.n_patches, dtype=np.float64)
        for start in range(0, self.weight.size, self.event_chunk_size):
            stop = min(start + self.event_chunk_size, self.weight.size)
            result += np.bincount(
                self.receiver[start:stop],
                weights=self.weight[start:stop] * x[self.origin[start:stop]],
                minlength=self.n_patches,
            )
        return result

    def symmetric_matvec(self, x: np.ndarray) -> np.ndarray:
        """Symmetrised area-weighted exchange matrix applied to x, S x."""

        area = self.patch_area_m2
        return 0.5 * (area * self.raw_matvec(x) + self.raw_rmatvec(area * x)) + self.support_diagonal_fraction * area * x

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Projected view factors applied to x, F_p x."""

        return (self.scale / self.patch_area_m2) * self.symmetric_matvec(self.scale * x)

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Transpose of the projected view factors applied to x, F_p^T x."""

        return self.scale * self.symmetric_matvec(self.scale * (x / self.patch_area_m2))

    def reciprocity_error(self) -> float:
        """Largest relative defect of A_i F_p,ij = A_j F_p,ji over all sampled pairs."""

        n, d, area = self.n_patches, self.scale, self.patch_area_m2
        pair = self.origin.astype(np.int64) * n + self.receiver
        reverse = self.receiver.astype(np.int64) * n + self.origin
        index = np.minimum(np.searchsorted(pair, reverse), pair.size - 1)
        w_reverse = np.where(pair[index] == reverse, self.weight[index], 0.0)
        i, j = self.origin, self.receiver
        s_ij = 0.5 * (area[i] * self.weight + area[j] * w_reverse)
        s_ji = 0.5 * (area[j] * w_reverse + area[i] * self.weight)
        diagonal = i == j
        s_ij[diagonal] += self.support_diagonal_fraction * area[i[diagonal]]
        s_ji[diagonal] += self.support_diagonal_fraction * area[j[diagonal]]
        joint_ij = area[i] * ((d[i] / area[i]) * s_ij * d[j])
        joint_ji = area[j] * ((d[j] / area[j]) * s_ji * d[i])
        return float(np.max(np.abs(joint_ij - joint_ji)) / max(float(np.max(np.abs(joint_ij))), _TINY))

    def _closure_error(self, y: np.ndarray) -> float:
        """max |d (S d) / A - 1| for d = exp(y)."""

        if not (np.all(np.isfinite(y)) and np.max(y) < _LOG_MAX and np.min(y) > _LOG_MIN):
            return float("inf")
        scale = np.exp(y)
        row = scale * self.symmetric_matvec(scale)
        if not np.all(np.isfinite(row)) or np.any(row <= 0.0):
            return float("inf")
        return float(np.max(np.abs(row / self.patch_area_m2 - 1.0)))

    def _balance(self, tolerance: float) -> tuple[np.ndarray, str, int]:
        """Solve d * (S d) = A for the diagonal factor d > 0."""

        n, log_area = self.n_patches, np.log(self.patch_area_m2)
        y0 = np.zeros(n, dtype=np.float64)
        if self._closure_error(y0) <= tolerance:
            return np.exp(y0), "none", 0
        iterations = 0

        def log_residual(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=np.float64)
            if not (np.all(np.isfinite(y)) and np.max(y) < _LOG_MAX and np.min(y) > _LOG_MIN):
                return np.full(n, np.inf)
            product = self.symmetric_matvec(np.exp(y))
            if not np.all(np.isfinite(product)) or np.any(product <= 0.0):
                return np.full(n, np.inf)
            return y + np.log(product) - log_area

        def count_iteration(_x: np.ndarray, _f: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        try:
            y = np.asarray(
                anderson(log_residual, y0, M=10, w0=0.01, maxiter=500, f_tol=tolerance, callback=count_iteration),
                dtype=np.float64,
            )
            if self._closure_error(y) <= min(max(10.0 * tolerance, 10.0 * np.finfo(float).eps), CLOSURE_TOLERANCE):
                return np.exp(y), "anderson", iterations
        except (NoConvergence, ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass

        y = y0.copy()
        for iteration in range(1, 20_001):
            scale = np.exp(y)
            row = scale * self.symmetric_matvec(scale)
            if not np.all(np.isfinite(row)) or np.any(row <= 0.0):
                break
            y = y + 0.5 * (log_area - np.log(row))
            error = self._closure_error(y)
            if not np.isfinite(error):
                break
            if error <= min(tolerance, CLOSURE_TOLERANCE):
                return np.exp(y), "multiplicative", iteration
        raise RuntimeError("diagonal balancing of the exchange matrix did not converge")


@dataclass
class RadiosityResult:
    """Emitted, absorbed and net (absorbed minus emitted) patch power."""

    emitted_power_W: np.ndarray
    absorbed_power_W: np.ndarray
    net_power_W: np.ndarray
    residual_power_W: float
    iterations: int

    @property
    def global_energy_residual_W(self) -> float:
        """Net power of all patches plus the power still in flight."""

        return float(np.sum(self.net_power_W) + self.residual_power_W)


def reflection_series(
    operator: ExchangeOperator,
    emitted_W: np.ndarray,
    emissivity: np.ndarray,
    *,
    relative_tolerance: float,
    max_reflections: int = 10000,
) -> RadiosityResult:
    """Absorbed power after all diffuse reflections of a (possibly signed) emission.

    Power in flight is passed to the receiving patches by F_p^T; each patch
    absorbs the fraction eps and reflects 1 - eps.  The series stops when the
    L1 norm of the power in flight falls below ``relative_tolerance`` times
    the L1 norm of the emission.
    """

    emitted = np.asarray(emitted_W, dtype=np.float64)
    absorbed = np.zeros_like(emitted)
    flight = emitted.copy()
    initial = max(float(np.sum(np.abs(emitted))), _TINY)
    for iteration in range(1, int(max_reflections) + 1):
        incident = operator.rmatvec(flight)
        absorbed += emissivity * incident
        flight = (1.0 - emissivity) * incident
        if float(np.sum(np.abs(flight))) <= float(relative_tolerance) * initial:
            break
    else:
        raise RuntimeError("radiosity reflection series did not converge")
    return RadiosityResult(emitted, absorbed, absorbed - emitted, float(np.sum(flight)), iteration)


def net_patch_power(
    operator: ExchangeOperator,
    temperature_K: np.ndarray,
    emissivity: np.ndarray,
    area_m2: np.ndarray,
    *,
    relative_tolerance: float,
    reference_temperature_K: float | None = None,
) -> RadiosityResult:
    """Net radiative power of every patch for given patch temperatures.

    With ``reference_temperature_K`` the isothermal state at that temperature,
    which carries zero net power, is subtracted before the series is summed:
    only the signed emission eps sigma A (T^4 - T_ref^4) is propagated, which
    is exact because the radiosity map is linear in the emission.
    """

    T = np.asarray(temperature_K, dtype=np.float64)
    if reference_temperature_K is None:
        emitted = emissivity * SIGMA * area_m2 * T**4
    else:
        ref = float(reference_temperature_K)
        emitted = emissivity * SIGMA * area_m2 * ((T - ref) * (T + ref) * (T * T + ref * ref))
    return reflection_series(operator, emitted, emissivity, relative_tolerance=relative_tolerance)


def net_patch_power_jvp(
    operator: ExchangeOperator,
    temperature_K: np.ndarray,
    delta_temperature_K: np.ndarray,
    emissivity: np.ndarray,
    area_m2: np.ndarray,
    *,
    relative_tolerance: float,
) -> np.ndarray:
    """Derivative of the net patch power along a patch temperature perturbation."""

    derivative = 4.0 * emissivity * SIGMA * area_m2 * temperature_K**3 * delta_temperature_K
    response = reflection_series(operator, derivative, emissivity, relative_tolerance=relative_tolerance)
    return response.absorbed_power_W - derivative
