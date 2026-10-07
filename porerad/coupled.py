"""Prescribed-field radiation channels and the strictly coupled radiation-conduction solve.

The net radiative power of every internal patch is assigned to the solid cell
that owns it.  Under a prescribed temperature field this gives the wall and
internal radiation channels q_h, q_c and q_i directly.  In the strictly coupled
problem the internal temperature is unknown and the cell balance

    R(T) = A T - b - Q_rad(T) = 0

is solved by a Newton-Krylov method: each Newton correction comes from
flexible GMRES applied to the Jacobian-vector product, right-preconditioned by
a fixed number of Jacobi-preconditioned conjugate-gradient steps on the
conduction matrix plus the local emission derivative, and is accepted by a
backtracking line search that keeps the field within the wall temperatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conduction import ConductionSystem
from .constants import SIGMA
from .exchange import ExchangeOperator, RadiosityResult, net_patch_power, net_patch_power_jvp
from .view_factors import COLD, HOT, INTERNAL, SurfacePatches

_TINY = np.finfo(np.float64).tiny

NEWTON_MAX_ITERATIONS = 8
NEWTON_RESIDUAL_W = 1.0e-13
NEWTON_STEP_K = 1.0e-10
GMRES_RESTART = 30
GMRES_MAX_ITERATIONS = 90
GMRES_RTOL = 1.0e-8
GMRES_ATOL = 1.0e-13
RADIOSITY_RTOL_NEWTON = 1.0e-8
RADIOSITY_RTOL_FINAL = 1.0e-13
PCG_STEPS = (10, 20)
"""Preconditioner steps of the first attempt and of the restart from the same initial field."""


# Patch to cell mapping

def patch_temperature(patches: SurfacePatches, T_cells: np.ndarray, T_hot_K: float, T_cold_K: float) -> np.ndarray:
    """Wall temperatures on wall patches and the owning-cell temperature on internal patches."""

    T = np.empty(patches.n_patches, dtype=np.float64)
    internal = patches.patch_class == INTERNAL
    T[patches.patch_class == HOT] = float(T_hot_K)
    T[patches.patch_class == COLD] = float(T_cold_K)
    T[internal] = np.asarray(T_cells, dtype=np.float64).ravel()[patches.patch_cell[internal]]
    return T


def cell_power(patches: SurfacePatches, patch_power_W: np.ndarray) -> np.ndarray:
    """Sum the power of the internal patches into their owning solid cells."""

    internal = patches.patch_class == INTERNAL
    power = np.zeros(patches.solid.size, dtype=np.float64)
    np.add.at(power, patches.patch_cell[internal], patch_power_W[internal])
    return power.reshape(patches.solid.shape)


def emission_diagonal(patches: SurfacePatches, T_cells: np.ndarray, epsilon: float) -> np.ndarray:
    """Local emission derivative sum_p 4 eps sigma A_p T^3 of every cell (W/K)."""

    internal = patches.patch_class == INTERNAL
    owner = patches.patch_cell[internal].astype(np.int64)
    T = np.asarray(T_cells, dtype=np.float64).ravel()[owner]
    diagonal = np.zeros(patches.solid.size, dtype=np.float64)
    np.add.at(diagonal, owner, 4.0 * float(epsilon) * SIGMA * patches.patch_area_m2[internal] * T**3)
    return diagonal


@dataclass
class RadiationState:
    """Patch temperatures, net patch power and net cell power of one temperature field."""

    patch_temperature_K: np.ndarray
    patch_net_power_W: np.ndarray
    cell_net_power_W: np.ndarray
    radiosity: RadiosityResult


def evaluate_radiation(
    operator: ExchangeOperator,
    patches: SurfacePatches,
    T_cells: np.ndarray,
    T_hot_K: float,
    T_cold_K: float,
    epsilon: float,
    *,
    relative_tolerance: float,
    reference_temperature_K: float | None = None,
) -> RadiationState:
    """Net radiative power of patches and cells; all surfaces share the emissivity eps."""

    T_patch = patch_temperature(patches, T_cells, T_hot_K, T_cold_K)
    emissivity = np.full(patches.n_patches, float(epsilon))
    radiosity = net_patch_power(
        operator, T_patch, emissivity, patches.patch_area_m2,
        relative_tolerance=relative_tolerance, reference_temperature_K=reference_temperature_K,
    )
    return RadiationState(T_patch, radiosity.net_power_W, cell_power(patches, radiosity.net_power_W), radiosity)


def radiation_jvp(
    operator: ExchangeOperator,
    patches: SurfacePatches,
    T_patch: np.ndarray,
    dT_cells: np.ndarray,
    epsilon: float,
    *,
    relative_tolerance: float,
) -> np.ndarray:
    """Change of the net cell power for a cell temperature perturbation (walls fixed)."""

    internal = patches.patch_class == INTERNAL
    dT_patch = np.zeros(patches.n_patches, dtype=np.float64)
    dT_patch[internal] = np.asarray(dT_cells, dtype=np.float64).ravel()[patches.patch_cell[internal]]
    emissivity = np.full(patches.n_patches, float(epsilon))
    patch_jvp = net_patch_power_jvp(
        operator, T_patch, dT_patch, emissivity, patches.patch_area_m2, relative_tolerance=relative_tolerance
    )
    return cell_power(patches, patch_jvp)


def cross_section_area(patches: SurfacePatches) -> float:
    """Projected wall area N_y N_z (H/N_x)^2."""

    nx, ny, nz = patches.solid.shape
    return float(ny * nz) * (patches.H_m / float(nx)) ** 2


# Prescribed temperature field

def linear_cell_temperature(shape: tuple[int, int, int], T_mean_K: float, delta_T_K: float) -> np.ndarray:
    """Linear field T_h + (T_c - T_h) s/H evaluated at the cell centres."""

    nx = int(shape[0])
    axial = (T_mean_K + delta_T_K / 2.0) - (np.arange(nx, dtype=np.float64) + 0.5) * delta_T_K / nx
    return np.broadcast_to(axial[:, None, None], tuple(int(n) for n in shape)).copy()


def coarse_source(cell_power_W: np.ndarray, bins: int = 8) -> np.ndarray:
    """Cell power summed over a bins^3 grid of equal blocks."""

    array = np.asarray(cell_power_W, dtype=np.float64)
    parts = [np.array_split(np.arange(n), min(bins, n)) for n in array.shape]
    return np.asarray(
        [float(np.sum(array[np.ix_(xs, ys, zs)])) for xs in parts[0] for ys in parts[1] for zs in parts[2]]
    ).reshape(tuple(len(p) for p in parts))


@dataclass
class PrescribedChannels:
    """Radiation channels (W m^-2) under the prescribed linear field, q_h + q_c + q_i = 0."""

    T_mean_K: float
    delta_T_K: float
    epsilon: float
    q_h: float
    q_c: float
    q_i: float
    closure_rel: float
    coarse_source: np.ndarray = field(repr=False)


def prescribed_field_channels(
    operator: ExchangeOperator, patches: SurfacePatches, T_mean_K: float, delta_T_K: float, epsilon: float
) -> PrescribedChannels:
    """Net power leaving the hot wall, the cold wall and the internal surfaces.

    Walls are held at T_m +/- dT/2 and every internal surface takes the
    linear field at its owning cell.  One radiosity evaluation to a relative
    tolerance of 1e-13 is made with the isothermal state at T_m subtracted.
    """

    state = evaluate_radiation(
        operator, patches, linear_cell_temperature(patches.solid.shape, T_mean_K, delta_T_K),
        T_mean_K + delta_T_K / 2.0, T_mean_K - delta_T_K / 2.0, epsilon,
        relative_tolerance=RADIOSITY_RTOL_FINAL, reference_temperature_K=T_mean_K,
    )
    area = cross_section_area(patches)
    net = state.patch_net_power_W
    q_h = -float(np.sum(net[patches.patch_class == HOT])) / area
    q_c = -float(np.sum(net[patches.patch_class == COLD])) / area
    q_i = -float(np.sum(net[patches.patch_class == INTERNAL])) / area
    closure = abs(q_h + q_c + q_i) / max(abs(q_h), abs(q_c), abs(q_i), 1.0e-30)
    return PrescribedChannels(
        float(T_mean_K), float(delta_T_K), float(epsilon), q_h, q_c, q_i, closure,
        coarse_source(state.cell_net_power_W) / area,
    )


def relative_difference(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), _TINY)


def ab_differences(set_a: list[PrescribedChannels], set_b: list[PrescribedChannels]) -> tuple[float, float]:
    """Largest A/B relative difference of the wall channels and of the 8^3 coarse source (L1)."""

    channel = max(
        relative_difference(getattr(a, name), getattr(b, name))
        for a, b in zip(set_a, set_b) for name in ("q_h", "q_c")
    )
    source = max(
        float(np.sum(np.abs(a.coarse_source - b.coarse_source)))
        / max(float(np.sum(np.abs(a.coarse_source))), float(np.sum(np.abs(b.coarse_source))), _TINY)
        for a, b in zip(set_a, set_b)
    )
    return channel, source


def append_required(channel_difference: float, source_difference: float) -> bool:
    """One symmetric append of a further ray block to each set when the sets disagree."""

    return channel_difference > 0.01 or source_difference > 0.05


# Strictly coupled solve

@dataclass
class GMRESResult:
    solution: np.ndarray
    converged: bool
    iterations: int
    true_residual_relative: float


def flexible_gmres(matvec, rhs: np.ndarray, preconditioner, *, restart: int, max_iterations: int,
                   relative_tolerance: float, absolute_tolerance: float) -> GMRESResult:
    """Right-preconditioned flexible GMRES with modified Gram-Schmidt and one reorthogonalisation.

    The preconditioned basis vectors are stored because the preconditioner
    may change between applications.  A small least-squares residual is
    confirmed with the true residual before the solution is accepted.
    """

    b = np.asarray(rhs, dtype=np.float64)
    n = b.size
    x = np.zeros_like(b)
    b_norm = max(float(np.linalg.norm(b)), _TINY)
    threshold = max(float(absolute_tolerance), float(relative_tolerance) * b_norm)
    eps = np.finfo(np.float64).eps
    total = 0
    while total < max_iterations:
        residual = b - matvec(x)
        beta = float(np.linalg.norm(residual))
        if beta <= threshold:
            return GMRESResult(x, True, total, beta / b_norm)
        width = min(restart, max_iterations - total)
        basis = np.zeros((width + 1, n))
        preconditioned = np.zeros((width, n))
        hessenberg = np.zeros((width + 1, width))
        projected_rhs = np.zeros(width + 1)
        basis[0] = residual / beta
        projected_rhs[0] = beta
        restart_from = None
        for column in range(width):
            preconditioned[column] = preconditioner(basis[column])
            work = matvec(preconditioned[column])
            for _ in range(2):
                for row in range(column + 1):
                    coefficient = float(np.dot(basis[row], work))
                    hessenberg[row, column] += coefficient
                    work -= coefficient * basis[row]
            next_norm = float(np.linalg.norm(work))
            hessenberg[column + 1, column] = next_norm
            if next_norm > eps * beta:
                basis[column + 1] = work / next_norm
            active = hessenberg[: column + 2, : column + 1]
            coefficients = np.linalg.lstsq(active, projected_rhs[: column + 2], rcond=None)[0]
            estimate = float(np.linalg.norm(projected_rhs[: column + 2] - active @ coefficients))
            total += 1
            if estimate <= threshold or next_norm <= eps * beta:
                candidate = x + coefficients @ preconditioned[: column + 1]
                true_norm = float(np.linalg.norm(b - matvec(candidate)))
                if true_norm <= threshold:
                    return GMRESResult(candidate, True, total, true_norm / b_norm)
                restart_from = candidate
                break
        if restart_from is not None:
            x = restart_from
            continue
        x += np.linalg.lstsq(hessenberg, projected_rhs, rcond=None)[0] @ preconditioned
    true_norm = float(np.linalg.norm(b - matvec(x)))
    return GMRESResult(x, true_norm <= threshold, total, true_norm / b_norm)


class FixedJacobiPCG:
    """A fixed number of Jacobi-preconditioned CG steps on P = A + diag(shift), from x = 0."""

    def __init__(self, system: ConductionSystem, shift: np.ndarray, steps: int) -> None:
        self.matrix, self.shift, self.steps = system.matrix, shift, int(steps)
        self.inverse_diagonal = 1.0 / (system.matrix.diagonal() + shift)

    def __call__(self, rhs: np.ndarray) -> np.ndarray:
        x = np.zeros_like(rhs)
        r = rhs.copy()
        z = self.inverse_diagonal * r
        p = z.copy()
        rz = r @ z
        for step in range(self.steps):
            Ap = self.matrix @ p + self.shift * p
            alpha = rz / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            z = self.inverse_diagonal * r
            if step + 1 < self.steps:
                rz_new = r @ z
                p = z + (rz_new / rz) * p
                rz = rz_new
        return x


@dataclass
class CoupledResult:
    """Converged coupled field, its radiation state and the iteration record."""

    temperature_K: np.ndarray
    state: RadiationState
    converged: bool
    newton_iterations: int
    linear_solves: list[dict] = field(default_factory=list)
    final_residual_inf_W: float = float("nan")
    preconditioner_steps: int = 0
    failure: str = ""

    @property
    def linear_residuals_pass(self) -> bool:
        """Every linear solve met ||r||_2 <= max(1e-13, 1e-8 ||b||_2)."""

        return bool(self.linear_solves) and all(
            s["converged"] and s["true_residual_l2_W"] <= max(1.0e-13, 1.0e-8 * s["rhs_l2_W"])
            for s in self.linear_solves
        )


def newton_krylov(
    system: ConductionSystem,
    operator: ExchangeOperator,
    patches: SurfacePatches,
    T_hot_K: float,
    T_cold_K: float,
    epsilon: float,
    T_initial: np.ndarray,
    preconditioner_steps: int,
) -> CoupledResult:
    """Newton-Krylov solve of A T - b - Q_rad(T) = 0 from the pure-conduction field.

    Converged when ||R||_inf <= 1e-13 W and the last update is at most
    1e-10 K.  Radiosity is summed to 1e-8 inside the iteration and to 1e-13
    for the returned state.  Trial fields outside [T_c, T_h] are rejected by
    the line search; no clipping is applied.
    """

    A = system.matrix
    b = system.boundary_rhs(T_hot_K, T_cold_K)
    lower, upper = float(T_cold_K), float(T_hot_K)
    pcg = FixedJacobiPCG(system, emission_diagonal(patches, T_initial, epsilon), preconditioner_steps)

    def radiation(T: np.ndarray, rtol: float) -> RadiationState:
        return evaluate_radiation(operator, patches, T, T_hot_K, T_cold_K, epsilon, relative_tolerance=rtol)

    T = np.asarray(T_initial, dtype=np.float64).copy()
    state = radiation(T, RADIOSITY_RTOL_NEWTON)
    residual = A @ T.ravel() - b - state.cell_net_power_W.ravel()
    target = NEWTON_RESIDUAL_W
    last_step = 0.0 if max(float(np.linalg.norm(residual, np.inf)), _TINY) <= target else float("inf")
    result = CoupledResult(T, state, False, 0, preconditioner_steps=int(preconditioner_steps))

    def finish(converged: bool, failure: str = "") -> CoupledResult:
        final = radiation(T, RADIOSITY_RTOL_FINAL)
        result.temperature_K, result.state, result.converged, result.failure = T, final, converged, failure
        result.final_residual_inf_W = float(np.linalg.norm(A @ T.ravel() - b - final.cell_net_power_W.ravel(), np.inf))
        return result

    for _ in range(NEWTON_MAX_ITERATIONS):
        residual_norm = float(np.linalg.norm(residual, np.inf))
        if residual_norm <= target and last_step <= NEWTON_STEP_K:
            return finish(True)
        T_patch = state.patch_temperature_K.copy()

        def jacobian(v: np.ndarray) -> np.ndarray:
            jvp = radiation_jvp(operator, patches, T_patch, v.reshape(system.shape), epsilon,
                                relative_tolerance=RADIOSITY_RTOL_NEWTON)
            return A @ v - jvp.ravel()

        gmres = flexible_gmres(jacobian, -residual, pcg, restart=GMRES_RESTART, max_iterations=GMRES_MAX_ITERATIONS,
                               relative_tolerance=GMRES_RTOL, absolute_tolerance=GMRES_ATOL)
        step = gmres.solution
        result.linear_solves.append({
            "converged": gmres.converged,
            "gmres_iterations": gmres.iterations,
            "rhs_l2_W": float(np.linalg.norm(residual)),
            "true_residual_l2_W": float(np.linalg.norm(jacobian(step) + residual)),
            "true_residual_relative": gmres.true_residual_relative,
        })
        if not gmres.converged:
            return finish(False, "linear solve did not converge")
        if residual_norm <= target and float(np.max(np.abs(step))) <= NEWTON_STEP_K:
            return finish(True)

        alpha, accepted = 1.0, False
        while alpha >= 2.0**-20:
            candidate = T.ravel() + alpha * step
            if np.all(np.isfinite(candidate)) and float(np.min(candidate)) >= lower and float(np.max(candidate)) <= upper:
                trial = candidate.reshape(system.shape)
                trial_state = radiation(trial, RADIOSITY_RTOL_NEWTON)
                trial_residual = A @ candidate - b - trial_state.cell_net_power_W.ravel()
                trial_norm = float(np.linalg.norm(trial_residual, np.inf))
                if trial_norm <= (1.0 - 1.0e-4 * alpha) * residual_norm:
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            return finish(False, "line search failed")
        last_step = float(np.max(np.abs(trial - T)))
        T, state, residual = trial, trial_state, trial_residual
        result.newton_iterations += 1
        if trial_norm <= target and last_step <= NEWTON_STEP_K:
            return finish(True)
    return finish(False, "maximum Newton iterations reached")


def solve_coupled(
    system: ConductionSystem,
    operator: ExchangeOperator,
    patches: SurfacePatches,
    T_hot_K: float,
    T_cold_K: float,
    epsilon: float,
    T_initial: np.ndarray,
) -> CoupledResult:
    """Coupled solve with 10 preconditioner steps, restarted from the same field with 20 if needed.

    The restart is made when the first attempt does not converge or any of
    its linear solves misses the true-residual bound.
    """

    for steps in PCG_STEPS:
        result = newton_krylov(system, operator, patches, T_hot_K, T_cold_K, epsilon, T_initial, steps)
        if result.converged and result.linear_residuals_pass:
            break
    return result


def wall_flux_metrics(
    system: ConductionSystem, patches: SurfacePatches, result: CoupledResult, T_hot_K: float, T_cold_K: float
) -> dict[str, float]:
    """Conductive, radiative and total wall fluxes (W m^-2) and the acceptance quantities of a coupled solution."""

    q_cond_hot, q_cond_cold = system.wall_fluxes(result.temperature_K, T_hot_K, T_cold_K)
    area = cross_section_area(patches)
    net = result.state.patch_net_power_W
    q_rad_hot = -float(np.sum(net[patches.patch_class == HOT])) / area
    q_rad_cold = float(np.sum(net[patches.patch_class == COLD])) / area
    q_total_hot, q_total_cold = q_cond_hot + q_rad_hot, q_cond_cold + q_rad_cold
    scale = max(abs(q_total_hot), abs(q_total_cold), _TINY)
    T = result.temperature_K
    return {
        "q_cond_hot_W_m2": q_cond_hot,
        "q_rad_hot_W_m2": q_rad_hot,
        "q_total_hot_W_m2": q_total_hot,
        "q_cond_cold_W_m2": q_cond_cold,
        "q_rad_cold_W_m2": q_rad_cold,
        "q_total_cold_W_m2": q_total_cold,
        "radiative_fraction_hot": q_rad_hot / q_total_hot,
        "k_eff_apparent_W_mK": q_total_hot * patches.H_m / (float(T_hot_K) - float(T_cold_K)),
        "wall_flux_mismatch_rel": abs(q_total_hot - q_total_cold) / scale,
        "energy_residual_rel": abs(result.state.radiosity.global_energy_residual_W) / max(scale * area, _TINY),
        "T_min_K": float(np.min(T)),
        "T_max_K": float(np.max(T)),
        "temperature_bound_violations": int(np.count_nonzero((T < float(T_cold_K)) | (T > float(T_hot_K)))),
    }


def acceptance_failures(result: CoupledResult, metrics: dict[str, float]) -> list[str]:
    """Acceptance conditions of a coupled solution that are not met (empty when accepted)."""

    checks = {
        "Newton iteration converged": result.converged,
        "linear residuals within max(1e-13, 1e-8 ||b||)": result.linear_residuals_pass,
        "wall-flux mismatch <= 0.5 %": metrics["wall_flux_mismatch_rel"] <= 0.005,
        "global energy residual <= 1e-3": metrics["energy_residual_rel"] <= 1.0e-3,
        "temperature within the wall temperatures": metrics["temperature_bound_violations"] == 0,
    }
    return [name for name, ok in checks.items() if not ok]
