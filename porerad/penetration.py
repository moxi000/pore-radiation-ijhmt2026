"""Penetration response of a slab of digital microstructure.

Rays enter the slab through the face x = 0 at uniformly distributed transverse
positions with Lambertian directions about the face normal and travel through
the pore space (periodic in the two transverse directions). At a solid surface
a ray is absorbed with probability eps, otherwise it is reflected into a
Lambertian direction about the surface normal and followed further, for at most
1000 reflections. A ray that leaves through x = 0 counts as reflected (R), one
that leaves through x = N as transmitted (T), one absorbed by the skeleton as
absorbed (A). Exchange between opaque diffuse surfaces is scale-free, so R, T
and A depend only on the geometry, the transfer axis and eps. The penetration
response is q_pen = q_pp T, where q_pp is the net flux between gray diffuse
parallel plates at T_h and T_c with the wall emissivity eps.

Two restart rules for a reflected ray are provided:

- "article": the rule behind the values reported in the article. A ray
  reflected for the first time restarts at its hit point; a ray reflected for
  the second or a later time restarts at its hit point advanced once more by
  the length of the preceding path segment along the preceding direction.
- "hit_point": every reflected ray restarts at its hit point.

In both rules the restart point is displaced by 1e-9 cells along the surface
normal. Lengths are in cells. The ray kernel is written in Taichi and runs on
the CPU.
"""

import math

import numpy as np

from .constants import SIGMA
from .properties import seed_average, solid_along_x

SEEDS = (321, 654)
N_RAYS = 500_000
MAX_BOUNCES = 1000
RESTART_RULES = ("article", "hit_point")


def parallel_plate_flux(T_hot: float, T_cold: float, epsilon: float) -> float:
    """Net radiative flux between gray diffuse parallel plates of emissivity eps, W m^-2."""
    eps = float(epsilon)
    return float(SIGMA * (float(T_hot) ** 4 - float(T_cold) ** 4) / (1.0 / eps + 1.0 / eps - 1.0))


def penetration_flux(transmittance: float, epsilon: float, T_mean: float, delta_T: float = 10.0) -> float:
    """q_pen = q_pp T at mean temperature T_mean and wall temperature difference delta_T, W m^-2."""
    T_hot = float(T_mean) + 0.5 * float(delta_T)
    T_cold = float(T_mean) - 0.5 * float(delta_T)
    return float(parallel_plate_flux(T_hot, T_cold, epsilon) * transmittance)


def _build_kernel(ti, hit_point: bool):
    """Compile the ray kernel for one restart rule."""

    @ti.func
    def _hash_u32(x):
        x ^= x >> ti.u32(16)
        x *= ti.u32(0x7FEB352D)
        x ^= x >> ti.u32(15)
        x *= ti.u32(0x846CA68B)
        x ^= x >> ti.u32(16)
        return x

    @ti.func
    def _rand01(state):
        state = _hash_u32(state + ti.u32(0x9E3779B9))
        value = ti.cast(state & ti.u32(0x00FFFFFF), ti.f64) / ti.cast(ti.u32(0x01000000), ti.f64)
        return state, value

    @ti.func
    def _trace(solid_field, ox, oy, oz, dx0, dy0, dz0, sx, sy, sz):
        """Cell-by-cell traversal; fate 0 = solid hit, 1 = exit at x = 0, 2 = exit at x = N."""
        norm = ti.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
        dx = dx0 / norm
        dy = dy0 / norm
        dz = dz0 / norm
        i = ti.cast(ti.floor(ox), ti.i32)
        j = ti.cast(ti.floor(oy), ti.i32)
        k = ti.cast(ti.floor(oz), ti.i32)
        if dx < 0.0 and ti.abs(ox - ti.floor(ox + 0.5)) <= 1.0e-12:
            i -= 1
        if dy < 0.0 and ti.abs(oy - ti.floor(oy + 0.5)) <= 1.0e-12:
            j -= 1
        if dz < 0.0 and ti.abs(oz - ti.floor(oz + 0.5)) <= 1.0e-12:
            k -= 1
        step_x = ti.cast(0, ti.i32)
        step_y = ti.cast(0, ti.i32)
        step_z = ti.cast(0, ti.i32)
        t_max_x = 1.0e300
        t_max_y = 1.0e300
        t_max_z = 1.0e300
        t_delta_x = 1.0e300
        t_delta_y = 1.0e300
        t_delta_z = 1.0e300
        safe_dx = ti.select(ti.abs(dx) > 0.0, dx, 1.0)
        safe_dy = ti.select(ti.abs(dy) > 0.0, dy, 1.0)
        safe_dz = ti.select(ti.abs(dz) > 0.0, dz, 1.0)
        if dx > 0.0:
            step_x = 1
            t_max_x = (ti.cast(i + 1, ti.f64) - ox) / safe_dx
            t_delta_x = 1.0 / safe_dx
        elif dx < 0.0:
            step_x = -1
            t_max_x = (ti.cast(i, ti.f64) - ox) / safe_dx
            t_delta_x = -1.0 / safe_dx
        if dy > 0.0:
            step_y = 1
            t_max_y = (ti.cast(j + 1, ti.f64) - oy) / safe_dy
            t_delta_y = 1.0 / safe_dy
        elif dy < 0.0:
            step_y = -1
            t_max_y = (ti.cast(j, ti.f64) - oy) / safe_dy
            t_delta_y = -1.0 / safe_dy
        if dz > 0.0:
            step_z = 1
            t_max_z = (ti.cast(k + 1, ti.f64) - oz) / safe_dz
            t_delta_z = 1.0 / safe_dz
        elif dz < 0.0:
            step_z = -1
            t_max_z = (ti.cast(k, ti.f64) - oz) / safe_dz
            t_delta_z = -1.0 / safe_dz
        t = 0.0
        nxn = 0.0
        nyn = 0.0
        nzn = 0.0
        fate = ti.cast(-1, ti.i32)
        max_steps = 10 * sx * sy * sz + 10
        steps = 0
        while fate == -1 and steps < max_steps:
            steps += 1
            if i < 0:
                fate = 1
            elif i >= sx:
                fate = 2
            else:
                jj = j % sy
                kk = k % sz
                if solid_field[i, jj, kk] != ti.u8(0):
                    fate = 0
                else:
                    t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                    crossed_x = ti.abs(t_max_x - t_next) <= 1.0e-12
                    crossed_y = ti.abs(t_max_y - t_next) <= 1.0e-12
                    crossed_z = ti.abs(t_max_z - t_next) <= 1.0e-12
                    if crossed_x:
                        i += step_x
                        t_max_x += t_delta_x
                        nxn = -ti.cast(step_x, ti.f64)
                        nyn = 0.0
                        nzn = 0.0
                    if crossed_y:
                        j += step_y
                        t_max_y += t_delta_y
                        if not crossed_x:
                            nxn = 0.0
                            nyn = -ti.cast(step_y, ti.f64)
                            nzn = 0.0
                    if crossed_z:
                        k += step_z
                        t_max_z += t_delta_z
                        if not crossed_x and not crossed_y:
                            nxn = 0.0
                            nyn = 0.0
                            nzn = -ti.cast(step_z, ti.f64)
                    t = t_next
        if fate == -1:
            fate = 3
        return fate, t, nxn, nyn, nzn

    @ti.func
    def _lambertian(state, nxn, nyn, nzn):
        state, xi1 = _rand01(state)
        state, xi2 = _rand01(state)
        helper_x = 1.0
        helper_y = 0.0
        helper_z = 0.0
        if ti.abs(nxn) > 0.9:
            helper_x = 0.0
            helper_y = 1.0
            helper_z = 0.0
        ux = helper_y * nzn - helper_z * nyn
        uy = helper_z * nxn - helper_x * nzn
        uz = helper_x * nyn - helper_y * nxn
        un = ti.sqrt(ux * ux + uy * uy + uz * uz)
        ux /= un
        uy /= un
        uz /= un
        vx = nyn * uz - nzn * uy
        vy = nzn * ux - nxn * uz
        vz = nxn * uy - nyn * ux
        radius = ti.sqrt(xi1)
        phi = 2.0 * math.pi * xi2
        cx = radius * ti.cos(phi)
        cy = radius * ti.sin(phi)
        cz = ti.sqrt(ti.max(0.0, 1.0 - xi1))
        return state, cx * ux + cy * vx + cz * nxn, cx * uy + cy * vy + cz * nyn, cx * uz + cy * vz + cz * nzn

    @ti.kernel
    def _kernel(solid_field: ti.template(), counter_field: ti.template(), sx: ti.i32, sy: ti.i32, sz: ti.i32,
                n: ti.i32, seed_u32: ti.u32, solid_eps: ti.f64, max_bounces: ti.i32):
        """Counts reflected (0), transmitted (1) and absorbed (2) rays, and rays stopped at the reflection limit (3)."""
        for ray in range(n):
            state = _hash_u32(seed_u32 + ti.cast(ray + 1, ti.u32) * ti.u32(747796405))
            state, ry = _rand01(state)
            state, rz = _rand01(state)
            ox = 0.0
            oy = ry * ti.cast(sy, ti.f64)
            oz = rz * ti.cast(sz, ti.f64)
            state, dx, dy, dz = _lambertian(state, 1.0, 0.0, 0.0)
            fate, dist, nxn, nyn, nzn = _trace(solid_field, ox, oy, oz, dx, dy, dz, sx, sy, sz)
            if fate != 0:
                if fate == 1:
                    ti.atomic_add(counter_field[0], 1)
                elif fate == 2:
                    ti.atomic_add(counter_field[1], 1)
                else:
                    ti.atomic_add(counter_field[3], 1)
                continue
            if nxn * nxn + nyn * nyn + nzn * nzn <= 1.0e-24:
                nxn = -dx
                nyn = -dy
                nzn = -dz
            if ti.static(hit_point):
                ox = ox + dx * dist
                oy = oy + dy * dist
                oz = oz + dz * dist
            hit = True
            bounces = 0
            while hit:
                state, absorb_u = _rand01(state)
                if absorb_u < solid_eps:
                    ti.atomic_add(counter_field[2], 1)
                    hit = False
                else:
                    prev_dx = dx
                    prev_dy = dy
                    prev_dz = dz
                    state, dx, dy, dz = _lambertian(state, nxn, nyn, nzn)
                    bounces += 1
                    if bounces >= max_bounces:
                        ti.atomic_add(counter_field[3], 1)
                        hit = False
                    else:
                        if ti.static(hit_point):
                            ox = ox + nxn * 1.0e-9
                            oy = oy + nyn * 1.0e-9
                            oz = oz + nzn * 1.0e-9
                        else:
                            ox = ox + prev_dx * dist + nxn * 1.0e-9
                            oy = oy + prev_dy * dist + nyn * 1.0e-9
                            oz = oz + prev_dz * dist + nzn * 1.0e-9
                        fate, dist, nxn, nyn, nzn = _trace(solid_field, ox, oy, oz, dx, dy, dz, sx, sy, sz)
                        if fate == 0:
                            if nxn * nxn + nyn * nyn + nzn * nzn <= 1.0e-24:
                                nxn = -dx
                                nyn = -dy
                                nzn = -dz
                            ox = ox + dx * dist
                            oy = oy + dy * dist
                            oz = oz + dz * dist
                        else:
                            if fate == 1:
                                ti.atomic_add(counter_field[0], 1)
                            elif fate == 2:
                                ti.atomic_add(counter_field[1], 1)
                            else:
                                ti.atomic_add(counter_field[3], 1)
                            hit = False

    return _kernel


def _binomial(count: int, n: int) -> tuple[float, float]:
    """Fraction count / n and its binomial standard error."""
    p = float(count) / float(n)
    return p, float(math.sqrt(max(p * (1.0 - p), 0.0) / n))


def trace_penetration(phases: np.ndarray, axis: str, epsilon: float, *, n_rays: int = N_RAYS, seed: int = SEEDS[0],
                      restart: str = "article", max_bounces: int = MAX_BOUNCES) -> dict:
    """R, T, A with binomial standard errors for one geometry, axis, eps and seed."""
    if restart not in RESTART_RULES:
        raise ValueError(f"restart must be one of {RESTART_RULES}")
    import taichi as ti

    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)
    solid = solid_along_x(phases, axis)
    nx, ny, nz = map(int, solid.shape)
    n = int(n_rays)
    kernel = _build_kernel(ti, restart == "hit_point")
    solid_field = ti.field(dtype=ti.u8, shape=solid.shape)
    counter_field = ti.field(dtype=ti.i64, shape=(4,))
    solid_field.from_numpy(solid)
    counter_field.fill(0)
    kernel(solid_field, counter_field, nx, ny, nz, n, int(seed) & 0xFFFFFFFF, float(epsilon), int(max_bounces))
    counts = [int(c) for c in counter_field.to_numpy()]
    ti.reset()

    out = {"n_rays": n, "terminated": counts[3]}
    for name, count in (("reflectance", counts[0]), ("transmittance", counts[1]), ("absorptance", counts[2])):
        out[name], out[f"{name}_se"] = _binomial(count, n)
    return out


def penetration_response(phases: np.ndarray, axis: str, epsilon: float, *, seeds=SEEDS, n_rays: int = N_RAYS,
                         restart: str = "article") -> dict:
    """R, T, A of one geometry, axis and emissivity, averaged over independent seeds."""
    runs = [trace_penetration(phases, axis, epsilon, n_rays=n_rays, seed=s, restart=restart) for s in seeds]
    out = {"n_rays": int(sum(r["n_rays"] for r in runs)), "terminated": int(sum(r["terminated"] for r in runs))}
    for key in ("reflectance", "transmittance", "absorptance"):
        out[key], out[f"{key}_se"] = seed_average([r[key] for r in runs], [r[f"{key}_se"] for r in runs])
    return out
