"""Equivalent radiative properties identified from pore-scale ray statistics.

Rays enter the slab at normal incidence through the face x = 0 at uniformly
distributed transverse positions and travel through the pore space of the
digital microstructure (periodic in the two transverse directions). The first
path segment of every ray gives either an interception by the solid or a
right-censored exposure for a ray that crosses the slab unintercepted; the
directional extinction coefficient is the right-censored maximum-likelihood
estimate of the rate of the survival law exp(-beta l). At a solid surface a ray
is absorbed with probability eps, otherwise it is reflected into a Lambertian
direction about the surface normal and followed further; the albedo omega is
the reflected fraction of all surface interactions and the asymmetry factor g
the mean cosine between incoming and reflected directions. Lengths are in cells.

Two restart rules for a reflected ray are provided. With restart="article"
(default), the rule behind the values reported in the article, a ray reflected
for the first time restarts at its hit point, and a ray reflected for the
second or a later time restarts at its hit point advanced once more by the
length of the preceding path segment along the preceding direction. With
restart="hit_point" every reflected ray restarts at its hit point. In both
rules the restart point is displaced by 1e-9 cells along the surface normal.
The first path segment, and therefore beta, is the same under both rules.

The ray kernel is written in Taichi and runs on the CPU.
"""

from __future__ import annotations

import math

import numpy as np

SEEDS = (123, 456)
RESTART_RULES = ("article", "hit_point")
N_RAYS = 200_000
MAX_BOUNCES = 1000


def solid_along_x(phases: np.ndarray, axis: str) -> np.ndarray:
    """Return the solid indicator with the transfer axis moved to the first index."""
    solid = np.asarray(phases) == 1
    return np.ascontiguousarray(np.moveaxis(solid, "xyz".index(axis), 0).astype(np.uint8))


def right_censored_mle(events: np.ndarray, censored: np.ndarray) -> tuple[float, float]:
    """Rate estimate events / total exposure and its standard error."""
    exposure = float(np.sum(events) + np.sum(censored))
    n_events = int(events.size)
    if exposure <= 0.0 or n_events == 0:
        return 0.0, 0.0
    return float(n_events / exposure), float(math.sqrt(n_events) / exposure)


def trace_properties(phases: np.ndarray, axis: str, epsilon: float, *, n_rays: int = N_RAYS, seed: int = SEEDS[0],
                     max_bounces: int = MAX_BOUNCES, restart: str = "article") -> dict:
    """Identify beta (per cell), omega and g for one geometry, transfer axis, emissivity and seed."""
    import taichi as ti

    if restart not in RESTART_RULES:
        raise ValueError(f"restart must be one of {RESTART_RULES}")
    hit_point = restart == "hit_point"

    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)

    solid = solid_along_x(phases, axis)
    nx, ny, nz = map(int, solid.shape)
    n = int(n_rays)
    eps = float(epsilon)
    solid_field = ti.field(dtype=ti.u8, shape=solid.shape)
    event_field = ti.field(dtype=ti.f64, shape=(n,))
    censored_field = ti.field(dtype=ti.f64, shape=(n,))
    counter_field = ti.field(dtype=ti.i64, shape=(2,))
    sum_field = ti.field(dtype=ti.f64, shape=(2,))
    solid_field.from_numpy(solid)
    event_field.from_numpy(np.full(n, -1.0))
    censored_field.from_numpy(np.full(n, -1.0))
    seed_u32 = int(seed) & 0xFFFFFFFF

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
    def _trace(ox, oy, oz, dx0, dy0, dz0, sx, sy, sz):
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
    def _kernel():
        for ray in range(n):
            state = _hash_u32(ti.u32(seed_u32) + ti.cast(ray + 1, ti.u32) * ti.u32(747796405))
            state, ry = _rand01(state)
            state, rz = _rand01(state)
            ox = 0.0
            oy = ry * ti.cast(ny, ti.f64)
            oz = rz * ti.cast(nz, ti.f64)
            dx = 1.0
            dy = 0.0
            dz = 0.0
            fate, dist, nxn, nyn, nzn = _trace(ox, oy, oz, dx, dy, dz, nx, ny, nz)
            if fate == 0:
                if nxn * nxn + nyn * nyn + nzn * nzn <= 1.0e-24:
                    nxn = -dx
                    nyn = -dy
                    nzn = -dz
                event_field[ray] = dist
                if ti.static(hit_point):
                    ox = ox + dx * dist
                    oy = oy + dy * dist
                    oz = oz + dz * dist
            else:
                censored_field[ray] = dist
                continue
            hit = True
            bounces = 0
            while hit:
                ti.atomic_add(counter_field[0], 1)
                state, absorb_u = _rand01(state)
                if absorb_u < eps:
                    hit = False
                else:
                    prev_dx = dx
                    prev_dy = dy
                    prev_dz = dz
                    state, dx, dy, dz = _lambertian(state, nxn, nyn, nzn)
                    mu = prev_dx * dx + prev_dy * dy + prev_dz * dz
                    ti.atomic_add(counter_field[1], 1)
                    ti.atomic_add(sum_field[0], mu)
                    ti.atomic_add(sum_field[1], mu * mu)
                    bounces += 1
                    if bounces >= int(max_bounces):
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
                        fate, dist, nxn, nyn, nzn = _trace(ox, oy, oz, dx, dy, dz, nx, ny, nz)
                        if fate == 0:
                            if nxn * nxn + nyn * nyn + nzn * nzn <= 1.0e-24:
                                nxn = -dx
                                nyn = -dy
                                nzn = -dz
                            ox = ox + dx * dist
                            oy = oy + dy * dist
                            oz = oz + dz * dist
                        else:
                            hit = False

    _kernel()
    event_lengths = event_field.to_numpy()
    censored_lengths = censored_field.to_numpy()
    counters = counter_field.to_numpy()
    sums = sum_field.to_numpy()
    ti.reset()

    events = event_lengths[event_lengths >= 0.0]
    censored = censored_lengths[censored_lengths >= 0.0]
    beta, beta_se = right_censored_mle(events, censored)
    interactions, scattered = int(counters[0]), int(counters[1])
    omega = scattered / interactions if interactions else 0.0
    omega_se = math.sqrt(omega * (1.0 - omega) / interactions) if interactions else 0.0
    g, g_se = 0.0, 0.0
    if scattered > 0:
        g = float(sums[0] / scattered)
        if scattered > 1:
            g_se = math.sqrt(max(0.0, float((sums[1] - scattered * g * g) / (scattered - 1))) / scattered)
    return {"n_rays": n, "beta": beta, "beta_se": beta_se, "omega": omega, "omega_se": omega_se, "g": g, "g_se": g_se}


def seed_average(values, ses):
    """Mean over seeds; the standard error combines within-seed and between-seed variance."""
    values, ses = np.asarray(values, float), np.asarray(ses, float)
    mean = float(values.mean())
    within = float(np.sum(ses**2) / values.size**2)
    between = float(values.var(ddof=1) / values.size) if values.size > 1 else 0.0
    return mean, float(np.sqrt(max(0.0, within + between)))


def identify_properties(phases: np.ndarray, axis: str, epsilon: float, *, seeds=SEEDS, n_rays: int = N_RAYS,
                        restart: str = "article") -> dict:
    """Properties of one geometry, axis and emissivity, averaged over independent seeds."""
    runs = [trace_properties(phases, axis, epsilon, n_rays=n_rays, seed=s, restart=restart) for s in seeds]
    out = {"n_rays": int(sum(r["n_rays"] for r in runs))}
    for key in ("beta", "omega", "g"):
        mean, se = seed_average([r[key] for r in runs], [r[f"{key}_se"] for r in runs])
        out[key], out[f"{key}_se"] = mean, se
    out["beta_ci95_low"] = out["beta"] - 1.96 * out["beta_se"]
    out["beta_ci95_high"] = out["beta"] + 1.96 * out["beta_se"]
    return out
