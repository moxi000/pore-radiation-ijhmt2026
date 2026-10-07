"""Digital microstructures of the four structures of the paper.

OF open-cell foam, FI random fibers, FA fibers aligned with x, TL tortuous
layered. Every microstructure lives on a periodic N x N x N grid of cells with
array axis 0 = x; phase arrays are uint8 with 1 = solid and 0 = pore. The solid
fraction phi_s is set by bisection on the strut or fiber radius (OF, FI, FA) or
by a quantile threshold of a continuous field (TL).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

from .descriptors import scalar_descriptors

STRUCTURES = ("OF", "FI", "FA", "TL")
CORE_N = 128
CORE_PHIS = (0.05, 0.10, 0.15)
PHI_TOL = 3e-3
MIN_PORE_CLUSTER = 0.99

FIBER_LINES = {"FI": {0.05: 6, 0.10: 10, 0.15: 14}, "FA": {0.05: 6, 0.10: 12, 0.15: 18}}
FIBER_LENGTH = {"FI": 9.0, "FA": 10.0}
FA_THETA_SIGMA = 0.05
FOAM_NODES = {0.05: 24, 0.10: 32, 0.15: 42}
FOAM_NEIGHBOURS = 3
NODE_TO_STRUT = 1.2
TL_LAYERS = {0.05: 9, 0.10: 7, 0.15: 9}
TL_NOISE = 0.12
TL_BASE_N = 192

# Matched fiber pairs k = 1, 2, 3: (seed, number of fibers) of the FA and FI member.
MATCHED_PAIRS = {1: {"FA": (7, 40), "FI": (0, 12)},
                 2: {"FA": (6, 52), "FI": (8, 14)},
                 3: {"FA": (2, 32), "FI": (4, 6)}}

_EX = np.array([1.0, 0.0, 0.0])
_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])


# Shape stamping on the periodic grid

def _matmul(a, b):
    """a @ b; large products may set spurious floating-point flags on some BLAS builds."""
    with np.errstate(all="ignore"):
        return a @ b


def _frame(z, helper):
    """Rotation matrix with columns (x, y, z) for the unit axis z."""
    if abs(float(np.dot(helper, z))) > 0.9:
        helper = _EY
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def _long_cylinder(N, r, half_length, R, center=(0.0, 0.0, 0.0)):
    """Cylinder longer than the cell, wrapped on the periodic grid.

    The axis is sampled at unit steps, each sample is rounded to the nearest
    lattice node (half to even), and every node is thickened by the cells of a
    one-cell-thick disk of radius r normal to the axis.
    """
    axis = _matmul(R, _EZ)
    axis /= float(np.linalg.norm(axis))
    n_steps = max(1, int(math.ceil(2.0 * half_length)))
    t = np.arange(n_steps, dtype=float) - n_steps / 2.0
    pts = t[:, None] * axis[None, :] + np.asarray(center, dtype=float)[None, :]
    nodes = np.unique(np.rint(pts).astype(np.int64) % N, axis=0)
    Rq = _frame(axis / float(np.linalg.norm(axis)), _EX)
    pad = int(math.ceil(r)) + 2
    g = np.arange(-pad, pad + 1, dtype=np.int64)
    off = np.stack([a.ravel() for a in np.meshgrid(g, g, g, indexing="ij")], axis=1)
    local = _matmul(Rq.T, off.T.astype(float)).T
    disk = (((local[:, 0] / r) ** 2 + (local[:, 1] / r) ** 2 <= 1.0 + 1e-12)
            & (np.abs(local[:, 2]) <= 0.5 + 1e-12))
    idx = (nodes[:, None, :] + off[disk][None, :, :]) % N
    mask = np.zeros(N ** 3, dtype=bool)
    mask[((idx[:, :, 0] * N + idx[:, :, 1]) * N + idx[:, :, 2]).ravel()] = True
    return mask.reshape((N, N, N))


def shape_mask(N, center, radii, kind, R):
    """Cells whose centres lie inside a rotated ellipsoid or cylinder on the periodic grid.

    Local axes are the columns of R; radii are (r_x, r_y, r_z) with r_z the
    half length of a cylinder. Shapes shorter than the cell use minimum-image
    distances. A cylinder longer than the cell is stamped by
    :func:`_long_cylinder`; for an integer centre it is built at the origin and
    shifted by whole cells.
    """
    cx, cy, cz = center
    rx, ry, rz = radii
    if kind == "cylinder" and float(2.0 * rz) > float(N):
        ci = tuple(int(round(c)) % N for c in center)
        if any(abs(c - float(i)) > 1e-9 for c, i in zip(center, ci)):
            return _long_cylinder(N, rx, rz, R, center)
        base = _long_cylinder(N, rx, rz, R)
        return base if ci == (0, 0, 0) else np.roll(base, shift=ci, axis=(0, 1, 2))
    half = float(N) / 2.0
    x = (np.arange(N)[:, None, None] - cx + half) % float(N) - half
    y = (np.arange(N)[None, :, None] - cy + half) % float(N) - half
    z = (np.arange(N)[None, None, :] - cz + half) % float(N) - half
    x3, y3, z3 = np.broadcast_arrays(x, y, z)
    local = _matmul(R.T, np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=0))
    xl, yl, zl = (local[i].reshape((N, N, N)) for i in range(3))
    if kind == "ellipsoid":
        return (xl / rx) ** 2 + (yl / ry) ** 2 + (zl / rz) ** 2 <= 1.0
    if kind == "cylinder":
        return ((xl / rx) ** 2 + (yl / ry) ** 2 <= 1.0) & (np.abs(zl) <= rz)
    raise ValueError(kind)


def _bisect_radius(render, target_phi, r_lo, r_hi, tol=PHI_TOL, max_iter=24):
    """Bisection on a radius r until |phi_s(r) - target| <= tol.

    Stops at the first radius within tol, otherwise returns the closest one.
    Returns (r, phases).
    """
    lo, hi, best = float(r_lo), float(r_hi), None
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        phases = render(mid)
        phi = float(np.mean(phases == 1))
        err = abs(phi - float(target_phi))
        if best is None or err < best[0]:
            best = (err, mid, phases)
        if err <= tol:
            break
        if phi < float(target_phi):
            lo = mid
        else:
            hi = mid
    return best[1], best[2]


# Fibers (FI, FA)

def _fiber_axis(rng, aligned):
    """Unnormalized fiber axis, isotropic or tilted from x by theta ~ N(0, 0.05) with uniform azimuth."""
    if aligned:
        theta = rng.normal(0.0, FA_THETA_SIGMA)
        az = rng.uniform(0.0, 2.0 * math.pi)
        return np.array([math.cos(theta), math.sin(theta) * math.cos(az), math.sin(theta) * math.sin(az)])
    return rng.normal(size=3)


def fiber_lines(N, seed, n_lines, aligned, continuous_centers=False):
    """Centres and rotations of n_lines straight fibers drawn from one random stream.

    For each fiber the centre is drawn first (a lattice node, or a uniform
    point when continuous_centers is set), then the axis.
    """
    rng = np.random.default_rng(int(seed))
    centers, rotations = [], []
    for _ in range(int(n_lines)):
        c = rng.random(3) * float(N) if continuous_centers else rng.integers(0, N, size=3)
        centers.append(tuple(float(v) for v in c))
        v = _fiber_axis(rng, aligned)
        if not aligned:
            v = v / np.linalg.norm(v)
        rotations.append(_frame(v / np.linalg.norm(v), _EZ))
    return centers, rotations


def render_fibers(N, centers, rotations, radius, length_factor):
    """Solid phase of periodic fibers of radius r and length length_factor * N."""
    phases = np.zeros((N, N, N), dtype=np.uint8)
    half = 0.5 * float(length_factor) * N
    for c, R in zip(centers, rotations):
        phases[shape_mask(N, c, (radius, radius, half), "cylinder", R)] = 1
    return phases


# Open-cell foam (OF)

def foam_graph(N, seed, n_nodes, k=FOAM_NEIGHBOURS):
    """Uniform random nodes joined to their k nearest periodic neighbours."""
    rng = np.random.default_rng(int(seed))
    points = rng.uniform(0, N, size=(int(n_nodes), 3))
    _, nn = cKDTree(points, boxsize=N).query(points, k=int(k) + 1)
    edges = {tuple(sorted((int(i), int(j)))) for i, row in enumerate(nn) for j in row[1:] if int(i) != int(j)}
    return points, sorted(edges)


def _strut(N, p0, p1, r):
    """Cylindrical strut of radius r between two nodes along the minimum-image vector."""
    vec = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    for a in range(3):
        if vec[a] > N / 2:
            vec[a] -= N
        elif vec[a] < -N / 2:
            vec[a] += N
    L = float(np.linalg.norm(vec))
    center = tuple((np.asarray(p0, dtype=float) + 0.5 * vec) % N)
    return shape_mask(N, center, (r, r, 0.5 * L), "cylinder", _frame(vec / L, _EZ))


def render_foam(N, points, edges, strut_radius, node_radius):
    """Solid phase of spherical nodes and cylindrical struts."""
    phases = np.zeros((N, N, N), dtype=np.uint8)
    for p in points:
        phases[shape_mask(N, tuple(p.tolist()), (node_radius,) * 3, "ellipsoid", np.eye(3))] = 1
    for a, b in edges:
        phases[_strut(N, points[a], points[b], strut_radius)] = 1
    return phases


# Tortuous layered (TL)

def lamellar_field(N, seed, n_layers, waviness, noise_amp=TL_NOISE):
    """Continuous layered field of n_layers wavy layers normal to x plus smoothed Gaussian noise."""
    rng = np.random.default_rng(int(seed))
    x = np.arange(N, dtype=float)[:, None, None]
    y = np.arange(N, dtype=float)[None, :, None]
    z = np.arange(N, dtype=float)[None, None, :]
    base = np.cos(2.0 * np.pi * n_layers * x / float(N))
    ph1 = rng.uniform(0.0, 2 * np.pi)
    ph2 = rng.uniform(0.0, 2 * np.pi)
    wav = float(waviness) * (0.65 * np.sin(2 * np.pi * y / float(N) + ph1)
                             + 0.35 * np.cos(4 * np.pi * z / float(N) + ph2))
    field = base + 0.35 * np.sin(2 * np.pi * n_layers * (x + wav) / float(N))
    noise = rng.normal(scale=float(noise_amp), size=(N, N, N))
    noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0) + np.roll(noise, 1, 1)
             + np.roll(noise, -1, 1) + np.roll(noise, 1, 2) + np.roll(noise, -1, 2)) / 7.0
    return field + noise


def threshold(field, target_phi):
    """Solid where the field reaches its (1 - phi_s) quantile."""
    q = np.quantile(field.ravel(), 1.0 - float(target_phi))
    return (field >= q).astype(np.uint8)


# Builders

def build(structure, N, target_phi, seed, continuous_centers=False):
    """Digital microstructure of one structure at a target solid fraction."""
    phi = round(float(target_phi), 2)
    if structure in ("FI", "FA"):
        centers, rotations = fiber_lines(N, seed, FIBER_LINES[structure][phi], structure == "FA",
                                         continuous_centers)
        lf = FIBER_LENGTH[structure]
        return _bisect_radius(lambda r: render_fibers(N, centers, rotations, r, lf), target_phi, 0.1, 4.0)[1]
    if structure == "OF":
        points, edges = foam_graph(N, seed, FOAM_NODES[phi])
        s = max(1.0, float(N) / 48.0)
        return _bisect_radius(lambda r: render_foam(N, points, edges, r, float(NODE_TO_STRUT * r)),
                              target_phi, 0.2 * s, 4.0 * s)[1]
    if structure == "TL":
        return threshold(lamellar_field(N, seed, TL_LAYERS[phi], 1.4 + 3.0 * float(target_phi)), target_phi)
    raise ValueError(structure)


def accepted(structure, d, target_phi):
    """Acceptance rules of the core library.

    A realization is retained if |phi_s - target| <= 3e-3 and it shows the
    signature of its structure. For OF the pore phase percolates along x, y and z
    and its largest 26-connected cluster holds at least 99% of the pore volume.
    With the mean axial pore chords l_x, l_y, l_z, TL needs max/min >= 2, FA needs
    l_x / mean(l_y, l_z) >= 2 and FI needs max/min < 2.
    """
    if abs(float(d["phi_solid"]) - float(target_phi)) > PHI_TOL:
        return False
    c = d["mean_pore_chord"]
    if structure == "OF":
        return all(d["pore_percolates"]) and d["pore_largest_cluster_fraction"] >= MIN_PORE_CLUSTER
    if structure == "TL":
        return max(c) / max(min(c), 1e-12) >= 2.0
    if structure == "FA":
        return c[0] / max(0.5 * (c[1] + c[2]), 1e-12) >= 2.0
    if structure == "FI":
        return max(c) / max(min(c), 1e-12) < 2.0
    raise ValueError(structure)


def core_realizations(structure, target_phi, n_accept=4, max_trials=80):
    """One cell of the core library, built at N = 128 from seeds 0, 1, 2, ... until n_accept are accepted.

    Returns a list of (seed, phases).
    """
    out = []
    for seed in range(int(max_trials)):
        phases = build(structure, CORE_N, target_phi, seed)
        if accepted(structure, scalar_descriptors(phases), target_phi):
            out.append((seed, phases))
            if len(out) == int(n_accept):
                break
    return out


def canonical(structure, N):
    """Canonical microstructure at phi_s = 0.10 and seed 0 on an N^3 grid.

    FI and FA place the fiber centres uniformly in the cell, so that the same
    fibers scale with N. TL is one field drawn at N = 192, resampled to N by
    periodic cubic splines and thresholded at N. OF follows :func:`build`; at
    N = 128 it equals OF_core_phi0.10_s0.
    """
    if structure == "TL":
        field = lamellar_field(TL_BASE_N, 0, TL_LAYERS[0.10], 1.4 + 3.0 * 0.10)
        if N != TL_BASE_N:
            field = zoom(field, float(N) / float(TL_BASE_N), order=3, mode="grid-wrap", grid_mode=True,
                         prefilter=True)
        return threshold(field, 0.10)
    return build(structure, N, 0.10, 0, continuous_centers=True)


def matched(structure, pair, N=CORE_N):
    """Member FA or FI of matched fiber pair k = 1, 2, 3 at phi_s = 0.10.

    Both members use fibers of length 9N. Centres come from one uniform stream
    and axes from a second stream; bisection on the radius stops at
    |phi_s - 0.10| <= 5e-4.
    """
    seed, n_lines = MATCHED_PAIRS[pair][structure]
    aligned = structure == "FA"
    centers = [tuple(float(v) for v in c)
               for c in (np.random.default_rng(seed + 1000003).random((64, 3)) * N)[:n_lines]]
    rng = np.random.default_rng(seed + (3000003 if aligned else 2000003))
    rotations = []
    for _ in range(n_lines):
        v = _fiber_axis(rng, aligned)
        rotations.append(_frame(v / np.linalg.norm(v), _EZ))
    return _bisect_radius(lambda r: render_fibers(N, centers, rotations, r, 9.0), 0.1, 0.12, 7.0,
                          tol=5e-4, max_iter=32)[1]
