"""Surface patches, Lambertian ray sets and first-hit exchange counts.

The solid-pore interface of a digital microstructure and the two walls normal
to the transfer axis are divided into square surface patches, one per exposed
cell face.  Each patch emits rays from points sampled uniformly over its face
in Lambertian directions, and every ray is followed in a straight line through
the transparent pore space to the first surface it meets.  The integer number
n_ij of rays leaving patch i that first reach patch j gives the Monte Carlo
view-factor estimate F_ij = n_ij / R_i, with R_i the rays emitted by patch i.

All arrays use the orientation in which the transfer axis is x: the hot wall
lies at x = 0, the cold wall at x = N_x, and the lateral directions y and z
are periodic.  Positions are measured in cell widths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from numba import njit, prange

HOT, COLD, INTERNAL = 0, 1, 2
"""Patch classes: hot wall, cold wall, solid-pore interface."""

_INF = 1.0e300
_TIE_TOLERANCE = 1.0e-12
_FACE_EDGE_INSET = 1.0e-12
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_FACE_NORMALS = np.asarray(
    ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)), dtype=np.float64
)


def orient_along_x(array: np.ndarray, axis: str) -> np.ndarray:
    """Return ``array`` with the requested transfer axis moved to the first index."""

    return np.ascontiguousarray(np.moveaxis(np.asarray(array), _AXIS_INDEX[axis], 0))


@dataclass
class SurfacePatches:
    """Patch table of one digital microstructure oriented along x.

    ``patch_class`` is HOT, COLD or INTERNAL.  An internal patch is the face
    ``patch_face`` (0..5 for +x, -x, +y, -y, +z, -z) of the solid cell
    ``patch_cell`` (flat C index).  ``cell_face_patch``, ``hot_wall_patch``
    and ``cold_wall_patch`` map a hit face back to its patch index.
    """

    solid: np.ndarray
    patch_class: np.ndarray
    patch_cell: np.ndarray
    patch_face: np.ndarray
    patch_origin: np.ndarray
    patch_normal: np.ndarray
    patch_area_m2: np.ndarray
    cell_face_patch: np.ndarray
    hot_wall_patch: np.ndarray
    cold_wall_patch: np.ndarray
    H_m: float

    @property
    def n_patches(self) -> int:
        return int(self.patch_class.size)


def build_surface_patches(phases_x: np.ndarray, H_m: float) -> SurfacePatches:
    """Enumerate wall and interface patches of an x-oriented phase array (1 = solid).

    Wall patches come first, ordered by (j, k) with the hot patch before the
    cold patch; a wall patch exists only where the adjacent boundary cell is
    pore.  Internal patches follow in (i, j, k, face) order.  Lateral
    neighbours are periodic, and solid faces on the x boundaries are not
    patches because the walls receive the radiation there.  Every patch has
    the area (H/N_x)^2.
    """

    solid = np.ascontiguousarray((np.asarray(phases_x) == 1).astype(np.uint8))
    nx, ny, nz = map(int, solid.shape)
    area = (float(H_m) / nx) ** 2

    wall_j, wall_k, wall_class = np.nonzero(np.stack((solid[0] == 0, solid[-1] == 0), axis=-1))
    n_wall = int(wall_class.size)
    wall_origin = np.empty((n_wall, 3), dtype=np.float64)
    wall_origin[:, 0] = np.where(wall_class == 0, 1.0e-9, float(nx) - 1.0e-9)
    wall_origin[:, 1] = wall_j.astype(np.float64) + 0.5
    wall_origin[:, 2] = wall_k.astype(np.float64) + 0.5
    wall_normal = np.zeros((n_wall, 3), dtype=np.float64)
    wall_normal[:, 0] = np.where(wall_class == 0, 1.0, -1.0)

    exposed = np.zeros((nx, ny, nz, 6), dtype=bool)
    exposed[:-1, :, :, 0] = (solid[:-1] != 0) & (solid[1:] == 0)
    exposed[1:, :, :, 1] = (solid[1:] != 0) & (solid[:-1] == 0)
    exposed[..., 2] = (solid != 0) & (np.roll(solid, -1, axis=1) == 0)
    exposed[..., 3] = (solid != 0) & (np.roll(solid, 1, axis=1) == 0)
    exposed[..., 4] = (solid != 0) & (np.roll(solid, -1, axis=2) == 0)
    exposed[..., 5] = (solid != 0) & (np.roll(solid, 1, axis=2) == 0)
    cell_i, cell_j, cell_k, face = np.nonzero(exposed)
    n_internal = int(face.size)
    internal_cell = ((cell_i * ny + cell_j) * nz + cell_k).astype(np.int32)
    internal_normal = _FACE_NORMALS[face]
    internal_origin = np.column_stack((cell_i, cell_j, cell_k)).astype(np.float64) + 0.5
    internal_origin[internal_normal > 0.0] += 0.5 + 1.0e-9
    internal_origin[internal_normal < 0.0] -= 0.5 + 1.0e-9
    internal_origin[:, 1] %= ny
    internal_origin[:, 2] %= nz

    n_patches = n_wall + n_internal
    cell_face_patch = np.full((nx * ny * nz, 6), -1, dtype=np.int32)
    cell_face_patch[internal_cell, face] = np.arange(n_wall, n_patches, dtype=np.int32)
    hot_wall_patch = np.full((ny, nz), -1, dtype=np.int32)
    cold_wall_patch = np.full((ny, nz), -1, dtype=np.int32)
    wall_id = np.arange(n_wall, dtype=np.int32)
    hot = wall_class == 0
    hot_wall_patch[wall_j[hot], wall_k[hot]] = wall_id[hot]
    cold_wall_patch[wall_j[~hot], wall_k[~hot]] = wall_id[~hot]
    return SurfacePatches(
        solid=solid,
        patch_class=np.concatenate((wall_class.astype(np.int8), np.full(n_internal, INTERNAL, np.int8))),
        patch_cell=np.concatenate((np.full(n_wall, -1, np.int32), internal_cell)),
        patch_face=np.concatenate((np.full(n_wall, -1, np.int8), face.astype(np.int8))),
        patch_origin=np.ascontiguousarray(np.vstack((wall_origin, internal_origin))),
        patch_normal=np.ascontiguousarray(np.vstack((wall_normal, internal_normal))),
        patch_area_m2=np.full(n_patches, area, dtype=np.float64),
        cell_face_patch=cell_face_patch,
        hot_wall_patch=hot_wall_patch,
        cold_wall_patch=cold_wall_patch,
        H_m=float(H_m),
    )


def generate_rays(
    patches: SurfacePatches, patch_start: int, patch_end: int, rays_per_patch: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Ray origins and directions for patches ``patch_start`` to ``patch_end - 1``.

    Each ray consumes four uniforms of one PCG64 stream: two place the origin
    uniformly on the patch face and two give a cosine-weighted (Lambertian)
    direction about the patch normal.  The stream is advanced by
    ``patch_start * rays_per_patch * 4`` draws, so any partition of the
    patches reproduces the same rays as a single pass over all of them.
    """

    p0, p1, rays = int(patch_start), int(patch_end), int(rays_per_patch)
    normals = patches.patch_normal[p0:p1]
    generator = np.random.PCG64(int(seed))
    generator.advance(p0 * rays * 4)
    uniforms = np.random.Generator(generator).random((p1 - p0, rays, 4), dtype=np.float64)

    origins = np.repeat(patches.patch_origin[p0:p1, None, :], rays, axis=1)
    tangent_a = np.clip(uniforms[:, :, 0], _FACE_EDGE_INSET, 1.0 - _FACE_EDGE_INSET) - 0.5
    tangent_b = np.clip(uniforms[:, :, 1], _FACE_EDGE_INSET, 1.0 - _FACE_EDGE_INSET) - 0.5
    normal_x = normals[:, 0] != 0.0
    normal_y = (~normal_x) & (normals[:, 1] != 0.0)
    normal_z = ~(normal_x | normal_y)
    origins[normal_x, :, 1] += tangent_a[normal_x]
    origins[normal_x, :, 2] += tangent_b[normal_x]
    origins[normal_y, :, 0] += tangent_a[normal_y]
    origins[normal_y, :, 2] += tangent_b[normal_y]
    origins[normal_z, :, 0] += tangent_a[normal_z]
    origins[normal_z, :, 1] += tangent_b[normal_z]
    origins[:, :, 1] %= patches.solid.shape[1]
    origins[:, :, 2] %= patches.solid.shape[2]

    radius = np.sqrt(uniforms[:, :, 2])
    phi = 2.0 * np.pi * uniforms[:, :, 3]
    local_a = radius * np.cos(phi)
    local_b = radius * np.sin(phi)
    cos_theta = np.sqrt(1.0 - uniforms[:, :, 2])
    directions = np.empty((p1 - p0, rays, 3), dtype=np.float64)
    for axis, (first, second) in enumerate(((1, 2), (0, 2), (0, 1))):
        for sign in (1.0, -1.0):
            mask = normals[:, axis] == sign
            directions[mask, :, axis] = sign * cos_theta[mask]
            directions[mask, :, first] = local_a[mask]
            directions[mask, :, second] = sign * local_b[mask]
    directions = np.ascontiguousarray(directions.reshape(-1, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.ascontiguousarray(origins.reshape(-1, 3)), directions


@njit(cache=False)
def _initial_cell(position: float, direction: float) -> int:
    index = int(math.floor(position))
    if direction < 0.0 and abs(position - round(position)) <= _TIE_TOLERANCE:
        index -= 1
    return index


@njit(cache=False)
def _axis_state(position: float, direction: float, index: int) -> tuple[int, float, float]:
    if direction > 0.0:
        return 1, (index + 1.0 - position) / direction, 1.0 / direction
    if direction < 0.0:
        return -1, (index - position) / direction, -1.0 / direction
    return 0, _INF, _INF


@njit(cache=False, parallel=True)
def _trace_kernel(solid, origins, directions, hot_map, cold_map, cell_face_map):
    """Cell-by-cell traversal (Amanatides-Woo) to the first solid face or wall."""

    n_rays = directions.shape[0]
    receiver = np.full(n_rays, -1, dtype=np.int32)
    failed = np.zeros(n_rays, dtype=np.uint8)
    nx, ny, nz = solid.shape
    for ray in prange(n_rays):
        ox, oy, oz = origins[ray, 0], origins[ray, 1], origins[ray, 2]
        dx, dy, dz = directions[ray, 0], directions[ray, 1], directions[ray, 2]
        if dx == 0.0:
            failed[ray] = 1
            continue
        i, j, k = _initial_cell(ox, dx), _initial_cell(oy, dy), _initial_cell(oz, dz)
        sx, tx, ddx = _axis_state(ox, dx, i)
        sy, ty, ddy = _axis_state(oy, dy, j)
        sz, tz, ddz = _axis_state(oz, dz, k)
        time = 0.0
        face = -1
        done = False
        crossing_scale = 8.0 * nx * (1.0 + (abs(dy) + abs(dz)) / abs(dx)) + 100.0
        max_steps = max(10 * solid.size + 10, int(min(crossing_scale, 100_000_000.0)))
        for _ in range(max_steps):
            if i < 0 or i >= nx:
                jj = int(math.floor(oy + dy * time)) % ny
                kk = int(math.floor(oz + dz * time)) % nz
                receiver[ray] = hot_map[jj, kk] if i < 0 else cold_map[jj, kk]
                done = True
                break
            jj, kk = j % ny, k % nz
            if solid[i, jj, kk] != 0:
                receiver[ray] = cell_face_map[(i * ny + jj) * nz + kk, face] if face >= 0 else -1
                done = True
                break
            next_time = min(tx, ty, tz)
            cross_x = abs(tx - next_time) <= _TIE_TOLERANCE
            cross_y = abs(ty - next_time) <= _TIE_TOLERANCE
            cross_z = abs(tz - next_time) <= _TIE_TOLERANCE
            if cross_x:
                i += sx
                tx += ddx
                face = 1 if sx > 0 else 0
            if cross_y:
                j += sy
                ty += ddy
                if not cross_x:
                    face = 3 if sy > 0 else 2
            if cross_z:
                k += sz
                tz += ddz
                if not cross_x and not cross_y:
                    face = 5 if sz > 0 else 4
            time = next_time
        if not done or receiver[ray] < 0:
            failed[ray] = 1
    return receiver, failed


def trace_first_hit(patches: SurfacePatches, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Index of the first patch hit by each ray."""

    receiver, failed = _trace_kernel(
        patches.solid, origins, directions,
        patches.hot_wall_patch, patches.cold_wall_patch, patches.cell_face_patch,
    )
    if np.any(failed):
        raise RuntimeError(f"{int(np.count_nonzero(failed))} rays did not resolve a receiving patch")
    return receiver


def first_hit_counts(
    patches: SurfacePatches, rays_per_patch: int, seed: int, *, patch_chunk: int = 16384
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer counts (origin, receiver, n) of one seeded ray block, sorted by edge."""

    n = patches.n_patches
    tables = []
    for start in range(0, n, int(patch_chunk)):
        end = min(start + int(patch_chunk), n)
        origins, directions = generate_rays(patches, start, end, rays_per_patch, seed)
        receiver = trace_first_hit(patches, origins, directions)
        origin = np.repeat(np.arange(start, end, dtype=np.int64), int(rays_per_patch))
        pair, count = np.unique(origin * n + receiver.astype(np.int64), return_counts=True)
        tables.append(((pair // n).astype(np.int32), (pair % n).astype(np.int32), count.astype(np.int64)))
    return pool_counts(tables, n)


def pool_counts(tables, n_patches: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum integer edge counts of several ray blocks into one sorted edge list."""

    n = int(n_patches)
    pair = np.concatenate([o.astype(np.int64) * n + r.astype(np.int64) for o, r, _ in tables])
    count = np.concatenate([np.asarray(c, dtype=np.int64) for _, _, c in tables])
    order = np.argsort(pair, kind="stable")
    pair, count = pair[order], count[order]
    start = np.r_[0, np.flatnonzero(pair[1:] != pair[:-1]) + 1]
    unique = pair[start]
    return (unique // n).astype(np.int32), (unique % n).astype(np.int32), np.add.reduceat(count, start)


def count_sha256(origin: np.ndarray, receiver: np.ndarray, count: np.ndarray, rays_per_origin: np.ndarray) -> str:
    """SHA256 of a pooled count table, comparable with the hashes listed in the case tables.

    The edge list must be sorted by (origin, receiver) as returned by
    ``pool_counts``.  For origin (int32), receiver (int32), count (int64)
    and rays per origin (int64), in this order, the digest takes the array
    name, the NumPy dtype string, the shape as int64 and the C-order bytes.
    """

    digest = hashlib.sha256()
    for name, array in (
        ("origin_patch_id", np.asarray(origin, dtype=np.int32)),
        ("receiver_patch_id", np.asarray(receiver, dtype=np.int32)),
        ("count", np.asarray(count, dtype=np.int64)),
        ("rays_per_origin", np.asarray(rays_per_origin, dtype=np.int64)),
    ):
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
