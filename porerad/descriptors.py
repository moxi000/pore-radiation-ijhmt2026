"""Morphology descriptors of a two-phase digital microstructure.

Phase arrays hold 1 for solid and 0 for pore; lengths are in cells. The scalar
descriptors (phi_s, A_s, L_m, mean axial pore chords, pore percolation and the
largest pore cluster) treat the box faces as ends of the sample. The
directional descriptors (periodic chord-length distribution, lineal-path
function L_a(l), interface-normal fractions f_a) use the periodic cell.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99)
LINEAL_SAMPLES = (1, 2, 4, 8, 16, 32, 64, 128)


def _lines(mask, axis):
    """All axis-parallel lines of a 3-D boolean array as the rows of a 2-D array."""
    x = np.moveaxis(np.asarray(mask, bool), axis, -1)
    return x.reshape(int(np.prod(x.shape[:-1])), x.shape[-1])


def _runs(rows):
    """Lengths and row indices of the runs of True along each row."""
    M, N = rows.shape
    pad = np.zeros((M, N + 2), bool)
    pad[:, 1:-1] = rows
    d = np.diff(pad.astype(np.int8), axis=1)
    sr, sc = np.nonzero(d == 1)
    _, ec = np.nonzero(d == -1)
    return (ec - sc).astype(np.int32), sr.astype(np.int32)


# Scalar descriptors

def interface_area(solid):
    """Number of cell faces between solid and pore inside the box, A_int."""
    m = np.asarray(solid, bool)
    return (int(np.count_nonzero(m[:-1, :, :] != m[1:, :, :]))
            + int(np.count_nonzero(m[:, :-1, :] != m[:, 1:, :]))
            + int(np.count_nonzero(m[:, :, :-1] != m[:, :, 1:])))


def mean_pore_chords(pore):
    """Mean length of the pore runs along x, y and z, runs ending at the box faces."""
    return tuple(float(np.mean(_runs(_lines(pore, a))[0])) for a in range(3))


def pore_connectivity(pore):
    """Pore percolation along x, y, z and the largest-cluster fraction (26-connected)."""
    lbl, _ = label(pore, structure=np.ones((3, 3, 3), bool))
    perc = []
    for a in range(3):
        common = np.intersect1d(np.take(lbl, 0, axis=a), np.take(lbl, lbl.shape[a] - 1, axis=a))
        perc.append(bool(np.any(common != 0)))
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return tuple(perc), float(sizes.max() / max(1, int(pore.sum())))


def scalar_descriptors(phases):
    """phi_s, A_s = A_int/N^3, L_m = 4 V_pore/A_int, mean axial pore chords and pore connectivity."""
    p = np.asarray(phases)
    solid = p == 1
    pore = p == 0
    perc, largest = pore_connectivity(pore)
    area = interface_area(solid)
    return {
        "phi_solid": float(solid.mean()),
        "interfacial_area_density": float(area / p.size),
        "mean_beam_length": float(4.0 * float(np.sum(pore)) / float(area)),
        "mean_pore_chord": mean_pore_chords(pore),
        "pore_percolates": perc,
        "pore_largest_cluster_fraction": largest,
    }


# Directional descriptors

def periodic_chords(pore, axis):
    """Pore chords along one axis of the periodic cell.

    Runs touching both ends of a line are merged into one chord; a line lying
    entirely in the pore gives one chord of length N. Returns the chord lengths,
    the number of all-pore lines, the number of lines and N.
    """
    rows = _lines(pore, axis)
    M, N = rows.shape
    allp = rows.all(axis=1)
    lengths, run_rows = _runs(rows)
    keep = np.ones(len(lengths), bool)
    merged = []
    wrap = np.where(rows[:, 0] & rows[:, -1] & (~allp))[0]
    if len(wrap):
        order = np.argsort(run_rows, kind="stable")
        uniq, first, count = np.unique(run_rows[order], return_index=True, return_counts=True)
        pos = {int(r): (int(f), int(c)) for r, f, c in zip(uniq, first, count)}
        for r in wrap:
            f, c = pos[int(r)]
            inds = order[f:f + c]
            if len(inds) >= 2:
                i0, i1 = inds[0], inds[-1]
                merged.append(int(lengths[i0] + lengths[i1]))
                keep[i0] = False
                keep[i1] = False
    chords = np.concatenate([lengths[keep], np.asarray(merged, np.int32)])
    return chords, int(allp.sum()), M, N


def lineal_path(counts, n_all_pore, n_lines, N):
    """L_a(l), l = 1..N, from the chord-count histogram of the periodic cell."""
    lengths = np.arange(len(counts), dtype=np.int64)
    seg = np.arange(1, N + 1, dtype=np.int64)
    hits = (np.maximum(lengths[None, :] - seg[:, None] + 1, 0) * counts[None, :]).sum(axis=1)
    return (hits + n_all_pore * (seg - 1)) / (n_lines * N)


def interface_face_counts(solid):
    """Solid-pore faces of the periodic cell whose normal is parallel to x, y and z."""
    return np.array([np.count_nonzero(solid != np.roll(solid, -1, axis=i)) for i in range(3)],
                    dtype=np.int64)


def chord_length_distribution(counts):
    """Chord probability, survival probability and pore-length fraction for l = 1..N."""
    n = counts.sum()
    total_length = np.sum(np.arange(len(counts)) * counts)
    surv = np.cumsum(counts[::-1])[::-1]
    seg = np.arange(1, len(counts), dtype=np.int64)
    return counts[1:] / n, surv[1:] / n, seg * counts[1:] / total_length


def directional_descriptors(phases):
    """Per-axis directional descriptors for x, y, z.

    Returns three dicts with the summary values (Supplementary Data S1), the
    lineal-path function ``lineal_path`` (l = 1..N) and the chord-count
    histogram ``chord_counts`` (index = chord length in cells).
    """
    pore = np.asarray(phases) == 0
    solid = ~pore
    N = pore.shape[0]
    fc = interface_face_counts(solid)
    f = fc / fc.sum()
    out = []
    for axis in range(3):
        chords, allp, nlines, n = periodic_chords(pore, axis)
        counts = np.bincount(chords, minlength=N + 1)[:N + 1]
        lp = lineal_path(counts, allp, nlines, n)
        vals = chords.astype(float)
        q = np.quantile(vals, list(QUANTILES))
        below = lp <= np.exp(-1)
        row = {
            "interface_face_count_axis": int(fc[axis]),
            "interface_face_count_total": int(fc.sum()),
            "interface_normal_fraction_axis": float(f[axis]),
            "interface_normal_anisotropy": float(f.max() / f.min()),
            "all_pore_line_count": int(allp),
            "total_axis_lines": int(nlines),
            "all_pore_line_fraction": float(allp / nlines),
            "lineal_integral_cells": float(lp.sum()),
            "lineal_1e_length_cells": float(np.where(below)[0][0] + 1) if np.any(below) else float(N),
            "pore_chord_cells_count": int(len(chords)),
            "pore_chord_cells_mean": float(vals.mean()),
            "pore_chord_cells_std": float(vals.std()),
            "pore_chord_cells_max": float(vals.max()),
            "pore_chord_cells_cv": float(vals.std() / vals.mean()),
        }
        for qq, v in zip(QUANTILES, q):
            row[f"pore_chord_cells_q{int(round(100 * qq))}"] = float(v)
        row["pore_chord_cells_q95_over_mean"] = float(q[3] / vals.mean())
        row["pore_chord_cells_q99_over_mean"] = float(q[4] / vals.mean())
        for l in LINEAL_SAMPLES:
            row[f"lineal_path_L{l:03d}"] = float(lp[l - 1])
        row["lineal_path"] = lp
        row["chord_counts"] = counts
        out.append(row)
    return out
