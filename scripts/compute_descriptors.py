"""Directional morphology descriptors of the 48 core microstructures.

Writes the four tables of the article's Supplementary Data: S1 per-axis
summaries (144 rows), S2 lineal-path functions L_a(l) (18432 rows), S3 periodic
pore chord-length distributions (18432 rows) and S4 interface-normal
statistics (144 rows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from porerad.descriptors import chord_length_distribution, directional_descriptors  # noqa: E402

ORDER = ["family", "target_phi", "seed", "axis"]
NORMAL_COLUMNS = ["interface_face_count_axis", "interface_face_count_total",
                  "interface_normal_fraction_axis", "interface_normal_anisotropy"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "descriptors",
                    help="directory for the four CSV tables")
    args = ap.parse_args()

    table = pd.read_csv(ROOT / "data" / "microstructures.csv", keep_default_na=False)
    summary, lineal, chords = [], [], []
    for r in table[table["set"] == "core"].itertuples(index=False):
        phases = np.load(ROOT / r.file)["phases"]
        for axis, d in zip("xyz", directional_descriptors(phases)):
            key = {"label": r.id, "family": r.structure, "target_phi": r.target_phi, "seed": r.seed,
                   "N": r.N, "axis": axis}
            lp, counts = d.pop("lineal_path"), d.pop("chord_counts")
            l = np.arange(1, len(lp) + 1)
            prob, surv, frac = chord_length_distribution(counts)
            summary.append({**key, **d})
            lineal.append(pd.DataFrame({**key, "segment_length_cells": l, "lineal_path_probability": lp}))
            chords.append(pd.DataFrame({**key, "chord_length_cells": l, "chord_count": counts[1:],
                                        "chord_probability": prob, "chord_survival_probability": surv,
                                        "pore_length_fraction": frac}))

    s1 = pd.DataFrame(summary).sort_values(ORDER).reset_index(drop=True)
    s2 = pd.concat(lineal).sort_values(ORDER + ["segment_length_cells"]).reset_index(drop=True)
    s3 = pd.concat(chords).sort_values(ORDER + ["chord_length_cells"]).reset_index(drop=True)
    s4 = s1[["label", "family", "target_phi", "seed", "N", "axis"] + NORMAL_COLUMNS]

    args.outdir.mkdir(parents=True, exist_ok=True)
    s1.to_csv(args.outdir / "S1_directional_descriptor_summaries.csv", index=False)
    s2.to_csv(args.outdir / "S2_lineal_path_functions.csv", index=False)
    s3.to_csv(args.outdir / "S3_pore_chord_length_distributions.csv", index=False)
    s4.to_csv(args.outdir / "S4_interface_normal_statistics.csv", index=False)
    print(f"{len(s1)} summaries, {len(s2)} lineal-path rows, {len(s3)} chord rows written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
