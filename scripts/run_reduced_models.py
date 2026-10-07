"""Evaluate the reduced radiation models against the prescribed-field reference.

--set main   36 geometry-axis cases at T_m = 700 K, eps = 0.6 (400 cells, 32 ordinates per hemisphere).
             Writes the predictions and E_bc of every case (as results/reduced_models_main.csv) and
             prints the median E_bc of each combination with its 95% geometry-block bootstrap interval.
--set range  all 324 reference conditions, T_m = 400-1000 K and eps = 0.3-0.9 (160 cells, 16 ordinates).
             Writes the median E_bc per structure, condition and combination
             (as results/reduced_models_range_medians.csv).

Inputs are results/prescribed_field.csv, the radiative properties and data/microstructures.csv.
With --restart hit_point the property table of results/hit_point_restart/ is used (see
porerad/properties.py for the two restart rules). Outputs go to output/ unless --out is given.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from porerad.reduced_models import Channels, geometry_block_bootstrap, predict, property_table  # noqa: E402

SETTINGS = {"main": dict(n_cells=400, n_half=32, tol=1e-11), "range": dict(n_cells=160, n_half=16, tol=1e-10)}


def evaluate(reference, props, cfg):
    pmap = {(r.microstructure, r.transfer_axis): r._asdict() for r in props.itertuples(index=False)}
    rows = []
    for r in reference.itertuples(index=False):
        p = pmap[(r.microstructure, r.transfer_axis)]
        ref = Channels(float(r.q_h_W_m2), float(r.q_c_W_m2), float(r.q_i_W_m2))
        base = {"microstructure": r.microstructure, "structure": r.structure, "target_phi": p["target_phi"],
                "transfer_axis": r.transfer_axis, "T_mean_K": r.T_mean_K, "epsilon": r.epsilon,
                "ref_q_h_W_m2": ref.q_h, "ref_q_c_W_m2": ref.q_c, "ref_q_i_W_m2": ref.q_i}
        for pred in predict(p, ref, float(r.T_mean_K), float(r.epsilon), **cfg):
            rows.append(base | pred)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETTINGS), default="main")
    ap.add_argument("--out", type=Path, help="output CSV (default: a file in output/)")
    ap.add_argument("--restart", choices=("article", "hit_point"), default="article",
                    help="which property table to use")
    args = ap.parse_args()
    results = ROOT / "results" if args.restart == "article" else ROOT / "results" / "hit_point_restart"
    out_dir = ROOT / "output" if args.restart == "article" else ROOT / "output" / "hit_point_restart"

    reference = pd.read_csv(ROOT / "results" / "prescribed_field.csv")
    reference = reference[reference.group == "medoid"]
    if args.set == "main":
        reference = reference[(reference.T_mean_K == 700.0) & (reference.epsilon == 0.6)]
    reference = reference.sort_values(["structure", "microstructure", "transfer_axis", "T_mean_K", "epsilon"])
    props = property_table(pd.read_csv(results / "radiative_properties.csv"),
                           pd.read_csv(ROOT / "data" / "microstructures.csv"))
    table = evaluate(reference, props, SETTINGS[args.set])

    if args.set == "main":
        out = args.out or out_dir / "reduced_models_main.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        summary = geometry_block_bootstrap(table)
        with pd.option_context("display.width", 200):
            print(summary.to_string(index=False, float_format=lambda v: f"{100 * v:.1f}%"))
    else:
        medians = (table.groupby(["model", "property_protocol", "scattering", "structure", "T_mean_K", "epsilon"],
                                 as_index=False)
                   .agg(n_cases=("E_bc", "size"), median_E_bc=("E_bc", "median"), max_E_bc=("E_bc", "max")))
        out = args.out or out_dir / "reduced_models_range_medians.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        medians.to_csv(out, index=False)
    print(f"{len(reference)} reference cases, {len(table)} predictions -> {out}")


if __name__ == "__main__":
    main()
