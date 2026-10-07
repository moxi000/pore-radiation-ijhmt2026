"""Penetration response R, T, A of one microstructure, transfer axis and emissivity.

Two independent seeds of 500 000 rays each are averaged. The output row has the
columns of results/penetration_response.csv (--restart article, the default)
and results/hit_point_restart/penetration_response.csv (--restart hit_point);
the column `retained` is copied from that table, since the Monte Carlo
tolerance checks of the Supplementary Material use all ray-number tiers and
seeds of the penetration calculations. The penetration response at a mean
temperature T_m with wall temperature difference dT follows as q_pen = q_pp T
(porerad.penetration.penetration_flux).

    python scripts/run_penetration.py --microstructure OF_core_phi0.10_s0 --axis x --epsilon 0.6
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from porerad.penetration import penetration_flux, penetration_response  # noqa: E402

TABLES = {"article": ROOT / "results" / "penetration_response.csv",
          "hit_point": ROOT / "results" / "hit_point_restart" / "penetration_response.csv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--microstructure", required=True)
    ap.add_argument("--axis", required=True, choices=("x", "y", "z"))
    ap.add_argument("--epsilon", required=True, type=float, choices=(0.3, 0.6, 0.9))
    ap.add_argument("--restart", choices=("article", "hit_point"), default="article",
                    help="restart rule for reflected rays (see porerad/penetration.py)")
    ap.add_argument("--out", type=Path, help="output CSV (default: print)")
    args = ap.parse_args()

    index = pd.read_csv(ROOT / "data" / "microstructures.csv").set_index("id")
    meta = index.loc[args.microstructure]
    phases = np.load(ROOT / meta["file"])["phases"]
    p = penetration_response(phases, args.axis, args.epsilon, restart=args.restart)

    retained = ""
    if TABLES[args.restart].is_file():
        table = pd.read_csv(TABLES[args.restart])
        match = table[(table.microstructure == args.microstructure) & (table.transfer_axis == args.axis)
                      & np.isclose(table.epsilon, args.epsilon)]
        if len(match) == 1:
            retained = bool(match.retained.iloc[0])
    row = pd.DataFrame([{
        "microstructure": args.microstructure, "structure": meta["structure"], "target_phi": meta["target_phi"],
        "transfer_axis": args.axis, "epsilon": args.epsilon, "n_rays": p["n_rays"],
        "reflectance": p["reflectance"], "reflectance_se": p["reflectance_se"],
        "transmittance": p["transmittance"], "transmittance_se": p["transmittance_se"],
        "absorptance": p["absorptance"], "absorptance_se": p["absorptance_se"], "retained": retained}])
    if args.out:
        row.to_csv(args.out, index=False)
    print(row.to_string(index=False))
    q = penetration_flux(p["transmittance"], args.epsilon, T_mean=700.0, delta_T=10.0)
    print(f"q_pen at T_m = 700 K and dT = 10 K is {q:.4f} W m^-2; {p['terminated']} rays reached the reflection limit")


if __name__ == "__main__":
    main()
