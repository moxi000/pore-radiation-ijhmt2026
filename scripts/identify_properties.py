"""Identify beta, omega and g for one microstructure, transfer axis and emissivity.

Two independent seeds of 200 000 rays are averaged. The output row has the
columns of results/radiative_properties.csv (--restart article, the default)
and results/hit_point_restart/radiative_properties.csv (--restart hit_point).

    python scripts/identify_properties.py --microstructure TL_core_phi0.10_s0 --axis x --epsilon 0.6
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from porerad.properties import identify_properties  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--microstructure", required=True)
    ap.add_argument("--axis", required=True, choices=("x", "y", "z"))
    ap.add_argument("--epsilon", required=True, type=float, choices=(0.3, 0.6, 0.9))
    ap.add_argument("--restart", choices=("article", "hit_point"), default="article",
                    help="restart rule for reflected rays (see porerad/properties.py)")
    ap.add_argument("--out", type=Path, help="output CSV (default: print)")
    args = ap.parse_args()

    index = pd.read_csv(ROOT / "data" / "microstructures.csv").set_index("id")
    meta = index.loc[args.microstructure]
    phases = np.load(ROOT / meta["file"])["phases"]
    p = identify_properties(phases, args.axis, args.epsilon, restart=args.restart)
    table = pd.DataFrame([{
        "microstructure": args.microstructure, "structure": meta["structure"], "target_phi": meta["target_phi"],
        "transfer_axis": args.axis, "epsilon": args.epsilon, "n_rays": p["n_rays"],
        "beta_per_cell": p["beta"], "beta_se_per_cell": p["beta_se"],
        "beta_ci95_low_per_cell": p["beta_ci95_low"], "beta_ci95_high_per_cell": p["beta_ci95_high"],
        "omega": p["omega"], "omega_se": p["omega_se"], "g": p["g"], "g_se": p["g_se"]}])
    if args.out:
        table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
