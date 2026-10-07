"""Regenerate the digital microstructures listed in data/microstructures.csv.

Every array is rebuilt from the generators in porerad.geometry: the 48 core
microstructures (seeds tried in turn until four per structure and solid fraction
satisfy the acceptance rules), the canonical microstructures at N = 128, 160 and
192, and the three matched fiber pairs. The SHA256 of each array (uint8, C
order) is compared with the column phase_sha256, and every verified array is
written to <outdir>/<id>.npz (compressed, key "phases"). The column file lists
the arrays shipped with the repository; the canonical arrays at N = 160 and
192 are produced here on demand.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from porerad import geometry  # noqa: E402

SETS = ("core", "canonical", "matched")


def _core(structure, phi):
    return [(f"{structure}_core_phi{phi:.2f}_s{seed}", p)
            for seed, p in geometry.core_realizations(structure, phi)]


def _canonical(structure, N):
    return [(f"{structure}_canonical_phi0.10_N{N}", geometry.canonical(structure, N))]


def _matched(structure, pair):
    return [(f"{structure}_matched_pair{pair}", geometry.matched(structure, pair))]


def _jobs(selected):
    jobs = []
    if "core" in selected:
        jobs += [(_core, s, phi) for s in geometry.STRUCTURES for phi in geometry.CORE_PHIS]
    if "canonical" in selected:
        jobs += [(_canonical, s, N) for s in geometry.STRUCTURES for N in (128, 160, 192)]
    if "matched" in selected:
        jobs += [(_matched, s, k) for k in (1, 2, 3) for s in ("FA", "FI")]
    return sorted(jobs, key=lambda j: j[1] != "OF")  # the foam takes longest


def _run(job):
    f, a, b = job
    return f(a, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=("all",) + SETS, default="all", help="which microstructures to rebuild")
    ap.add_argument("--outdir", type=Path, default=ROOT / "data" / "microstructures",
                    help="directory for the .npz files")
    args = ap.parse_args()

    selected = SETS if args.set == "all" else (args.set,)
    table = pd.read_csv(ROOT / "data" / "microstructures.csv", keep_default_na=False).set_index("id")
    expected = set(table.index[table["set"].isin(selected)])
    args.outdir.mkdir(parents=True, exist_ok=True)

    verified = set()
    with ProcessPoolExecutor() as ex:
        for result in ex.map(_run, _jobs(selected)):
            for mid, phases in result:
                phases = np.ascontiguousarray(phases, dtype=np.uint8)
                digest = hashlib.sha256(phases.tobytes(order="C")).hexdigest()
                ok = mid in expected and digest == table.at[mid, "phase_sha256"]
                if ok:
                    verified.add(mid)
                    np.savez_compressed(args.outdir / f"{mid}.npz", phases=phases)
                print(f"{mid:32s} {'ok' if ok else 'MISMATCH'}", flush=True)

    missing = sorted(expected - verified)
    print(f"{len(verified)} of {len(expected)} microstructures verified")
    if missing:
        print("not reproduced: " + ", ".join(missing))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
