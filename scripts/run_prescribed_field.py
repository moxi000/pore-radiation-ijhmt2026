"""Radiation channels of one prescribed-field case listed in data/cases_prescribed_field.csv.

The two walls are held at T_m +/- dT/2 with dT = 20 K and every internal solid
surface takes the linear temperature field between them.  The rays of the
two ray sets A and B (and of the appended blocks where the case table records
an append) are regenerated from the stored seeds, and q_h, q_c and q_i are
evaluated on the projected view-factor operator of all sets combined, for
every condition (T_m, eps) of the case.  With --ab both sets are also
evaluated on their own, the append rule is applied to their difference, and
the largest relative A/B difference of the wall channels q_h and q_c over the
conditions of the case is reported.

    python scripts/run_prescribed_field.py pf_OF_core_phi0.10_s2_x --out prescribed_rerun.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from porerad import coupled, exchange, view_factors  # noqa: E402

DELTA_T_K = 20.0
COLUMNS = [
    "case_id", "group", "structure", "microstructure", "transfer_axis", "T_mean_K", "delta_T_K", "epsilon",
    "q_h_W_m2", "q_c_W_m2", "q_i_W_m2", "ab_wall_channel_rel_difference",
]
START = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - START:8.1f} s] {message}", file=sys.stderr, flush=True)


def read_case(case_id: str) -> dict[str, str]:
    with open(ROOT / "data" / "cases_prescribed_field.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == case_id:
                return row
    raise SystemExit(f"unknown case {case_id!r}; see data/cases_prescribed_field.csv")


def load_phases(directory: Path, microstructure: str) -> np.ndarray:
    path = directory / f"{microstructure}.npz"
    if not path.is_file():
        raise SystemExit(f"{path} not found; generate it with scripts/make_microstructures.py")
    with np.load(path) as stored:
        return np.asarray(stored["phases"], dtype=np.uint8)


def write_rows(path: Path | None, rows: list[dict]) -> None:
    """Write rows to stdout, or merge them into a CSV file replacing rows of the same case."""

    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return
    kept = []
    if path.is_file():
        with open(path, newline="") as handle:
            kept = [r for r in csv.DictReader(handle) if r["case_id"] not in {row["case_id"] for row in rows}]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept + rows)


def build_operator(patches, counts, seeds, rays_per_patch):
    """Projected operator of the given ray blocks; blocks not yet traced are traced first."""

    for seed in seeds:
        if seed not in counts:
            counts[seed] = view_factors.first_hit_counts(patches, rays_per_patch, seed)
            log(f"traced ray block of seed {seed}")
    origin, receiver, count = view_factors.pool_counts([counts[s] for s in seeds], patches.n_patches)
    rays = np.full(patches.n_patches, rays_per_patch * len(seeds), dtype=np.int64)
    operator = exchange.ExchangeOperator(origin, receiver, count, rays, patches.patch_area_m2)
    log(f"operator of {len(seeds)} blocks: {count.size} sampled patch pairs, balancing {operator.balance_method} "
        f"({operator.balance_iterations} iterations), largest row-sum deviation {operator.row_sum_error:.2e}, "
        f"relative reciprocity deviation {operator.reciprocity_error():.2e}")
    return operator


def channels(operator, patches, conditions):
    return [coupled.prescribed_field_channels(operator, patches, T, DELTA_T_K, eps) for T, eps in conditions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case_id")
    parser.add_argument("--ab", action="store_true", help="also evaluate ray sets A and B separately")
    parser.add_argument("--out", type=Path, help="result CSV (rows of the same case are replaced)")
    parser.add_argument("--microstructure-dir", type=Path, default=ROOT / "data" / "microstructures")
    args = parser.parse_args()

    case = read_case(args.case_id)
    rpp = int(case["rays_per_patch_per_block"])
    seeds = {key: [int(s) for s in case[key].split(";")]
             for key in ("seeds_A", "seeds_B", "append_seeds_A", "append_seeds_B")}
    append = case["append_used"] == "true"
    conditions = [(float(T), float(eps)) for T, eps in (item.split("/") for item in case["conditions"].split(";"))]

    phases = view_factors.orient_along_x(load_phases(args.microstructure_dir, case["microstructure"]),
                                         case["transfer_axis"])
    if hashlib.sha256(phases.tobytes()).hexdigest() != case["phase_sha256_x"]:
        raise SystemExit("phase array differs from the one listed for this case")
    patches = view_factors.build_surface_patches(phases, float(case["H_m"]))
    log(f"{args.case_id}: {phases.shape} cells, {patches.n_patches} surface patches")

    counts = {}
    ab_difference = ""
    if args.ab:
        set_a = channels(build_operator(patches, counts, seeds["seeds_A"], rpp), patches, conditions)
        set_b = channels(build_operator(patches, counts, seeds["seeds_B"], rpp), patches, conditions)
        channel, source = coupled.ab_differences(set_a, set_b)
        append_rule = coupled.append_required(channel, source)
        log(f"sets A and B: wall-channel difference {channel:.3e}, coarse-source L1 difference {source:.3e}, "
            f"append {'required' if append_rule else 'not required'}")
        if append_rule != append:
            log("the append decision differs from the case table; the decision of the rule is used")
            append = append_rule
        if append:
            set_a = channels(build_operator(patches, counts, seeds["seeds_A"] + seeds["append_seeds_A"], rpp),
                             patches, conditions)
            set_b = channels(build_operator(patches, counts, seeds["seeds_B"] + seeds["append_seeds_B"], rpp),
                             patches, conditions)
            channel, source = coupled.ab_differences(set_a, set_b)
            log(f"appended sets: wall-channel difference {channel:.3e}, coarse-source L1 difference {source:.3e}")
        ab_difference = channel

    final_seeds = seeds["seeds_A"] + seeds["seeds_B"]
    if append:
        final_seeds += seeds["append_seeds_A"] + seeds["append_seeds_B"]
    final = build_operator(patches, counts, final_seeds, rpp)
    rows = []
    for result in channels(final, patches, conditions):
        if result.closure_rel > 1.0e-4:
            raise SystemExit(f"channel balance not closed: {result.closure_rel:.2e}")
        rows.append({
            **{key: case[key] for key in ("case_id", "group", "structure", "microstructure", "transfer_axis")},
            "T_mean_K": f"{result.T_mean_K:g}", "delta_T_K": f"{DELTA_T_K:g}",
            "epsilon": result.epsilon, "q_h_W_m2": result.q_h, "q_c_W_m2": result.q_c, "q_i_W_m2": result.q_i,
            "ab_wall_channel_rel_difference": ab_difference,
        })
        log(f"T_m {result.T_mean_K:g} K, eps {result.epsilon:g}: q_h {result.q_h:.6f}, q_c {result.q_c:.6f}, "
            f"q_i {result.q_i:.6f} W m^-2, balance {result.closure_rel:.1e}")
    write_rows(args.out, rows)
    log("done")


if __name__ == "__main__":
    main()
