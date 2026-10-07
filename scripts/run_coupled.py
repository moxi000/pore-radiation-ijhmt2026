"""Solve one strictly coupled radiation-conduction case listed in data/cases_coupled.csv.

The rays of the two independently seeded ray sets A and B are regenerated
from the stored seeds, the projected view-factor operator of the combined
sets is built, and the steady coupled field is solved from the pure-conduction
field.  The integer first-hit counts of each set are checked against the
hashes stored in the case table where these are listed.  With --ab the case
is also solved with set A alone and with set B alone and their differences
are reported.

    python scripts/run_coupled.py coupled_OF_x_T700_eps0.6_N128 --out coupled_rerun.csv
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

from porerad import conduction, coupled, exchange, view_factors  # noqa: E402
from porerad.constants import K_GAS, K_SOLID  # noqa: E402

COLUMNS = [
    "case_id", "group", "structure", "microstructure", "transfer_axis", "H_m", "N_axis", "T_mean_K", "delta_T_K",
    "epsilon", "q_cond_hot_W_m2", "q_rad_hot_W_m2", "q_total_hot_W_m2", "q_cond_cold_W_m2", "q_rad_cold_W_m2",
    "q_total_cold_W_m2", "radiative_fraction_hot", "k_eff_apparent_W_mK", "wall_flux_mismatch_rel",
    "energy_residual_rel", "newton_updates", "T_min_K", "T_max_K",
]
START = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - START:8.1f} s] {message}", file=sys.stderr, flush=True)


def read_case(case_id: str) -> dict[str, str]:
    with open(ROOT / "data" / "cases_coupled.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == case_id:
                return row
    raise SystemExit(f"unknown case {case_id!r}; see data/cases_coupled.csv")


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


def pooled_counts(patches, counts, seeds, rays_per_patch, expected_sha256=""):
    """Counts of several ray blocks summed into one set, checked against a stored hash if given."""

    origin, receiver, count = view_factors.pool_counts([counts[s] for s in seeds], patches.n_patches)
    rays = np.full(patches.n_patches, rays_per_patch * len(seeds), dtype=np.int64)
    if expected_sha256:
        match = view_factors.count_sha256(origin, receiver, count, rays) == expected_sha256
        log(f"first-hit counts of seeds {seeds}: {'identical to' if match else 'DIFFERENT FROM'} the stored hash")
    return origin, receiver, count, rays


def build_operator(patches, pooled):
    operator = exchange.ExchangeOperator(*pooled, patches.patch_area_m2)
    log(f"operator: {pooled[2].size} sampled patch pairs, balancing {operator.balance_method} "
        f"({operator.balance_iterations} iterations), largest row-sum deviation {operator.row_sum_error:.2e}, "
        f"relative reciprocity deviation {operator.reciprocity_error():.2e}")
    return operator


def solve(system, operator, patches, T_hot, T_cold, epsilon, T_initial):
    result = coupled.solve_coupled(system, operator, patches, T_hot, T_cold, epsilon, T_initial)
    metrics = coupled.wall_flux_metrics(system, patches, result, T_hot, T_cold)
    log(f"coupled solve: converged {result.converged}, {result.newton_iterations} Newton updates, "
        f"FGMRES iterations {[s['gmres_iterations'] for s in result.linear_solves]}, "
        f"{result.preconditioner_steps} preconditioner steps, final residual {result.final_residual_inf_W:.2e} W")
    failures = coupled.acceptance_failures(result, metrics)
    if result.failure:
        failures.insert(0, result.failure)
    if failures:
        raise SystemExit("solution not accepted: " + "; ".join(failures))
    return result, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case_id")
    parser.add_argument("--ab", action="store_true", help="also solve with ray sets A and B separately")
    parser.add_argument("--out", type=Path, help="result CSV (rows of the same case are replaced)")
    parser.add_argument("--microstructure-dir", type=Path, default=ROOT / "data" / "microstructures")
    args = parser.parse_args()

    case = read_case(args.case_id)
    axis, H, eps = case["transfer_axis"], float(case["H_m"]), float(case["epsilon"])
    T_mean, delta_T = float(case["T_mean_K"]), float(case["delta_T_K"])
    T_hot, T_cold = T_mean + delta_T / 2.0, T_mean - delta_T / 2.0
    rpp = int(case["rays_per_patch_per_block"])
    seeds = {name: [int(s) for s in case[f"seeds_{name}"].split(";")] for name in ("A", "B")}

    phases = view_factors.orient_along_x(load_phases(args.microstructure_dir, case["microstructure"]), axis)
    phases = np.ascontiguousarray(np.concatenate([phases] * int(case["repeats_along_axis"]), axis=0))
    if phases.shape[0] != int(case["N_axis"]):
        raise SystemExit(f"microstructure has {phases.shape[0]} cells along the transfer axis")
    if hashlib.sha256(phases.tobytes()).hexdigest() != case["phase_sha256_x"]:
        raise SystemExit("phase array differs from the one listed for this case")
    patches = view_factors.build_surface_patches(phases, H)
    log(f"{args.case_id}: {phases.shape} cells, {patches.n_patches} surface patches")

    counts = {s: view_factors.first_hit_counts(patches, rpp, s) for s in seeds["A"] + seeds["B"]}
    log(f"traced {len(counts)} ray blocks of {rpp} rays per patch")
    sets = {name: pooled_counts(patches, counts, seeds[name], rpp, case[f"raw_count_sha256_{name}"]) for name in "AB"}
    final = build_operator(patches, pooled_counts(patches, counts, seeds["A"] + seeds["B"], rpp))
    del counts

    system = conduction.assemble_conduction(np.where(phases == 1, K_SOLID, K_GAS), H)
    T_initial = conduction.solve_conduction(system, T_hot, T_cold)
    log("pure-conduction field solved")
    result, metrics = solve(system, final, patches, T_hot, T_cold, eps, T_initial)

    row = {key: case[key] for key in ("case_id", "group", "structure", "microstructure", "transfer_axis", "H_m",
                                      "N_axis", "T_mean_K", "delta_T_K", "epsilon")}
    row.update({key: metrics[key] for key in COLUMNS if key in metrics})
    row["newton_updates"] = result.newton_iterations
    write_rows(args.out, [row])
    if args.ab:
        solved = {name: solve(system, build_operator(patches, sets[name]), patches, T_hot, T_cold, eps, T_initial)
                  for name in "AB"}
        (result_a, metrics_a), (result_b, metrics_b) = solved["A"], solved["B"]
        dT = float(np.max(np.abs(result_a.temperature_K - result_b.temperature_K)))
        dq = coupled.relative_difference(metrics_a["q_total_hot_W_m2"], metrics_b["q_total_hot_W_m2"])
        print(f"ray sets A and B: largest temperature difference {dT:.4g} K, "
              f"relative difference of the total flux {dq:.4g}")
    log("done")


if __name__ == "__main__":
    main()
