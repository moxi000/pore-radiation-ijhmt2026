"""Recompute values quoted in the article from the tables in results/ and data/.

Each output line gives the location in the article, the quantity, the printed
value and the value recomputed here, rounded as printed. With --restart
hit_point the property, penetration and reduced-model values are taken from
results/hit_point_restart/ (see the README).

    python scripts/check_paper_values.py
    python scripts/check_paper_values.py --restart hit_point --out check.csv
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from porerad.descriptors import directional_descriptors  # noqa: E402
from porerad.reduced_models import geometry_block_bootstrap, property_table  # noqa: E402

ROWS = []
S4 = ("OF", "FI", "FA", "TL")


def add(location, quantity, printed, value, fmt):
    ROWS.append({"location": location, "quantity": quantity, "printed": printed,
                 "recomputed": fmt.format(*value) if isinstance(value, tuple) else fmt.format(value)})


def coupled_checks(c, sets):
    canon = c[c.group == "canonical"]

    def q(s, axis="x", T=700.0, eps=0.6, col="q_total_hot_W_m2"):
        r = canon[(canon.structure == s) & (canon.transfer_axis == axis) & (canon.T_mean_K == T) & (canon.epsilon == eps)]
        return float(r[col].iloc[0])

    printed = {"OF": (144.2, 407.4, 551.6, 73.9, 0.276, 561.9, 595.4, 1.08),
               "FI": (184.6, 202.2, 386.8, 52.3, 0.193, 357.2, 363.7, 1.08),
               "FA": (196.7, 346.0, 542.7, 63.8, 0.271, 398.6, 389.5, 1.39),
               "TL": (181.5, 28.9, 210.5, 13.7, 0.105, 394.8, 408.3, 1.94)}
    for s in S4:
        p = printed[s]
        tot = [q(s, a) for a in "xyz"]
        values = (q(s, col="q_cond_hot_W_m2"), q(s, col="q_rad_hot_W_m2"), tot[0],
                  100 * q(s, col="radiative_fraction_hot"), q(s, col="k_eff_apparent_W_mK"), tot[1], tot[2], max(tot) / min(tot))
        names = ("q_cond", "q_rad", "q_total", "q_rad/q_total (%)", "k_eff_app", "q_total y", "q_total z", "q_max/q_min")
        fmts = ("{:.1f}", "{:.1f}", "{:.1f}", "{:.1f}", "{:.3f}", "{:.1f}", "{:.1f}", "{:.2f}")
        for name, pv, v, f in zip(names, p, values, fmts):
            add("Table 3", f"{s} {name}", pv, v, f)
    x = {s: q(s) for s in S4}
    add("Sec. 3.2", "spread of q_total between structures", 2.62, max(x.values()) / min(x.values()), "{:.2f}")
    add("Sec. 3.2", "TL q_rad y, z", "248.1, 228.1", (q("TL", "y", col="q_rad_hot_W_m2"), q("TL", "z", col="q_rad_hot_W_m2")), "{:.1f}, {:.1f}")
    add("Sec. 3.2", "TL q_rad max/min over axes", 8.6, q("TL", "y", col="q_rad_hot_W_m2") / q("TL", col="q_rad_hot_W_m2"), "{:.1f}")
    tl_cond = [q("TL", a, col="q_cond_hot_W_m2") for a in "xyz"]
    add("Sec. 3.2", "TL q_cond range over axes", "146.7-181.5", (min(tl_cond), max(tl_cond)), "{:.1f}-{:.1f}")

    def growth(s, key):
        lo = q(s, T=400.0) if key == "T" else q(s, eps=0.3)
        hi = q(s) if key == "T" else q(s, eps=0.9)
        rlo = q(s, T=400.0, col="q_rad_hot_W_m2") if key == "T" else q(s, eps=0.3, col="q_rad_hot_W_m2")
        rhi = q(s, col="q_rad_hot_W_m2") if key == "T" else q(s, eps=0.9, col="q_rad_hot_W_m2")
        return hi / lo, (rhi - rlo) / (hi - lo), hi - lo, rhi - rlo

    for s, f, share in zip(S4, (3.4, 2.9, 3.1, 2.4), (86, 67, 77, 81)):
        g, sh, _, _ = growth(s, "T")
        add("Sec. 3.2", f"{s} q_total 700 K / 400 K", f, g, "{:.1f}")
        if s == "TL":
            add("Sec. 3.2", "TL conductive share of the increase 400-700 K (%)", share, 100 * (1 - sh), "{:.0f}")
        else:
            add("Sec. 3.2", f"{s} radiative share of the increase 400-700 K (%)", share, 100 * sh, "{:.0f}")
    expo = []
    for s in S4:
        T = np.array([400.0, 550.0, 700.0])
        r = np.array([q(s, T=t, col="q_rad_hot_W_m2") for t in T])
        expo.append(np.polyfit(np.log(T), np.log(r), 1)[0])
    add("Sec. 3.2", "fitted exponent of q_rad over 400-700 K", "3.03-3.20", (min(expo), max(expo)), "{:.2f}-{:.2f}")
    for s, f in zip(S4, (2.4, 1.6, 1.9, 1.9)):
        add("Sec. 3.2", f"{s} q_total eps 0.9 / eps 0.3", f, growth(s, "eps")[0], "{:.1f}")
    dcond = [abs(q(s, eps=0.9, col="q_cond_hot_W_m2") - q(s, eps=0.3, col="q_cond_hot_W_m2")) for s in ("OF", "FI", "FA")]
    add("Sec. 3.2", "largest change of q_cond, eps 0.3-0.9, OF/FI/FA", 15.4, max(dcond), "{:.1f}")
    add("Sec. 3.2", "TL q_cond at eps 0.3 and 0.9", "133.3, 241.2",
        (q("TL", eps=0.3, col="q_cond_hot_W_m2"), q("TL", eps=0.9, col="q_cond_hot_W_m2")), "{:.1f}, {:.1f}")
    g, sh, _, drad = growth("TL", "eps")
    add("Sec. 3.2", "TL conductive share of the increase, eps 0.3-0.9 (%)", 78, 100 * (1 - sh), "{:.0f}")
    add("Sec. 3.2", "TL radiative increase, eps 0.3-0.9", 30.4, drad, "{:.1f}")
    x400 = {s: q(s, T=400.0) for s in S4}
    add("Sec. 3.2", "spread of q_total at 400 K", 2.02, max(x400.values()) / min(x400.values()), "{:.2f}")

    thick = c[c.group == "thickness"].set_index("structure")
    for s, r in zip(S4, (0.74, 0.59, 0.73, 0.53)):
        add("Sec. 3.2", f"{s} q_total 20 mm / 10 mm", r, float(thick.loc[s, "q_total_hot_W_m2"]) / q(s), "{:.2f}")
    for s, pv in (("OF", "0.74, 0.72"), ("TL", "0.53, 0.53")):
        add("Sec. 3.2", f"{s} q_rad and q_cond ratio 20 mm / 10 mm", pv,
            (float(thick.loc[s, "q_rad_hot_W_m2"]) / q(s, col="q_rad_hot_W_m2"),
             float(thick.loc[s, "q_cond_hot_W_m2"]) / q(s, col="q_cond_hot_W_m2")), "{:.2f}, {:.2f}")

    m = c[c.group == "matched"].set_index(["structure", "transfer_axis"])
    for col, name in (("q_total_hot_W_m2", "total"), ("q_rad_hot_W_m2", "radiative"), ("q_cond_hot_W_m2", "conductive")):
        dx = m.loc[("FA", "x"), col] - m.loc[("FI", "x"), col]
        dy = m.loc[("FA", "y"), col] - m.loc[("FI", "y"), col]
        pv = {"total": "+27.8, -85.2, -113.0", "radiative": "+21.5, -35.5, -57.0", "conductive": "+6.4, -49.7, -56.0"}[name]
        add("Sec. 3.3", f"matched pair FA-FI {name}: x, y, interaction", pv, (dx, dy, dy - dx), "{:+.1f}, {:+.1f}, {:+.1f}")
    add("Sec. 3.3", "matched pair q_total FA x, FI x, FA y, FI y", "397.7, 369.9, 247.3, 332.5",
        tuple(float(m.loc[k, "q_total_hot_W_m2"]) for k in (("FA", "x"), ("FI", "x"), ("FA", "y"), ("FI", "y"))),
        "{:.1f}, {:.1f}, {:.1f}, {:.1f}")

    res = c[c.group == "resolution"].set_index(["structure", "N_axis"])
    for s, pv in (("OF", "+0.27%, +0.42%"), ("FA", "-0.32%, +0.60%"), ("FI", "+1.60%, +1.81%"), ("TL", "-1.43%, -4.62%")):
        add("SI Table F1", f"{s} change of q_total at N = 160, 192", pv,
            tuple(100 * (float(res.loc[(s, n), "q_total_hot_W_m2"]) / q(s) - 1) for n in (160, 192)), "{:+.2f}%, {:+.2f}%")
    r192 = {s: float(res.loc[(s, 192), "q_total_hot_W_m2"]) for s in S4}
    add("SI Sec. F.1", "spread of q_total at N = 192", 2.76, max(r192.values()) / min(r192.values()), "{:.2f}")
    add("SI Sec. F.1", "TL radiative fraction at N = 192", 0.106, float(res.loc[("TL", 192), "radiative_fraction_hot"]), "{:.3f}")
    main = c[c.group != "resolution"]
    add("Sec. 2.5", "largest wall-flux mismatch", "<= 1.3e-5", main.wall_flux_mismatch_rel.max(), "{:.1e}")
    add("Sec. 2.5", "largest global energy residual", "<= 5e-10", main.energy_residual_rel.max(), "{:.1e}")
    add("Sec. 2.6", "independent ray sets, largest temperature and total-flux difference", "<= 0.06 K, <= 0.14%",
        (sets.max_temperature_difference_K.max(), 100 * sets.total_flux_rel_difference.max()), "{:.2f} K, {:.2f}%")


def prescribed_field_checks(pf, ms):
    main = pf[(pf.group == "medoid") & (pf.T_mean_K == 700.0) & (pf.epsilon == 0.6)].copy()
    main["abs_qh"] = main.q_h_W_m2.abs()
    ratio = main.groupby(["structure", "microstructure"]).abs_qh.agg(lambda s: s.max() / s.min())
    for s, pv in (("OF", "1.10 (1.06-1.29)"), ("FI", "1.07 (1.06-1.11)"), ("FA", "1.42 (1.32-1.73)"), ("TL", "4.96 (2.11-5.78)")):
        r = ratio.loc[s]
        add("Sec. 3.1", f"{s} median (range) of max/min q_h over axes", pv, (r.median(), r.min(), r.max()), "{:.2f} ({:.2f}-{:.2f})")
    tl = main[main.microstructure == "TL_core_phi0.10_s0"].set_index("transfer_axis")
    add("Sec. 3.1", "TL phi 0.10 q_h along x, y, z", "56.7, 281.1, 234.9", tuple(tl.q_h_W_m2), "{:.1f}, {:.1f}, {:.1f}")
    add("Sec. 3.1", "TL phi 0.10 x: q_h, q_c, q_i", "56.7, -133.9, 77.2",
        (tl.loc["x", "q_h_W_m2"], tl.loc["x", "q_c_W_m2"], tl.loc["x", "q_i_W_m2"]), "{:.1f}, {:.1f}, {:.1f}")
    of = main[(main.microstructure == "OF_core_phi0.10_s2") & (main.transfer_axis == "x")].iloc[0]
    add("Sec. 3.1", "OF phi 0.10 x: q_i and |q_c|", "44.0, 425.1", (of.q_i_W_m2, abs(of.q_c_W_m2)), "{:.1f}, {:.1f}")
    keys = pf[(pf.T_mean_K == 700.0) & (pf.epsilon == 0.6)].drop_duplicates("case_id")
    add("Sec. 2.6", "A/B sets, largest wall-channel difference over 48 cases: median, max", "0.10%, 0.62%",
        (100 * keys.ab_wall_channel_rel_difference.median(), 100 * keys.ab_wall_channel_rel_difference.max()), "{:.2f}%, {:.2f}%")

    mp = pf[pf.group == "matched"].set_index(["microstructure", "transfer_axis"]).q_h_W_m2
    for k, pv in zip((1, 2, 3), ("+30.4, -0.9, -31.3", "+13.1, -33.8, -46.9", "-12.5, -75.7, -63.2")):
        dx = mp[(f"FA_matched_pair{k}", "x")] - mp[(f"FI_matched_pair{k}", "x")]
        dy = mp[(f"FA_matched_pair{k}", "y")] - mp[(f"FI_matched_pair{k}", "y")]
        add("SI Table C4", f"pair {k} q_h contrast x, y, interaction", pv, (dx, dy, dy - dx), "{:+.1f}, {:+.1f}, {:+.1f}")
    msi = ms.set_index("id")
    for k, pv in zip((1, 2, 3), ("8.0e-05, 0.18%, 0.19%", "1.9e-04, 0.24%, 0.22%", "7.2e-04, 0.39%, 0.31%")):
        a, b = msi.loc[f"FA_matched_pair{k}"], msi.loc[f"FI_matched_pair{k}"]
        add("SI Table C5", f"pair {k} |d phi_s|, d A_s, d L_m", pv,
            (abs(a.realized_phi - b.realized_phi),
             100 * abs(a.interfacial_area_density_per_cell / b.interfacial_area_density_per_cell - 1),
             100 * abs(a.mean_beam_length_cells / b.mean_beam_length_cells - 1)), "{:.1e}, {:.2f}%, {:.2f}%")
    cr = [r.mean_pore_chord_x_cells / (0.5 * (r.mean_pore_chord_y_cells + r.mean_pore_chord_z_cells))
          for r in (msi.loc["FA_matched_pair1"], msi.loc["FI_matched_pair1"])]
    add("Sec. 3.3", "pair 1 mean pore chord along x / transverse, FA and FI", "5.3, 1.1", tuple(cr), "{:.1f}, {:.1f}")

    medoids = ms[ms.medoid == 1]
    rows = []
    for r in medoids.itertuples(index=False):
        d = directional_descriptors(np.load(ROOT / r.file)["phases"])
        for axis, dd in zip("xyz", d):
            rows.append({"microstructure": r.id, "structure": r.structure, "target_phi": r.target_phi, "transfer_axis": axis,
                         "f": dd["interface_normal_fraction_axis"], "lam": dd["lineal_integral_cells"],
                         "L64": dd["lineal_path_L064"], "q95": dd["pore_chord_cells_q95"], "mean": dd["pore_chord_cells_mean"]})
    d = pd.DataFrame(rows).merge(main[["microstructure", "transfer_axis", "abs_qh"]], on=["microstructure", "transfer_axis"])
    norm = d.groupby("microstructure").transform("mean", numeric_only=True)
    d["qh_n"] = d.abs_qh / norm.abs_qh
    add("Fig. 3 caption", "Spearman: f_a vs normalized q_h", -0.65, spearmanr(d.f, d.qh_n).statistic, "{:+.2f}")
    add("Fig. 3 caption", "Spearman: Lambda_a vs normalized q_h", 0.53, spearmanr(d.lam, d.qh_n).statistic, "{:+.2f}")
    add("Fig. 3 caption", "Spearman: l_95 vs |q_h|", 0.71, spearmanr(d.q95, d.abs_qh).statistic, "{:+.2f}")
    p10 = d[d.target_phi == 0.10].set_index(["structure", "transfer_axis"])
    for s, pv in (("OF", "76.4, 76.1, 76.9"), ("FI", "35.8, 37.6, 34.8"), ("FA", "109.2, 44.8, 44.1"), ("TL", "28.7, 103.1, 104.5")):
        add("Sec. 3.1", f"{s} phi 0.10 Lambda_a x, y, z", pv, tuple(p10.loc[(s, a), "lam"] for a in "xyz"), "{:.1f}, {:.1f}, {:.1f}")
    for s, pv in (("FA", "0.85, 0.29, 0.28"), ("TL", "0.17, 0.79, 0.81")):
        add("Sec. 3.1", f"{s} phi 0.10 L_a(64) x, y, z", pv, tuple(p10.loc[(s, a), "L64"] for a in "xyz"), "{:.2f}, {:.2f}, {:.2f}")
    add("Sec. 3.1", "FA phi 0.10 mean pore chord x, y, z", "113.6, 40.5, 42.0",
        tuple(p10.loc[("FA", a), "mean"] for a in "xyz"), "{:.1f}, {:.1f}, {:.1f}")
    add("Sec. 3.1", "FA phi 0.10 f_a x, y, z", "0.033, 0.49, 0.48", tuple(p10.loc[("FA", a), "f"] for a in "xyz"), "{:.3f}, {:.2f}, {:.2f}")
    add("Sec. 3.1", "TL phi 0.10 f_a x, y", "0.59, 0.20", (p10.loc[("TL", "x"), "f"], p10.loc[("TL", "y"), "f"]), "{:.2f}, {:.2f}")


def property_and_model_checks(props, rm, rng_medians, penetration, ms, restart):
    loc = "" if restart == "article" else " (hit_point restart)"
    props = props[props.microstructure.isin(ms.loc[ms.medoid == 1, "id"])]
    pt = property_table(props, ms)
    ratio = pt.groupby(["microstructure", "structure"]).beta_axis.agg(lambda s: s.max() / s.min()).groupby("structure")
    for s, pv in (("OF", "<= 1.2"), ("FI", "<= 1.2"), ("FA", "11.6-18.1"), ("TL", "8.3-27.9")):
        add("Sec. 2.4", f"{s} largest/smallest axis extinction", pv, (ratio.min()[s], ratio.max()[s]), "{:.1f}-{:.1f}")
    add("Sec. 2.4" + loc, "largest |omega - (1 - eps)|", "<= 0.0026", (props.omega - (1 - props.epsilon)).abs().max(), "{:.4f}")
    s = pt.drop_duplicates("microstructure")
    fac = 1 - 0.4 * s.g_scalar_eps06
    add("Sec. 2.4" + loc, "1 - omega g at 700 K, eps 0.6", "1.246-1.258", (fac.min(), fac.max()), "{:.3f}-{:.3f}")
    cauchy = s.beta_cauchy / (s.beta_scalar * fac)
    add("Sec. 2.4" + loc, "beta_C / transport extinction", "3-14", (cauchy.min(), cauchy.max()), "{:.0f}-{:.0f}")
    tau = s.beta_scalar * 128
    add("Sec. 3.4", "optical thickness tau", "0.6-5.1", (tau.min(), tau.max()), "{:.1f}-{:.1f}")

    boot = geometry_block_bootstrap(rm).set_index(["model", "property_protocol", "scattering"])

    def med(model, protocol, scattering):
        return 100 * boot.loc[(model, protocol, scattering), "median_E_bc"]

    add("Sec. 3.4", "DOM isotropic, scalar beta: median E_bc (%)", 28, med("DOM", "identified_scalar", "isotropic"), "{:.0f}")
    b = boot.loc[("DOM", "identified_scalar", "transport")]
    add("Sec. 3.4" + loc, "DOM transport, scalar beta: median E_bc and 95% interval (%)", "12 (7-27)",
        (100 * b.median_E_bc, 100 * b.ci95_low, 100 * b.ci95_high), "{:.0f} ({:.0f}-{:.0f})")
    tr = rm[(rm.model == "DOM") & (rm.scattering == "transport")]
    fam = tr[tr.property_protocol == "identified_scalar"].groupby("structure").E_bc.median()
    add("Sec. 3.4" + loc, "DOM transport structure medians OF, FI, FA, TL (%)", "7, 8, 23, 62",
        tuple(100 * fam[x] for x in S4), "{:.0f}, {:.0f}, {:.0f}, {:.0f}")
    iso = rm[(rm.model == "DOM") & (rm.scattering == "isotropic") & (rm.property_protocol == "identified_scalar")]
    k = ["microstructure", "transfer_axis"]
    j = tr[tr.property_protocol == "identified_scalar"].merge(iso, on=k, suffixes=("_t", "_i"))
    add("Sec. 3.4" + loc, "cases in which the transport correction lowers E_bc", "36 of 36", int((j.E_bc_t < j.E_bc_i).sum()), "{:d} of 36")
    p = tr.pivot_table(index=k, columns="property_protocol", values="E_bc")
    bt = tr.pivot_table(index=k, columns="property_protocol", values="beta_per_cell")
    rr = bt.identified_axis / bt.identified_scalar
    gain = p.identified_scalar - p.identified_axis
    add("Sec. 3.4" + loc, "blocked axes improved, open axes worsened", "8 of 8, 9 of 9",
        (int(((rr > 1.2) & (gain > 0)).sum()), int((rr > 1.2).sum()), int(((rr < 1 / 1.2) & (gain < 0)).sum()), int((rr < 1 / 1.2).sum())),
        "{:d} of {:d}, {:d} of {:d}")
    add("Sec. 3.4" + loc, "Spearman: beta ratio vs error reduction", 0.97, spearmanr(np.log10(rr), gain).statistic, "{:.2f}")
    add("Sec. 3.4" + loc, "DOM transport, axis beta: median E_bc (%)", 15.5, med("DOM", "identified_axis", "transport"), "{:.1f}")
    add("Sec. 3.4", "mean beam length estimate: median E_bc (%)", 32, med("MBL", "mean_beam_length", "none"), "{:.0f}")
    add("Sec. 3.4", "Rosseland, Cauchy: median E_bc (%)", 30, med("Rosseland", "cauchy", "none"), "{:.0f}")
    add("Sec. 3.4" + loc, "Rosseland, identified transport: median E_bc (%)", 295,
        med("Rosseland", "identified_scalar", "transport"), "{:.0f}")
    ros = rm[(rm.model == "Rosseland") & (rm.property_protocol == "identified_scalar")]
    add("Sec. 3.4" + loc, "Rosseland, identified: hot-channel over-prediction, cases and median factor", "36 of 36, about 4",
        (int((ros.pred_q_h_W_m2 > ros.ref_q_h_W_m2).sum()), (ros.pred_q_h_W_m2 / ros.ref_q_h_W_m2).median()), "{:d} of 36, {:.1f}")
    rc = rm[(rm.model == "Rosseland") & (rm.property_protocol == "cauchy")]
    add("Sec. 3.4", "Rosseland, Cauchy: median signed hot-wall deviation (%)", -6, 100 * rc.signed_hot_rel.median(), "{:.0f}")
    rmed = rng_medians[(rng_medians.model == "DOM") & (rng_medians.property_protocol == "identified_scalar")
                       & (rng_medians.scattering == "transport")].groupby("structure").median_E_bc
    for s, pv in (("OF", "7"), ("FI", "7-17"), ("FA", "15-24"), ("TL", "46-77")):
        lo, hi = 100 * rmed.min()[s], 100 * rmed.max()[s]
        add("Sec. 3.4" + loc, f"{s} range-check medians (%)", pv, (lo, hi), "{:.0f}" if round(lo) == round(hi) else "{:.0f}-{:.0f}")

    n_ret = int(penetration.retained.sum())
    add("Sec. 2.3" + loc, "penetration solutions retained of 432", 411, n_ret, "{:d}")
    add("Sec. 2.3" + loc, "condition combinations", 4932, 12 * n_ret, "{:d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restart", choices=("article", "hit_point"), default="article")
    ap.add_argument("--out", type=Path, help="also write the table to this CSV")
    args = ap.parse_args()
    res = ROOT / "results"
    alt = res if args.restart == "article" else res / "hit_point_restart"
    ms = pd.read_csv(ROOT / "data" / "microstructures.csv")

    coupled_checks(pd.read_csv(res / "coupled.csv"), pd.read_csv(res / "coupled_ray_set_comparison.csv"))
    prescribed_field_checks(pd.read_csv(res / "prescribed_field.csv"), ms)
    property_and_model_checks(pd.read_csv(alt / "radiative_properties.csv"), pd.read_csv(alt / "reduced_models_main.csv"),
                              pd.read_csv(alt / "reduced_models_range_medians.csv"),
                              pd.read_csv(alt / "penetration_response.csv"), ms, args.restart)

    table = pd.DataFrame(ROWS)
    if args.out:
        table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 80, "display.max_rows", 500):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
