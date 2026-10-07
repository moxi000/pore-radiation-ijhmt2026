"""Reduced radiation models assessed against the pore-scale prescribed-field response.

All models are evaluated for a homogenized gray slab of N cells under the same
prescribed linear temperature field and gray diffuse walls as the pore-scale
reference. Channels follow the sign convention of the paper: q_h > 0 is net
emission from the hot wall, q_c < 0 net absorption at the cold wall, q_i the net
exchange of the medium, and q_h + q_c + q_i = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import SIGMA

EPSILONS = (0.3, 0.6, 0.9)

# The nine combinations of model, property protocol and scattering treatment,
# in the order used for the geometry-block bootstrap.
COMBINATIONS = (
    ("DOM", "cauchy", "isotropic"),
    ("DOM", "identified_axis", "isotropic"),
    ("DOM", "identified_axis", "transport"),
    ("DOM", "identified_scalar", "isotropic"),
    ("DOM", "identified_scalar", "transport"),
    ("MBL", "mean_beam_length", "none"),
    ("Rosseland", "cauchy", "none"),
    ("Rosseland", "identified_axis", "transport"),
    ("Rosseland", "identified_scalar", "transport"),
)


@dataclass
class Channels:
    q_h: float
    q_c: float
    q_i: float


def _half_range_ordinates(n_half: int):
    """Double Gauss-Legendre quadrature with n_half ordinates on each hemisphere."""
    x, w = np.polynomial.legendre.leggauss(n_half)
    mu_pos = 0.5 * (x + 1.0)
    w_pos = 0.5 * w
    mu = np.concatenate([-mu_pos[::-1], mu_pos])
    weights = np.concatenate([w_pos[::-1], w_pos])
    return mu, weights


def solve_dom_slab(tau0, omega, eps_wall, T_hot, T_cold, *, n_cells=400, n_half=32, tol=1e-12, max_iter=20000):
    """Discrete ordinates solution of a gray isotropically scattering slab.

    Step characteristics in optical depth, source iteration on the scattering
    term, gray diffuse walls at T_hot and T_cold, and the medium held at the
    linear temperature profile between them.
    """
    mu, w = _half_range_ordinates(n_half)
    npos = n_half
    dt = tau0 / n_cells
    tc = (np.arange(n_cells) + 0.5) * dt
    T = T_hot + (T_cold - T_hot) * tc / tau0
    Ib = SIGMA * T**4 / np.pi
    Ib_h = SIGMA * T_hot**4 / np.pi
    Ib_c = SIGMA * T_cold**4 / np.pi
    rho = 1.0 - eps_wall

    Jbar = Ib.copy()
    q_in_hot = np.pi * Ib_c
    q_in_cold = np.pi * Ib_h
    I_hot_out = eps_wall * Ib_h + rho * q_in_hot / np.pi
    I_cold_out = eps_wall * Ib_c + rho * q_in_cold / np.pi

    mp = mu[npos:]
    e = np.exp(-dt / mp)
    for _ in range(max_iter):
        S = (1.0 - omega) * Ib + omega * Jbar
        Ipos_faces = np.empty((npos, n_cells + 1))
        Ipos_faces[:, 0] = I_hot_out
        Icell_pos = np.empty((npos, n_cells))
        for k in range(n_cells):
            Iin = Ipos_faces[:, k]
            Ipos_faces[:, k + 1] = Iin * e + S[k] * (1.0 - e)
            Icell_pos[:, k] = S[k] + (Iin - S[k]) * (1.0 - e) * mp / dt
        Ineg_faces = np.empty((npos, n_cells + 1))
        Ineg_faces[:, n_cells] = I_cold_out
        Icell_neg = np.empty((npos, n_cells))
        for k in range(n_cells - 1, -1, -1):
            Iin = Ineg_faces[:, k + 1]
            Ineg_faces[:, k] = Iin * e + S[k] * (1.0 - e)
            Icell_neg[:, k] = S[k] + (Iin - S[k]) * (1.0 - e) * mp / dt
        Jnew = 0.5 * (w[npos:, None] * Icell_pos + w[npos:, None] * Icell_neg).sum(axis=0)
        q_in_hot_new = 2.0 * np.pi * np.sum(w[npos:] * mu[npos:] * Ineg_faces[:, 0])
        q_in_cold_new = 2.0 * np.pi * np.sum(w[npos:] * mu[npos:] * Ipos_faces[:, n_cells])
        I_hot_out_new = eps_wall * Ib_h + rho * q_in_hot_new / np.pi
        I_cold_out_new = eps_wall * Ib_c + rho * q_in_cold_new / np.pi
        err = max(
            np.max(np.abs(Jnew - Jbar)) / max(np.max(np.abs(Jnew)), 1e-300),
            abs(I_hot_out_new - I_hot_out) / max(abs(I_hot_out_new), 1e-300),
            abs(I_cold_out_new - I_cold_out) / max(abs(I_cold_out_new), 1e-300),
        )
        Jbar = Jnew
        I_hot_out, I_cold_out = I_hot_out_new, I_cold_out_new
        q_in_hot, q_in_cold = q_in_hot_new, q_in_cold_new
        if err < tol:
            break

    q_emit_hot = 2.0 * np.pi * np.sum(w[npos:] * mu[npos:] * np.full(npos, I_hot_out))
    q_h = q_emit_hot - q_in_hot
    q_emit_cold = 2.0 * np.pi * np.sum(w[npos:] * mu[npos:] * np.full(npos, I_cold_out))
    q_cold_face = q_in_cold - q_emit_cold
    return Channels(float(q_h), float(-q_cold_face), float(q_cold_face - q_h))


def dom_channels(beta, n_axis, omega, g, eps_wall, T_mean, *, delta_T=20.0, transport=True, n_cells=400, n_half=32, tol=1e-11):
    """Homogenized DOM; with transport=True the extinction and albedo are rescaled by
    beta_tr = beta (1 - omega g) and omega_tr = omega (1 - g) / (1 - omega g)."""
    tau = beta * float(n_axis)
    omega_eff = omega
    if transport:
        omega_eff = omega * (1.0 - g) / (1.0 - omega * g)
        tau *= 1.0 - omega * g
    return solve_dom_slab(tau, omega_eff, eps_wall, T_mean + 0.5 * delta_T, T_mean - 0.5 * delta_T,
                          n_cells=n_cells, n_half=n_half, tol=tol)


def _t3_channels(prefactor, T_mean, delta_T):
    """Boundary channels of a flux q(x) = prefactor * T(x)^3 evaluated at each wall."""
    T_hot = T_mean + 0.5 * delta_T
    T_cold = T_mean - 0.5 * delta_T
    q_h = prefactor * T_hot**3
    q_cold_face = prefactor * T_cold**3
    return Channels(float(q_h), float(-q_cold_face), float(q_cold_face - q_h))


def rosseland_channels(beta, n_axis, omega, g, T_mean, *, delta_T=20.0, transport=True):
    """Rosseland-type approximation, k_rad = 16 sigma T^3 / (3 beta_eff)."""
    tau = beta * float(n_axis)
    if transport:
        tau *= 1.0 - omega * g
    return _t3_channels(16.0 * SIGMA * delta_T / (3.0 * tau), T_mean, delta_T)


def mean_beam_length_channels(mean_beam_length_cells, n_axis, eps_wall, T_mean, *, delta_T=20.0):
    """Mean beam length estimate, k_rad = 4 sigma T^3 L_m eps / (2 - eps)."""
    eps_eff = eps_wall / (2.0 - eps_wall)
    prefactor = 4.0 * SIGMA * eps_eff * (mean_beam_length_cells / float(n_axis)) * delta_T
    return _t3_channels(prefactor, T_mean, delta_T)


def boundary_channel_error(pred: Channels, ref: Channels) -> float:
    """Combined boundary-channel error E_bc of Eq. (4)."""
    return (abs(pred.q_h - ref.q_h) + abs(pred.q_c - ref.q_c)) / (abs(ref.q_h) + abs(ref.q_c))


def property_table(properties: pd.DataFrame, microstructures: pd.DataFrame) -> pd.DataFrame:
    """Equivalent properties of each geometry and transfer axis.

    beta_axis is the identified extinction of the axis (averaged over the three
    emissivities, for which it is identical to within sampling), beta_scalar the
    trace invariant (beta_x + beta_y + beta_z) / 3, beta_cauchy = 3 / L_m, and
    g is kept per axis and emissivity, with its axis mean as the scalar value.
    """
    b = (properties.groupby(["microstructure", "structure", "target_phi", "transfer_axis"], as_index=False)
         .agg(beta_axis=("beta_per_cell", "mean"),
              beta_axis_ci95_low=("beta_ci95_low_per_cell", "mean"),
              beta_axis_ci95_high=("beta_ci95_high_per_cell", "mean")))
    g = (properties.pivot_table(index=["microstructure", "transfer_axis"], columns="epsilon", values="g", aggfunc="mean")
         .reset_index())
    g.columns = ["microstructure", "transfer_axis"] + [f"g_axis_eps{int(round(float(c) * 10)):02d}" for c in g.columns[2:]]
    t = b.merge(g, on=["microstructure", "transfer_axis"], validate="one_to_one")
    agg = {"beta_scalar": ("beta_axis", "mean"),
           "beta_scalar_ci95_low": ("beta_axis_ci95_low", "mean"),
           "beta_scalar_ci95_high": ("beta_axis_ci95_high", "mean")}
    for e in EPSILONS:
        agg[f"g_scalar_eps{int(e * 10):02d}"] = (f"g_axis_eps{int(e * 10):02d}", "mean")
    s = t.groupby(["microstructure", "structure", "target_phi"], as_index=False).agg(**agg)
    t = t.merge(s, on=["microstructure", "structure", "target_phi"], validate="many_to_one")
    lm = microstructures.set_index("id")["mean_beam_length_cells"]
    t["mean_beam_length_cells"] = t["microstructure"].map(lm)
    t["beta_cauchy"] = 3 / t["mean_beam_length_cells"]
    return t


def protocol_values(row, eps, protocol):
    """(beta, g) of a property protocol at emissivity eps."""
    suffix = f"eps{int(round(eps * 10)):02d}"
    if protocol == "cauchy":
        return float(row["beta_cauchy"]), 0.0
    if protocol == "identified_scalar":
        return float(row["beta_scalar"]), float(row[f"g_scalar_{suffix}"])
    if protocol == "identified_axis":
        return float(row["beta_axis"]), float(row[f"g_axis_{suffix}"])
    raise KeyError(protocol)


def predict(row, ref: Channels, T_mean, eps, *, n_axis=128, n_cells=400, n_half=32, tol=1e-11):
    """Predictions and errors of the nine combinations for one reference case."""
    omega = 1.0 - eps
    out = []
    for model, protocol, scattering in COMBINATIONS:
        if model == "MBL":
            beta, g = np.nan, 0.0
            pred = mean_beam_length_channels(float(row["mean_beam_length_cells"]), n_axis, eps, T_mean)
        else:
            beta, g = protocol_values(row, eps, protocol)
            if model == "DOM":
                pred = dom_channels(beta, n_axis, omega, g, eps, T_mean, transport=scattering == "transport",
                                    n_cells=n_cells, n_half=n_half, tol=tol)
            else:
                pred = rosseland_channels(beta, n_axis, omega, g, T_mean, transport=scattering == "transport")
        out.append({
            "model": model, "property_protocol": protocol, "scattering": scattering,
            "beta_per_cell": beta, "g": g, "omega": omega,
            "pred_q_h_W_m2": pred.q_h, "pred_q_c_W_m2": pred.q_c, "pred_q_i_W_m2": pred.q_i,
            "E_bc": boundary_channel_error(pred, ref),
            "signed_hot_rel": (pred.q_h - ref.q_h) / max(abs(ref.q_h), 1e-300),
            "signed_cold_rel": (pred.q_c - ref.q_c) / max(abs(ref.q_c), 1e-300),
        })
    return out


def geometry_block_bootstrap(table: pd.DataFrame, *, n_boot=50000, seed=20260805):
    """Median E_bc and its 95% interval obtained by resampling whole geometries
    together with their three transfer axes."""
    rng = np.random.default_rng(seed)
    geoms = np.array(sorted(table["microstructure"].unique()))
    rows = []
    for model, protocol, scattering in COMBINATIONS:
        s = table[(table.model == model) & (table.property_protocol == protocol) & (table.scattering == scattering)]
        mat = (s.pivot(index="microstructure", columns="transfer_axis", values="E_bc")
               .reindex(index=geoms, columns=["x", "y", "z"]).to_numpy(float))
        idx = rng.integers(0, len(geoms), size=(n_boot, len(geoms)))
        medians = np.median(mat[idx].reshape(n_boot, -1), axis=1)
        rows.append({"model": model, "property_protocol": protocol, "scattering": scattering,
                     "median_E_bc": s.E_bc.median(), "ci95_low": np.quantile(medians, 0.025),
                     "ci95_high": np.quantile(medians, 0.975), "max_E_bc": s.E_bc.max()})
    return pd.DataFrame(rows)
