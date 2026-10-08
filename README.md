# pore-radiation-ijhmt2026

Code and data for the article

> W. Yan, K. Zhu, Y. Huang. Pore-scale radiative heat transfer and accuracy of reduced radiation models in porous and fibrous insulation materials. *International Journal of Heat and Mass Transfer* (2026). https://doi.org/10.1016/j.ijheatmasstransfer.2026.129687

The repository contains the digital microstructures of the four morphology archetypes (open-cell foam OF, random fiber FI, aligned fiber FA, tortuous layered TL), the code that generates them and evaluates their directional descriptors, the pore-scale surface-radiation and strictly coupled radiation-conduction solver, the identification of equivalent radiative properties, the reduced radiation models, and the case-level results behind the numbers reported in the article.

## Contents

| Path | Content |
|---|---|
| `porerad/geometry.py` | Generation of the four structures, matching of the solid volume fraction and retention rules (Sec. 2.1) |
| `porerad/descriptors.py` | Solid volume fraction, interfacial area density, mean beam length, pore chords, lineal-path function, interface-normal fractions (Secs. 2.1, 2.2) |
| `porerad/view_factors.py` | Surface patches, ray generation and first-hit tracing (Sec. 2.5) |
| `porerad/exchange.py` | Exchange matrix with row closure and area-weighted reciprocity, gray diffuse radiosity (Sec. 2.5) |
| `porerad/conduction.py` | Cell-centred finite-volume conduction (Sec. 2.5) |
| `porerad/coupled.py` | Prescribed-field channels q_h, q_c, q_i and the Newton-Krylov radiation-conduction solution (Secs. 2.3, 2.5) |
| `porerad/properties.py` | Directional extinction, albedo and asymmetry factor from pore-scale ray statistics (Sec. 2.4) |
| `porerad/penetration.py` | Penetration response of the slab (Sec. 2.3) |
| `porerad/reduced_models.py` | Homogenized DOM, Rosseland-type approximation, mean beam length estimate and E_bc (Secs. 2.3, 2.4) |
| `scripts/` | One entry point per step |
| `data/microstructures/` | The digital microstructures at N = 128 (48 core geometries, 4 canonical geometries of the coupled comparison, 3 matched aligned/random fiber pairs) |
| `data/microstructures.csv` | Identifier, structure, solid volume fraction, interfacial area density, mean beam length, axial mean pore chords, SHA-256 of every phase array |
| `data/cases_*.csv` | Conditions, ray counts per patch and random seeds of every prescribed-field and coupled case; for the canonical and resolution cases also the SHA-256 of the integer first-hit counts of each ray set |
| `results/` | Case-level results (see below) |

The directional descriptors of all 48 core geometries (lineal-path functions, chord-length distributions, interface-normal statistics) accompany the article as Supplementary Data; `scripts/compute_descriptors.py` regenerates those tables.

## Installation

Python 3.10 to 3.12.

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions with which the repository was checked against the archived results. The radiative-property and penetration kernels use Taichi on the CPU; all other parts use NumPy, SciPy and Numba. On macOS, NumPy builds linked to Accelerate may print harmless "encountered in matmul" warnings.

## Reproducing the results

The property, penetration, prescribed-field and coupled scripts act on one geometry or case; the others act on the fixed tables in `data/` and `results/`. Case identifiers are listed in `data/cases_coupled.csv` and `data/cases_prescribed_field.csv`. Run times refer to a 14-core desktop CPU. With `--ab` the prescribed-field and coupled scripts also solve with ray sets A and B separately and report their difference.

```bash
# Regenerate the microstructures and check every phase array against its SHA-256 (a few minutes).
# This also writes the N = 160 and 192 arrays used by the resolution cases.
python scripts/make_microstructures.py

# Directional descriptors of the 48 core geometries (Supplementary Data S1-S4)
python scripts/compute_descriptors.py --outdir results/descriptors

# Equivalent radiative properties of one geometry, axis and emissivity (seconds)
python scripts/identify_properties.py --microstructure TL_core_phi0.10_s0 --axis x --epsilon 0.6

# Penetration response of one geometry, axis and emissivity (seconds)
python scripts/run_penetration.py --microstructure OF_core_phi0.10_s0 --axis x --epsilon 0.9

# Prescribed-field reference of one geometry-axis case at its conditions (1 to 15 minutes)
python scripts/run_prescribed_field.py pf_TL_core_phi0.10_s0_x

# Strictly coupled radiation-conduction solution of one case (10 to 50 minutes and 4 to 9 GB of memory
# at N = 128 and for the 20 mm slabs; the N = 160 and 192 cases take longer and need more memory)
python scripts/run_coupled.py coupled_OF_x_T700_eps0.6_N128

# Reduced-model comparison (about one minute); the tables are written to output/
python scripts/run_reduced_models.py --set main
python scripts/run_reduced_models.py --set range

# Values quoted in the article, recomputed from the tables in results/
python scripts/check_paper_values.py
```

The production calculations ran the ray tracing and the radiosity linear algebra on GPUs (Taichi and CuPy). This repository contains CPU implementations of the same algorithms. They reproduce the archived integer first-hit counts exactly and the archived prescribed-field and coupled fluxes to round-off.

## Results

| File | Content |
|---|---|
| `results/coupled.csv` | 44 strictly coupled solutions: the canonical condition with its axis, temperature and emissivity slices, the N = 160 and 192 resolution checks, the matched pair, and the 20 mm repeated-microstructure slabs |
| `results/coupled_ray_set_comparison.csv` | Differences between coupled solutions obtained with independently seeded ray sets of each operator (set A against set B, and the combined set against a third set C) at the stated conditions |
| `results/prescribed_field.csv` | 336 prescribed-field solutions (36 geometry-axis cases at nine (T_m, ε) conditions and the 12 matched-pair cases), with the largest relative difference of the wall channels between ray sets A and B |
| `results/radiative_properties.csv` | beta, omega and g of the 48 core geometries, three axes and three emissivities |
| `results/penetration_response.csv` | Reflected, transmitted and absorbed fractions of the 432 penetration solutions, and whether each met the Monte Carlo tolerances |
| `results/reduced_models_main.csv` | Predictions and E_bc of the nine combinations of model and property protocol for the 36 cases at 700 K, ε = 0.6 |
| `results/reduced_models_range_medians.csv` | Median E_bc per structure and condition over 400–1000 K and ε = 0.3–0.9 |
| `results/fullpath_crosscheck.csv` | Channel values of the independent full-path Monte Carlo cross-check (Supplementary Material, Table C3) |
| `results/hit_point_restart/` | The same property, penetration and reduced-model tables with the alternative restart rule described below |

Fluxes are in W m^-2, lengths in cells of the digital microstructure (cell size h = H/N, 78 micrometres at H = 10 mm and N = 128).

## Restart rule for reflected rays

In the property and penetration calculations a ray that is reflected at a solid surface starts its next segment from the surface. The kernel used for the article (`restart="article"`, the default here) restarts a ray reflected for the first time at its hit point, and a ray reflected for the second or a later time at its hit point advanced once more along the preceding direction by the length of the preceding segment. With `restart="hit_point"` every reflected segment starts at its hit point. This changes the asymmetry factor g by about +0.01 at ε = 0.6 and +0.04 at ε = 0.3, and lowers the transmitted fraction of the fiber and layered structures at low emissivity. The extinction coefficient, the prescribed-field and coupled solutions and the geometry do not depend on this rule, and the albedo agrees within its sampling error. Both sets of results are provided. The table lists the values quoted in the article that differ at the printed precision.

| Quantity (section of the article) | `article` | `hit_point` |
|---|---|---|
| 1 − ωg at 700 K, ε = 0.6 (Sec. 2.4) | 1.246–1.258 | 1.241–1.256 |
| Median E_bc, transport-corrected DOM with the axis-specific extinction (Sec. 3.4) | 15.5% | 15.7% |
| Median E_bc, Rosseland-type approximation with the identified transport extinction (Secs. 1, 3.4, 4) | 295% | 296% |
| Range-check medians of the transport-corrected DOM, random fiber (Sec. 3.4) | 7–17% | 7–20% |
| Penetration solutions within the Monte Carlo tolerances, and the condition combinations formed from them (Sec. 2.3) | 411 of 432; 4932 | 407 of 432; 4884 |

The other values recomputed by `python scripts/check_paper_values.py --restart hit_point`, including every value in the abstract, are the same at the printed precision. The tables in `results/hit_point_restart/` have the same columns as those in `results/`.

## Data available on request

The integer first-hit counts of every ray set listed in `data/` can be regenerated exactly from its seeds (`porerad.view_factors.count_sha256` gives the hash listed there for the canonical and resolution cases). The three-dimensional field exports, other intermediate data and the GPU implementation used for the production runs are available from the corresponding author on reasonable request.

## License

The code is licensed under the GNU General Public License v3.0 (`LICENSE`) and the data in `data/` and `results/` under the Creative Commons Attribution 4.0 International license (`LICENSE-DATA`). Please cite the article when using either.

