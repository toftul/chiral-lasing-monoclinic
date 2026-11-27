# Observation of Chiral Lasing from Monoclinic Resonant Metasurfaces

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24574076.svg)](https://doi.org/10.5281/zenodo.24574076)

Code and data behind the figures of the Letter and its Supplemental Material (SM).

| Path | Content |
|---|---|
| `compute/fem.py` | NGSolve: resonant states, m, σ_top, σ_bot, CD_mode, C₂ parity, energy shares, transmission |
| `compute/rcwa.py` | grcwa: angle-resolved transmission maps with the effective loss κ |
| `figures/*.py` | one script per figure, reads `data/`, writes `figures/output/` |
| `data/computed/` | output of `compute/` |
| `data/measured/` | measured data |

## Figures

| Script | Figure |
|---|---|
| `figures/knobs.py` | Fig. 2: bands, C₂ parity, Q and CD_mode against δ and φ |
| `figures/linear.py` | Fig. 3: transmission spectra and maps, γ_rad, γ_abs, γ_tot |
| `figures/lasing.py` | Fig. 4: decay rates, pump curves, far fields, lasing maps |
| `figures/sm_transmission.py` | SM: bands and measured transmission of Design 1 and its mirror image |
| `figures/sm_pl.py` | SM: angle-resolved photoluminescence |
| `figures/sm_farfield.py` | SM: far fields of the A-band laser with analyzers |
| `figures/sm_interference.py` | SM: Michelson self-interference |

## Run

The figures take about 30 s (Python 3.12).

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd figures && for f in *.py; do [ "$f" = common.py ] || ../.venv/bin/python "$f"; done
```

The computed data are already in `data/computed/`.  To recompute them you need NGSolve (tested with 6.2.2608,
Python 3.14, 36 cores).

```sh
cd compute
python3 fem.py path design1      # also: path design2, knobs, kgrid design1|design2, gamma design1|design2, spectrum
python3 rcwa.py
```

Each `fem.py` task takes 5 min to 1 h (see its docstring), `rcwa.py` takes 2–3 h.

## Lattice angle

The code uses a₂ = b(x̂ cos φ + ŷ sin φ) with φ = 75° for the fabricated sample.  The manuscript and the figures
label it 180° − φ = 105°.

## Notation

| Manuscript | Code |
|---|---|
| a, b, φ, d, δ | `a`, `b`, `phi`, `d`, `delta` (µm, rad) |
| pillar height, slab thickness | `h`, `t` |
| n + iκ | `n_pillar`, `n_slab`, `KAPPA` |
| k_B, Ω_n = ω_n − iγ_n, Q_n = ω_n/(2γ_n) | `k_B`, `k0` = Ω_n/c, columns `lambda_nm`, `Q` |
| m_R, m_L, m′_R, m′_L [SM Eqs. (S9)–(S12)] | `coupling()`, columns `mR`, `mL`, `mpR`, `mpL` |
| σ_top, σ_bot [SM Eq. (S15)], CD_mode [Eq. (1)] | `sigma()`, `cd_mode()`, columns `sigma_top`, `sigma_bot`, `CD_mode` |
| γ_rad = ½\|m_n\|² [SM Eq. (S14)] | column `sum_rule` = ½\|m_n\|²/γ_n |
| N_n [SM Eq. (S6)] | `norm()` (N_n/ε₀) |
| p_n, Ĉ₂ F_n = p_n F_n | `p` in `knobs/*.npz` |
| γ_abs/ω = κ (S_pillar/n_pillar + S_slab/n_slab) | `figures/common.py: absorption_share()` |
| ψ_n, w, w_i, D_i [Eq. (3), SM Eq. (S31)] | `figures/lasing.py: packet()`, `curvature()` |
| T_ij (output i, input j) | `transmission()`, `t[i, j]` with 0 = R, 1 = L |

## Data

### `data/computed/`

| File | Task |
|---|---|
| `bands_design{1,2}.csv` | `fem.py path`: modes along Y ← Γ → X |
| `knobs/{delta,phi}_*.npz` | `fem.py knobs`: Γ states of Design 1 against δ (φ = 75°) and φ (δ = 40 nm) |
| `kgrid_design{1,2}.npz` | `fem.py kgrid`: modes on a k-grid around Γ, for the far field |
| `gamma_design{1,2}.csv` | `fem.py gamma`: Γ states with the shares of ε\|E\|² in pillars and slab |
| `spectrum_design1.npz` | `fem.py spectrum`: T_ij at θ_x = 0.2° |
| `transmission_map_design1.npz` | `rcwa.py`: T_ij against θ_x and λ |

These files come from the original project scripts.  The code here reproduces them at every point checked.

### `data/measured/`

| File | Content | Source |
|---|---|---|
| `design1_spectrum_theta0.2.csv` | T_LL, T_RR, T_RL, T_LR at θ_x = 0.2° | raw |
| `design1_transmission_map.csv` | T_L, T_R against θ_x, λ | raw |
| `design1_Q.csv` | Q against θ_x | **digitized** from the authors' plot |
| `design2_Q.csv` | Q against δ (d = 255 nm), θ = 0 | raw |
| `design{1,2}_pump.csv` | LCP and RCP emission against pump fluence | raw |
| `design{1,2}_lasing_spectrum.csv` | emission spectrum above threshold | raw |
| `design{1,2}_lasing_map.csv` | angle-resolved lasing emission | raw |
| `design{1,2}_farfield.png` | far field with the authors' Stokes ellipses | **crop** of the authors' figure |
| `sm_transmission_maps.npz` | T_L, T_R of Design 1 and its mirror image, Γ-X and Γ-Y | **decoded** from the authors' figure |
| `sm_pl_maps.npz` | PL maps, LCP, RCP, CD | **decoded** from the authors' figure |
| `sm_farfield/`, `sm_interference/` | camera frames | **crops** of the authors' figures |

For B₁ of Design 2 at δ = 30 nm the raw file gives Q = 1705 and the authors' plot 1763.  So the fit gives
κ = 6.24 × 10⁻⁴, and the manuscript quotes 6.1 × 10⁻⁴.  The figures use the manuscript values (`KAPPA` in
`compute/fem.py`).

## License

The code and the data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`).
