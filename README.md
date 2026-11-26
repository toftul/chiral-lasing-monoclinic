# Observation of chiral lasing with monoclinic resonant metasurfaces

Code and data behind the figures of the Letter and its Supplemental Material (SM): the finite-element eigenmodes and
coupling parameters, the RCWA and finite-element transmission, the loss and far-field models, and the measured data.

| Path | Content |
|---|---|
| `compute/fem.py` | NGSolve: resonant states, coupling parameters m, σ_top, σ_bot, CD_mode, C₂ parity, energy shares, transmission |
| `compute/rcwa.py` | grcwa: angle-resolved transmission maps with the effective loss κ |
| `figures/*.py` | One script per figure; they read `data/` and write `figures/output/` |
| `data/computed/` | Output of `compute/` |
| `data/measured/` | Measured data (see the provenance below) |

## Figures

| Script | Figure |
|---|---|
| `figures/knobs.py` | Fig. 2: bands, C₂ parity, Q and CD_mode against δ and φ |
| `figures/linear.py` | Fig. 3: transmission spectra and maps, decay rates γ_rad, γ_abs, γ_tot |
| `figures/lasing.py` | Fig. 4: decay rates, pump curves, far fields [Eq. (3)], lasing maps |
| `figures/sm_transmission.py` | SM: computed bands and measured transmission of Design 1 and of its mirror image |
| `figures/sm_pl.py` | SM: angle-resolved photoluminescence |
| `figures/sm_farfield.py` | SM: far fields of the A-band laser with analyzers |
| `figures/sm_interference.py` | SM: Michelson self-interference |

Fig. 1, the schematics of Figs. 3(a), 4(a), 4(b), the insets of Fig. 2(a), and the SM figures of fabrication, SEM,
setup and the interferometer sketch are drawings and are not made here.  The scripts leave their slots empty.

The scripts print the numbers quoted in the text: the effective loss κ, the thresholds (166.6 and 178.3 µJ/cm²),
the lasing wavelengths (724.2 and 722.7 nm), (I_R − I_L)/(I_R + I_L) at the lasing peak (−0.88 and +0.96), the pump
radius w (158 µm, quoted as 160 µm) and σ_bot averaged over the computed far field (−0.97 and +0.88).

## Run

Figures (about 30 s in total), tested with Python 3.12:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd figures && for f in *.py; do [ "$f" = common.py ] || ../.venv/bin/python "$f"; done
```

Computed data (optional; the results are in `data/computed/`), tested on 36 cores:

```sh
cd compute
python3 fem.py path design1      # also: path design2, knobs, kgrid design1|design2, gamma design1|design2, spectrum
python3 rcwa.py
```

`fem.py` needs NGSolve (tested with 6.2.2608, Python 3.14).  The run time of each task is in the docstring of
`fem.py` (from 5 min to 1 h); `rcwa.py` takes 2–3 h.

## Lattice angle

The code uses the formula of the main text, a₂ = b(x̂ cos φ + ŷ sin φ).  With this formula the fabricated cell
(SEM, SM) has **φ = 75°** (a₂ leans to +x), and `compute/` uses this value.  The manuscript calls this sample
φ = 105°, the obtuse angle of the cell, so the figures label every angle as 180° − φ.  The mirror image (the SM
sample "φ = 75°") is φ = 105° in the code.  Mirroring x → −x keeps the bands and Q and reverses every CD_mode.

## Notation

| Manuscript | Code |
|---|---|
| a, b, φ, d, δ | `a`, `b`, `phi`, `d`, `delta` (lengths in µm, angles in rad) |
| pillar height, slab thickness | `h`, `t` |
| n + iκ (effective loss) | `n_pillar`, `n_slab`, `KAPPA` |
| k_B, Ω_n = ω_n − iγ_n, Q_n = ω_n/(2γ_n) | `k_B`, `k0` = Ω_n/c, columns `lambda_nm`, `Q` |
| m_R, m_L, m′_R, m′_L [SM Eqs. (S9)–(S12)] | `coupling()`, columns `mR`, `mL`, `mpR`, `mpL` |
| σ_top, σ_bot [SM Eq. (S15)], CD_mode [Eq. (1)] | `sigma()`, `cd_mode()`, columns `sigma_top`, `sigma_bot`, `CD_mode` |
| γ_rad = ½\|m_n\|² [SM Eq. (S14)] | column `sum_rule` = ½\|m_n\|²/γ_n (1 below the diffraction threshold) |
| N_n [SM Eq. (S6)] | `norm()` (N_n/ε₀) |
| p_n, Ĉ₂ F_n = p_n F_n | `p` in `knobs/*.npz` |
| γ_abs/ω = κ (S_pillar/n_pillar + S_slab/n_slab) | `figures/common.py: absorption_share()` |
| ψ_n, w, w_i, D_i [Eq. (3), SM Eq. (S31)] | `figures/lasing.py: packet()`, `curvature()` |
| T_ij (output i, input j), t[i, j] with 0 = R, 1 = L | `transmission()` |

## Data

### `data/computed/`

| File | Task |
|---|---|
| `bands_design{1,2}.csv` | `fem.py path`: every mode along Y ← Γ → X |
| `knobs/{delta,phi}_*.npz` | `fem.py knobs`: Γ states of Design 1 against δ (at φ = 75°) and φ (at δ = 40 nm) |
| `kgrid_design{1,2}.npz` | `fem.py kgrid`: modes on a k-grid around Γ, for the far field |
| `gamma_design{1,2}.csv` | `fem.py gamma`: Γ states with the shares of ε\|E\|² in the pillars and the slab |
| `spectrum_design1.npz` | `fem.py spectrum`: T_ij at θ_x = 0.2° with κ |
| `transmission_map_design1.npz` | `rcwa.py`: T_ij against θ_x and λ with κ |

These results come from the original project scripts, which use the same numerics.  The code here reproduces them
at the points checked: at k_x a/2π = 0.01 (Design 1) the same λ, Q and CD_mode, and |m| to 10⁻¹⁰ relative; at
(δ, φ) = (40 nm, 85°) the same states and parity p; the Γ shares of Design 2 to six digits; the transmission spectrum
exactly at two wavelengths; the RCWA map to 5 × 10⁻¹³ at three points.  `kgrid` uses the same solver as `path`.
σ_top and σ_bot are computed from m [SM Eq. (S15)].  The original scripts took them from the zero-order field on
planes above and below the structure; the two agree to 10⁻².

### `data/measured/`

| File | Content | Source |
|---|---|---|
| `design1_spectrum_theta0.2.csv` | T_LL, T_RR, T_RL, T_LR at θ_x = 0.2° | raw |
| `design1_transmission_map.csv` | T_L, T_R against θ_x, λ | raw |
| `design1_Q.csv` | measured Q against θ_x | **digitized** from the vector paths of the authors' Q plot |
| `design2_Q.csv` | measured Q against δ (d = 255 nm), θ = 0 | raw |
| `design{1,2}_pump.csv` | LCP and RCP emission against pump fluence | raw |
| `design{1,2}_lasing_spectrum.csv` | emission spectrum above threshold | raw |
| `design{1,2}_lasing_map.csv` | angle-resolved lasing emission | raw |
| `design{1,2}_farfield.png` | far field of the laser, with the authors' Stokes ellipses | **crop** of the authors' figure, 220 dpi |
| `sm_transmission_maps.npz` | T_L, T_R of Design 1 and its mirror image, Γ-X and Γ-Y | **decoded** from the raster of the authors' figure with its colour bar; Design 1 Γ-X agrees with `design1_transmission_map.csv` to rms 0.009 |
| `sm_pl_maps.npz` | PL maps, LCP, RCP, CD, and the authors' colour maps | **decoded** the same way |
| `sm_farfield/`, `sm_interference/` | camera frames | **crops** of the authors' figures, 600 dpi |

The measured Q of B₁ in Design 2 at δ = 30 nm is 1705 in the raw file and 1763 in the authors' plot.  With 1705 the
fit gives κ = 6.24 × 10⁻⁴ for Design 2; the manuscript quotes 6.1 × 10⁻⁴ (from 1763).  The figures use the manuscript
values (`KAPPA` in `compute/fem.py`) and print the fit next to them.  For Design 1 the fit gives 8.01 × 10⁻⁴
(manuscript 8.0 × 10⁻⁴).
