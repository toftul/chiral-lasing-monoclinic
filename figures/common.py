"""Data, constants and model quantities shared by the figure scripts."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
MEASURED, COMPUTED, OUTPUT = ROOT/'data'/'measured', ROOT/'data'/'computed', ROOT/'figures'/'output'
sys.path.insert(0, str(ROOT/'compute'))
from fem import a, b, phi, n_h, n_pillar, n_slab, KAPPA, helicity_basis, lattice  # noqa: E402,F401

BLUE, RED = '#2B59C3', '#D62728'
PML_MAX = 0.1                     # modes with more of |E|^2 in the PML are PML modes
# Gamma states in the window (gamma_design*.csv): B2 RCP, A the BIC, B1 LCP
MODES = {'design1': {'B2': 717.485, 'A': 725.351, 'B1': 726.819},
         'design2': {'B2': 722.482, 'B1': 733.647, 'A': 734.487}}
NAME = {'A': 'A', 'B1': r'B$_1$', 'B2': r'B$_2$'}


def style():
    plt.style.use(ROOT/'figures'/'paper.mplstyle')
    OUTPUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUTPUT/f'{name}.pdf')
    print('wrote', OUTPUT/f'{name}.pdf')


def paper_angle(phi_deg):
    """The manuscript's label of the lattice angle phi of a2 = b (x cos(phi) + y sin(phi)) (README)."""
    return 180.0 - phi_deg


def csv(name):
    return np.genfromtxt(name, delimiter=',', names=True)


def diffraction_threshold(s, phi=phi):
    """Wavelength (nm) below which a first order propagates in the host at k_B = (s, 0) 2 pi/a for s >= 0 and
    (0, -s) 2 pi/a for s < 0."""
    G = 2*np.pi*np.linalg.inv(np.array(lattice(phi))).T                        # reciprocal vectors as rows
    G = np.array([m*G[0] + n*G[1] for m in range(-3, 4) for n in range(-3, 4) if (m, n) != (0, 0)])
    s = np.atleast_1d(s)
    k = 2*np.pi/a*np.stack([np.clip(s, 0, None), np.clip(-s, 0, None)], -1)
    return 1e3*n_h*2*np.pi/np.linalg.norm(k[:, None] - G[None], axis=-1).min(1)


def theta_deg(s, lam_nm):
    """Angle in air of the zero order at k_B = |s| 2 pi/a."""
    return np.degrees(np.arcsin(np.abs(s)*lam_nm*1e-3/a))


def bands(design):
    """Modes along Y <- Gamma -> X without PML modes and the diffraction region; s = kx (X branch), -ky (Y)."""
    d = csv(COMPUTED/f'bands_{design}.csv')
    s = np.where(d['ky_a_2pi'] > 0, -d['ky_a_2pi'], d['kx_a_2pi'])
    keep = (d['pml_share'] < PML_MAX) & (d['lambda_nm'] > diffraction_threshold(s))
    return s[keep], d[keep]


def follow(design, mode, s_max=0.015):
    """One band along Gamma-X, followed from Gamma by the nearest wavelength: s, lambda (nm), Q."""
    d = csv(COMPUTED/f'bands_{design}.csv')
    d = d[(d['pml_share'] < PML_MAX) & (d['ky_a_2pi'] == 0)]
    lam, out = MODES[design][mode], []
    for v in np.round(np.arange(0, s_max + 1e-9, 0.001), 3):
        c = np.flatnonzero(np.isclose(d['kx_a_2pi'], v))
        j = c[np.argmin(abs(d['lambda_nm'][c] - lam))]
        lam = d['lambda_nm'][j]
        out.append((v, lam, abs(d['Q'][j])))
    return np.array(out).T


def absorption_share(design):
    """S_pillar/n_pillar + S_slab/n_slab of each Gamma state: gamma_abs/omega = kappa times this, to first order in
    kappa (Im eps = 2 n kappa; S = share of eps|E|^2)."""
    d = csv(COMPUTED/f'gamma_{design}.csv')
    out = {}
    for m, lam in MODES[design].items():
        j = np.argmin(abs(d['lambda_nm'] - lam))
        assert abs(d['lambda_nm'][j] - lam) < 0.01
        out[m] = d['share_pillar'][j]/n_pillar + d['share_slab'][j]/n_slab
    return out


def measured_Q(design):
    """Measured Q used for the loss fit: Design 1 at theta_x = 0.1 deg, Design 2 at delta = 30 nm, theta = 0."""
    if design == 'design1':
        d = np.genfromtxt(MEASURED/'design1_Q.csv', delimiter=',', names=True, dtype=None, encoding=None)
        return {m: d['Q'][d['mode'] == m][np.argmin(d['theta_deg'][d['mode'] == m])] for m in ('A', 'B2', 'B1')}
    d = csv(MEASURED/'design2_Q.csv')
    i = np.flatnonzero(d['delta_nm'] == 30)[0]
    return {'B2': d['Q_B2'][i], 'B1': d['Q_B1'][i]}


def kappa_fit(design):
    """Effective loss kappa: least squares in ln Q of 1/Q = 1/Q_rad + 2 kappa S over the measured modes, with Q_rad
    at k_x a/2pi = 0.0012 (theta_x ~ 0.1 deg) from the k-grid and S of absorption_share."""
    d = np.load(COMPUTED/f'kgrid_{design}.npz')
    rows = d['rows'][np.flatnonzero(np.all(np.isclose(d['k'], [0.0012, 0.0]), 1))[0]]
    rows = rows[rows[:, 5] < PML_MAX]
    q_meas, S = measured_Q(design), absorption_share(design)
    q_rad = {m: abs(rows[np.argmin(abs(rows[:, 0] - MODES[design][m])), 1]) for m in q_meas}
    cost = lambda kappa: sum((np.log(1/(1/q_rad[m] + 2*kappa*S[m])) - np.log(q))**2 for m, q in q_meas.items())
    return minimize_scalar(cost, bounds=(1e-5, 1e-2), method='bounded', options=dict(xatol=1e-9)).x
