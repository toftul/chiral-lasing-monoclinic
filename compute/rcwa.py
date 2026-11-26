"""Angle-resolved transmission of Design 1 with the effective loss, by RCWA (grcwa), for the model maps of Fig. 3(d).

Same structure as fem.py: a1 = a x, a2 = b (x cos(phi) + y sin(phi)), phi = 75 deg (README, "Lattice angle");
pillars n = 2.2 + i kappa as 10 staircase layers of the truncated cones on a 400 x 400 grid, slab n = 2.1 + i kappa,
host n_h = 1.46 on both sides, kappa = 8.003e-4.  Light comes from the substrate at the angle theta (in air) in the
xz plane (Gamma-X).  t[i, j]: zero-order transmitted amplitude in e_i for unit input in e_j (0 = R, 1 = L, the
helicity basis of fem.py), against the same stack with every layer n_h; T_ij = |t_ij|^2.

Run:  python rcwa.py            -> ../data/computed/transmission_map_design1.npz   (~2-3 h on 36 cores)
"""

import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')   # parallel over jobs

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import grcwa
import grcwa.rcwa as _gr

from fem import DESIGNS, KAPPA, h, helicity_basis, lattice, n_h, n_pillar, n_slab, t, taper

OUT = Path(__file__).resolve().parents[1]/'data'/'computed'
N_G, N_LAYERS, N_GRID = 201, 10, 400
THETA = np.round(np.arange(0.0, 4.0001, 0.1), 3)        # deg, in air
LAM = np.round(np.arange(700.0, 740.0001, 0.05), 3)      # nm
a1, a2 = lattice()


def pillar_layers(d, delta, kappa):
    """eps on the grid (i/N) a1 + (j/N) a2 of each pillar layer, bottom first."""
    f1, f2 = np.meshgrid(np.arange(N_GRID)/N_GRID, np.arange(N_GRID)/N_GRID, indexing='ij')
    layers = []
    for i in range(N_LAYERS):
        dr = taper*(1 - 2*(i + 0.5)/N_LAYERS)                       # cone radius at the layer's mid height
        eps = np.full((N_GRID, N_GRID), n_h**2, dtype=complex)
        for centre, diameter in (((0.0, 0.0), d + delta), ((0.5, 0.5), d - delta)):
            g1, g2 = (f1 - centre[0] + 0.5) % 1 - 0.5, (f2 - centre[1] + 0.5) % 1 - 0.5
            x, y = g1*a1[0] + g2*a2[0], g1*a1[1] + g2*a2[1]
            eps[x**2 + y**2 < (diameter/2 + dr)**2] = (n_pillar + 1j*kappa)**2
        layers.append(eps)
    return layers


def stack(lam_um, theta_deg, layers, kappa, reference):
    theta_sub = np.arcsin(np.sin(np.radians(theta_deg))/n_h)       # grcwa takes the angle in the incident medium
    obj = grcwa.obj(N_G, list(a1), list(a2), 1/lam_um, theta_sub, 0.0, verbose=0)
    obj.Add_LayerUniform(0.0, n_h**2)
    obj.Add_LayerUniform(t, n_h**2 if reference else (n_slab + 1j*kappa)**2)
    if reference:
        obj.Add_LayerUniform(h, n_h**2)
    else:
        for _ in layers:
            obj.Add_LayerGrid(h/len(layers), N_GRID, N_GRID)
    obj.Add_LayerUniform(0.0, n_h**2)
    obj.Init_Setup()
    assert np.all(obj.G[0] == 0), 'order 0 is not the first G'
    if not reference:
        obj.GridLayer_geteps(np.concatenate([g.ravel() for g in layers]))
    return obj


def zero_order(obj):
    """Transmitted zero-order E for grcwa's two inputs (p and s)."""
    # build the S-matrix once and reuse it for both inputs: grcwa rebuilds it in every solve (3x slower)
    S11, S12, S21, S22 = _gr.GetSMatrix(0, obj.Layer_N - 1, obj.q_list, obj.phi_list, obj.kp_list, obj.thickness_list)
    original = _gr.SolveExterior
    _gr.SolveExterior = lambda a0, bN, *_: (S11 @ a0 + S12 @ bN, S21 @ a0 + S22 @ bN)
    try:
        out = []
        for p, s in ((1.0, 0.0), (0.0, 1.0)):
            obj.MakeExcitationPlanewave(p, 0.0, s, 0.0, order=0)
            E, _ = obj.Solve_FieldFourier(obj.Layer_N - 1, 0.0)
            out.append(np.array([E[0][0], E[1][0], E[2][0]]))
    finally:
        _gr.SolveExterior = original
    return out


def transmission(lam_nm, theta_deg, layers, kappa):
    """t (2 x 2) in the helicity basis; the two runs share it, so grcwa's own p/s labels drop out."""
    lam_um = lam_nm*1e-3
    k0 = 2*np.pi/lam_um
    k = np.array([k0*np.sin(np.radians(theta_deg)), 0.0, 0.0])
    k[2] = np.sqrt((n_h*k0)**2 - k[0]**2)
    to_rl = lambda E: np.column_stack([[e.conj() @ v for e in helicity_basis(k)] for v in E])
    E_out = to_rl(zero_order(stack(lam_um, theta_deg, layers, kappa, False)))
    E_ref = to_rl(zero_order(stack(lam_um, theta_deg, layers, kappa, True)))
    return E_out @ np.linalg.inv(E_ref)


_layers = []


def _job(args):
    theta, lam = args
    if not _layers:
        _layers.extend(pillar_layers(*DESIGNS['design1'], KAPPA['design1']))
    return [transmission(l, theta, _layers, KAPPA['design1']) for l in lam]


def main():
    chunks = np.array_split(LAM, 9)
    jobs = [(th, c) for th in THETA for c in chunks]
    # shortcut: no resume, a crash loses the run (~2-3 h); write per-job files if that happens
    with Pool(os.cpu_count(), maxtasksperchild=40) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    tt = np.array([np.concatenate(res[i*len(chunks):(i + 1)*len(chunks)]) for i in range(len(THETA))])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'transmission_map_design1.npz', theta_deg=THETA, lambda_nm=LAM, kappa=KAPPA['design1'],
                        t=tt)


if __name__ == '__main__':
    main()
