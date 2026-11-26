"""Resonant states and transmission of the monoclinic metasurface with the finite-element method (NGSolve).

Structure (main text; SM, "Fabricated structures and simulation model"), lengths in um
  lattice   a1 = a x,  a2 = b (x cos(phi) + y sin(phi)),  a = 0.500, b = 0.410, phi = 75 deg
            (README, "Lattice angle": the manuscript calls this cell phi = 105 deg)
  pillars   TiO2, n = 2.2, height h = 0.190 on top of the slab; diameter d + delta at the cell corners and
            d - delta at the cell centre (a1 + a2)/2, at mid-height; truncated cones, 0.020 wider at the base and
            0.020 narrower at the top
  slab      Ta2O5, n = 2.1, thickness t = 0.195
  host      n_h = 1.46 below (fused silica) and above (DMSO)
  Design 1: d = 0.235, delta = 0.040;  Design 2: d = 0.255, delta = 0.030

Method.  Nedelec elements on one unit cell, Bloch-periodic side faces, graded PML in z, time dependence
exp(-i omega t).  Mode n has Omega_n = omega_n - i gamma_n and Q_n = omega_n/(2 gamma_n).  The coupling parameters
m_n = (m_R, m_L, m'_R, m'_L) follow SM Eqs. (S9)-(S12), sigma_top, sigma_bot SM Eq. (S15) and CD_mode main-text
Eq. (1).  Check of SM Eq. (S14): sum_rule = |m_n|^2/(2 gamma_n) = 1 below the diffraction threshold (lossless).

Run with a Python that has NGSolve (tested with 6.2.2608); each task writes into ../data/computed:
  python3 fem.py path design1          bands along Y <- Gamma -> X          -> bands_design1.csv     (~1 h, 36 cores)
  python3 fem.py path design2                                               -> bands_design2.csv
  python3 fem.py knobs                 Gamma states against delta and phi    -> knobs/*.npz            (~6 min)
  python3 fem.py kgrid design1         bands on a k-grid around Gamma        -> kgrid_design1.npz     (~1 h)
  python3 fem.py kgrid design2                                              -> kgrid_design2.npz
  python3 fem.py gamma design1         Gamma states, shares of eps|E|^2      -> gamma_design1.csv     (~5 min)
  python3 fem.py gamma design2                                              -> gamma_design2.csv
  python3 fem.py spectrum             T_ij of Design 1 at theta_x = 0.2 deg, loss kappa -> spectrum_design1.npz
"""

import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')   # parallel over k, not BLAS

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

OUT = Path(__file__).resolve().parents[1]/'data'/'computed'

a, b, phi = 0.500, 0.410, np.radians(75.0)
h, t, taper = 0.190, 0.195, 0.010
n_pillar, n_slab, n_h = 2.2, 2.1, 1.46
DESIGNS = {'design1': (0.235, 0.040), 'design2': (0.255, 0.030)}        # d, delta
KAPPA = {'design1': 8.003e-4, 'design2': 6.124e-4}   # effective loss, fitted to the measured Q (figures/common.py)
c = 299792458e6                                      # um/s

ORDER, H_STRUCT, H_HOST, PAD = 3, 0.06, 0.15, 2.5    # element order, mesh sizes, host between structure and PML
L_PML, ALPHA = 0.5, 6.0                              # PML thickness, largest Im of the stretch
COLUMNS = ('lambda_nm,Q,sigma_top,sigma_bot,CD_mode,pml_share,'
           'mR_re,mR_im,mL_re,mL_im,mpR_re,mpR_im,mpL_re,mpL_im,sum_rule')


def lattice(phi=phi):
    return np.array([a, 0.0]), b*np.array([np.cos(phi), np.sin(phi)])


def permittivity(kappa=0.0):
    """Relative permittivity of each mesh region; kappa: effective loss, n + i kappa in pillars and slab."""
    return {'pmlb': n_h**2, 'subs': n_h**2, 'slab': (n_slab + 1j*kappa)**2, 'pillar': (n_pillar + 1j*kappa)**2,
            'hostp': n_h**2, 'hostt': n_h**2, 'pmlt': n_h**2}


# ---------------------------------------------------------------- geometry and mesh

def build_mesh(d, delta, phi=phi):
    from netgen.occ import (Axes, Cone, Glue, IdentificationType, ListOfShapes, OCCGeometry, Vec, WorkPlane, X, Z,
                            gp_Trsf)
    from ngsolve import Mesh
    a1, a2 = lattice(phi)

    def layer(z0, z1):
        wp = WorkPlane(Axes((0, 0, z0), n=Z, h=X))
        wp.MoveTo(0, 0).LineTo(*a1).LineTo(*(a1 + a2)).LineTo(*a2).Close()
        return wp.Face().Extrude(z1 - z0)

    def pillar(centre, diameter):
        return Cone(Axes((*centre, t), n=Z, h=X), diameter/2 + taper, diameter/2 - taper, h, 2*np.pi)

    zs = [-PAD - L_PML, -PAD, 0.0, t, t + h, t + h + PAD, t + h + PAD + L_PML]
    pml_b, subs, slab, pl, host_t, pml_t = [layer(zs[i], zs[i + 1]) for i in range(6)]
    cones = pillar((a1 + a2)/2, d - delta)
    for corner in (np.zeros(2), a1, a2, a1 + a2):
        cones = cones + pillar(corner, d + delta)
    parts = [(pml_b, 'pmlb', H_HOST), (subs, 'subs', H_HOST), (slab, 'slab', H_STRUCT),
             (pl - cones, 'hostp', H_STRUCT), (pl*cones, 'pillar', H_STRUCT), (host_t, 'hostt', H_HOST),
             (pml_t, 'pmlt', H_HOST)]
    for s, name, hmax in parts:
        s.mat(name)
        s.maxh = hmax
    shape = Glue([p[0] for p in parts])
    shape.faces.Max(Z).name = 'ztop'
    shape.faces.Min(Z).name = 'zbot'

    # side faces r = p a1 + q a2 with p or q = 0, 1, identified by the lattice vectors
    lat = np.column_stack([a1, a2])
    sides = {key: [] for key in ('p0', 'p1', 'q0', 'q1')}
    for f in shape.faces:
        bb = f.bounding_box
        if bb[1][2] - bb[0][2] < 1e-9:
            continue
        p, q = np.linalg.solve(lat, [f.center.x, f.center.y])
        for key, val in (('p0', p), ('p1', p - 1), ('q0', q), ('q1', q - 1)):
            if abs(val) < 1e-7:
                sides[key].append(f)
    ListOfShapes(sides['p0']).Identify(ListOfShapes(sides['p1']), 'pa1', IdentificationType.PERIODIC,
                                       gp_Trsf.Translation(Vec(*a1, 0)))
    ListOfShapes(sides['q0']).Identify(ListOfShapes(sides['q1']), 'pa2', IdentificationType.PERIODIC,
                                       gp_Trsf.Translation(Vec(*a2, 0)))
    mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=H_HOST))
    mesh.Curve(ORDER)
    return mesh, zs, (a1, a2)


def bloch_space(mesh, k_B):
    """Nedelec space with E(r + a_i) = exp(i k_B.a_i) E(r) on the side faces; PEC behind the PMLs."""
    from ngsolve import HCurl, Periodic
    ids, pts = mesh.ngmesh.GetIdentifications(), mesh.ngmesh.Points()
    shifts = [np.mean([np.array(pts[j].p) - np.array(pts[i].p) for i, j, n in ids if n == nr], axis=0)[:2]
              for nr in sorted({n for _, _, n in ids})]       # netgen does not keep the order of the Identify calls
    return Periodic(HCurl(mesh, order=ORDER, complex=True, dirichlet='zbot|ztop'),
                    phase=[np.exp(1j*k_B @ s) for s in shifts])


def materials(mesh, zs, kappa=0.0):
    """eps and 1/mu of the stretched coordinates (PML along z), and the plain relative permittivity."""
    from ngsolve import CoefficientFunction, IfPos, z
    xi_t, xi_b = (z - zs[5])/L_PML, (zs[1] - z)/L_PML
    s = IfPos(z - zs[5], 1 + 1j*ALPHA*xi_t**2, IfPos(zs[1] - z, 1 + 1j*ALPHA*xi_b**2, 1.0))
    eps_r = mesh.MaterialCF(permittivity(kappa))
    eps = CoefficientFunction((eps_r*s, 0, 0, 0, eps_r*s, 0, 0, 0, eps_r/s), dims=(3, 3))
    nu = CoefficientFunction((1/s, 0, 0, 0, 1/s, 0, 0, 0, s), dims=(3, 3))
    return eps, nu, eps_r


# ---------------------------------------------------------------- plane waves

def helicity_basis(k):
    """(e_R, e_L) = (e_p -+ i e_s)/sqrt2 for the real wavevector k, e_s = z x k/|z x k| (y along z),
    e_p = e_s x k_hat [SM Eq. (S8)]; e_R is the RCP wave, helicity -1."""
    k_hat = k/np.linalg.norm(k)
    k_par = np.array([k[0], k[1], 0.0])
    if np.linalg.norm(k_par) < 1e-12:
        e_s = np.array([0.0, 1.0, 0.0])
    else:
        e_s = np.cross([0, 0, 1.0], k_par)/np.linalg.norm(k_par)
    e_p = np.cross(e_s, k_hat)
    return (e_p - 1j*e_s)/np.sqrt(2), (e_p + 1j*e_s)/np.sqrt(2)


def sigma(m_R, m_L):
    """Degree of circular polarization of a radiated wave, +1 for RCP [SM Eq. (S15)]."""
    return (abs(m_R)**2 - abs(m_L)**2)/(abs(m_R)**2 + abs(m_L)**2)


def cd_mode(sigma_top, sigma_bot):
    """Modal circular dichroism [main text Eq. (1)]."""
    return (sigma_top + sigma_bot)/(1 + sigma_top*sigma_bot)


# ---------------------------------------------------------------- eigenmodes and coupling parameters

def eigenmodes(mesh, zs, k_B, lam_shift, nev):
    """The nev resonant states nearest the wavelength lam_shift (um) at the Bloch vector k_B (1/um), by increasing
    wavelength: a GridFunction, k0 = Omega_n/c (1/um, complex) and the eigenvectors (columns)."""
    from ngsolve import BilinearForm, GridFunction, TaskManager, curl, dx
    fes = bloch_space(mesh, k_B)
    u, v = fes.TnT()
    eps, nu, _ = materials(mesh, zs)
    shift = (2*np.pi/lam_shift)**2
    mass = BilinearForm(fes)
    mass += (eps*u)*v*dx
    stiff = BilinearForm(fes)
    stiff += (nu*curl(u))*curl(v)*dx - shift*(eps*u)*v*dx
    with TaskManager():
        mass.Assemble()
        stiff.Assemble()
        inv = stiff.mat.Inverse(fes.FreeDofs(), inverse='umfpack')
    gf = GridFunction(fes)
    tmp, out = gf.vec.CreateVector(), gf.vec.CreateVector()

    def op(vec):
        gf.vec.FV().NumPy()[:] = vec
        tmp.data = mass.mat*gf.vec
        out.data = inv*tmp
        return out.FV().NumPy().copy()

    nd = fes.ndof
    mu, vecs = spla.eigs(spla.LinearOperator((nd, nd), matvec=op, dtype=complex), k=nev, which='LM', tol=1e-10)
    k0 = np.sqrt(shift + 1/mu)
    k0 = np.where(k0.real < 0, -k0, k0)
    order = np.argsort(-k0.real)
    return gf, k0[order], vecs[:, order]


def partner_quadrature(mesh, cell):
    """Quadrature points and weights over V (the cell without the PMLs), their C2 images about the cell centre,
    and the relative permittivity at the points."""
    from ngsolve import CoefficientFunction, Det, IntegrationRule, TET, VOL, specialcf, x, y, z
    rule = IntegrationRule(TET, 2*ORDER + 2)
    pts = mesh.MapToAllElements(rule, VOL)
    w = np.tile(np.array(rule.weights), mesh.ne)*np.abs(Det(specialcf.JacobianMatrix(3))(pts)[:, 0])
    in_v = mesh.MaterialCF({'pmlt': 1, 'pmlb': 1}, default=0)(pts)[:, 0] < 0.5
    pts, w = pts[in_v], w[in_v]
    xyz = CoefficientFunction((x, y, z))(pts)
    centre = (cell[0] + cell[1])/2
    images = mesh(2*centre[0] - xyz[:, 0], 2*centre[1] - xyz[:, 1], xyz[:, 2])
    assert np.all(images['nr'] >= 0), 'C2 image outside the mesh'
    return pts, images, w, mesh.MaterialCF(permittivity())(pts)[:, 0].real


def norm(E, quad, cell, k_B, k0):
    """N_n/eps0 = int_V (eps E_bar.E + curl E_bar.curl E/k0^2) dV [SM Eq. (S6), H = curl E/(i omega mu0)], with the
    partner E_bar the C2 image of the mode; the parity p_n only changes the common phase of the four m."""
    from ngsolve import curl
    pts, images, w, eps = quad
    R = np.array([-1.0, -1.0, 1.0])                                    # rotation by pi about z
    E_r, curl_r = E(images)*R, curl(E)(images)*R
    N_centre = np.sum(w*(eps*np.sum(E(pts)*E_r, 1) + np.sum(curl(E)(pts)*curl_r, 1)/k0**2))
    return np.exp(-1j*k_B @ (cell[0] + cell[1]))*N_centre             # rotation about the centre -> the origin


def coupling(E, mesh, quad, cell, k_B, k0):
    """m_n = (m_R, m_L, m'_R, m'_L) in s^-1/2 of the mode E with k0 = Omega_n/c at the Bloch vector k_B
    [SM Eqs. (S11), (S12)]."""
    from ngsolve import Integrate, exp, x, y, z
    A_cell = abs(np.linalg.det(np.array(cell)))                        # a b sin(phi)
    k_n = n_h*k0                                                       # Omega_n/c_h
    k_z = np.sqrt(k_n**2 - k_B @ k_B)
    omega_n = c*k0.real
    N = norm(E, quad, cell, k_B, k0)
    prefactor = 1j*k0*np.sqrt(omega_n/(2*A_cell*N*k_z.real))          # sqrt(eps0/eps_h) A_n of SM Eq. (S12)
    m = []
    for sign in (1, -1):                                               # up: m_R, m_L; down: m'_R, m'_L
        wave = exp(-1j*(k_B[0]*x + k_B[1]*y + sign*k_z*z))
        overlap = sum((n**2 - n_h**2)*np.array(Integrate(E*wave, mesh, definedon=mesh.Materials(region),
                                                         order=2*ORDER + 2))
                      for region, n in (('slab', n_slab), ('pillar', n_pillar)))       # Delta eps = eps - eps_h
        k_real = np.array([*k_B, sign*np.sqrt((n_h*k0.real)**2 - k_B @ k_B)])
        m += [prefactor*(overlap @ e.conj()) for e in helicity_basis(k_real)]          # e(-k) = e(k)^*
    return np.array(m)


def solve(mesh, zs, cell, k_B, lam_shift, nev, quad):
    """One row of COLUMNS per mode at k_B (1/um); also the GridFunction and the eigenvectors."""
    from ngsolve import InnerProduct, Integrate
    gf, k0, vecs = eigenmodes(mesh, zs, k_B, lam_shift, nev)
    rows = []
    for i in range(len(k0)):
        gf.vec.FV().NumPy()[:] = vecs[:, i]
        E2 = InnerProduct(gf, gf)
        pml_share = Integrate(E2, mesh, definedon=mesh.Materials('pmlt|pmlb')).real/Integrate(E2, mesh).real
        m = coupling(gf, mesh, quad, cell, k_B, k0[i])
        s_top, s_bot = sigma(m[0], m[1]), sigma(m[2], m[3])
        gamma_n = -c*k0[i].imag
        rows.append((2*np.pi/k0[i].real*1e3, k0[i].real/(-2*k0[i].imag), s_top, s_bot, cd_mode(s_top, s_bot),
                     pml_share, *np.column_stack([m.real, m.imag]).ravel(), np.sum(abs(m)**2)/2/gamma_n))
    return rows, gf, vecs


# ---------------------------------------------------------------- driven problem: transmission

def transmission(mesh, zs, cell, lam_nm, theta_deg, kappa):
    """Zero-order transmission t[i, j] (i output, j input; 0 = R, 1 = L) for light from the substrate at the angle
    theta in air in the xz plane, against the bare host; T_ij = |t_ij|^2.  Scattered field in the host n_h."""
    from ngsolve import BilinearForm, CoefficientFunction, GridFunction, LinearForm, curl, dx, exp, x, y, z
    k0 = 2*np.pi/(lam_nm*1e-3)
    k_x = k0*np.sin(np.radians(theta_deg))
    k = np.array([k_x, 0.0, np.sqrt((n_h*k0)**2 - k_x**2)])
    fes = bloch_space(mesh, k[:2])
    u, v = fes.TnT()
    eps, nu, eps_r = materials(mesh, zs, kappa)
    lhs = BilinearForm(fes)
    lhs += (nu*curl(u))*curl(v)*dx - k0**2*(eps*u)*v*dx
    lhs.Assemble()
    inv = lhs.mat.Inverse(fes.FreeDofs(), inverse='umfpack')
    gf = GridFunction(fes)
    z_top = zs[4] + 0.05                       # close to the structure: the order-3 host mesh adds a phase error
    n = 32                                     # cell average on an n x n grid keeps only the zero order
    p, q = [g.ravel() for g in np.meshgrid((np.arange(n) + 0.5)/n, (np.arange(n) + 0.5)/n, indexing='ij')]
    px, py = p*cell[0][0] + q*cell[1][0], p*cell[0][1] + q*cell[1][1]
    basis = helicity_basis(k)
    tt = np.zeros((2, 2), complex)
    for j, e in enumerate(basis):
        f = LinearForm(fes)
        f += k0**2*(eps_r - n_h**2)*(CoefficientFunction(tuple(e))*exp(1j*(k[0]*x + k[1]*y + k[2]*z)))*v*dx(
            definedon=mesh.Materials('slab|pillar'))
        f.Assemble()
        gf.vec.data = inv*f.vec
        E0 = np.mean(gf(mesh(px, py, np.full(px.size, z_top)))*np.exp(-1j*k_x*px)[:, None], axis=0)
        E_t = E0 + e*np.exp(1j*k[2]*z_top)
        tt[:, j] = [e_i.conj() @ E_t*np.exp(-1j*k[2]*z_top) for e_i in basis]
    return tt


# ---------------------------------------------------------------- tasks

_worker = {}


def _init(d, delta, phi):
    import ngsolve
    ngsolve.SetNumThreads(1)
    mesh, zs, cell = build_mesh(d, delta, phi)
    _worker.update(mesh=mesh, zs=zs, cell=cell, quad=partner_quadrature(mesh, cell))


def _path_point(args):
    kx, ky, lam_shift = args
    w = _worker
    rows, _, _ = solve(w['mesh'], w['zs'], w['cell'], 2*np.pi/a*np.array([kx, ky]), lam_shift, 36, w['quad'])
    return [(kx, ky, *r) for r in rows]


def path(design):
    """Bands along Y <- Gamma -> X: k_B = (s, 0) 2 pi/a for s >= 0 and (0, -s) 2 pi/a for s < 0."""
    lam_shift = {'design1': 0.724, 'design2': 0.728}[design]
    s = np.round(np.arange(-0.061, 0.0505, 0.001), 6)
    jobs = [((v, 0.0) if v >= 0 else (0.0, -v)) + (lam_shift,) for v in s]
    with Pool(os.cpu_count(), initializer=_init, initargs=(*DESIGNS[design], phi)) as pool:
        rows = [r for res in pool.imap(_path_point, jobs) for r in res]
    np.savetxt(OUT/f'bands_{design}.csv', rows, delimiter=',', fmt='%.10g', comments='',
               header='kx_a_2pi,ky_a_2pi,' + COLUMNS)


def _knob(args):
    """Gamma states of Design 1 at one (delta, phi): rows, the C2 parity p_n and E at fixed points of the slab (to
    follow the states across the sweep); at (40 nm, 75 deg) also E on the plane z = t of one cell."""
    kind, value = args
    import ngsolve
    ngsolve.SetNumThreads(2)
    delta, phi_deg = (value*1e-3, 75.0) if kind == 'delta' else (0.040, value)
    mesh, zs, cell = build_mesh(DESIGNS['design1'][0], delta, np.radians(phi_deg))
    lam_shift = 0.724*np.sin(np.radians(phi_deg))/np.sin(np.radians(75.0))    # the cell area moves the modes
    rows, gf, vecs = solve(mesh, zs, cell, np.zeros(2), lam_shift, 30, partner_quadrature(mesh, cell))
    lat = np.column_stack(cell)

    def fold(r):                                                         # into the cell (Gamma: E is periodic)
        pq = np.linalg.solve(lat, r)
        return lat @ np.clip(pq - np.floor(pq), 1e-6, 1 - 1e-6)

    rng = np.random.default_rng(1)
    r = lat @ rng.uniform(0.02, 0.98, (2, 400))
    zz = rng.uniform(0.01, t - 0.01, 400)
    pts, pts_c2 = mesh(*r, zz), mesh(*fold(-r), zz)
    extra = {}
    if kind == 'delta' and value == 40.0:
        x = np.linspace(-0.02, cell[0][0] + cell[1][0] + 0.02, 131)
        y = np.linspace(-0.02, cell[1][1] + 0.02, 91)
        gx, gy = np.meshgrid(x, y)
        pts_map = mesh(*fold(np.vstack([gx.ravel(), gy.ravel()])), np.full(gx.size, t))
        extra = dict(x=x, y=y, E_map=[])
    u, p = [], []
    for i in range(vecs.shape[1]):
        gf.vec.FV().NumPy()[:] = vecs[:, i]
        E, E_c2 = gf(pts), gf(pts_c2)
        u.append(E.astype(np.complex64))
        # C2 eigenvalue from the in-plane components: (C2 E)_par(r) = -E_par(-r)
        p.append(-np.real(np.sum(E_c2[:, :2]*E[:, :2].conj()))/np.sum(abs(E[:, :2])**2))
        if extra:
            extra['E_map'].append(gf(pts_map).reshape(len(y), len(x), 3))
    (OUT/'knobs').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'knobs'/f'{kind}_{value:g}.npz', delta_nm=delta*1e3, phi_deg=phi_deg, rows=np.array(rows),
                        p=np.array(p), u=np.array(u), a1=cell[0], a2=cell[1], d=DESIGNS['design1'][0], taper=taper,
                        **extra)
    return kind, value


def knobs():
    """Design 1 at Gamma: delta = 2.5 ... 60 nm at phi = 75 deg, and phi = 90 ... 77.5 deg at delta = 40 nm."""
    jobs = [('delta', v) for v in (2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)] + \
           [('phi', v) for v in (90.0, 87.5, 85.0, 82.5, 80.0, 77.5)]
    with Pool(len(jobs)) as pool:
        for kind, value in pool.imap_unordered(_knob, jobs):
            print(kind, value, flush=True)


def _kgrid_point(args):
    """Rows at one k_B and u = E exp(-i k_B.r) at fixed points of the slab (to follow the band across k)."""
    kx, ky, lam_shift = args
    w = _worker
    k_B = 2*np.pi/a*np.array([kx, ky])
    rows, gf, vecs = solve(w['mesh'], w['zs'], w['cell'], k_B, lam_shift, 16, w['quad'])
    rng = np.random.default_rng(1)
    r = np.column_stack(w['cell']) @ rng.uniform(0.02, 0.98, (2, 300))
    pts = w['mesh'](*r, rng.uniform(0.01, t - 0.01, 300))
    u = []
    for i in range(vecs.shape[1]):
        gf.vec.FV().NumPy()[:] = vecs[:, i]
        u.append((gf(pts)*np.exp(-1j*(k_B @ r))[:, None]).astype(np.complex64))
    return kx, ky, rows, u


def kgrid(design):
    """Square k-grid around Gamma, half plane ky >= 0 (C2 gives -k_B); k in units of 2 pi/a."""
    dk, lam_shift = {'design1': (0.0006, 0.7245), 'design2': (0.0012, 0.7240)}[design]
    n = int(round(0.006/dk))
    ks = np.arange(-n, n + 1)*dk
    jobs = [(kx, ky, lam_shift) for ky in ks if ky >= -1e-12 for kx in ks if ky > 1e-12 or kx >= -1e-12]
    # shortcut: no resume, a crash loses the run (~1 h); write per-job files if that happens
    with Pool(os.cpu_count(), initializer=_init, initargs=(*DESIGNS[design], phi)) as pool:
        res = sorted(pool.imap_unordered(_kgrid_point, jobs), key=lambda r: (r[1], r[0]))
    np.savez_compressed(OUT/f'kgrid_{design}.npz', k=np.array([r[:2] for r in res]), dk=dk,
                        rows=np.array([r[2] for r in res]), u=np.array([r[3] for r in res]))


def gamma(design):
    """Gamma states with the shares of eps|E|^2 in the pillars and the slab (outside the PMLs), for gamma_abs."""
    from ngsolve import InnerProduct, Integrate
    import ngsolve
    ngsolve.SetNumThreads(os.cpu_count())
    mesh, zs, cell = build_mesh(*DESIGNS[design])
    lam_shift = {'design1': 0.7245, 'design2': 0.7280}[design]
    rows, gf, vecs = solve(mesh, zs, cell, np.zeros(2), lam_shift, 14, partner_quadrature(mesh, cell))
    eps = permittivity()
    out = []
    for row, vec in zip(rows, vecs.T):
        gf.vec.FV().NumPy()[:] = vec
        W = {r: eps[r].real*Integrate(InnerProduct(gf, gf), mesh, definedon=mesh.Materials(r)).real
             for r in ('pillar', 'slab', 'hostp', 'hostt', 'subs')}
        total = sum(W.values())
        out.append((row[0], row[1], row[4], row[5], W['pillar']/total, W['slab']/total))
    np.savetxt(OUT/f'gamma_{design}.csv', out, delimiter=',', fmt='%.6g', comments='',
               header='lambda_nm,Q,CD_mode,pml_share,share_pillar,share_slab')


def _spectrum_point(lam):
    w = _worker
    return transmission(w['mesh'], w['zs'], w['cell'], lam, 0.2, KAPPA['design1'])


def spectrum():
    """T_ij of Design 1 at theta_x = 0.2 deg with the effective loss: a 0.25 nm grid, denser across each line
    (centre and lossy linewidth lambda/Q of the eigenmodes B2, A, B1)."""
    lam = [np.arange(715.0, 730.0001, 0.25)]
    for centre, Q in ((717.46, 2118), (725.11, 2500), (727.06, 1500)):
        lam.append(centre + centre/Q/2*np.tan(np.pi*np.linspace(-0.95, 0.95, 25)/2))
    lam = np.unique(np.round(np.concatenate(lam), 4))
    lam = lam[(lam >= 715.0) & (lam <= 730.0)]
    with Pool(os.cpu_count(), initializer=_init, initargs=(*DESIGNS['design1'], phi)) as pool:
        tt = pool.map(_spectrum_point, lam, chunksize=4)
    np.savez(OUT/'spectrum_design1.npz', lambda_nm=lam, theta_deg=0.2, kappa=KAPPA['design1'], t=np.array(tt))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {'path': path, 'knobs': knobs, 'kgrid': kgrid, 'gamma': gamma, 'spectrum': spectrum}[sys.argv[1]](*sys.argv[2:])
    print(f'{sys.argv[1:]}: {time.time() - t0:.0f} s')
