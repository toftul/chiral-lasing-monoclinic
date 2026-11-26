"""Fig. 4: chiral lasing of Design 1 (A band, LCP) and Design 2 (B2, RCP).

(a), (b) slots for the setup and the lasing sketch (drawn by hand, not here).
(c), (g) Computed decay rates at theta_x = 0.2 deg, gamma_abs (light) + gamma_rad (dark); star: the lasing mode.
(d), (h) Measured LCP and RCP emission against pump fluence; dotted: threshold from a linear fit.
(e), (i) Far field of the laser: measured (authors' camera image) and computed from main-text Eq. (3),
         I(k) ~ |psi_n(k_B)|^2 (|m'_R|^2 + |m'_L|^2), with the anisotropic Gaussian packet of SM Eq. (S31),
         |psi_n|^2 ~ exp(-K_1^2 w_1^2 - K_2^2 w_2^2), w_i = (w^2 |D_i|/(2 g_n))^(1/4), g_n = gamma_n(0) from the
         measured Q.  The pump radius w is fitted to the measured lobe separation of Design 1 and used for both.
         White ellipses: polarization of the substrate-side wave m'_R e_R + m'_L e_L.
(f), (j) Measured angle-resolved lasing emission, LCP and RCP; dashed: computed bands along Gamma-X.
"""

from collections import deque

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, ConnectionPatch
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from common import (BLUE, COMPUTED, KAPPA, MEASURED, MODES, NAME, PML_MAX, RED, a, absorption_share, csv, follow,
                    helicity_basis, kappa_fit, measured_Q, n_h, save, style, theta_deg)

LASER = {'design1': ('A', 'L'), 'design2': ('B2', 'R')}            # lasing mode and its handedness
COLOUR = {'A': 'k', 'B2': RED, 'B1': BLUE}
LIGHT = {'A': '0.75', 'B2': '#f2b3b3', 'B1': '#b3c4ec'}
OVERLAY = {'A': 'w', 'B2': '#ff9b9b', 'B1': '#9fbfff'}              # mode colours on the dark maps
STAR = '#ff00e5'
THETA_RATES = 0.2                  # deg
LOBE_SEPARATION = 0.172            # deg, measured between the two lobes of Design 1 (LCP analyzer image)
R_FOV = 0.5                        # deg, radius of the measured far-field disks (from their 0.2 deg scale bar)
FIT_DEG = 0.1                      # deg, range of the quadratic fit of omega_n(k_B) at Gamma
N_FINE, ELL_MIN = 241, 0.1

# layout, inches
W, H = 6.2, 3.49
D_FF, GAP_FF = 0.60, 0.05
ROW_H = 2*D_FF + GAP_FF
ROW_Y = {'design1': 2.07, 'design2': 0.31}
X_RATE, W_RATE, X_IO, W_IO, X_FF, X_MAP, W_MAP, GAP_MAP = 2.32, 0.44, 3.14, 0.74, 3.96, 5.05, 0.50, 0.03


def axes(fig, x, y, w, h):
    return fig.add_axes([x/W, y/H, w/W, h/H])


def letter(fig, x, y, s):
    fig.text(x/W, y/H, f'({s})', ha='left', va='bottom')


# ---------------------------------------------------------------- measured data

def threshold(P, I):
    """Linear fit of the emission above 0.2 of its maximum, extrapolated to the median level below it."""
    above = I > 0.2*I.max()
    slope, offset = np.polyfit(P[above], I[above], 1)
    return (np.median(I[P < P[above].min() - 0.5]) - offset)/slope


def lasing_map(design):
    d = csv(MEASURED/f'{design}_lasing_map.csv')
    th, lam = np.unique(d['theta_deg']), np.unique(d['lambda_nm'])
    assert np.allclose(d['theta_deg'][:len(th)], th)
    return th, lam, d['I_L'].reshape(len(lam), len(th)), d['I_R'].reshape(len(lam), len(th))


# ---------------------------------------------------------------- far-field model

def track(design):
    """The lasing band on the k-grid around Gamma, followed from Gamma by the overlap of the sampled fields, with
    the C2 images filled in: k (2 pi/a), and lambda (nm), Q, m (4 complex), sigma_bot on the square grid."""
    d = np.load(COMPUTED/f'kgrid_{design}.npz')
    k, rows, u, dk = d['k'], d['rows'], d['u'], float(d['dk'])
    ij = {(int(round(kx/dk)), int(round(ky/dk))): n for n, (kx, ky) in enumerate(k)}
    g = rows[ij[(0, 0)]]
    ok = np.flatnonzero(g[:, 5] < PML_MAX)
    picked = {(0, 0): int(ok[np.argmin(abs(g[ok, 0] - MODES[design][LASER[design][0]]))])}
    queue = deque([(0, 0)])
    while queue:                                                       # breadth first over the half plane
        cur = queue.popleft()
        uc = u[ij[cur]][picked[cur]].ravel()
        for nb in ((cur[0] + 1, cur[1]), (cur[0] - 1, cur[1]), (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)):
            if nb in picked or nb not in ij:
                continue
            ok = np.flatnonzero(rows[ij[nb]][:, 5] < PML_MAX)
            cand = u[ij[nb]][ok].reshape(len(ok), -1)
            picked[nb] = int(ok[np.argmax(abs(cand.conj() @ uc)**2/np.sum(abs(cand)**2, 1))])
            queue.append(nb)
    n = max(i for i, _ in ij)
    ks = np.arange(-n, n + 1)*dk
    band = np.full((ks.size, ks.size, 15), np.nan)                     # [iy, ix, column]
    for iy, j in enumerate(range(-n, n + 1)):
        for ix, i in enumerate(range(-n, n + 1)):
            key = (i, j) if (i, j) in picked else (-i, -j)            # C2: the same state at -k_B
            band[iy, ix] = rows[ij[key]][picked[key]]
    m = band[..., 6:14:2] + 1j*band[..., 7:14:2]
    return dict(ks=ks, lam=band[..., 0], Q=abs(band[..., 1]), m=m, sigma_bot=band[..., 3])


def fine_grid(band):
    """The band on an N_FINE x N_FINE grid: omega a/2 pi c, |m'_R|^2 + |m'_L|^2, sigma_bot and the Stokes S1, S2 of
    the substrate-side wave; theta from k_B at the Gamma wavelength."""
    ks = band['ks']
    J = np.zeros(band['lam'].shape + (2,), complex)
    for iy, ky in enumerate(ks):
        for ix, kx in enumerate(ks):
            k_par = 2*np.pi/a*np.array([kx, ky])
            k_dn = np.array([*k_par, -np.sqrt((n_h*2*np.pi/(band['lam'][iy, ix]*1e-3))**2 - k_par @ k_par)])
            e_R, e_L = helicity_basis(k_dn)
            J[iy, ix] = (band['m'][iy, ix, 2]*e_R + band['m'][iy, ix, 3]*e_L)[:2]
    S0 = np.sum(abs(J)**2, -1)
    S1, S2 = (abs(J[..., 0])**2 - abs(J[..., 1])**2)/S0, 2*np.real(J[..., 0]*J[..., 1].conj())/S0
    kf = np.linspace(ks[0], ks[-1], N_FINE)
    KX, KY = np.meshgrid(kf, kf)
    pts = np.column_stack([KY.ravel(), KX.ravel()])
    f = lambda F, method='cubic': RegularGridInterpolator((ks, ks), F, method=method)(pts).reshape(KX.shape)
    lam0 = band['lam'][ks.size//2, ks.size//2]
    return dict(kf=kf, th=theta_deg(kf, lam0)*np.sign(kf), w0=a/(lam0*1e-3), w=f(a/(band['lam']*1e-3)),
                m2=np.clip(f(np.sum(abs(band['m'][..., 2:])**2, -1)), 0, None),
                sigma_bot=f(np.nan_to_num(band['sigma_bot']), 'linear'), S1=f(np.nan_to_num(S1), 'linear'),
                S2=f(np.nan_to_num(S2), 'linear'))


def curvature(F):
    """D_i/omega_n (um^2) of omega_n(k_B) ~ omega_n(0) + D_1 K_1^2 + D_2 K_2^2 near Gamma, and the principal axes."""
    k = F['kf']*2*np.pi/a
    KX, KY = np.meshgrid(k, k)
    near = np.hypot(*np.meshgrid(F['th'], F['th'])) <= FIT_DEG
    X = np.column_stack([KX[near]**2, 2*KX[near]*KY[near], KY[near]**2])
    c = np.linalg.lstsq(X, (F['w'][near] - F['w0'])/F['w0'], rcond=None)[0]
    return np.linalg.eigh(np.array([[c[0], c[1]], [c[1], c[2]]]))


def packet(F, w, Q):
    """|psi_n(k_B)|^2 of SM Eq. (S31) for the pump radius w (um); g_n = gamma_n(0) = omega_n/(2 Q)."""
    D, R = curvature(F)
    w_i = (w**2*np.abs(D)*Q)**0.25                                    # (w^2 |D_i|/(2 g_n))^(1/4), D in omega_n
    k = F['kf']*2*np.pi/a
    KX, KY = np.meshgrid(k, k)
    K1, K2 = KX*R[0, 0] + KY*R[1, 0], KX*R[0, 1] + KY*R[1, 1]
    return np.exp(-(K1*w_i[0])**2 - (K2*w_i[1])**2)


def lobe_separation(th, I):
    """Distance (deg) between the two maxima of I along theta_x, refined by a parabola through each."""
    c = th.size//2
    v, out = I[c], []
    for lo, hi in ((0, c), (c + 1, th.size)):
        i = lo + int(np.argmax(v[lo:hi]))
        out.append(th[i] + 0.5*(v[i - 1] - v[i + 1])/(v[i - 1] - 2*v[i] + v[i + 1])*(th[1] - th[0]))
    return out[1] - out[0]


def far_fields():
    """Computed far fields of both lasers, with w fitted on Design 1."""
    F = {d: fine_grid(track(d)) for d in LASER}
    Q = {d: measured_Q(d)[LASER[d][0]] for d in LASER}
    ws = np.linspace(40.0, 400.0, 361)
    seps = [lobe_separation(F['design1']['th'], F['design1']['m2']*packet(F['design1'], w, Q['design1'])) for w in ws]
    w = float(np.interp(LOBE_SEPARATION, seps[::-1], ws[::-1]))
    print(f'pump radius w = {w:.1f} um (lobe separation {LOBE_SEPARATION} deg); measured Q {Q}')
    out = {}
    for d, f in F.items():
        I = f['m2']*packet(f, w, Q[d])
        TX, TY = np.meshgrid(f['th'], f['th'])
        disk = np.hypot(TX, TY) <= R_FOV
        print(f'{d}: sigma_bot averaged over the far field {np.sum((I*f["sigma_bot"])[disk])/np.sum(I[disk]):+.3f}')
        out[d] = (f, I/I[disk].max())
    return out


def ellipse(ax, x, y, S1, S2, sigma, size=0.022):
    """Polarization ellipse: orientation atan2(S2, S1)/2, axis ratio tan(arcsin(sigma)/2)."""
    t = np.linspace(0, 2*np.pi, 30)
    psi, ratio = 0.5*np.arctan2(S2, S1), np.tan(abs(0.5*np.arcsin(np.clip(sigma, -1, 1))))
    ex, ey = size*np.cos(t), size*ratio*np.sin(t)
    ax.plot(x + ex*np.cos(psi) - ey*np.sin(psi), y + ex*np.sin(psi) + ey*np.cos(psi), color='w', lw=0.2)


def disk(ax, image, extent, bar_x):
    im = ax.imshow(image, extent=extent, origin='lower' if image.ndim == 2 else 'upper', cmap='inferno', vmin=0,
                   vmax=1, interpolation='bilinear')
    im.set_clip_path(Circle((0, 0), R_FOV, transform=ax.transData))
    ax.add_patch(Circle((0, 0), R_FOV, fc='none', ec='k', lw=0.5))
    if bar_x is not None:
        ax.plot(bar_x, [-0.4 if bar_x[0] < 0 else -0.25]*2, color='w', lw=1.2, solid_capstyle='butt')
    ax.set_xlim(-R_FOV*1.03, R_FOV*1.03)
    ax.set_ylim(-R_FOV*1.03, R_FOV*1.03)
    ax.set_aspect('equal')
    ax.axis('off')


# ---------------------------------------------------------------- panels

def panel_rates(ax, design):
    u, kappa, S = 1e4, KAPPA[design], absorption_share(design)
    ax.text(0.07, 0.97, rf'$\theta_x={THETA_RATES:g}$°', transform=ax.transAxes, ha='left', va='top')
    for i, mode in enumerate(('A', 'B2', 'B1')):
        s, lam, Q = follow(design, mode)
        g_rad, g_abs = float(np.interp(THETA_RATES, theta_deg(s, lam), 1/(2*Q))), kappa*S[mode]
        ax.bar(i, u*g_abs, width=0.62, color=LIGHT[mode], lw=0)
        ax.bar(i, u*g_rad, bottom=u*g_abs, width=0.62, color=COLOUR[mode], lw=0)
        if mode == 'B1':
            ax.text(i, u*g_abs/2, 'abs', rotation=90, ha='center', va='center', fontsize=6)
            ax.text(i, u*(g_abs + g_rad/2), 'rad', rotation=90, ha='center', va='center', color='w', fontsize=6)
        if mode == LASER[design][0]:
            ys = u*(g_abs + g_rad) + 0.45
            ax.plot([i], [ys], marker='*', color=STAR, ms=7.5, mec='k', mew=0.4, clip_on=False, zorder=5)
            ax.annotate('', xy=(i, ys + 0.3), xytext=(i, ys + 1.2),
                        arrowprops=dict(arrowstyle='-|>', color='k', lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
            ax.text(max(i, 0.3), ys + 1.25, 'lasing', ha='center', va='bottom', fontsize=6)
        print(f'{design} {mode}: gamma/omega at {THETA_RATES} deg: rad {g_rad:.2e}, abs {g_abs:.2e}, '
              f'tot {g_rad + g_abs:.2e}')
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAME[m] for m in ('A', 'B2', 'B1')])
    for tick, m in zip(ax.get_xticklabels(), ('A', 'B2', 'B1')):
        tick.set_color(COLOUR[m])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, 5.4)
    ax.set_yticks([0, 2, 4])
    ax.set_ylabel(r'Decay rate, $\gamma/\omega$ ($10^{-4}$)', labelpad=2)


def panel_pump(ax, design):
    d = csv(MEASURED/f'{design}_pump.csv')
    P = d['pump_uJ_cm2']
    ax.plot(P, d['I_L']/1e5, '-o', color=BLUE, ms=2.6, lw=0.8)
    ax.plot(P, d['I_R']/1e5, '-s', color=RED, mfc='w', ms=2.6, lw=0.8)
    P_th = threshold(P, d[f'I_{LASER[design][1]}'])
    s = csv(MEASURED/f'{design}_lasing_spectrum.csv')
    i = np.argmax(s[f'I_{LASER[design][1]}'])
    print(f'{design}: threshold {P_th:.1f} uJ/cm^2; lasing at {s["lambda_nm"][i]:.1f} nm with '
          f'(I_R - I_L)/(I_R + I_L) = {(s["I_R"][i] - s["I_L"][i])/(s["I_R"][i] + s["I_L"][i]):+.2f}')
    ax.axvline(P_th, color='0.55', lw=0.6, ls=':')
    ax.text(P_th, 2.15, f' {P_th:.1f}', color='0.3', ha='left', va='top')
    ax.set_ylim(0, 2.2)
    ax.set_xlim(*{'design1': (158.5, 175.5), 'design2': (168.5, 184.5)}[design])
    ax.set_xticks({'design1': [160, 165, 170, 175], 'design2': [170, 175, 180]}[design])
    ax.set_xlabel(r'Pump fluence ($\mu$J/cm$^2$)', labelpad=1)
    ax.set_ylabel(r'Emission ($10^5$ counts/s)', labelpad=2)
    labels = {'design1': (('LCP', 170.9, 0.62, 'left'), ('RCP', 174.2, 0.36, 'right')),
              'design2': (('RCP', 177.6, 0.42, 'right'), ('LCP', 183.3, 0.2, 'right'))}[design]
    for pol, x, y, ha in labels:
        ax.text(x, y, pol, color=BLUE if pol == 'LCP' else RED, ha=ha, va='center')


def panel_maps(fig, design, y, top):
    th, lam, I_L, I_R = lasing_map(design)
    vmax = max(I_L.max(), I_R.max())
    l0, l1 = {'design1': (715.0, 730.0), 'design2': (720.0, 737.0)}[design]
    for k, (I, name, col) in enumerate(((I_L, 'LCP', BLUE), (I_R, 'RCP', RED))):
        ax = axes(fig, X_MAP + k*(W_MAP + GAP_MAP), y, W_MAP, ROW_H)
        ax.pcolormesh(th, lam, I/vmax, cmap='inferno', vmin=0, vmax=1, shading='nearest', rasterized=True)
        for mode in ('A', 'B2', 'B1'):
            s, lb, _ = follow(design, mode, s_max=0.009)
            t = theta_deg(s, lb)
            t, lb = np.concatenate([-t[:0:-1], t]), np.concatenate([lb[:0:-1], lb])   # C2: theta -> -theta
            ax.plot(t, lb, color=OVERLAY[mode], lw=0.6, ls=(0, (2.5, 1.5)))
            yl = float(np.interp(0.47, t, lb))
            if k == 1 and l0 < yl < l1 - 0.8:
                ax.text(0.47, yl + 0.35, NAME[mode], color=OVERLAY[mode], ha='right', va='bottom', fontsize=6)
        ax.text(0.07, 0.97, 'Exp.', transform=ax.transAxes, ha='left', va='top', color='w', fontsize=6)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(l0, l1)
        ax.set_yticks(np.arange(5*np.ceil(l0/5), l1 + 1e-9, 5))
        ax.set_xticks([-0.5, 0, 0.5])
        if top:
            ax.set_xticklabels([])
            ax.set_title(name, color=col, pad=2)
        else:
            ax.set_xticklabels(['$-0.5$', '0', ''] if k == 0 else ['', '0', '0.5'])
            ax.set_xlabel(r'$\theta_x$ (deg)', labelpad=1)
        if k == 1:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r'Wavelength, $\lambda$ (nm)', labelpad=2)


def main():
    style()
    plt.rcParams['savefig.bbox'] = 'standard'                       # axes are placed in page inches
    for d in LASER:
        print(f'{d}: kappa {KAPPA[d]:.3e} used, {kappa_fit(d):.3e} from the fit to the measured Q')
    ff = far_fields()
    fig = plt.figure(figsize=(W, H))
    letter(fig, 0.02, ROW_Y['design1'] + ROW_H + 0.03, 'a')
    letter(fig, 0.02, 1.45, 'b')
    letters = iter('cdefghij')
    sim_axes = {}
    for design, y in ROW_Y.items():
        top = design == 'design1'
        fig.text(1.90/W, (y + ROW_H/2)/H, design.replace('design', 'DESIGN '), rotation=90, ha='center', va='center',
                 fontsize=9, fontweight='bold')
        panel_rates(axes(fig, X_RATE, y, W_RATE, ROW_H), design)
        letter(fig, X_RATE - 0.30, y + ROW_H + 0.03, next(letters))
        panel_pump(axes(fig, X_IO, y, W_IO, ROW_H), design)
        letter(fig, X_IO - 0.33, y + ROW_H + 0.03, next(letters))

        ax_exp, ax_sim = axes(fig, X_FF, y + D_FF + GAP_FF, D_FF, D_FF), axes(fig, X_FF, y, D_FF, D_FF)
        img = np.asarray(Image.open(MEASURED/f'{design}_farfield.png').convert('RGB'))
        disk(ax_exp, img, (-R_FOV, R_FOV, -R_FOV, R_FOV), None)
        F, I = ff[design]
        th = F['th']
        disk(ax_sim, I, (th[0], th[-1], th[0], th[-1]), [0.22, 0.42] if top else [-0.1, 0.1])
        Ii = RegularGridInterpolator((th, th), I, bounds_error=False, fill_value=0)
        for iy in range(0, th.size, 12):                             # ellipses where the far field is bright
            for ix in range(0, th.size, 12):
                if np.hypot(th[ix], th[iy]) < 0.9*R_FOV and Ii([[th[iy], th[ix]]])[0] > ELL_MIN:
                    ellipse(ax_sim, th[ix], th[iy], F['S1'][iy, ix], F['S2'][iy, ix], F['sigma_bot'][iy, ix])
        for ax, txt in ((ax_exp, 'Exp.'), (ax_sim, 'Sim.')):
            ax.text(0.08, 0.72, txt, transform=ax.transAxes, ha='left', va='center', color='w', fontsize=6, zorder=10)
        mode, hand = LASER[design]
        ax_exp.text(1.06, 1.02, NAME[mode], transform=ax_exp.transAxes, ha='right', va='top',
                    color=BLUE if hand == 'L' else RED)
        sim_axes[design] = (ax_exp, ax_sim)
        letter(fig, X_FF - 0.06, y + ROW_H + 0.03, next(letters))
        panel_maps(fig, design, y, top)
        letter(fig, X_MAP - 0.43, y + ROW_H + 0.03, next(letters))

    gap_lo, gap_hi = ROW_Y['design2'] + ROW_H, ROW_Y['design1']
    xt, yt = X_FF + D_FF/2 + 0.22, (gap_lo + gap_hi)/2 + 0.03
    fig.text(xt/W, yt/H, 'lasing\npolarization', ha='center', va='center', linespacing=1.0)
    for ax, xy, y_end in ((sim_axes['design1'][1], (0.12, -0.38), yt + 0.13), (sim_axes['design2'][0], (0.11, 0.25),
                                                                               yt - 0.13)):
        fig.add_artist(ConnectionPatch(xyA=((xt - 0.02)/W, y_end/H), coordsA='figure fraction', xyB=xy,
                                       coordsB=ax.transData, color='k', lw=0.4,
                                       path_effects=[pe.withStroke(linewidth=1.4, foreground='w')]))
    xc = X_MAP + W_MAP + GAP_MAP/2
    cax = axes(fig, xc - 0.30, gap_lo + 0.21, 0.60, 0.05)
    bar = fig.colorbar(ScalarMappable(Normalize(0, 1), 'inferno'), cax=cax, orientation='horizontal', ticks=[])
    bar.outline.set_linewidth(0.4)
    fig.text(xc/W, (gap_lo + 0.30)/H, 'Emission', ha='center', va='bottom')
    fig.text((xc - 0.30)/W, (gap_lo + 0.19)/H, '0', ha='left', va='top')
    fig.text((xc + 0.30)/W, (gap_lo + 0.19)/H, 'max', ha='right', va='top')
    save(fig, 'lasing')


if __name__ == '__main__':
    main()
