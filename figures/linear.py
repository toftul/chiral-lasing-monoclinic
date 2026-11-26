"""Fig. 3: linear response of Design 1 (d = 235 nm, delta = 40 nm).

(a) slot for the sample schematic (drawn by hand, not here).
(b) Transmittance T_ij (output i, input j) at theta_x = 0.2 deg: measured (x) and computed with the effective loss
    kappa (lines; compute/fem.py spectrum).
(c) Measured T_L = T_LL + T_RL (LCP in), T_R = T_RR + T_LR (RCP in), CD_tot = (T_L - T_R)/(T_L + T_R); dots: the
    computed modes along Gamma-X, coloured by CD_mode.
(d) The same computed by RCWA with kappa (compute/rcwa.py); dashed: onset of the first diffraction order.
(e)-(g) Decay rates of the A band, B2 and B1: gamma_rad/omega = 1/(2 Q) (dashed, lossless band), gamma_abs/omega =
    kappa (S_pillar/n_pillar + S_slab/n_slab) (dotted, first order in kappa, shares at Gamma) and their sum
    gamma_tot (solid); symbols: measured 1/(2 Q).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import (BLUE, COMPUTED, KAPPA, MEASURED, RED, absorption_share, bands, csv, diffraction_threshold,
                    follow, kappa_fit, save, style, theta_deg)

CD_MAX = 0.2
GREY, MAGENTA = '0.55', '#ff00e5'
COLOUR = {'A': 'k', 'B2': RED, 'B1': BLUE}
MARKER = {'A': 'o', 'B2': 's', 'B1': '^'}
TITLE = {'A': 'A (BIC band)', 'B2': r'B$_2$', 'B1': r'B$_1$'}


def letter(ax, s, x=-0.2):
    ax.text(x, 1.02, f'({s})', transform=ax.transAxes, ha='left', va='bottom')


def grid_map(name, cols):
    """A measured map, theta running fastest: theta, lambda and each column as [lambda, theta]."""
    d = csv(MEASURED/name)
    th, lam = np.unique(d['theta_deg']), np.unique(d['lambda_nm'])
    assert np.allclose(d['theta_deg'][:len(th)], th)
    return th, lam, [d[c].reshape(len(lam), len(th)) for c in cols]


def panel_spectrum(ax):
    d = csv(MEASURED/'design1_spectrum_theta0.2.csv')
    m = np.load(COMPUTED/'spectrum_design1.npz')
    T = abs(m['t'])**2                                                 # t[i, j]: output i, input j; 0 = R, 1 = L
    for ch, (i, j), col in (('LR', (1, 0), GREY), ('RL', (0, 1), GREY), ('RR', (0, 0), RED), ('LL', (1, 1), BLUE)):
        ax.plot(d['lambda_nm'], d[f'T_{ch}'], 'x', color=col, ms=1.6 if col != GREY else 1.2, mew=0.4, alpha=0.8)
        ax.plot(m['lambda_nm'], T[:, i, j], '-', color=col, lw=0.9)
    ax.set_xlim(715, 730)
    ax.set_ylim(-0.03, 1.2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r'Wavelength, $\lambda$ (nm)', labelpad=1)
    ax.set_ylabel('Transmission')
    ax.set_title(rf'$\theta_x={float(m["theta_deg"]):g}$°', pad=2)
    for ch, x, y in (('RR', 715.4, 0.62), ('LL', 727.7, 0.55), ('LR', 718.6, 0.09), ('RL', 727.7, 0.09)):
        ax.text(x, y, rf'$T_\mathrm{{{ch}}}$', color={'RR': RED, 'LL': BLUE}.get(ch, GREY), ha='left', va='center')
    for txt, x, col in ((r'B$_2$', 717.75, RED), ('A', 724.95, 'k'), (r'B$_1$', 726.85, BLUE)):
        ax.text(x, 1.09, txt, color=col, ha='center', va='bottom')
    ax.legend(handles=[Line2D([], [], color='0.3', marker='x', ms=2.5, mew=0.5, ls='none', label='Exp.'),
                       Line2D([], [], color='0.3', lw=0.9, label='Sim. with effective loss')],
              loc='center left', bbox_to_anchor=(0.0, 0.27), handlelength=1.2, borderaxespad=0.1)


def panel_rates(ax, mode, measured, kappa):
    s, lam, Q = follow('design1', mode)
    th = theta_deg(s, lam)
    g_rad, g_abs = 1/(2*Q), kappa*absorption_share('design1')[mode]
    col, u = COLOUR[mode], 1e4
    ax.plot(th, u*g_rad, '--', color=col, lw=0.8)
    ax.axhline(u*g_abs, color=col, ls=':', lw=0.8)
    ax.plot(th, u*(g_rad + g_abs), '-', color=col, lw=1.1)
    v = measured[mode]
    ax.plot(v[:, 0], u/(2*v[:, 1]), MARKER[mode], color=col, mfc='w' if mode == 'A' else col, ms=3.2, mew=0.7,
            ls='none', clip_on=False, zorder=5)
    labels = {'A': dict(tot=(0.8, -1), abs=(0.45, -1), rad=(0.8, -1)),
              'B2': dict(tot=(1.02, 1), abs=(0.5, -1), rad=(0.2, 1)),
              'B1': dict(tot=(0.62, 1), abs=(0.62, 1), rad=(0.62, -1))}[mode]
    for kind, (x, side) in labels.items():                            # each rate written on its line
        y = {'tot': g_rad + g_abs, 'abs': g_abs + 0*g_rad, 'rad': g_rad}[kind]
        ax.text(x, u*float(np.interp(x, th, y)) + side*0.15, rf'$\gamma_\mathrm{{{kind}}}$', color=col,
                ha='center', va='bottom' if side > 0 else 'top')
    if mode == 'A':                                                    # the lowest measured rate
        p = (v[0, 0], u/(2*v[0, 1]))
        ax.scatter(*p, s=110, facecolors='none', edgecolors=MAGENTA, linestyles=':', linewidths=0.8, zorder=6,
                   clip_on=False)
        ax.annotate('lowest, best\ncandidate\nfor lasing', xy=p, xytext=(0.03, 0.98), textcoords='axes fraction',
                    ha='left', va='top', color=MAGENTA, fontsize=6, linespacing=1.1,
                    arrowprops=dict(arrowstyle='-', color=MAGENTA, lw=0.6, shrinkA=1, shrinkB=6, relpos=(0.1, 0.0)))
    ax.set_xlim(-0.03, 1.2)
    ax.set_ylim(0, 5.8)
    ax.set_yticks([0, 2, 4])
    ax.set_xticks([0, 0.5, 1.0])
    ax.text(1.0, 1.02, TITLE[mode], transform=ax.transAxes, ha='right', va='bottom', color=col)


def main():
    style()
    kappa = KAPPA['design1']
    print(f'kappa: {kappa:.3e} used, {kappa_fit("design1"):.3e} from the fit to the measured Q')
    fig = plt.figure(figsize=(6.2, 3.9), layout='constrained')
    g = fig.add_gridspec(3, 5, width_ratios=[1.62, 0.65, 0.65, 0.65, 1.2], height_ratios=[1, 0.045, 1])

    ax_a = fig.add_subplot(g[0:2, 0])
    ax_a.set_axis_off()
    letter(ax_a, 'a', x=0.0)
    ax_b = fig.add_subplot(g[2, 0])
    panel_spectrum(ax_b)
    letter(ax_b, 'b', x=-0.25)

    th_m, lam_m, (TL_m, TR_m) = grid_map('design1_transmission_map.csv', ('T_L', 'T_R'))
    keep = th_m >= -1e-9
    t = np.load(COMPUTED/'transmission_map_design1.npz')
    T = abs(t['t'])**2
    TL_s, TR_s = (T[..., 1, 1] + T[..., 0, 1]).T, (T[..., 0, 0] + T[..., 1, 0]).T
    s, d = bands('design1')
    x_branch = s >= 0
    s_thr = np.linspace(0, 0.07, 400)
    lam_thr = diffraction_threshold(s_thr)
    rows = ((th_m[keep], lam_m, [Z[:, keep] for Z in (TL_m, TR_m, (TL_m - TR_m)/(TL_m + TR_m))]),
            (t['theta_deg'], t['lambda_nm'], [TL_s, TR_s, (TL_s - TR_s)/(TL_s + TR_s)]))
    ims = {}
    for r, (th, lam, maps) in enumerate(rows):
        for c, (Z, title, col) in enumerate(zip(maps, (r'LCP in, $T_\mathrm{L}$', r'RCP in, $T_\mathrm{R}$',
                                                       r'$\mathrm{CD}_\mathrm{tot}$'), (BLUE, RED, 'k'))):
            ax = fig.add_subplot(g[2*r, 1 + c])
            cd = c == 2
            ims['CD' if cd else 'T'] = ax.pcolormesh(th, lam, Z, cmap='seismic' if cd else 'bone',
                                                     vmin=-CD_MAX if cd else 0, vmax=CD_MAX if cd else 1,
                                                     shading='nearest', rasterized=True)
            ax.set_xlim(0, 4)
            ax.set_ylim(700, 740)
            ax.set_xticks([0, 2, 4])
            if r == 1:
                ax.plot(theta_deg(s_thr, lam_thr), lam_thr, '--', color='0.6', lw=0.4)
                ax.set_xlabel(r'$\theta_x$ (deg)')
            else:
                ax.set_xticklabels([])
                ax.set_title(title, color=col, pad=2)
                if not cd:
                    ax.scatter(theta_deg(s[x_branch], d['lambda_nm'][x_branch]), d['lambda_nm'][x_branch],
                               c=d['CD_mode'][x_branch], s=1.2, cmap='bwr', vmin=-1, vmax=1, linewidths=0, alpha=0.9)
            if not cd:
                for m, x, y, va in (('B1', 1.2, 730.4, 'bottom'), ('A', 1.55, 722.0, 'top'),
                                    ('B2', 0.75, 715.6, 'top')):
                    ax.text(x, y, {'A': 'A (BIC)'}.get(m, TITLE[m]), color=COLOUR[m], ha='center', va=va)
            ax.text(0.06, 0.97, ('Exp.', 'Sim.')[r], transform=ax.transAxes, ha='left', va='top')
            if c == 0:
                ax.set_ylabel(r'Wavelength, $\lambda$ (nm)')
                letter(ax, 'cd'[r], x=-0.62)
            else:
                ax.set_yticklabels([])
    for key, cols, ticks, label in (('T', slice(1, 3), [0, 0.5, 1], 'Transmission'),
                                    ('CD', slice(3, 4), [-CD_MAX, 0, CD_MAX], r'$\mathrm{CD}_\mathrm{tot}$')):
        cax = fig.add_subplot(g[1, cols])
        bar = fig.colorbar(ims[key], cax=cax, orientation='horizontal', ticks=ticks)
        bar.set_label(label, labelpad=1)
        bar.outline.set_linewidth(0.4)
        cax.tick_params(length=1.5, pad=1)
    print(f'model CD_tot reaches {np.nanmax(abs(rows[1][2][2])):.2f} (colour scale clipped at {CD_MAX})')

    q = np.genfromtxt(MEASURED/'design1_Q.csv', delimiter=',', names=True, dtype=None, encoding=None)
    measured = {m: np.column_stack([q['theta_deg'][q['mode'] == m], q['Q'][q['mode'] == m]]) for m in COLOUR}
    rates = g[:, 4].subgridspec(3, 1)
    for k, (mode, let) in enumerate(zip(('A', 'B2', 'B1'), 'efg')):
        ax = fig.add_subplot(rates[k])
        panel_rates(ax, mode, measured, kappa)
        letter(ax, let)
        if k == 2:
            ax.set_xticklabels(['0', '0.5', '1'])
            ax.set_xlabel(r'$\theta_x$ (deg)')
        else:
            ax.set_xticklabels([])
        if k == 1:
            ax.set_ylabel(r'Decay rate, $\gamma/\omega$ ($10^{-4}$)')
    save(fig, 'linear')


if __name__ == '__main__':
    main()
