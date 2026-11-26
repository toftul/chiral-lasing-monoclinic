"""SM figure: computed bands and measured angle-resolved transmission of Design 1 and of its mirror image.

(a) Design 1 along Gamma-X, (b) along Gamma-Y, (c), (d) the mirror image (x -> -x: same bands and Q, opposite
CD_mode).  Sim.: every computed mode, coloured by CD_mode, marker area ~ linewidth 1/Q, open circle: the BIC, grey:
a first diffraction order is open; the k >= 0 branch is drawn at +theta and -theta (C2).  Exp.: measured T_L and
T_R, read from the raster of the authors' figure (README, data provenance).  Angles on the figure are the
manuscript's labels (README, "Lattice angle").
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from common import (BLUE, MEASURED, NAME, RED, bands, diffraction_threshold, lattice, paper_angle, phi, save, style,
                    theta_deg)

PHI = {'design1': np.degrees(phi), 'mirror': 180.0 - np.degrees(phi)}
HAND = {'B1': -1, 'A': -1, 'B2': +1}                          # handedness of each mode in Design 1
SEEN = {'TL': {'design1': ('B1', 'A'), 'mirror': ('B2',)}, 'TR': {'design1': ('B2',), 'mirror': ('B1', 'A')}}
# mode name (theta, lambda) and arrow tip, per kind of panel and direction
ARROWS = {('sim', 'x'): {'B1': (2.9, 727.6, 0.45, 727.9), 'A': (0.0, 720.0, 0.0, 724.7),
                        'B2': (0.0, 712.4, 0.0, 717.0)},
          ('exp', 'x'): {'B1': (2.9, 729.0, 0.7, 728.2), 'A': (-3.0, 726.4, -1.6, 723.6),
                        'B2': (0.0, 723.5, 0.0, 719.2)},
          ('sim', 'y'): {'B1': (3.2, 729.3, 1.3, 727.2), 'A': (-1.5, 720.7, -1.5, 723.9),
                        'B2': (0.0, 722.2, 0.0, 718.0)},
          ('exp', 'y'): {'B1': (2.9, 729.0, 1.6, 726.6), 'A': (-3.2, 725.8, -2.6, 722.5),
                        'B2': (0.0, 723.5, 0.0, 719.0)}}
Q_LEGEND = (1e3, 3e3, 1e4, 3e4)


def area(Q):
    return np.maximum(4.5**2*1e3/Q, 1.0)


def q_label(q):
    m, e = f'{q:.0e}'.split('e')
    return rf'$Q = 10^{int(e)}$' if m == '1' else rf'$Q = {m}\times10^{int(e)}$'


def mark(ax, mode, spec, colour):
    x, y, xt, yt = spec
    ax.annotate(NAME[mode], xy=(xt, yt), xytext=(x, y), ha='center', va='center', color=colour, zorder=4,
                arrowprops=dict(arrowstyle='-|>', lw=0.5, color=colour, shrinkA=1, shrinkB=0.5, mutation_scale=5))


def cell_inset(ax, phi_deg):
    """The unit cell: pillars d + delta at the corners, d - delta at the centre."""
    a1, a2 = (np.array(v)/0.5 for v in lattice(np.radians(phi_deg)))
    corners = np.array([[0, 0], a1, a1 + a2, a2])
    poly = Polygon(corners, closed=True, fc='#eeeeee', ec='none', zorder=1)
    ax.add_patch(poly)
    for c, r, col in [(c, 0.275, '#a3a3a3') for c in corners] + [((a1 + a2)/2, 0.195, '#f5f5f5')]:
        p = Circle(c, r, fc=col, ec='#7a7a7a', lw=0.3, zorder=2)
        ax.add_patch(p)
        p.set_clip_path(poly)
    ax.plot(*np.vstack([corners, corners[:1]]).T, color='k', lw=0.4, zorder=3)
    ax.set_xlim(corners[:, 0].min() - 0.05, corners[:, 0].max() + 0.05)
    ax.set_ylim(-0.05, a2[1] + 0.05)
    ax.set_aspect('equal')
    ax.set_axis_off()


def main():
    style()
    maps = np.load(MEASURED/'sm_transmission_maps.npz')
    s_all, d_all = bands('design1')
    fig = plt.figure(figsize=(6.2, 3.9), layout='constrained')
    g = fig.add_gridspec(3, 7, width_ratios=[1, 0.72, 0.72, 0.25, 1, 0.72, 0.72], height_ratios=[1, 1, 0.085])
    for k, (letter, sample, direction) in enumerate((('a', 'design1', 'x'), ('b', 'design1', 'y'),
                                                     ('c', 'mirror', 'x'), ('d', 'mirror', 'y'))):
        r, c0 = k//2, 0 if k % 2 == 0 else 4
        sign = 1 if sample == 'design1' else -1                    # the mirror image reverses CD_mode
        branch = s_all >= 0 if direction == 'x' else s_all <= 0
        s, lam, Q, cd = s_all[branch], d_all['lambda_nm'][branch], abs(d_all['Q'][branch]), d_all['CD_mode'][branch]
        th = theta_deg(s, lam)
        bic = (s == 0) & (Q > 1e5)
        ax = fig.add_subplot(g[r, c0])
        s_thr = np.linspace(0, 0.07, 400)*(1 if direction == 'x' else -1)
        lam_thr = diffraction_threshold(s_thr)
        for side in (1, -1):
            ax.fill_between(side*theta_deg(s_thr, lam_thr), 710, lam_thr, color='0.9', lw=0, zorder=0)
            ax.plot(side*theta_deg(s_thr, lam_thr), lam_thr, color='0.5', ls='--', lw=0.5, zorder=0)
            o = np.argsort(-area(Q))
            o = o[~bic[o]]
            dots = ax.scatter(side*th[o], lam[o], s=area(Q[o]), c=sign*cd[o], cmap='bwr', vmin=-1, vmax=1,
                              linewidths=0)
        ax.scatter([0], lam[bic][:1], s=area(Q[bic][:1]) + 14, facecolors='none', edgecolors='0.3', linewidths=0.5,
                   zorder=3)
        ax.set_xlim(-4, 4)
        ax.set_ylim(710, 730)
        ax.set_ylabel(r'Wavelength, $\lambda$ (nm)')
        ax.text(0.06, 0.97, 'Sim.', transform=ax.transAxes, ha='left', va='top')
        ax.text(-0.5, 1.02, f'({letter})', transform=ax.transAxes, ha='left', va='bottom')
        cell_inset(ax.inset_axes([0.02, 0.0, 0.34, 0.17]), PHI[sample])
        ax.set_title(rf'$\varphi={paper_angle(PHI[sample]):g}^\circ$', pad=2)
        for m, spec in ARROWS[('sim', direction)].items():
            mark(ax, m, spec, BLUE if HAND[m]*sign < 0 else RED)
        row = [ax]
        for j, (ch, title, col) in enumerate((('TL', r'LCP in, $T_\mathrm{L}$', BLUE),
                                              ('TR', r'RCP in, $T_\mathrm{R}$', RED))):
            axm = fig.add_subplot(g[r, c0 + 1 + j], sharey=ax)
            im = axm.imshow(maps[f'{sample}_{direction}_{ch}'], origin='lower', extent=[-4, 4, 710, 730], aspect='auto',
                            cmap='bone', vmin=0, vmax=1, interpolation='bilinear', rasterized=True)
            axm.set_title(title, color=col, pad=2)
            axm.text(0.06, 0.97, 'Exp.', transform=axm.transAxes, ha='left', va='top')
            axm.tick_params(labelleft=False)
            for m in SEEN[ch][sample]:
                mark(axm, m, ARROWS[('exp', direction)][m], BLUE if HAND[m]*sign < 0 else RED)
            row.append(axm)
        for axi in row:
            axi.set_xticks([-4, -2, 0, 2, 4] if axi is ax else [-4, 0, 4])
            axi.set_xlabel(rf'$\theta_{direction}$ (deg)')

    for cell, mappable, ticks, label in ((g[2, 0], dots, [-1, 0, 1], r'$\mathrm{CD}_\mathrm{mode}$'),
                                         (g[2, 1:3], im, [0, 0.5, 1], 'Transmission')):
        bar = fig.colorbar(mappable, cax=fig.add_subplot(cell), orientation='horizontal', ticks=ticks)
        bar.outline.set_linewidth(0.5)
        bar.set_label(label, labelpad=1)
    lax = fig.add_subplot(g[2, 4:7])
    lax.set_axis_off()
    lax.legend([Line2D([], [], ls='', marker='o', color='0.3', markeredgewidth=0, markersize=np.sqrt(area(q)))
                for q in Q_LEGEND], [q_label(q) for q in Q_LEGEND], loc='center', ncol=4, handletextpad=0.1,
               columnspacing=0.6, borderaxespad=0, fontsize=6)
    save(fig, 'sm_transmission')


if __name__ == '__main__':
    main()
