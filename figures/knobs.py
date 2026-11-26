"""Fig. 2: two nearly independent knobs, the dimerization delta (Q) and the lattice angle phi (handedness).

(a) Bands of Design 1 along Y <- Gamma -> X, coloured by CD_mode [Eq. (1)], marker area ~ linewidth 1/Q; grey: a
    first diffraction order is open; open circle: the BIC.
(b) C2 parity p_n of the three Gamma states: |E_par| (grey) and Re E_par (arrows) on the plane of the pillar bases.
(c) Q and CD_mode of B1, B2 at Gamma against delta;  (d) against phi.  The states are followed from
    (delta, phi) = (40 nm, 75 deg) by the overlap of their fields (data/computed/knobs, compute/fem.py knobs).
Angles on the figure are the manuscript's labels, 180 deg - phi (README).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter, ScalarFormatter
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment

from common import COMPUTED, MODES, NAME, PML_MAX, bands, diffraction_threshold, paper_angle, save, style

KNOBS = COMPUTED/'knobs'
GREYS = LinearSegmentedColormap.from_list('white_grey', ['white', '0.42'])
BLUE, RED = plt.get_cmap('seismic')(0.1), plt.get_cmap('seismic')(0.9)   # CD_mode -1 and +1


def area(Q):
    """Marker area (pt^2) ~ linewidth 1/Q."""
    return 0.55*np.maximum(5.5**2*1e3/Q, 0.6**2)


def reference():
    """The (40 nm, 75 deg) solution and the index of each Gamma state in it."""
    d = np.load(KNOBS/'delta_40.npz')
    idx = {m: int(np.argmin(abs(d['rows'][:, 0] - lam))) for m, lam in MODES['design1'].items()}
    for m, i in idx.items():
        assert (d['p'][i] > 0) == (m == 'A'), f'{m}: p = {d["p"][i]:+.2f}'      # A even, B odd
    return d, idx


def follow(kind, values):
    """Q, CD_mode and field overlap of B1, B2 against the knob, followed from the reference both ways among the
    C2-odd states."""
    d0, idx = reference()
    ref = 40.0 if kind == 'delta' else 75.0
    names = ('B1', 'B2')
    out = {m: {ref: (abs(d0['rows'][idx[m], 1]), d0['rows'][idx[m], 4], 1.0)} for m in names}
    vals = sorted(values)
    i0 = vals.index(ref)
    for branch in (vals[i0 + 1:], vals[:i0][::-1]):
        prev = np.array([d0['u'][idx[m]].ravel() for m in names])
        for v in branch:
            d = np.load(KNOBS/f'{kind}_{v:g}.npz')
            keep = np.flatnonzero((d['rows'][:, 5] < PML_MAX) & (d['p'] < 0))
            u = d['u'][keep].reshape(len(keep), -1)
            O = abs(prev.conj() @ u.T)**2/np.outer(np.sum(abs(prev)**2, 1), np.sum(abs(u)**2, 1))
            ia, ib = linear_sum_assignment(-O)
            for i, j in zip(ia, ib):
                out[names[i]][v] = (abs(d['rows'][keep[j], 1]), d['rows'][keep[j], 4], O[i, j])
            prev = u[ib[np.argsort(ia)]]
    res = {}
    for m in names:
        x = np.array(sorted(out[m]))
        res[m] = (x, *np.array([out[m][v] for v in x]).T)
    return res


def panel_bands(fig, ax):
    s, d = bands('design1')
    Q = abs(d['Q'])
    ss = np.linspace(-0.061, 0.032, 600)
    ax.fill_between(ss, 710, diffraction_threshold(ss), color='0.9', lw=0, zorder=0)
    ax.plot(ss, diffraction_threshold(ss), color='0.55', ls='--', lw=0.5)
    ax.axvline(0, color='0.6', ls=':', lw=0.5, zorder=0)
    bic = (s == 0) & (Q > 1e6)
    o = np.argsort(-area(Q))
    o = o[~bic[o]]
    sc = ax.scatter(s[o], d['lambda_nm'][o], s=area(Q[o]), c=d['CD_mode'][o], cmap='seismic', vmin=-1, vmax=1,
                    linewidths=0)
    ax.scatter(s[bic], d['lambda_nm'][bic], s=area(Q[bic]) + 6, facecolors='none', edgecolors='0.4', linewidths=0.6,
               zorder=3)
    ax.set_xlim(-0.061, 0.032)
    ax.set_ylim(710, 732)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):g}'))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.text(0.0275, 716.0, 'diffraction', color='0.4', rotation=90, ha='center', va='center')
    ax.set_xlabel(r'$k_y a/2\pi$  $\leftarrow\Gamma\rightarrow$  $k_x a/2\pi$')
    ax.set_ylabel(r'Wavelength, $\lambda$ (nm)')
    for m, xy in (('B1', (0.013, 725.6)), ('A', (-0.008, 721.8)), ('B2', (-0.014, 715.4))):
        ax.annotate(NAME[m] + (' (BIC)' if m == 'A' else ''), xy=(0.0, MODES['design1'][m]), xytext=xy, ha='center',
                    va='center', arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3', shrinkA=2, shrinkB=1.5))
    cax = ax.inset_axes([0.52, 1.035, 0.46, 0.035])
    bar = fig.colorbar(sc, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
    bar.outline.set_linewidth(0.4)
    cax.xaxis.set_ticks_position('top')
    cax.tick_params(length=1.5, pad=1)
    cax.text(-0.06, 0.5, r'$\mathrm{CD}_\mathrm{mode}$', transform=cax.transAxes, ha='right', va='center')


def parity_map(ax, d, i, title):
    """|E_par| and Re E_par of state i on the plane z = t (pillar bases) inside one cell; + marks the C2 centre."""
    a1, a2, x, y = d['a1'], d['a2'], d['x'], d['y']
    E = d['E_map'][i].astype(complex)
    E *= np.exp(-0.5j*np.angle(np.sum(E*E)))                    # global phase: sum E.E real and positive
    mag = np.hypot(abs(E[..., 0]), abs(E[..., 1]))
    cell = Polygon(np.array([[0, 0], a1, a1 + a2, a2]), closed=True, fc='none', ec='k', lw=0.6, zorder=4)
    img = ax.imshow(mag/mag.max(), extent=(x[0], x[-1], y[0], y[-1]), origin='lower', cmap=GREYS, vmin=0, vmax=1,
                    interpolation='bilinear', rasterized=True)
    ax.add_patch(cell)
    img.set_clip_path(cell)
    d_cn, delta = float(d['d']), float(d['delta_nm'])*1e-3
    for centre, diameter in [(c, d_cn + delta) for c in (np.zeros(2), a1, a2, a1 + a2)] + [((a1 + a2)/2, d_cn - delta)]:
        circ = Circle(centre, diameter/2 + float(d['taper']), fc='none', ec='0.3', lw=0.4, ls=(0, (1.5, 1.2)), zorder=3)
        ax.add_patch(circ)
        circ.set_clip_path(cell)
    ss, tt = np.meshgrid((np.arange(6) + 0.5)/6, (np.arange(5) + 0.5)/5)
    px, py = ss*a1[0] + tt*a2[0], ss*a1[1] + tt*a2[1]
    U, V = [RegularGridInterpolator((y, x), E[..., k].real)(np.column_stack([py.ravel(), px.ravel()])) for k in (0, 1)]
    n = np.hypot(U, V).max()
    ax.quiver(px.ravel(), py.ravel(), U/n, V/n, color='k', scale=7.2, scale_units='width', width=0.016, headwidth=3.2,
              headlength=3.2, headaxislength=2.9, pivot='mid', zorder=5)
    ax.plot(*(a1 + a2)/2, ls='', marker='+', ms=5, mew=1.0, color=RED, zorder=6)
    ax.set_xlim(-0.01, a1[0] + a2[0] + 0.01)
    ax.set_ylim(-0.01, a2[1] + 0.01)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, pad=1.5)
    return img


def knob_panels(top, bot, data, x, log):
    for m, col in (('B1', BLUE), ('B2', RED)):
        v, Q, cd, overlap = data[m]
        top.plot(x(v), Q, 'o-', color=col, ms=2.4, lw=0.9)
        bot.plot(x(v), cd, 'o-', color=col, ms=2.4, lw=0.9)
        if (overlap < 0.9).any():
            print(f'{m}: field overlap below 0.9 at {v[overlap < 0.9]}')
    top.set_yscale('log')
    bot.set_ylim(-1.18, 1.18)
    bot.set_yticks([-1, 0, 1])
    bot.axhline(0, color='0.85', lw=0.5, zorder=0)
    if log:
        for ax in (top, bot):
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator([3, 10, 30]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())


def main():
    style()
    fig = plt.figure(figsize=(6.2, 2.35), layout='constrained')
    sfa, sfb, sfc, sfd = fig.subfigures(1, 4, width_ratios=[1.62, 0.95, 1.3, 1.13], wspace=0.0)
    panel_bands(sfa, sfa.subplots())

    mos = sfb.subplot_mosaic([['cb'], ['B1'], ['A'], ['B2']], height_ratios=[0.07, 1, 1, 1],
                             gridspec_kw=dict(hspace=0.02))
    mos['cb'].axis('off')
    d, idx = reference()
    for m in ('B1', 'A', 'B2'):
        p = d['p'][idx[m]]
        img = parity_map(mos[m], d, idx[m], rf'{NAME[m]}, {"even" if p > 0 else "odd"}, $p={round(p):+d}$')
        print(f'{m}: lambda {d["rows"][idx[m], 0]:.2f} nm, p = {p:+.4f}')
    cax = mos['cb'].inset_axes([0.3, 0.0, 0.55, 1.0])
    bar = sfb.colorbar(img, cax=cax, orientation='horizontal', ticks=[0, 1])
    bar.outline.set_linewidth(0.4)
    cax.xaxis.set_ticks_position('top')
    cax.tick_params(length=1.5, pad=1)
    cax.text(-0.06, 0.5, r'$|E_\parallel|$', transform=cax.transAxes, ha='right', va='center')

    ct = sfc.add_subplot(2, 1, 1)
    cb = sfc.add_subplot(2, 1, 2, sharex=ct)
    dd = follow('delta', [2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    knob_panels(ct, cb, dd, lambda v: v, log=True)
    q40 = np.interp(40, dd['B2'][0], dd['B2'][1])
    xx = np.array([2.0, 70.0])
    ct.plot(xx, 1.8*q40*(40/xx)**2, color='0.55', ls='--', lw=0.6, zorder=0)
    ct.text(12, 2.9*q40*(40/12)**2, r'$\propto\delta^{-2}$', color='0.4', ha='left', va='bottom')
    for ax in (ct, cb):
        ax.set_xlim(2.0, 70.0)
        ax.axvline(40, color='0.6', ls=':', lw=0.5, zorder=0)
    ct.tick_params(labelbottom=False)
    ct.set_ylabel(r'$Q$-factor')
    cb.set_ylabel(r'$\mathrm{CD}_\mathrm{mode}$')
    cb.set_xlabel(r'Dimerization, $\delta$ (nm)')
    ct.set_title(rf'$\varphi={paper_angle(75):g}$°', pad=2)

    pt = sfd.add_subplot(2, 1, 1, sharey=ct)
    pb = sfd.add_subplot(2, 1, 2, sharex=pt, sharey=cb)
    dp = follow('phi', [90.0, 87.5, 85.0, 82.5, 80.0, 77.5, 75.0])
    knob_panels(pt, pb, dp, paper_angle, log=False)
    for ax in (pt, pb):
        ax.set_xlim(89, 106)
        ax.xaxis.set_major_locator(FixedLocator([90, 95, 100, 105]))
        ax.tick_params(labelleft=False)
    pt.tick_params(labelbottom=False)
    pb.set_xlabel(r'Lattice angle, $\varphi$ (deg)')
    pt.set_title(r'$\delta=40$ nm', pad=2)
    lo = min(v[1].min() for data in (dd, dp) for v in data.values())
    hi = max(v[1].max() for data in (dd, dp) for v in data.values())
    ct.set_ylim(lo/2.5, hi*5)
    for ax, m, x, y in ((cb, 'B2', 2.6, 0.72), (cb, 'B1', 2.6, -0.72), (ct, 'B2', 3.6, 8e6), (ct, 'B1', 3.6, 6e4),
                        (pb, 'B2', 98.5, 0.45), (pb, 'B1', 98.5, -0.45), (pt, 'B2', 101.0, 2.6e4),
                        (pt, 'B1', 101.0, 1.1e3)):
        ax.text(x, y, NAME[m], color=RED if m == 'B2' else BLUE, ha='center' if ax is pt else 'left', va='center')

    fig.canvas.draw()
    for sf, letter in zip((sfa, sfb, sfc, sfd), 'abcd'):
        fig.text(fig.transFigure.inverted().transform(sf.bbox)[0, 0] + 0.002, 0.995, f'({letter})', va='top')
    save(fig, 'knobs')
    for name, data, x in (('delta (nm)', dd, lambda v: v), ('phi, manuscript (deg)', dp, paper_angle)):
        for m, (v, Q, cd, _) in data.items():
            print(name, m, ' '.join(f'{xv:g}: Q {q:.0f} CD {c:+.4f}' for xv, q, c in zip(x(v), Q, cd)))


if __name__ == '__main__':
    main()
