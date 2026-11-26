"""SM figure: measured angle-resolved photoluminescence along Gamma-X of Design 1 (a) and Design 2 (b), LCP and RCP
detection, and CD = (I_RCP - I_LCP)/(I_RCP + I_LCP).  The maps are read from the raster of the authors' figure with
its own colour bars and colour maps (README, data provenance).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from common import BLUE, MEASURED, NAME, RED, save, style

PL_MAX = 4e3                                             # counts/s, top of the authors' colour bar
COLOUR = {'B1': BLUE, 'A': BLUE, 'B2': RED}
# (sample, channel, mode): mode name (theta, lambda) and arrow tip
LABELS = {('design1', 'LCP', 'B1'): (2.9, 735.5, 1.3, 730.3), ('design1', 'LCP', 'A'): (-3.3, 719.0, -1.5, 724.0),
          ('design1', 'RCP', 'B2'): (0.7, 726.5, 0.7, 720.6),
          ('design2', 'LCP', 'A'): (3.0, 738.0, 1.0, 732.6), ('design2', 'LCP', 'B1'): (3.3, 724.5, 2.0, 729.0),
          ('design2', 'RCP', 'B2'): (0.65, 730.5, 0.65, 724.3)}


def main():
    style()
    d = np.load(MEASURED/'sm_pl_maps.npz')
    pl_cmap, cd_cmap = ListedColormap(d['pl_cmap']), ListedColormap(d['cd_cmap'])
    fig = plt.figure(figsize=(6.2, 2.45), layout='constrained')
    g = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 0.22, 1, 1, 1], height_ratios=[1, 0.1])
    for k, (letter, sample, d_nm) in enumerate((('a', 'design1', 235), ('b', 'design2', 255))):
        first = None
        for j, (ch, col) in enumerate((('LCP', BLUE), ('RCP', RED), ('CD', 'k'))):
            ax = fig.add_subplot(g[0, 4*k + j], sharey=first)
            first = first or ax
            cd = ch == 'CD'
            im = ax.imshow(d[f'{sample}_{ch}'], origin='lower', extent=[-4, 4, 700, 740], aspect='auto',
                           cmap=cd_cmap if cd else pl_cmap, vmin=-1 if cd else 0, vmax=1 if cd else PL_MAX,
                           interpolation='bilinear', rasterized=True)
            if cd:
                cd_im = im
            else:
                pl_im = im
            ax.set_xticks([-4, 0, 4])
            ax.set_xlabel(r'$\theta_x$ (deg)')
            ax.set_title(ch, color=col, pad=2)
            if j == 0:
                ax.set_ylabel(r'Wavelength, $\lambda$ (nm)')
                ax.text(-0.42, 1.02, f'({letter})', transform=ax.transAxes, ha='left', va='bottom')
                ax.text(0.05, 0.97, rf'$d={d_nm}$ nm', transform=ax.transAxes, ha='left', va='top')
            else:
                ax.tick_params(labelleft=False)
            for (s, c, m), (x, y, xt, yt) in LABELS.items():
                if (s, c) == (sample, ch):
                    ax.annotate(NAME[m], xy=(xt, yt), xytext=(x, y), ha='center', va='center', color=COLOUR[m],
                                arrowprops=dict(arrowstyle='-|>', lw=0.5, color=COLOUR[m], shrinkA=1, shrinkB=0.5,
                                                mutation_scale=5))
    pb = fig.colorbar(pl_im, cax=fig.add_subplot(g[1, 1]), orientation='horizontal', ticks=[0, 2e3, 4e3])
    pb.ax.set_xticklabels(['0', '2', '4'])
    pb.set_label(r'PL (counts/s, $10^3$)', labelpad=1)
    cb = fig.colorbar(cd_im, cax=fig.add_subplot(g[1, 5]), orientation='horizontal', ticks=[-1, 0, 1])
    cb.set_label('CD', labelpad=1)
    for bar in (pb, cb):
        bar.outline.set_linewidth(0.5)
    save(fig, 'sm_pl')


if __name__ == '__main__':
    main()
