"""SM figure: measured far fields of the A-band laser of Design 1 (a) and of its mirror image (b).  First column:
the emission with the polarization ellipses of the authors' Stokes polarimetry; then the emission through a linear
analyzer at 90, 135, 0 and 45 deg and through an LCP and an RCP analyzer.  Camera frames from the authors' figure
(README, data provenance); scale bar 0.2 deg.  Angles on the figure are the manuscript's labels (README).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from common import MEASURED, paper_angle, phi, save, style

FRAMES = MEASURED/'sm_farfield'
SETTINGS = ('total', 'pol90', 'pol135', 'pol0', 'pol45', 'LCP', 'RCP')


def frame(ax, path):
    im = ax.imshow(np.asarray(Image.open(path).convert('RGB')), extent=(-1, 1, -1, 1))
    im.set_clip_path(Circle((0, 0), 1, transform=ax.transData))
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    style()
    fig = plt.figure(figsize=(6.2, 2.3))
    g = fig.add_gridspec(4, 7, height_ratios=[0.28, 1, 0.16, 1], wspace=0.06, hspace=0.04)
    for j, setting in enumerate(SETTINGS):
        ax = fig.add_subplot(g[0, j])
        ax.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')                                        # true analyzer angles
        if setting.startswith('pol'):                              # double arrow along the analyzer axis
            t = np.radians(float(setting[3:]))
            ax.annotate('', xy=(0.9*np.cos(t), 0.9*np.sin(t)), xytext=(-0.9*np.cos(t), -0.9*np.sin(t)),
                        arrowprops=dict(arrowstyle='<|-|>', lw=0.6, color='k', mutation_scale=4))
        elif setting != 'total':
            ax.text(0, 0, setting, ha='center', va='center')
    for r, (sample, phi_deg, letter) in enumerate((('design1', np.degrees(phi), 'a'),
                                                   ('mirror', 180 - np.degrees(phi), 'b'))):
        for j, setting in enumerate(SETTINGS):
            ax = fig.add_subplot(g[1 + 2*r, j])
            frame(ax, FRAMES/f'{sample}_{setting}.png')
            if j == 0:
                ax.text(-1.0, 1.08, f'({letter})  $\\varphi={paper_angle(phi_deg):g}^\\circ$', ha='left', va='bottom')
    cax = fig.add_subplot(g[2, 5:7])
    cax.imshow(np.asarray(Image.open(FRAMES/'colorbar.png').convert('RGB')), extent=(0, 1, 0, 1), aspect='auto')
    cax.set_position([cax.get_position().x0 + 0.03, cax.get_position().y0 + 0.01, 0.12, 0.02])
    cax.set_xticks([0, 1])
    cax.set_xticklabels(['0', 'max'])
    cax.set_yticks([])
    cax.tick_params(length=0, pad=1)
    save(fig, 'sm_farfield')


if __name__ == '__main__':
    main()
