"""SM figure: self-interference of the two lasers in a Michelson interferometer, (a) slot for the interferometer
sketch (drawn by hand, not here), (b) camera frames of the A-band laser (Design 1) and of the B2 laser (Design 2)
from the authors' figure (README, data provenance); scale bar 0.2 deg.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

from common import MEASURED, RED, BLUE, save, style


def main():
    style()
    fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.0), width_ratios=[1.4, 1, 1])
    axs[0].axis('off')
    axs[0].text(0, 1.02, '(a)', transform=axs[0].transAxes, ha='left', va='bottom')
    for ax, name, title, col in ((axs[1], 'design1_A', 'A', BLUE), (axs[2], 'design2_B2', r'B$_2$', RED)):
        im = ax.imshow(np.asarray(Image.open(MEASURED/'sm_interference'/f'{name}.png').convert('RGB')),
                       extent=(-1, 1, -1, 1))
        im.set_clip_path(Circle((0, 0), 1, transform=ax.transData))
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, color=col, pad=2)
    axs[1].text(-0.15, 1.02, '(b)', transform=axs[1].transAxes, ha='left', va='bottom')
    save(fig, 'sm_interference')


if __name__ == '__main__':
    main()
