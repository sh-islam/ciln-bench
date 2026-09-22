"""Fig 6 (DM per-sample loss bimodality): reference fig06 kde_lines variant —
clean vs noisy loss densities, 3 settings x 2 phases (warmup ep10 / final
ep99), noisy-image scenario — with the paper viridis loss palette and
pub-ready TMLR 2x3 grid. Data: dm_losses_<setting>_N_<phase>.csv (seed 0).

Run: python fig06_dm.py <csv-dir> <out-dir>
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pub_ready_plots as prp
from scipy.stats import gaussian_kde

from plot_style_ciln import LOSS_PALETTE, GRID_COLOR, make_grid2, style_axes, darken_color, save_pdf

SETTINGS = [("frost_sev1", "frost (1), 10.1%"), ("contrast_sev5", "contrast (5), 47.3%"),
            ("glass_blur_sev3", "glass-blur (3), 64.6%")]
PHASES = [("warmup", "warmup"), ("final", "final")]


def load(csv_dir, key, phase):
    clean, noisy = [], []
    with open(Path(csv_dir) / f"dm_losses_{key}_N_{phase}.csv") as f:
        for r in csv.DictReader(f):
            (noisy if r["is_noisy"] == "1" else clean).append(float(r["loss"]))
    return np.asarray(clean), np.asarray(noisy)


def draw(ax, clean, noisy, *, legend, ylabel, title=None, xlabel=True):
    upper = np.percentile(np.concatenate([clean, noisy]), 99)
    xs = np.linspace(0, upper, 400)
    cd, nd = gaussian_kde(clean)(xs), gaussian_kde(noisy)(xs)
    for d, key in ((cd, "clean"), (nd, "noisy")):
        c = LOSS_PALETTE[key]
        ax.fill_between(xs, 0, d, color=c, alpha=0.14, linewidth=0)
        ax.plot(xs, d, color=darken_color(c, 0.72), lw=1.0, label=key)
    ax.set_xlim(0, upper)
    ax.set_ylim(bottom=0)
    if xlabel:
        ax.set_xlabel("per-sample loss")
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel("density")
    if legend:
        ax.legend(loc="upper right", framealpha=0.95, handlelength=1.0,
                  borderpad=0.3, labelspacing=0.3)
    style_axes(ax)


def main(csv_dir, out_dir):
    fig, axs = make_grid2(6, extra_top_line=True, aspect=0.21, hspace=1.1, width_frac=0.765)
    # one setting per row: warmup left, final right; column headers once on top row
    panels = [(key, slabel, phase, plabel)
              for key, slabel in SETTINGS for phase, plabel in PHASES]
    for i, (ax, (key, slabel, phase, plabel)) in enumerate(zip(axs, panels)):
        clean, noisy = load(csv_dir, key, phase)
        draw(ax, clean, noisy, legend=(i == 0), ylabel=(i % 2 == 0),
             title=None, xlabel=(i >= 4))
    # warmup/final column headers once, at the very top
    for col, (_ph, plabel) in enumerate(PHASES):
        pc = axs[col].get_position()
        fig.text((pc.x0 + pc.x1) / 2, axs[0].get_position().y1 + 0.115,
                 plabel, ha="center", va="bottom", fontsize=9)
    # one setting label per row, centered over the panel pair
    for r, (_key, slabel) in enumerate(SETTINGS):
        pl, pr = axs[2 * r].get_position(), axs[2 * r + 1].get_position()
        x = (pl.x0 + pr.x1) / 2
        y = pl.y1 + 0.025
        fig.text(x, y, slabel, ha="center", va="bottom", fontsize=9)
    save_pdf(fig, Path(out_dir) / "dm_grid.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
