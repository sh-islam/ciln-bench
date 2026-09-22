"""TV panels (paper Fig 4): reference fig04 layout — x=severity level, one bar
per corruption, GU (PL-IDN) reference lines with low/med labels — with the
new verified TV data, paper families, paper viridis palette, pub-ready
TMLR sizing.

Data: tv_cifar_all_settings.csv (S variant; verified digit-for-digit against
the paper characterisation table on all 10 matched settings).
Run:  python fig04_tv.py <path-to-tv_cifar_all_settings.csv> <out-dir>
"""
import csv
import sys
from pathlib import Path

import numpy as np

from plot_style_ciln import (
    GROUP_STEP, GROUP_WIDTH, INNER_BAR_GAP, VIRIDIS4, BAR_LINEWIDTH,
    REF_LINE_COLOR, make_panel, style_axes, darken_color, save_pdf,
)

GU_TV = {"low": 0.180, "med": 0.301}          # paper: 0.180±0.005, 0.301±0.008
REF_LABEL = {"low": "low (0.18)", "med": "med (0.30)"}

CANONICAL = {
    "gaussian-noise": "gaussian", "shot-noise": "shot", "impulse-noise": "impulse",
    "defocus-blur": "defocus", "glass-blur": "glass", "motion-blur": "motion",
    "zoom-blur": "zoom", "elastic-transform": "elastic",
    "jpeg-compression": "jpeg", "pixelate": "pixelate", "contrast": "contrast",
    "fog": "fog", "frost": "frost", "snow": "snow", "brightness": "brightness",
}

# paper families
SUPER = [
    ("Noise",   ["gaussian-noise", "shot-noise", "impulse-noise"],            0.478),
    ("Blur",    ["defocus-blur", "glass-blur", "motion-blur", "zoom-blur"],   0.456),
    ("Weather", ["fog", "frost", "snow", "brightness"],                       0.478),
    ("Digital", ["contrast", "elastic-transform", "pixelate", "jpeg-compression"], 0.456),
]
SEVS = [1, 3, 5]


def main(csv_path, out_dir):
    tv = {}
    for r in csv.DictReader(open(csv_path)):
        if r["variant"] != "S":
            continue
        tv[(r["corruption"], int(r["severity"]))] = float(r["tv_mean"])

    for name, subs, width_frac in SUPER:
        fig, ax = make_panel(width_frac)
        n_sub = len(subs)
        bar_w = (GROUP_WIDTH - INNER_BAR_GAP * (n_sub - 1)) / n_sub
        xs = np.arange(len(SEVS)) * GROUP_STEP
        panel_max = max(GU_TV.values())
        for j, sub in enumerate(subs):
            vals = [tv.get((sub, sv), 0) for sv in SEVS]
            panel_max = max(panel_max, max(vals))
            c = VIRIDIS4[j]
            ax.bar(
                xs + (j - (n_sub - 1) / 2) * (bar_w + INNER_BAR_GAP),
                vals, width=bar_w, color=c, label=CANONICAL.get(sub, sub),
                edgecolor=darken_color(c), linewidth=BAR_LINEWIDTH,
            )
        for lvl, ref in GU_TV.items():
            ax.axhline(y=ref, linestyle="--", alpha=0.95,
                       color=REF_LINE_COLOR, linewidth=0.9, zorder=5)
            ax.text(xs[-1] + GROUP_WIDTH / 2, ref + 0.008, REF_LABEL[lvl],
                    color=REF_LINE_COLOR, va="bottom", ha="right",
                    fontsize=5.5, fontweight="bold", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=0.5))
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in SEVS])
        ax.set_xlabel("level")
        ax.set_ylabel("TV")
        ax.set_ylim(0, max(0.42, panel_max * 1.12))
        ax.legend(loc="upper left", ncol=1, framealpha=0.95,
                  handlelength=1.2, borderpad=0.35, labelspacing=0.35)
        style_axes(ax, grid_axis="y")
        save_pdf(fig, Path(out_dir) / f"tv_{name.lower()}.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
