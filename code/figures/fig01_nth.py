"""NTH panels (paper Fig 3): reference fig01 layout — x=severity level, one bar
per corruption, in-panel legend — with thesis families, thesis viridis
palette, and pub-ready TMLR sizing.

Data: v2_all_native.json (m2_frobenius == thesis NTH, verified against the
thesis characterisation table).
Run:  python fig01_nth.py <path-to-v2_all_native.json> <out-dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

from plot_style_ciln import (
    GROUP_STEP, GROUP_WIDTH, INNER_BAR_GAP, VIRIDIS4, BAR_LINEWIDTH,
    make_panel, style_axes, darken_color, save_pdf,
)

CANONICAL = {
    "gaussian_noise": "gaussian", "shot_noise": "shot", "impulse_noise": "impulse",
    "defocus_blur": "defocus", "glass_blur": "glass", "motion_blur": "motion",
    "zoom_blur": "zoom", "elastic_transform": "elastic",
    "jpeg_compression": "jpeg", "pixelate": "pixelate", "contrast": "contrast",
    "fog": "fog", "frost": "frost", "snow": "snow", "brightness": "brightness",
}

# thesis families (brightness -> Weather, elastic/jpeg -> Digital, no Geometric)
SUPER = [
    ("Noise",   ["gaussian_noise", "shot_noise", "impulse_noise"],            0.478),
    ("Blur",    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],   0.456),
    ("Weather", ["fog", "frost", "snow", "brightness"],                       0.478),
    ("Digital", ["contrast", "elastic_transform", "pixelate", "jpeg_compression"], 0.456),
]
SEVS = [1, 3, 5]


def main(json_path, out_dir):
    v2 = json.load(open(json_path))["settings"]
    data = {}
    for s in v2:
        nm = s["name"]
        if "_sev" not in nm:
            continue
        sub, sev = nm.rsplit("_sev", 1)
        data.setdefault(sub, {})[int(sev)] = s["m2_frobenius"]

    y_max = max(
        data.get(sub, {}).get(sv, 0)
        for _n, subs, _w in SUPER for sub in subs for sv in SEVS
    ) * 1.32

    for name, subs, width_frac in SUPER:
        fig, ax = make_panel(width_frac)
        n_sub = len(subs)
        bar_w = (GROUP_WIDTH - INNER_BAR_GAP * (n_sub - 1)) / n_sub
        xs = np.arange(len(SEVS)) * GROUP_STEP
        for j, sub in enumerate(subs):
            vals = [data.get(sub, {}).get(sv, 0) for sv in SEVS]
            c = VIRIDIS4[j]
            ax.bar(
                xs + (j - (n_sub - 1) / 2) * (bar_w + INNER_BAR_GAP),
                vals, width=bar_w, color=c, label=CANONICAL.get(sub, sub),
                edgecolor=darken_color(c), linewidth=BAR_LINEWIDTH,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in SEVS])
        ax.set_xlabel("level")
        ax.set_ylabel("NTH")
        ax.set_ylim(0, y_max)
        ax.legend(loc="upper left", ncol=2, framealpha=0.95,
                  handlelength=1.2, columnspacing=0.9,
                  borderpad=0.35, labelspacing=0.35)
        style_axes(ax, grid_axis="y")
        save_pdf(fig, Path(out_dir) / f"nth_{name.lower()}.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
