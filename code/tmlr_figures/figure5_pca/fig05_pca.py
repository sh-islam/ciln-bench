"""Fig 5: ten CILN settings closest to CIFAR-10H in PCA space.

Rebuilt from the released projections (tv_pca_2d_data/pca_projections.csv:
the all-49 PCA fit; verified to reproduce the released coordinates exactly).
Explained variance from pca_explained_variance.csv (69%/15%). y/x limits
extended so the PL-IDN-H marker and label print in full.

Run: python fig05_pca.py <tv_pca_2d_data-dir> <out-dir>
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pub_ready_plots as prp

from plot_style_ciln import PASTEL, GRID_COLOR, darken_color, save_pdf

TOP10 = [
    ("brightness_sev1", "brightness", 1), ("defocus_blur_sev1", "defocus-blur", 1),
    ("fog_sev1", "fog", 1), ("contrast_sev1", "contrast", 1),
    ("brightness_sev3", "brightness", 3), ("frost_sev1", "frost", 1),
    ("pixelate_sev1", "pixelate", 1), ("fog_sev3", "fog", 3),
    ("snow_sev1", "snow", 1), ("brightness_sev5", "brightness", 5),
]
CORR_COLORS = {
    "brightness": PASTEL[1], "fog": PASTEL[7], "frost": PASTEL[0],
    "snow": PASTEL[9], "defocus-blur": PASTEL[4], "contrast": PASTEL[3],
    "pixelate": PASTEL[2],
}
SEV_MARKERS = {1: "o", 3: "s", 5: "^"}
PLIDN_COLOR = darken_color(PASTEL[4], 0.52)  # dark pastel purple (palette-derived)
REF_COLOR = "#8b0000"    # prof fig04 ref red — one accent red paper-wide


def main(data_dir, out_dir):
    d = Path(data_dir)
    proj = {r["name"]: (float(r["PC1"]), float(r["PC2"]))
            for r in csv.DictReader(open(d / "pca_projections.csv"))}
    ev = {r["component"]: float(r["explained_variance_ratio"])
          for r in csv.DictReader(open(d / "pca_explained_variance.csv"))}

    rc, w, _h = prp.get_mpl_rcParams(layout=prp.Layout.TMLR, width_frac=0.9,
                                     height_frac=0.2)
    rc = dict(rc)
    rc.update({"grid.color": GRID_COLOR, "grid.linewidth": 0.4,
               "axes.edgecolor": "black", "legend.fontsize": 7,
               "axes.titlesize": 9})
    plt.rcParams.update(rc)
    fig, ax = plt.subplots(figsize=(3.08, 1.82))

    seen = set()
    for name, corr, sev in TOP10:
        x, y = proj[name]
        c = CORR_COLORS[corr]
        ax.scatter(x, y, marker=SEV_MARKERS[sev], s=64, color=c,
                   edgecolor=darken_color(c), linewidth=0.8, zorder=3,
                   label=corr if corr not in seen else None)
        seen.add(corr)

    for key, lab in [("PL-IDN_low", "PL-IDN-L"), ("PL-IDN_medium", "PL-IDN-M"),
                     ("PL-IDN_high", "PL-IDN-H")]:
        x, y = proj[key]
        ax.scatter(x, y, marker="+", s=130, color=PLIDN_COLOR,
                   linewidth=1.8, zorder=3)
        if lab == "PL-IDN-M":
            kw = dict(xytext=(-2, -9), ha="right", va="top")
        elif lab == "PL-IDN-L":
            kw = dict(xytext=(7, -1), ha="left", va="center")
        else:
            kw = dict(xytext=(7, -1), ha="left", va="center")
        ax.annotate(lab, (x, y), textcoords="offset points", fontsize=7,
                    color=PLIDN_COLOR, **kw)

    x, y = proj["CIFAR-10H"]
    ax.scatter(x, y, marker="x", s=110, color=REF_COLOR, linewidth=2.2,
               zorder=4)
    ax.annotate("CIFAR-10H", (x, y), xytext=(6, -3), textcoords="offset points",
                ha="left", va="top", fontsize=7, color=REF_COLOR)

    # severity legend entries (gray markers)
    for sev, m in SEV_MARKERS.items():
        ax.scatter([], [], marker=m, s=52, color="#bdbdbd",
                   edgecolor="#6e6e6e", linewidth=0.8, label=f"sev {sev}")

    ax.set_xlabel(f"PC1 ({ev["PC1"] * 100:.0f}% var.)")
    ax.set_ylabel(f"PC2 ({ev["PC2"] * 100:.0f}% var.)")
    # extended limits: PL-IDN-H (y=0.036) and its label print in full
    ax.set_ylim(0.004, 0.044)
    ax.margins(x=0.05)
    ax.set_xlim(-0.048, 0.004)
    ax.legend(loc="upper left", ncol=2, framealpha=0.95,
              handletextpad=0.25, columnspacing=0.9, borderpad=0.45,
              labelspacing=0.45)
    ax.grid(True, alpha=1.0)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    save_pdf(fig, Path(out_dir) / "pca_top10_pub.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
