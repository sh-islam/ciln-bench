"""Single-row (1x4) variants of the NTH and TV figures, full \\textwidth."""
import csv, json, sys
from pathlib import Path
import numpy as np
from plot_style_ciln import (GROUP_STEP, GROUP_WIDTH, INNER_BAR_GAP, VIRIDIS4,
                             BAR_LINEWIDTH, REF_LINE_COLOR, make_grid2, make_row,
                             style_axes, darken_color, save_pdf)
from fig01_nth import SUPER as NTH_SUPER, CANONICAL as NTH_CANON
from fig04_tv import SUPER as TV_SUPER, CANONICAL as TV_CANON, GU_TV, REF_LABEL
SEVS = [1, 3, 5]

def draw_bars(ax, subs, get_val, canon):
    n_sub = len(subs)
    bar_w = (GROUP_WIDTH - INNER_BAR_GAP*(n_sub-1)) / n_sub
    xs = np.arange(len(SEVS)) * GROUP_STEP
    mx = 0
    for j, sub in enumerate(subs):
        vals = [get_val(sub, sv) for sv in SEVS]
        mx = max(mx, max(vals))
        c = VIRIDIS4[j]
        ax.bar(xs + (j-(n_sub-1)/2)*(bar_w+INNER_BAR_GAP), vals, width=bar_w,
               color=c, label=canon.get(sub, sub),
               edgecolor=darken_color(c), linewidth=BAR_LINEWIDTH)
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in SEVS])
    ax.set_xlabel("level")
    return xs, mx

def nth_row(json_path, out):
    v2 = json.load(open(json_path))["settings"]
    data = {}
    for s in v2:
        if "_sev" not in s["name"]: continue
        sub, sev = s["name"].rsplit("_sev", 1)
        data.setdefault(sub, {})[int(sev)] = s["m2_frobenius"]
    fig, axs = make_grid2(3, ncols=3, width_frac=0.88)
    families = [s for s in NTH_SUPER if s[0] != "Noise"]
    for ax, (name, subs, _w) in zip(axs, families):
        _, mx = draw_bars(ax, subs, lambda s, v: data.get(s, {}).get(v, 0), NTH_CANON)
        ax.set_ylim(0, mx * 1.32)
        ax.set_title(name)
        ax.legend(loc="upper left", ncol=2, framealpha=0.95, handlelength=1.2,
                  columnspacing=0.8, borderpad=0.3, labelspacing=0.35)
        style_axes(ax)
    axs[0].set_ylabel("NTH")
    save_pdf(fig, Path(out)/"nth_row.pdf")

def tv_row(csv_path, out):
    tv = {}
    for r in csv.DictReader(open(csv_path)):
        if r["variant"] == "S":
            tv[(r["corruption"], int(r["severity"]))] = float(r["tv_mean"])
    fig, axs = make_grid2(3, ncols=3, width_frac=0.88)
    families = [s for s in TV_SUPER if s[0] != "Noise"]
    for i, (ax, (name, subs, _w)) in enumerate(zip(axs, families)):
        xs, mx = draw_bars(ax, subs, lambda s, v: tv.get((s, v), 0), TV_CANON)
        ax.set_ylim(0, max(mx, max(GU_TV.values())) * 1.14)
        ax.set_title(name)
        for lvl, ref in GU_TV.items():
            ax.axhline(y=ref, linestyle="--", alpha=0.95,
                       color="black", linewidth=0.8, zorder=5)
            # label on every panel, right end, sitting on top of the line
            ax.text(xs[-1] + GROUP_WIDTH/2, ref + 0.008, REF_LABEL[lvl],
                    color="black", va="bottom", ha="right",
                    fontsize=6.5, fontweight="bold", zorder=6)
        ax.legend(loc="upper left", ncol=1, framealpha=0.95, handlelength=1.0,
                  borderpad=0.3, labelspacing=0.3)
        style_axes(ax)
    axs[0].set_ylabel("TV")
    save_pdf(fig, Path(out)/"tv_row.pdf")

if __name__ == "__main__":
    nth_row(sys.argv[1], sys.argv[3]); tv_row(sys.argv[2], sys.argv[3])
