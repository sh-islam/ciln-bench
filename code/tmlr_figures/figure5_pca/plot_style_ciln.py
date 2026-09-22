"""Shared plotting style: reference reproduction-script structure,
CILN thesis viridis palette, pub-ready-plots TMLR standardization."""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pub_ready_plots as prp

# ---- palette (seaborn "pastel" hexes, plot_style.py picks) ----
# PASTEL = sns.color_palette("pastel").as_hex(); hardcoded to avoid the dep.
PASTEL = ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff",
          "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"]
VIRIDIS4 = [PASTEL[0], PASTEL[1], PASTEL[2], PASTEL[4]]  # bar series (lab FIG1/FIG4 picks 0,1,2,4)
SEV_COLORS = {1: PASTEL[0], 3: PASTEL[2], 5: PASTEL[3]}
SEV_PALETTE = {"clean": PASTEL[7], 1: PASTEL[0], 3: PASTEL[2], 5: PASTEL[3]}  # lab SEVERITY_PALETTE 7,0,2,3
LOSS_PALETTE = {"clean": PASTEL[0], "noisy": PASTEL[3]}  # lab: clean blue, noisy red
REF_LINE_COLOR = "#8b0000"  # lab fig04
PANEL_ASPECT = 5.5 / 6.6  # lab PANEL_SIZE proportions

# ---- geometry constants kept from the original scripts ----
GROUP_STEP = 1.25
GROUP_WIDTH = 0.70
INNER_BAR_GAP = 0.035
BAR_LINEWIDTH = 0.5
GRID_COLOR = "0.86"

TMLR_TEXTWIDTH_IN = 6.5


def make_panel(width_frac, height_frac=0.215):
    """One pub-ready TMLR panel sized as a width_frac slot of \\textwidth."""
    rc, w, h = prp.get_mpl_rcParams(
        layout=prp.Layout.TMLR, width_frac=width_frac, height_frac=height_frac
    )
    rc = dict(rc)
    rc.update({
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.4,
        "axes.edgecolor": "black",
    })
    plt.rcParams.update(rc)
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def make_row(ncols, height_frac=0.165):
    """One full-textwidth pub-ready TMLR row of ncols panels, shared y."""
    rc, w, h = prp.get_mpl_rcParams(
        layout=prp.Layout.TMLR, width_frac=1.0, height_frac=height_frac
    )
    rc = dict(rc)
    rc.update({
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.4,
        "axes.edgecolor": "black",
        "legend.fontsize": 5,
        "axes.titlesize": 7,
    })
    plt.rcParams.update(rc)
    fig, axs = plt.subplots(1, ncols, figsize=(w, h), sharey=True,
                            gridspec_kw={"wspace": 0.09})
    return fig, axs


def style_axes(ax, *, grid_axis="y"):
    ax.grid(True, alpha=1.0, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.tick_params(color="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")


def darken_color(color, amount=0.55):
    rgb = mcolors.to_rgb(color)
    return tuple(max(0.0, c * amount) for c in rgb)


def save_pdf(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {path}")


def make_grid2(n_panels, *, aspect=PANEL_ASPECT, hspace=0.42, wspace=0.55,
               sharey=False, extra_top_line=False, width_frac=0.70, ncols=2):
    """Pub-ready TMLR figure, panels in rows of 2; odd count -> last centered.

    Width and rc come from pub_ready_plots at width_frac=1.0; height is set so
    each panel's drawn axes box has the lab PANEL_SIZE aspect (measured, not
    guessed). Returns (fig, [axes in panel order]).
    """
    import numpy as _np
    nrows = (n_panels + ncols - 1) // ncols
    rc, w, _h = prp.get_mpl_rcParams(layout=prp.Layout.TMLR,
                                     width_frac=width_frac, height_frac=0.2)
    rc = dict(rc)
    rc.update({
        "grid.color": GRID_COLOR, "grid.linewidth": 0.4,
        "axes.edgecolor": "black", "legend.fontsize": 7,
        "axes.titlesize": 9,
    })
    plt.rcParams.update(rc)

    def build(h):
        fig = plt.figure(figsize=(w, h))
        gs = fig.add_gridspec(nrows, 2 * ncols, hspace=hspace, wspace=wspace)
        axes, ref = [], None
        for i in range(n_panels):
            r, c = divmod(i, ncols)
            span = (slice(1, 3) if (ncols == 2 and i == n_panels - 1 and n_panels % 2 == 1)
                    else slice(2 * c, 2 * c + 2))
            ax = fig.add_subplot(gs[r, span], sharey=ref if sharey else None)
            if ref is None:
                ref = ax
            axes.append(ax)
        return fig, axes

    # measure once with dummy decorations, then solve height for target aspect
    trial_h = 2.4 * nrows
    fig, axes = build(trial_h)
    for ax in axes:
        ax.set_title("Xg\nXg" if extra_top_line else "Xg")
        ax.set_xlabel("Xg")
    axes[0].set_ylabel("Xg")
    fig.canvas.draw()
    b = axes[0].get_position()
    aw, ah = b.width * w, b.height * trial_h
    plt.close(fig)
    # axes height scales ~linearly with fig height (margins are fractions)
    target_ah = aw * aspect
    h = trial_h * target_ah / ah
    return build(h)
