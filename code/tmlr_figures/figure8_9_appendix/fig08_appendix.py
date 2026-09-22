"""Appendix Fig 8 rows (Adult + AG-News characterization), standardized.

Row 1 (fig8_dist_row.pdf): corruption-induced label distributions —
adult mcar, adult mnar, agnews butter-fingers, agnews front-truncation.
Clean = TRUE labels (thesis convention). Adult sampled labels use the
reference fig02 rng voter-pick protocol (adult was in that script); AG-News
uses MAJORITY VOTE of per-voter argmax — validated: reproduces the thesis
sev-5 flip-distribution table (tab:agnews_flip_dist) to <5e-5 on all 24
cells.

Row 2 (fig8_nth_row.pdf): NTH by family — adult Missing (mcar/mar/mnar),
adult Value (gaussian, scaling), agnews Character (butter-fingers),
agnews Structural (front-truncation). NTH computed with the one-hot
argmax implementation validated 12/12 vs v2_all_native.json on CIFAR and
6/6 vs thesis prose AG-News values.

Run: python fig08_appendix.py <fig02_data> <fig8_data> <out-dir>
"""
import sys
from pathlib import Path

import numpy as np

from plot_style_ciln import (
    GROUP_STEP, GROUP_WIDTH, INNER_BAR_GAP, VIRIDIS4, BAR_LINEWIDTH,
    SEV_PALETTE, make_grid2, style_axes, darken_color, save_pdf,
)

V_ADULT = ["xgboost_dummyna", "mlp", "ft_transformer", "tabpfn", "catboost"]
V_AGNEWS = ["fasttext", "distilbert", "roberta", "sbert"]
ADULT_CLASSES = [r"$\leq$50K", r"$>$50K"]
AGNEWS_CLASSES = ["World", "Sports", "Business", "Sci-Tech"]
SEVS = [1, 3, 5]
CANONICAL = {
    "missing_mcar": "mcar", "missing_mar": "mar", "missing_mnar": "mnar",
    "gaussian_noise": "gaussian", "scaling": "scaling",
    "butter_fingers": "butter-fingers", "front_truncation": "front-trunc",
}

FIG02 = None  # fig02 bundle data root (has adult mcar/mnar/gaussian/scaling)
FIG8 = None    # fig8 bundle root (has adult mar + all agnews)


def nl_dir(dataset, family, sev):
    """Locate a severity dir across the two bundles."""
    for root in (FIG8, FIG02):
        d = root / dataset / family / f"severity_{sev}" / "noisy_label_train"
        if (d / "labels.npy").exists():
            return d
    raise FileNotFoundError(f"{dataset}/{family}/sev{sev}")


def clean_labels(dataset):
    for root in (FIG8, FIG02):
        p = root / "clean" / dataset / "noisylabeltrain_clean" / "labels.npy"
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(f"clean labels for {dataset}")


def rng_pick_labels(d, voters):
    """reference fig02 protocol (validated for cifar/mnist/adult)."""
    sms = [np.load(d / f"softmax_{v}.npy") for v in voters]
    n = sms[0].shape[0]
    rng = np.random.default_rng(0)
    vi = rng.integers(0, len(sms), size=n)
    out = np.empty(n, dtype=np.int64)
    for i, sm in enumerate(sms):
        m = vi == i
        if m.any():
            out[m] = sm[m].argmax(1)
    return out


def majority_labels(d, voters, n_classes):
    """Thesis AG-News protocol (validated vs tab:agnews_flip_dist)."""
    am = np.stack([np.load(d / f"softmax_{v}.npy").argmax(1) for v in voters])
    counts = np.apply_along_axis(np.bincount, 0, am, minlength=n_classes)
    return counts.argmax(0)


def label_share(labels, n_classes):
    return np.bincount(labels, minlength=n_classes) / len(labels) * 100


def nth(d, voters, y, n_classes):
    sms = [np.load(d / f"softmax_{v}.npy") for v in voters]
    pbar = np.stack([np.eye(n_classes)[s.argmax(1)] for s in sms]).mean(0)
    tot = 0.0
    for k in range(n_classes):
        m = y == k
        cm = pbar[m].mean(0)
        tot += ((pbar[m] - cm) ** 2).sum()
    return tot / len(y)


def sev_label(v):
    return "clean" if v == "clean" else f"level {v}"


def draw_dist(ax, title, dataset, family, voters, classes, proto, *, xlabel=True,
              legend, ylabel):
    n_classes = len(classes)
    dists = [("clean", label_share(clean_labels(dataset), n_classes))]
    for sev in SEVS:
        d = nl_dir(dataset, family, sev)
        lab = (majority_labels(d, voters, n_classes) if proto == "majority"
               else rng_pick_labels(d, voters))
        dists.append((sev, label_share(lab, n_classes)))
    x = np.arange(n_classes) * GROUP_STEP
    n = len(dists)
    w = (GROUP_WIDTH - INNER_BAR_GAP * (n - 1)) / n
    for i, (lvl, dist) in enumerate(dists):
        c = SEV_PALETTE[lvl]
        ax.bar(x + (i - (n - 1) / 2) * (w + INNER_BAR_GAP), dist, width=w,
               color=c, label=sev_label(lvl),
               edgecolor=darken_color(c), linewidth=BAR_LINEWIDTH)
    top = max(max(d) for _l, d in dists)
    ax.set_ylim(0, top * (1.45 if legend else 1.12))
    ax.set_xticks(x)
    rot = 20 if len(classes) > 2 else 0
    ax.set_xticklabels(classes, fontsize=7, rotation=rot,
                       ha="right" if rot else "center")
    if xlabel:
        ax.set_xlabel("class")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel("share (%)")
    if legend:
        ax.legend(loc="upper right", framealpha=0.95, handlelength=1.0,
                  borderpad=0.3, labelspacing=0.3)
    style_axes(ax)


def draw_nth(ax, title, dataset, families, voters, y, n_classes, *,
             legend, ylabel, y_max):
    n_sub = len(families)
    bar_w = (GROUP_WIDTH - INNER_BAR_GAP * (n_sub - 1)) / n_sub
    xs = np.arange(len(SEVS)) * GROUP_STEP
    for j, fam in enumerate(families):
        vals = [nth(nl_dir(dataset, fam, sv), voters, y, n_classes)
                for sv in SEVS]
        c = VIRIDIS4[j]
        ax.bar(xs + (j - (n_sub - 1) / 2) * (bar_w + INNER_BAR_GAP), vals,
               width=bar_w, color=c, label=CANONICAL.get(fam, fam),
               edgecolor=darken_color(c), linewidth=BAR_LINEWIDTH)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in SEVS])
    ax.set_xlabel("level")
    ax.set_title(title)
    ax.set_ylim(0, y_max)
    if ylabel:
        ax.set_ylabel("NTH")
    if legend is not None:
        ax.legend(loc=legend, framealpha=0.95, handlelength=1.2,
                  borderpad=0.35, labelspacing=0.35)
    style_axes(ax, grid_axis="y")


def main(fig02_data, fig8_data, out_dir):
    global FIG02, FIG8
    FIG02, FIG8 = Path(fig02_data), Path(fig8_data)
    out = Path(out_dir)

    # ---- Row 1: label distributions ----
    fig, axs = make_grid2(3, ncols=3, width_frac=0.88)  # 1x3 family standard
    draw_dist(axs[0], "Adult mnar", "adult", "missing_mnar", V_ADULT,
              ADULT_CLASSES, "rng", legend=True, ylabel=True)
    draw_dist(axs[1], "AG-News butter-fingers", "agnews", "butter_fingers",
              V_AGNEWS, AGNEWS_CLASSES, "majority", legend=False, ylabel=False)
    draw_dist(axs[2], "AG-News front-trunc", "agnews", "front_truncation",
              V_AGNEWS, AGNEWS_CLASSES, "majority", legend=False, ylabel=False)
    save_pdf(fig, out / "fig8_dist_row.pdf")

    # ---- Row 2: NTH ----
    ya, yg = clean_labels("adult"), clean_labels("agnews")
    specs = [
        ("Adult Value", "adult", ["gaussian_noise", "scaling"], V_ADULT,
         ya, 2),
        ("AG-News Character", "agnews", ["butter_fingers"], V_AGNEWS, yg, 4),
        ("AG-News Structural", "agnews", ["front_truncation"], V_AGNEWS,
         yg, 4),
    ]
    fig, axs = make_grid2(3, ncols=3, width_frac=0.88)
    for i, (title, ds, fams, v, y, nc) in enumerate(specs):
        y_max = 1.48 * max(nth(nl_dir(ds, f, sv), v, y, nc)
                           for f in fams for sv in SEVS)
        draw_nth(axs[i], title, ds, fams, v, y, nc,
                 legend="upper left",
                 ylabel=i % 2 == 0, y_max=y_max)
    # single-corruption agnews panels: name the corruption in the title
    save_pdf(fig, out / "fig8_nth_row.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
