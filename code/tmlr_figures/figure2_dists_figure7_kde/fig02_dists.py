"""Fig 2 (label distributions) + appendix kde (sorted-frequency) panels:
reference fig02 logic — sampled labels via the PL-IDN protocol (seed 0) from raw
voter softmaxes — with the paper viridis severity palette and pub-ready TMLR
row layout matching Figs 3/4.

Run: python fig02_dists.py <fig02-data-dir> <out-dir>
"""
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from plot_style_ciln import (
    GROUP_STEP, GROUP_WIDTH, INNER_BAR_GAP, BAR_LINEWIDTH, SEV_PALETTE,
    make_grid2, make_row, style_axes, darken_color, save_pdf,
)
SORTED_LINESTYLES = {"clean": (0, (1.2, 1.6)), 1: "solid", 5: "solid"}

CIFAR_CLASSES = ["plane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MNIST_CLASSES = [str(i) for i in range(10)]
ADULT_CLASSES = ["<=50K", ">50K"]
V_CIFAR = ["resnet20", "wrn28_10", "deit3_small", "clip"]
V_MNIST = ["lenet5", "mlp", "resnet20", "deit3_small"]
V_ADULT = ["xgboost_dummyna", "mlp", "ft_transformer", "tabpfn", "catboost"]

CANONICAL = {"gaussian_noise": "gaussian", "impulse_noise": "impulse",
             "missing_mcar": "mcar", "missing_mnar": "mnar", "rotate": "rotate"}

# main Fig 2 row (matches the paper's current panel selection)
MAIN_ROW = [
    ("cifar", "brightness", V_CIFAR, CIFAR_CLASSES, "cifar10"),
    ("cifar", "pixelate", V_CIFAR, CIFAR_CLASSES, "cifar10"),
    ("mnist", "brightness", V_MNIST, MNIST_CLASSES, "mnist"),
]
# appendix kde row (matches paper fig:sorted_frequency_profiles)
KDE_ROW = [
    ("cifar", "pixelate", V_CIFAR, CIFAR_CLASSES, "cifar10"),
    ("mnist", "rotate", V_MNIST, MNIST_CLASSES, "mnist"),
    ("mnist", "brightness", V_MNIST, MNIST_CLASSES, "mnist"),
]
# appendix adult dist panels (row of two, half-width)
ADULT_ROW = [
    ("adult", "missing_mcar", V_ADULT, ADULT_CLASSES, "adult"),
    ("adult", "missing_mnar", V_ADULT, ADULT_CLASSES, "adult"),
]

DATA = None  # set in main


def sampled_labels(nl_dir, voters):
    sms = [np.load(nl_dir / f"softmax_{v}.npy") for v in voters]
    n = sms[0].shape[0]
    rng = np.random.default_rng(0)
    vi = rng.integers(0, len(sms), size=n)
    labels = np.empty(n, dtype=np.int64)
    for i, sm in enumerate(sms):
        m = vi == i
        if m.any():
            labels[m] = sm[m].argmax(axis=1)
    return labels


def label_share(labels, n_classes):
    return np.bincount(labels, minlength=n_classes) / len(labels) * 100


def panel_dists(family, voters, root, n_classes, sevs=(1, 3, 5)):
    # clean = TRUE labels (paper convention), not voter-sampled labels
    clean = np.load(DATA / "clean" / root / "noisylabeltrain_clean" / "labels.npy")
    dists = [("clean", label_share(clean, n_classes))]
    for sev in sevs:
        d = DATA / root / family / f"severity_{sev}" / "noisy_label_train"
        if (d / f"softmax_{voters[0]}.npy").exists():
            dists.append((sev, label_share(sampled_labels(d, voters), n_classes)))
    return dists


def entropy_bits(dist):
    p = np.asarray(dist, float) / 100
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def sev_label(v):
    return "clean" if v == "clean" else f"level {v}"


def draw_dist(ax, dataset, family, voters, classes, root, *, legend, ylabel, xlabel=True):
    dists = panel_dists(family, voters, root, len(classes))
    x = np.arange(len(classes)) * GROUP_STEP
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
    rot = 45 if len(classes) == 10 else 0
    ax.set_xticklabels(classes, rotation=rot,
                       ha="right" if rot else "center", fontsize=7)
    if xlabel:
        ax.set_xlabel("class")
    ax.set_title(f"{dataset} {CANONICAL.get(family, family)}")
    if ylabel:
        ax.set_ylabel("share (%)")
    if legend:
        ax.legend(loc="upper left", framealpha=0.95, handlelength=1.0,
                  borderpad=0.3, labelspacing=0.3)
    style_axes(ax)


def draw_sorted(ax, dataset, family, voters, classes, root, *, legend, ylabel):
    dists = panel_dists(family, voters, root, len(classes), sevs=(1, 5))
    x = np.arange(1, len(classes) + 1)
    mx = 0
    for lvl, dist in dists:
        c = SEV_PALETTE[lvl]
        srt = np.sort(dist)
        xs = np.linspace(x.min(), x.max(), 300)
        ys = np.clip(PchipInterpolator(x, srt)(xs), 0, None)
        mx = max(mx, ys.max())
        ax.fill_between(xs, 0, ys, color=c, alpha=0.14, linewidth=0)
        ax.plot(xs, ys, color=darken_color(c), lw=1.0,
                linestyle=SORTED_LINESTYLES[lvl],
                label=f"{sev_label(lvl)} H={entropy_bits(dist):.2f}")
    ax.set_xlim(1, len(classes))
    ax.set_ylim(0, mx * 1.12 if mx else 1)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=4.5)
    ax.set_xlabel("class frequency rank")
    ax.set_title(f"{dataset} {CANONICAL.get(family, family)}")
    if ylabel:
        ax.set_ylabel("share (%)")
    if legend:
        ax.legend(loc="upper left", framealpha=0.95, handlelength=1.0,
                  borderpad=0.3, labelspacing=0.3)
    style_axes(ax)


def main(data_dir, out_dir):
    global DATA
    DATA = Path(data_dir)
    out = Path(out_dir)

    fig, axs = make_grid2(3, ncols=3, width_frac=0.88)  # 1x3 family standard
    for i, (ax, spec) in enumerate(zip(axs, MAIN_ROW)):
        draw_dist(ax, *spec, legend=(i == 0), ylabel=(i == 0))
    save_pdf(fig, out / "dist_row.pdf")

    fig, axs = make_grid2(3, ncols=3, width_frac=0.88)  # 1x3 family standard
    for i, (ax, spec) in enumerate(zip(axs, KDE_ROW)):
        draw_sorted(ax, *spec, legend=True, ylabel=(i == 0))
    save_pdf(fig, out / "kde_row.pdf")

    fig, axs = make_grid2(2)
    for i, (ax, spec) in enumerate(zip(axs, ADULT_ROW)):
        draw_dist(ax, *spec, legend=(i == 0), ylabel=(i % 2 == 0))
    save_pdf(fig, out / "dist_adult_row.pdf")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
