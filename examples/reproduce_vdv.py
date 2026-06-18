"""Reproduce the VDV (= paper's ``nth``) numbers from results/v2_all_native.json.

For each of the 45 CIFAR-10 settings we recompute the VDV from the released
voter softmaxes and compare against the pre-computed JSON. With matched
inputs the numbers are bit-identical.

Usage:

    python examples/reproduce_vdv.py --data-root PATH

``--data-root`` should point to the ``settings/`` directory inside a
HuggingFace download. The layout under it is::

    <root>/<corruption>_sev<n>/noisy_label_train/
        labels.npy
        softmax_resnet20.npy
        softmax_wrn28_10.npy
        softmax_deit3_small.npy
        softmax_clip.npy

E.g. after downloading ``sh-islam/ciln-bench-cifar10`` to
``./ciln-bench-cifar10/``, use ``--data-root ./ciln-bench-cifar10/settings``.

This is the 22,500-row NLT split that the ICDE paper's headline numbers were
computed on. (A separate 10,000-row test-split copy of the same voters is
used only by ``reproduce_tv.py`` for the CIFAR-10H comparison, because
CIFAR-10H itself is 10,000 test images.)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code" / "analyze"))

from public.metrics import vote_distribution_variance, soft_label_from_voters

V2_VOTERS = ["resnet20", "wrn28_10", "deit3_small", "clip"]


def load_setting(setting_dir: Path):
    argmax = np.stack(
        [np.load(setting_dir / f"softmax_{v}.npy").argmax(axis=1) for v in V2_VOTERS],
        axis=1,
    )
    p = soft_label_from_voters(argmax, n_classes=10)
    true_y = np.load(setting_dir / "labels.npy")
    return p, true_y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data-root", type=Path, required=True,
        help="Path to a directory containing <corr>/severity_<n>/"
             "noisy_label_train/ with voter softmaxes and labels "
             "(e.g. release_v1/cifar10).",
    )
    args = ap.parse_args()

    expected = {r["name"]: r for r in json.load(
        open(REPO / "results" / "v2_all_native.json")
    )["settings"]}

    print(f"Recomputing VDV from {args.data_root}")
    print(f"  comparing to released numbers ({len(expected)} settings)\n")
    print(f"{'setting':<28}  {'VDV (ours)':>11}  {'VDV (json)':>11}  match")
    n_ok = n_total = 0
    for sdir in sorted(args.data_root.iterdir()):
        if not sdir.is_dir():
            continue
        nlt = sdir / "noisy_label_train"
        if not (nlt / "labels.npy").exists():
            continue
        name = sdir.name
        if name not in expected:
            continue
        p, true_y = load_setting(nlt)
        vdv = vote_distribution_variance(p, true_y)
        ej = expected[name]
        ok = vdv == ej["m2_frobenius"]
        n_total += 1
        n_ok += int(ok)
        print(f"{name:<28}  {vdv:11.6f}  {ej['m2_frobenius']:11.6f}  "
              f"{'OK' if ok else 'MISMATCH'}")

    print(f"\n{n_ok}/{n_total} settings reproduced bit-identically.")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
