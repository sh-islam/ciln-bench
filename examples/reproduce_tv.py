"""Reproduce the CIFAR-10H TV numbers from results/tv_vs_cifar10h_v2.json.

For each setting we recompute the bootstrapped mean per-image TV between the
benchmark soft labels and three references (upsampled CIFAR-10H, symmetric
flipping, CCN) and compare to the released JSON. With matched seeds the
numbers should be bit-identical.

Usage:

    python examples/reproduce_tv.py --data-root PATH --cifar10h PATH [--limit N]

``--data-root`` should point to the ``settings/`` directory inside a
HuggingFace download. Each setting has a ``test/`` subdir with voter
softmaxes against the 10,000 CIFAR-10 **test** images, which is what
CIFAR-10H aligns with.

E.g. after downloading ``sh-islam/ciln-bench-cifar10`` to
``./ciln-bench-cifar10/``, use ``--data-root ./ciln-bench-cifar10/settings``.

``--cifar10h`` should point to ``cifar10h-counts.npy`` (shape ``(10000, 10)``).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code" / "analyze"))

from public.tv import tv_vs_human
from public.metrics import soft_label_from_voters

V2_VOTERS = ["resnet20", "wrn28_10", "deit3_small", "clip"]


def load_p(setting_dir: Path):
    argmax = np.stack(
        [np.load(setting_dir / f"softmax_{v}.npy").argmax(axis=1) for v in V2_VOTERS],
        axis=1,
    )
    return soft_label_from_voters(argmax, n_classes=10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Test-split voter softmaxes; see module docstring.")
    ap.add_argument("--cifar10h",  type=Path, required=True,
                    help="Path to cifar10h-counts.npy (shape 10000x10).")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process this many settings (TV is slow).")
    args = ap.parse_args()

    expected = {r["name"]: r for r in json.load(
        open(REPO / "results" / "tv_vs_cifar10h_v2.json")
    )}
    human_counts = np.load(args.cifar10h)

    print(f"Recomputing TV vs CIFAR-10H from {args.data_root}")
    print(f"  comparing to released TV numbers ({len(expected)} settings)\n")
    print(f"{'setting':<28}  {'ours':>9}  {'sym':>9}  {'ccn':>9}  match")

    n_ok = n_total = 0
    for sdir in sorted(args.data_root.iterdir()):
        if not sdir.is_dir():
            continue
        test = sdir / "test"
        if not (test / "labels.npy").exists():
            continue
        name = sdir.name
        if name not in expected:
            continue
        p = load_p(test)
        true_y = np.load(test / "labels.npy")
        r = tv_vs_human(p, true_y, human_counts, n_voters=4, n_boot=50)
        ej = expected[name]
        ok = (r.tv_ours_mean == ej["tv_ours"]["mean"]
              and r.tv_symmetric_mean == ej["tv_symmetric"]["mean"]
              and r.tv_ccn_mean == ej["tv_ccn"]["mean"])
        n_total += 1
        n_ok += int(ok)
        print(f"{name:<28}  {r.tv_ours_mean:9.6f}  "
              f"{r.tv_symmetric_mean:9.6f}  {r.tv_ccn_mean:9.6f}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if args.limit and n_total >= args.limit:
            break

    print(f"\n{n_ok}/{n_total} settings reproduced bit-identically.")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
