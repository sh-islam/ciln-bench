"""Build avg_softmax.npy per setting for MNIST, CIFAR-10, and Adult.

For each setting (clean or corrupted, all severities), this script:
  1. Loads every voter's softmax_<voter>.npy in that setting directory.
  2. Averages them element-wise to a single (N, num_classes) array.
  3. Saves to <setting_dir>/avg_softmax.npy.

Voter pools per dataset (matches what was trained + evaluated):
  - mnist:   lenet5, mlp, resnet20, deit3_small         (4 voters)
  - cifar10: resnet20, wrn28_10, deit3_small            (3 voters)
  - adult:   xgboost, xgboost_dummyna, catboost, mlp,
             ft_transformer, tabpfn                     (6 voters)

This is mean-softmax aggregation (Marquardt-style soft-label distribution).
No entropy, no hard-vote tally — just E[p_voter](class | image).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JOURNAL_ROOT = HERE.parent

VOTERS = {
    "mnist":    ["lenet5", "mlp", "resnet20", "deit3_small"],
    "cifar10":  ["resnet20", "wrn28_10", "deit3_small", "clip"],
    "adult":    ["xgboost_dummyna", "catboost", "mlp", "ft_transformer", "tabpfn"],
}


def setting_dirs(output_root: Path, dataset: str):
    """Yield (label, setting_path) for clean splits + every corrupted setting/split.

    Clean splits: clean/<dataset>/cleanlabelvalid/ and clean/<dataset>/test/
    Corrupted splits: <dataset>/<corruption>/severity_<S>/{noisy_label_train, noisy_label_valid}/
    """
    clean_root = output_root / "clean" / dataset
    if clean_root.is_dir():
        for clean_split in ("cleanlabelvalid", "test"):
            d = clean_root / clean_split
            if d.is_dir():
                yield (f"clean/{clean_split}", d)
    dataset_dir = output_root / dataset
    if not dataset_dir.is_dir():
        return
    for corr_dir in sorted(dataset_dir.iterdir()):
        if not corr_dir.is_dir():
            continue
        for sev_dir in sorted(corr_dir.iterdir()):
            if not sev_dir.is_dir():
                continue
            for split_name in ("noisy_label_train", "noisy_label_valid"):
                d = sev_dir / split_name
                if d.is_dir():
                    yield (f"{corr_dir.name}/{sev_dir.name}/{split_name}", d)


def build_avg(setting_path: Path, voters: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load per-voter softmaxes, return (avg, voters_used)."""
    stacks = []
    used = []
    for v in voters:
        f = setting_path / f"softmax_{v}.npy"
        if not f.exists():
            continue
        stacks.append(np.load(f))
        used.append(v)
    if not stacks:
        return None, []
    return np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "adult"])
    ap.add_argument("--print-acc", action="store_true",
                    help="Print per-setting argmax-of-avg accuracy (sanity check)")
    args = ap.parse_args()

    output_root = JOURNAL_ROOT / f"output_seed{args.seed}"

    summary = {"datasets": {}, "per_setting": []}

    for dataset in args.datasets:
        voters = VOTERS[dataset]
        n_done = 0
        print(f"\n=== {dataset}  voters={voters}  ({len(voters)} voters) ===", flush=True)
        ds_records = []
        for label, sp in setting_dirs(output_root, dataset):
            avg, used = build_avg(sp, voters)
            if avg is None:
                print(f"  [SKIP] {dataset}/{label}: no per-voter softmaxes found", flush=True)
                continue
            np.save(sp / "avg_softmax.npy", avg)
            n_done += 1

            line = f"  [{dataset}/{label}] saved avg_softmax.npy  shape={avg.shape}  voters={len(used)}"
            if args.print_acc:
                labels_path = sp / "labels.npy"
                if labels_path.exists():
                    labels = np.load(labels_path)
                    pred = avg.argmax(axis=1)
                    acc = (pred == labels).mean()
                    line += f"  avg_acc={acc:.4f}"
                    ds_records.append({"setting": label, "n": int(len(avg)),
                                       "voters_used": used, "avg_acc": float(acc)})
            print(line, flush=True)
        summary["datasets"][dataset] = {"voters": voters, "n_settings_done": n_done}
        summary["per_setting"].extend([{"dataset": dataset, **r} for r in ds_records])

    out_path = output_root / "_avg_softmax_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== summary written to {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
