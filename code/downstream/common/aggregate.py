"""Aggregate downstream_multimodal results into a single CSV.

Reads results/{dataset}/*.json, aggregates over seeds, writes per-config
mean/std for best_test_acc, plus per-class recall summary.

Run any time to monitor progress:
  python aggregate.py
"""
from __future__ import annotations
import csv, glob, json, statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MULTI_ROOT = HERE.parent
RESULTS_ROOT = MULTI_ROOT / "results"
OUT_CSV = MULTI_ROOT / "aggregate.csv"


def load_all():
    rows = []
    for dataset_dir in sorted(RESULTS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for f in sorted(dataset_dir.glob("*.json")):
            try:
                d = json.loads(f.read_text())
            except Exception as e:
                print(f"skip {f.name}: {e}")
                continue
            rows.append(d)
    return rows


def key_of(r):
    return (r["dataset"], r["setting"], r.get("variant", "?"),
            r["method"], r["image_source"])


def main():
    rows = load_all()
    if not rows:
        print("no results found")
        return
    groups = defaultdict(list)
    for r in rows:
        groups[key_of(r)].append(r)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "setting", "variant", "method", "image_source",
                    "n_seeds", "mean_best_test_acc", "std_best_test_acc",
                    "sampled_noise_rate", "n_train",
                    "recall_class0", "recall_class1"])
        for k, group in sorted(groups.items()):
            accs = [g["best_test_acc"] for g in group]
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            nrs = [g.get("sampling_manifest", {}).get("sampled_noise_rate", 0) for g in group]
            ns = [g.get("n_train", 0) for g in group]
            # best per-class recall averaged across seeds (class-wise)
            recalls = defaultdict(list)
            for g in group:
                r = g.get("best_per_class_recall") or {}
                for c, v in r.items():
                    recalls[int(c)].append(float(v))
            r0 = statistics.mean(recalls[0]) if 0 in recalls and recalls[0] else 0.0
            r1 = statistics.mean(recalls[1]) if 1 in recalls and recalls[1] else 0.0
            w.writerow([*k, len(group),
                        f"{mean_acc:.4f}", f"{std_acc:.4f}",
                        f"{statistics.mean(nrs):.4f}", int(statistics.mean(ns)),
                        f"{r0:.4f}", f"{r1:.4f}"])
    print(f"wrote {OUT_CSV} ({sum(1 for _ in OUT_CSV.open()) - 1} configs, {len(rows)} raw runs)")


if __name__ == "__main__":
    main()
