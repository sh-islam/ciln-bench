"""Produce the corrupted Adult test sets.

6 corruption types x 3 severities = 18 settings.

For each setting we emit:
  output_seed{S}/adult/{corruption}/severity_{sev}/
      adult_corrupted.parquet   the corrupted dataframe (rows + columns preserved order)
      labels.npy                ground-truth labels (unchanged)
      params.jsonl              one JSON line per row with sampled corruption params
      manifest.json             setting-level metadata (master_seed, n, sha256, ...)

Per-row RNG: rng_seed = SHA256("{master_seed}|adult|{corruption}|{severity}|{row_idx}")[:8]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import adult_funcs as af
from seeding import derive_per_row_seed
from common import load_adult

DEFAULT_SEVERITIES = [1, 3, 5]
JOURNAL_ROOT = HERE.parent.parent
OUTPUT_ROOT_TEMPLATE = str(JOURNAL_ROOT / "output_seed{seed}")


def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Stats for MNAR: per-numerical quantile ranks, per-categorical frequencies.
    Computed on the train set (we then re-use for test-set corruption to avoid
    test-set leakage)."""
    stats = {}
    for c in af.NUMERICAL_COLS:
        sorted_vals = np.sort(df[c].values.astype(float))
        ranks = {}
        for v in df[c].unique():
            idx = np.searchsorted(sorted_vals, float(v))
            ranks[v] = idx / max(1, len(sorted_vals))
        stats[f"{c}__quantile_rank"] = ranks
    for c in af.CATEGORICAL_COLS:
        vc = df[c].astype(str).value_counts(normalize=True)
        stats[f"{c}__freq"] = vc.to_dict()
    return stats


VALID_SPLITS = ("test", "noisy_label_train", "noisy_label_valid")
SPLITS_INDEX_ROOT = JOURNAL_ROOT / "output_seed0" / "splits" / "adult"


def setting_dir(out_root: Path, corruption: str, severity: int, split: str) -> Path:
    return out_root / "adult" / corruption / f"severity_{severity}" / split


def is_done(setting_path: Path) -> bool:
    return (setting_path / "adult_corrupted.parquet").exists() and \
           (setting_path / "manifest.json").exists()


def _load_adult_split_indices(split: str) -> np.ndarray:
    """Load Gu-style split index array. `test` returns None (use full test set)."""
    if split == "test":
        return None
    name = split.replace("_label_", "label")  # noisy_label_train -> noisylabeltrain
    f = SPLITS_INDEX_ROOT / f"{name}_indices.npy"
    if not f.exists():
        raise FileNotFoundError(
            f"Gu-split Adult index missing for {split}: {f}. "
            f"Run `python splits/build_splits.py` first."
        )
    return np.load(f)


def produce_setting(df: pd.DataFrame, labels: np.ndarray, corruption: str,
                    severity: int, split: str, master_seed: int, out_root: Path,
                    dataset_stats: dict):
    setting_path = setting_dir(out_root, corruption, severity, split)
    setting_path.mkdir(parents=True, exist_ok=True)
    fn = af.CORRUPTION_FUNCTIONS[corruption]
    n = len(df)
    t0 = time.time()

    out_rows = []
    seeds = []
    all_params = []

    for i in range(n):
        row = df.iloc[i].to_dict()
        seed_i = derive_per_row_seed(master_seed, "adult", corruption, severity, i, split=split)
        rng = np.random.default_rng(seed_i)
        out_row, params = fn(row, severity, rng, dataset_stats=dataset_stats)
        out_rows.append(out_row)
        seeds.append(seed_i)
        all_params.append(params)

        if (i + 1) % 5000 == 0:
            print(f"  [{corruption}/sev{severity}] {i+1}/{n}", flush=True)

    out_df = pd.DataFrame(out_rows)
    out_df = out_df[df.columns.tolist()]
    parquet_path = setting_path / "adult_corrupted.parquet"
    out_df.to_parquet(parquet_path, index=False)
    np.save(setting_path / "labels.npy", labels)

    # Re-read parquet so per-row sha256 matches what a downstream user would
    # compute by hashing row N of the saved parquet. Without this round-trip,
    # int->float promotions (forced when a corruption introduces NaN in an int
    # column) make the in-memory dict differ from the parquet-roundtrip dict.
    reread_df = pd.read_parquet(parquet_path)
    params_path = setting_path / "params.jsonl"
    with open(params_path, "w") as pf:
        for i in range(n):
            row_after = reread_df.iloc[i].to_dict()
            canon = {k: ("__NaN__" if (isinstance(v, float) and np.isnan(v)) else _jsonable(v))
                     for k, v in row_after.items()}
            row_sha = hashlib.sha256(
                json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            pf.write(json.dumps({
                "row_idx": i,
                "rng_seed": int(seeds[i]),
                "params": _jsonable(all_params[i]),
                "output_sha256": row_sha,
            }) + "\n")

    manifest = {
        "dataset": "adult",
        "corruption": corruption,
        "severity": severity,
        "split": split,
        "master_seed": master_seed,
        "n_rows": n,
        "function_module": "adult_funcs",
        "function_name": corruption,
        "wall_time_sec": round(time.time() - t0, 2),
        "parquet_sha256": hashlib.sha256(
            open(setting_path / "adult_corrupted.parquet", "rb").read()
        ).hexdigest(),
    }
    with open(setting_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--severities", type=int, nargs="+", default=DEFAULT_SEVERITIES)
    ap.add_argument("--corruptions", type=str, default=None,
                    help="Comma-separated subset (else all 5)")
    ap.add_argument("--splits", type=str, nargs="+",
                    default=["noisy_label_train", "noisy_label_valid"],
                    choices=list(VALID_SPLITS),
                    help="Which Gu-style splits to corrupt. Default: noisy_label_train + noisy_label_valid")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_root = Path(OUTPUT_ROOT_TEMPLATE.format(seed=args.master_seed))
    out_root.mkdir(parents=True, exist_ok=True)

    corr_names = list(af.CORRUPTION_FUNCTIONS.keys())
    if args.corruptions:
        wanted = [c.strip() for c in args.corruptions.split(",")]
        corr_names = [c for c in corr_names if c in wanted]

    X_train, y_train, X_test, y_test = load_adult()
    # Dataset stats for MNAR: use CleanLabelTrain rows of the TRAIN set so we
    # don't leak NoisyLabelTrain values. (CleanLabelTrain is by definition disjoint
    # from NoisyLabelTrain and NoisyLabelValid.)
    clt_idx = _load_adult_split_indices("noisy_label_train")  # just to assert it exists
    from common import split_train_val  # now returns Gu-style CleanLabelTrain split
    X_clt, _, _, _ = split_train_val(X_train, y_train)
    print(f"[stats] computing MNAR stats from CleanLabelTrain ({len(X_clt)} rows)...", flush=True)
    stats = compute_dataset_stats(X_clt)

    # Pre-load each target split's df + labels (slice from X_train using indices).
    split_data = {}
    for sp in args.splits:
        if sp == "test":
            df, labels = X_test.reset_index(drop=True), y_test.values
        else:
            idx = _load_adult_split_indices(sp)
            df = X_train.iloc[idx].reset_index(drop=True)
            labels = y_train.iloc[idx].values
        split_data[sp] = (df, labels)
        print(f"[plan] {sp}: {len(df)} rows", flush=True)

    plan = []
    for sp in args.splits:
        for corr in corr_names:
            for sev in args.severities:
                plan.append((corr, sev, sp))
    print(f"[plan] {len(plan)} (corruption, severity, split) settings total", flush=True)

    n_done = 0
    n_skipped = 0
    t_start = time.time()
    for corr, sev, sp in plan:
        s_path = setting_dir(out_root, corr, sev, sp)
        if args.resume and is_done(s_path):
            print(f"  [SKIP] adult/{corr}/sev{sev}/{sp}", flush=True)
            n_skipped += 1
            continue
        df, labels = split_data[sp]
        print(f"=== [{n_done + n_skipped + 1}/{len(plan)}] adult/{corr}/sev{sev}/{sp} ===", flush=True)
        m = produce_setting(df, labels, corr, sev, sp, args.master_seed, out_root, stats)
        print(f"  done in {m['wall_time_sec']:.1f}s, sha={m['parquet_sha256'][:12]}...", flush=True)
        n_done += 1

    total = time.time() - t_start
    print(f"\n=== ALL DONE: {n_done} produced, {n_skipped} skipped in {total:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
