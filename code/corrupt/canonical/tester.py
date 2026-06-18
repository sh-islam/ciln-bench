"""Reproducibility tester for Adult corruptions.

For each corruption type at severity 5, run it twice on the same 100 Adult rows
with the same master_seed and compare byte-for-byte.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # for common.py

import adult_funcs as af
from seeding import get_rng
from common import load_adult


OUT = HERE.parent / "output_tester_adult"


def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Stats needed by MNAR (quantile ranks for numericals, frequencies for categoricals)."""
    stats = {}
    for c in af.NUMERICAL_COLS:
        # quantile rank of each unique value
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


def run_one(df: pd.DataFrame, corruption: str, severity: int, master_seed: int,
            dataset_stats: dict, n_rows: int = 100):
    fn = af.CORRUPTION_FUNCTIONS[corruption]
    out_rows = []
    out_params = []
    for i in range(n_rows):
        row = df.iloc[i].to_dict()
        rng = get_rng(master_seed, "adult", corruption, severity, i)
        out_row, params = fn(row, severity, rng, dataset_stats=dataset_stats)
        out_rows.append(out_row)
        out_params.append(params)
    out_df = pd.DataFrame(out_rows)
    return out_df, out_params


def df_sha256(df: pd.DataFrame) -> str:
    """Hash a dataframe's bytes for byte-equality comparison."""
    # serialize to a deterministic representation: sort columns, write csv as bytes
    s = df.to_csv(index=False, na_rep="__NaN__").encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=100)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== Tester: master_seed={args.master_seed} severity={args.severity} n_rows={args.n_rows} ===")

    X_train, y_train, _, _ = load_adult()
    df = X_train.iloc[:args.n_rows].reset_index(drop=True)
    stats = compute_dataset_stats(X_train)

    results = []
    for corr in af.CORRUPTION_FUNCTIONS:
        d1, p1 = run_one(df, corr, args.severity, args.master_seed, stats, args.n_rows)
        d2, p2 = run_one(df, corr, args.severity, args.master_seed, stats, args.n_rows)
        sha1, sha2 = df_sha256(d1), df_sha256(d2)
        same = sha1 == sha2
        same_params = json.dumps(p1, sort_keys=True, default=str) == json.dumps(p2, sort_keys=True, default=str)
        status = "PASS" if (same and same_params) else "FAIL"
        results.append((corr, status, sha1, sha2, same_params))
        # Save artifacts
        d = OUT / corr
        d.mkdir(parents=True, exist_ok=True)
        d1.to_csv(d / "run1.csv", index=False)
        d2.to_csv(d / "run2.csv", index=False)
        with open(d / "params_run1.json", "w") as f:
            json.dump(p1, f, indent=2, default=str)
        print(f"  [{corr:18s}] status={status}  sha1={sha1[:12]}...  sha2={sha2[:12]}...  same_params={same_params}")

    print(f"\nResult: {sum(1 for r in results if r[1] == 'PASS')}/{len(results)} PASS")


if __name__ == "__main__":
    main()
