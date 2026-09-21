"""Build Gu-style 5-way splits for CIFAR-10, MNIST, Adult.

For each dataset:
  1. Start from the original train + test.
  2. Carve a 10% validation set out of train (stratified, seed=0).
  3. Partition train_remaining → CleanLabelTrain + NoisyLabelTrain (50/50, stratified, seed=0).
  4. Partition val → CleanLabelValid + NoisyLabelValid (50/50, stratified, seed=0).
  5. Test stays untouched.

Saves per-dataset index arrays (relative to the original train / test set, so
downstream code can do `train_data[cleanlabeltrain_indices]` to recover the
right subset). Also writes a manifest with sizes and sha256 of each index file.

Output: output_seed0/splits/<dataset>/{cleanlabeltrain,cleanlabelvalid,noisylabeltrain,noisylabelvalid,test}_indices.npy + manifest.json

Reproducibility: all three partitions use sklearn.train_test_split with
random_state=0 and stratify=labels. Run twice → byte-identical indices.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
JOURNAL_ROOT = HERE.parent
sys.path.insert(0, str(JOURNAL_ROOT / "tabular"))

SEED = 0
VAL_FRACTION = 0.10        # 10% of train becomes val
GU_HALF = 0.50             # 50/50 within train and within val


def stratified_split(indices: np.ndarray, labels: np.ndarray, test_size: float, seed: int):
    """sklearn stratified split returning (left_idx, right_idx) within indices."""
    left, right = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels,
    )
    return np.sort(left), np.sort(right)


def build_cifar10_splits():
    from torchvision import datasets
    data_dir = str(JOURNAL_ROOT / "data_cache")
    train = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test = datasets.CIFAR10(root=data_dir, train=False, download=True)
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)
    return _build_image_partition(train_labels, test_labels, "cifar10")


def build_mnist_splits():
    from torchvision import datasets
    data_dir = str(JOURNAL_ROOT / "data_cache")
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)
    train_labels = np.array(train.targets)
    test_labels = np.array(test.targets)
    return _build_image_partition(train_labels, test_labels, "mnist")


def _build_image_partition(train_labels: np.ndarray, test_labels: np.ndarray, name: str):
    n_train = len(train_labels)
    n_test = len(test_labels)
    all_idx = np.arange(n_train)

    # Step 1: carve 10% val from train
    train_remain_idx, val_idx = stratified_split(all_idx, train_labels, test_size=VAL_FRACTION, seed=SEED)
    print(f"[{name}] after val carve: train_remain={len(train_remain_idx)}, val={len(val_idx)}")

    # Step 2: split train_remain 50/50 into CleanLabelTrain + NoisyLabelTrain
    clt_idx, nlt_idx = stratified_split(
        train_remain_idx, train_labels[train_remain_idx], test_size=GU_HALF, seed=SEED,
    )
    # Step 3: split val 50/50 into CleanLabelValid + NoisyLabelValid
    clv_idx, nlv_idx = stratified_split(
        val_idx, train_labels[val_idx], test_size=GU_HALF, seed=SEED,
    )

    return {
        "cleanlabeltrain_indices": clt_idx,
        "noisylabeltrain_indices": nlt_idx,
        "cleanlabelvalid_indices": clv_idx,
        "noisylabelvalid_indices": nlv_idx,
        "test_indices": np.arange(n_test),  # full test
        "n_train": n_train,
        "n_test": n_test,
    }


def build_adult_splits():
    from common import load_adult  # tabular/common.py
    X_train, y_train, X_test, y_test = load_adult()
    train_labels = y_train.values.astype(np.int64)
    n_train = len(train_labels)
    n_test = len(y_test)
    all_idx = np.arange(n_train)

    train_remain_idx, val_idx = stratified_split(all_idx, train_labels, test_size=VAL_FRACTION, seed=SEED)
    print(f"[adult] after val carve: train_remain={len(train_remain_idx)}, val={len(val_idx)}")
    clt_idx, nlt_idx = stratified_split(
        train_remain_idx, train_labels[train_remain_idx], test_size=GU_HALF, seed=SEED,
    )
    clv_idx, nlv_idx = stratified_split(
        val_idx, train_labels[val_idx], test_size=GU_HALF, seed=SEED,
    )

    return {
        "cleanlabeltrain_indices": clt_idx,
        "noisylabeltrain_indices": nlt_idx,
        "cleanlabelvalid_indices": clv_idx,
        "noisylabelvalid_indices": nlv_idx,
        "test_indices": np.arange(n_test),
        "n_train": n_train,
        "n_test": n_test,
    }


def save_splits(splits: dict, out_dir: Path, dataset: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": dataset,
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "gu_half_ratio": GU_HALF,
        "n_train_original": splits["n_train"],
        "n_test_original": splits["n_test"],
        "splits": {},
    }
    for key in ["cleanlabeltrain_indices", "noisylabeltrain_indices",
                "cleanlabelvalid_indices", "noisylabelvalid_indices", "test_indices"]:
        arr = splits[key]
        path = out_dir / f"{key}.npy"
        np.save(path, arr.astype(np.int64))
        # sha256 of file bytes
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        manifest["splits"][key] = {"n": int(len(arr)), "sha256": sha, "file": path.name}
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    out_root = JOURNAL_ROOT / "output_seed0" / "splits"
    out_root.mkdir(parents=True, exist_ok=True)

    for name, builder in [
        ("cifar10", build_cifar10_splits),
        ("mnist",   build_mnist_splits),
        ("adult",   build_adult_splits),
    ]:
        print(f"\n=== {name} ===")
        splits = builder()
        out_dir = out_root / name
        save_splits(splits, out_dir, name)
        # Pretty-print sizes
        for key in ["cleanlabeltrain_indices", "noisylabeltrain_indices",
                    "cleanlabelvalid_indices", "noisylabelvalid_indices", "test_indices"]:
            print(f"  {key}: {len(splits[key])}")
        print(f"  saved to {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
