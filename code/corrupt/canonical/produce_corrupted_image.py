"""Produce the corrupted test sets for CIFAR-10 and MNIST.

15 corruption types × 3 severities × 2 datasets = 90 (corruption, severity, dataset) settings.
Severities chosen: {1, 3, 5}. Default for the journal edition.

For each setting we emit:
  output_seed{S}/{dataset}/{corruption}/severity_{sev}/
      images.npy            (N, H, W[, 3]) uint8 — the corrupted images
      labels.npy            (N,) int64 — original ground-truth labels (unchanged)
      params.jsonl          one JSON line per image with {image_idx, rng_seed, params, output_sha256}
      manifest.json         setting-level metadata (master_seed, dataset, corruption, severity, N, function source)

Per-image RNG:
    rng_seed = SHA256("{master_seed}|{dataset}|{corruption}|{severity}|{image_idx}")[:8]
    rng = np.random.default_rng(rng_seed)

Reproducibility:
- Re-running with the same --master-seed reproduces every image byte-exact.
- A single image (dataset, corruption, severity, image_idx) can be reproduced
  independently of others using the per-image seed alone.
- The params.jsonl records: function-specific summary params (NOT full per-pixel
  randomness — too big), and the SHA-256 hash of the output pixel bytes so a
  reproducer can verify byte-equality of their rerun.

Usage:
    python produce_corrupted.py                                # all 90 settings, master_seed=0
    python produce_corrupted.py --datasets mnist               # MNIST only
    python produce_corrupted.py --severities 5                 # only sev 5
    python produce_corrupted.py --corruptions glass_blur,fog   # subset
    python produce_corrupted.py --master-seed 7                # alt seed
    python produce_corrupted.py --resume                       # skip settings already done
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
import torchvision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cifar_funcs as cf
import mnist_funcs as mf
from seeding import get_rng, derive_per_image_seed

DEFAULT_SEVERITIES = [1, 3, 5]
DATA_ROOT = HERE.parent.parent / "data"
OUTPUT_ROOT = HERE.parent / "output_seed{seed}"


VALID_SPLITS = ("test", "noisy_label_train", "noisy_label_valid")
SPLITS_INDEX_ROOT = HERE.parent / "output_seed0" / "splits"


def setting_dir(out_root: Path, dataset: str, corruption: str, severity: int, split: str) -> Path:
    return out_root / dataset / corruption / f"severity_{severity}" / split


def is_done(setting_path: Path) -> bool:
    return (setting_path / "images.npy").exists() and (setting_path / "manifest.json").exists()


class _IndexedDataset:
    """Wrap a torchvision dataset + an index array so iteration yields the
    selected subset in index-array order. Length = len(indices)."""
    def __init__(self, base, indices: np.ndarray):
        self.base = base
        self.indices = indices.astype(np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


def _load_split_indices(dataset: str, split: str) -> np.ndarray:
    """Load Gu-style split index array. `test` returns None (use full test set)."""
    if split == "test":
        return None
    f = SPLITS_INDEX_ROOT / dataset / f"{split.replace('_label_', 'label')}_indices.npy"
    # noisy_label_train -> noisylabeltrain_indices.npy (matches splits/build_splits.py naming)
    if not f.exists():
        raise FileNotFoundError(
            f"Gu-split index file missing for {dataset}/{split}: {f}. "
            f"Run `python splits/build_splits.py` first."
        )
    return np.load(f)


def load_dataset(dataset: str, split: str = "test"):
    """Return (dataset-like-thing, indices_or_None).

    For 'test' we use the official test split (no indexing).
    For 'noisy_label_train' / 'noisy_label_valid' we use the official TRAIN set
    sliced by the Gu-style indices.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    is_train = (split != "test")
    if dataset == "cifar10":
        base = torchvision.datasets.CIFAR10(root=str(DATA_ROOT), train=is_train, download=True)
    elif dataset == "mnist":
        base = torchvision.datasets.MNIST(root=str(DATA_ROOT), train=is_train, download=True)
    else:
        raise ValueError(dataset)
    if split == "test":
        return base
    indices = _load_split_indices(dataset, split)
    return _IndexedDataset(base, indices)


def produce_setting(dataset: str, corruption: str, severity: int, split: str,
                    master_seed: int, out_root: Path, ds_cache: dict, log_n_every: int = 1000):
    """Generate one (dataset, corruption, severity, split) and write to disk."""
    setting_path = setting_dir(out_root, dataset, corruption, severity, split)
    setting_path.mkdir(parents=True, exist_ok=True)

    # pick the corruption function
    fn_table = cf.CORRUPTION_FUNCTIONS if dataset == "cifar10" else mf.CORRUPTION_FUNCTIONS
    fn = fn_table[corruption]

    cache_key = (dataset, split)
    ds = ds_cache.setdefault(cache_key, load_dataset(dataset, split))
    n = len(ds)

    # output buffers
    if dataset == "cifar10":
        images_arr = np.empty((n, 32, 32, 3), dtype=np.uint8)
    else:
        images_arr = np.empty((n, 28, 28), dtype=np.uint8)
    labels_arr = np.empty((n,), dtype=np.int64)

    t0 = time.time()
    params_path = setting_path / "params.jsonl"
    pf = open(params_path, "w")

    for i in range(n):
        img, label = ds[i]
        labels_arr[i] = int(label)
        seed_i = derive_per_image_seed(master_seed, dataset, corruption, severity, i, split=split)
        rng = np.random.default_rng(seed_i)
        out, params = fn(img, severity, rng)
        out_u8 = np.uint8(np.clip(out, 0, 255))
        images_arr[i] = out_u8
        out_sha = hashlib.sha256(out_u8.tobytes()).hexdigest()
        pf.write(json.dumps({
            "image_idx": i,
            "rng_seed": int(seed_i),
            "params": _jsonable(params),
            "output_sha256": out_sha,
        }) + "\n")

        if (i + 1) % log_n_every == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n - i - 1) / (i + 1)
            print(f"  [{dataset}/{corruption}/sev{severity}/{split}] {i+1}/{n} ({elapsed:.1f}s elapsed, ETA {eta:.0f}s)", flush=True)
    pf.close()

    np.save(setting_path / "images.npy", images_arr)
    np.save(setting_path / "labels.npy", labels_arr)

    # setting-level manifest
    manifest = {
        "dataset": dataset,
        "corruption": corruption,
        "severity": severity,
        "split": split,
        "master_seed": master_seed,
        "n_images": n,
        "image_shape": list(images_arr.shape[1:]),
        "function_module": "cifar_funcs" if dataset == "cifar10" else "mnist_funcs",
        "function_name": corruption,
        "wall_time_sec": round(time.time() - t0, 2),
        "output_npy_sha256": hashlib.sha256(open(setting_path / "images.npy", "rb").read()).hexdigest(),
    }
    with open(setting_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _jsonable(obj):
    """Make np scalars/lists JSON-safe."""
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
    ap.add_argument("--datasets", type=str, nargs="+", default=["cifar10", "mnist"])
    ap.add_argument("--corruptions", type=str, default=None,
                    help="Comma-separated subset (else all 15 per dataset)")
    ap.add_argument("--splits", type=str, nargs="+",
                    default=["noisy_label_train", "noisy_label_valid"],
                    choices=list(VALID_SPLITS),
                    help="Which Gu-style splits to corrupt. Default: noisy_label_train + noisy_label_valid")
    ap.add_argument("--resume", action="store_true",
                    help="Skip settings that already have images.npy + manifest.json")
    args = ap.parse_args()

    out_root = Path(str(OUTPUT_ROOT).replace("{seed}", str(args.master_seed)))
    out_root.mkdir(parents=True, exist_ok=True)

    cifar_corruptions = list(cf.CORRUPTION_FUNCTIONS.keys())
    mnist_corruptions = list(mf.CORRUPTION_FUNCTIONS.keys())
    if args.corruptions:
        wanted = [c.strip() for c in args.corruptions.split(",")]
        cifar_corruptions = [c for c in cifar_corruptions if c in wanted]
        mnist_corruptions = [c for c in mnist_corruptions if c in wanted]

    plan = []
    for ds in args.datasets:
        corrs = cifar_corruptions if ds == "cifar10" else mnist_corruptions
        for c in corrs:
            for sev in args.severities:
                for sp in args.splits:
                    plan.append((ds, c, sev, sp))

    print(f"=== Plan: {len(plan)} (dataset, corruption, severity, split) settings ===", flush=True)
    print(f"    output root: {out_root}", flush=True)
    print(f"    master_seed: {args.master_seed}", flush=True)
    print(f"    severities:  {args.severities}", flush=True)
    print(f"    splits:      {args.splits}", flush=True)
    print()

    ds_cache = {}
    n_done = 0
    n_skipped = 0
    t_start = time.time()
    for ds, c, sev, sp in plan:
        setting_path = setting_dir(out_root, ds, c, sev, sp)
        if args.resume and is_done(setting_path):
            print(f"  [SKIP] {ds}/{c}/sev{sev}/{sp} (already done)", flush=True)
            n_skipped += 1
            continue
        print(f"=== [{n_done + n_skipped + 1}/{len(plan)}] {ds}/{c}/sev{sev}/{sp} ===", flush=True)
        m = produce_setting(ds, c, sev, sp, args.master_seed, out_root, ds_cache)
        print(f"  done in {m['wall_time_sec']:.1f}s, "
              f"images sha256={m['output_npy_sha256'][:12]}...", flush=True)
        n_done += 1

    total = time.time() - t_start
    print(f"\n=== ALL DONE: {n_done} produced, {n_skipped} skipped in {total:.1f}s "
          f"({total/60:.1f} min) ===", flush=True)


if __name__ == "__main__":
    main()
