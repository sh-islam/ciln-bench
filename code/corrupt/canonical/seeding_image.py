"""Per-image deterministic RNG derivation.

A corrupted image's RNG is seeded from a 5-tuple:
    (master_seed, dataset, corruption, severity, split, image_idx)

`split` ∈ {"test", "noisy_label_train", "noisy_label_valid"} identifies which
partition of the data we're corrupting. Including it in the hash means the
same image_idx in different splits gets a different seed, so corruption
artifacts can't bleed across splits.

Backward compat: split defaults to "test" which omits split from the hash
key, reproducing the original 4-tuple seeds for the test-side pipeline.

Given the master_seed plus these keys, the exact bytes of every random call
inside the corruption function are reproducible. Independent of:
- order of processing
- which other corruptions ran first
- multi-threading / multi-process
"""
from __future__ import annotations
import hashlib
import struct
import numpy as np


def derive_per_image_seed(master_seed: int, dataset: str, corruption: str,
                          severity: int, image_idx: int,
                          split: str = "test") -> int:
    """Returns a stable 64-bit unsigned int derived from SHA-256 of the inputs.

    For backward compat with the old test-side artifacts, split='test' uses
    the original 4-tuple key. Other splits insert the split name into the key.
    """
    if split == "test":
        key = f"{master_seed}|{dataset}|{corruption}|{severity}|{image_idx}".encode("utf-8")
    else:
        key = f"{master_seed}|{dataset}|{corruption}|{severity}|{split}|{image_idx}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    return struct.unpack("<Q", h[:8])[0]


def get_rng(master_seed: int, dataset: str, corruption: str,
            severity: int, image_idx: int, split: str = "test") -> np.random.Generator:
    """Convenience: SHA-derived seed -> seeded np.random.Generator."""
    seed = derive_per_image_seed(master_seed, dataset, corruption, severity, image_idx, split)
    return np.random.default_rng(seed)
