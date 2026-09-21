"""Per-row deterministic RNG derivation for tabular corruption.

A corrupted Adult row's RNG is seeded from a 6-tuple:
    (master_seed, dataset, corruption, severity, split, row_idx)

`split` ∈ {"test", "noisy_label_train", "noisy_label_valid"}.

Backward compat: split defaults to "test" which omits the split from the key,
reproducing the original 5-tuple seeds for the test-side pipeline.
"""
from __future__ import annotations
import hashlib
import struct
import numpy as np


def derive_per_row_seed(master_seed: int, dataset: str, corruption: str,
                        severity: int, row_idx: int,
                        split: str = "test") -> int:
    """Stable 64-bit unsigned int derived from SHA-256 of the inputs."""
    if split == "test":
        key = f"{master_seed}|{dataset}|{corruption}|{severity}|{row_idx}".encode("utf-8")
    else:
        key = f"{master_seed}|{dataset}|{corruption}|{severity}|{split}|{row_idx}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    return struct.unpack("<Q", h[:8])[0]


def get_rng(master_seed: int, dataset: str, corruption: str,
            severity: int, row_idx: int, split: str = "test") -> np.random.Generator:
    seed = derive_per_row_seed(master_seed, dataset, corruption, severity, row_idx, split)
    return np.random.default_rng(seed)
