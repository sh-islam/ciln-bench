"""Per-image / per-row deterministic seed derivation.

Every random number consumed by every corruption is derived from a single
hash key:

    SHA-256(master_seed | dataset | corruption | severity | [split |] index)

This means:
- Two runs of the same pipeline with the same master_seed produce
  bit-identical output.
- Running corruptions in a different order does not change the result.
- Multiprocessing does not change the result.

`split` is an optional component identifying which partition of the dataset
we are corrupting (e.g. "test" / "train"). It is included to prevent
artifact bleed-through across splits.
"""
from __future__ import annotations
import hashlib
import struct
import numpy as np


def derive_per_item_seed(
    master_seed: int,
    dataset: str,
    corruption: str,
    severity: int,
    index: int,
    split: str = "test",
) -> int:
    """Return a stable 64-bit unsigned integer derived from SHA-256 of inputs.

    For backward compatibility with the canonical pipeline, ``split='test'``
    omits the split component from the hash key, reproducing the original
    4-tuple keys used to generate the released benchmark.
    """
    if split == "test":
        key = f"{master_seed}|{dataset}|{corruption}|{severity}|{index}"
    else:
        key = f"{master_seed}|{dataset}|{corruption}|{severity}|{split}|{index}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return struct.unpack("<Q", digest[:8])[0]


def get_rng(
    master_seed: int,
    dataset: str,
    corruption: str,
    severity: int,
    index: int,
    split: str = "test",
) -> np.random.Generator:
    """Build a seeded :class:`numpy.random.Generator` for one item.

    Convenience wrapper around :func:`derive_per_item_seed`.
    """
    seed = derive_per_item_seed(master_seed, dataset, corruption, severity, index, split)
    return np.random.default_rng(seed)
