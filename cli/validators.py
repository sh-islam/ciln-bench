"""Validators — check user-supplied data BEFORE running any corruption.

If validation fails, the CLI prints a friendly error and re-prompts. Nothing
is written to disk until everything checks out.

We avoid hard-coding dataset assumptions. Instead we check generic invariants:
- image: numpy array, uint8/float, 2D grayscale or 3D RGB, height/width sane
- tabular: pandas DataFrame, has a label column the user identifies, dtype sane
"""
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np


# ---------------- Image ----------------
SUPPORTED_IMAGE_SIDES = {28, 32, 64, 224}     # easy to extend


def load_image_dataset(path: str) -> np.ndarray:
    """Load images from .npy. Accepts (N, H, W), (N, H, W, 1), or (N, H, W, 3).
    Returns a (N, H, W, C) uint8 array."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image dataset path does not exist: {path}")
    if p.suffix != ".npy":
        raise ValueError(f"Expected a .npy file. Got: {p.suffix}")
    arr = np.load(p)
    if arr.ndim == 3:
        # (N, H, W) grayscale -> add channel
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(
            f"Image array must be 3D (N,H,W) or 4D (N,H,W,C). Got shape {arr.shape}"
        )
    n, h, w, c = arr.shape
    if h != w:
        raise ValueError(
            f"Only square images supported (H must equal W). Got H={h}, W={w}"
        )
    if h not in SUPPORTED_IMAGE_SIDES:
        raise ValueError(
            f"Side length {h} not yet supported. Supported: {sorted(SUPPORTED_IMAGE_SIDES)}"
        )
    if c not in (1, 3):
        raise ValueError(f"Channel count must be 1 (grayscale) or 3 (RGB). Got C={c}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.5 else (arr * 255).astype(np.uint8)
    return arr


def summarize_image_dataset(arr: np.ndarray) -> str:
    n, h, w, c = arr.shape
    kind = "grayscale" if c == 1 else "RGB"
    return f"{n} {kind} images, {h}x{w}, dtype={arr.dtype}"


# ---------------- Tabular ----------------
def load_tabular_dataset(path: str):
    """Load a tabular dataset from .parquet or .csv. Returns a pandas DataFrame."""
    import pandas as pd
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tabular dataset path does not exist: {path}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Expected .parquet or .csv. Got: {p.suffix}")
    if df.empty:
        raise ValueError("Tabular dataset is empty.")
    return df


def detect_label_column(df) -> Optional[str]:
    """Heuristic: 'label', 'target', 'y', 'class' (case-insensitive)."""
    candidates = ["label", "target", "y", "class", "income"]
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None


def numeric_columns(df, exclude: List[str] = ()) -> List[str]:
    return [c for c in df.columns if c not in exclude and df[c].dtype.kind in ("i", "f")]


def categorical_columns(df, exclude: List[str] = ()) -> List[str]:
    return [c for c in df.columns
            if c not in exclude and df[c].dtype.kind not in ("i", "f")]


def summarize_tabular_dataset(df, label_col: Optional[str]) -> str:
    n, k = df.shape
    nums = numeric_columns(df, exclude=[label_col] if label_col else [])
    cats = categorical_columns(df, exclude=[label_col] if label_col else [])
    parts = [f"{n} rows, {k} columns"]
    if label_col: parts.append(f"label column: {label_col}")
    parts.append(f"numeric: {len(nums)} cols")
    parts.append(f"categorical: {len(cats)} cols")
    return "  ".join(parts)


# ---------------- Output path ----------------
def validate_output_path(path: str, allow_overwrite: bool = False) -> Path:
    """Make sure we won't silently clobber. Returns a Path object."""
    p = Path(path)
    if p.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Output path already exists: {path}. "
            "Use a new folder or pass --overwrite."
        )
    return p
