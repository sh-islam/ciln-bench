"""Apply corruptions — thin wrappers around our existing corruption pipelines.

NOTE: in the published repo this file should import from `code/corrupt/`.
For the draft it ships minimal stand-in implementations so the CLI is testable.
Replace these stand-ins with imports to the real pipelines once the directory
structure is in place.
"""
from typing import Tuple, List, Optional
import hashlib
import numpy as np


def _seed_for(setting_key: str, image_idx: int) -> int:
    h = hashlib.sha256(f"{setting_key}__{image_idx}".encode()).hexdigest()[:8]
    return int(h, 16) % (2**31)


# =========================================================
# Image corruptions
# =========================================================
def apply_image(img: np.ndarray, type_name: str, severity: int, rng_seed: int) -> np.ndarray:
    """Apply a single image corruption. img is HxWxC uint8.
    Stand-in implementations: replace with real per-corruption code from
    code/corrupt/image/. Returns HxWxC uint8."""
    rng = np.random.default_rng(rng_seed)
    if type_name == "gaussian_noise":
        sigma = {1: 0.04, 2: 0.06, 3: 0.08, 4: 0.10, 5: 0.12}[severity]
        x = img.astype(np.float32) / 255.0
        noisy = x + rng.normal(0, sigma, size=x.shape)
        return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    if type_name == "shot_noise":
        scale = {1: 60, 2: 25, 3: 12, 4: 5, 5: 3}[severity]
        x = img.astype(np.float32) / 255.0
        out = rng.poisson(x * scale).astype(np.float32) / scale
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
    if type_name == "impulse_noise":
        p = {1: 0.03, 2: 0.06, 3: 0.09, 4: 0.17, 5: 0.27}[severity]
        out = img.copy()
        mask = rng.random(out.shape[:2]) < p
        salt = rng.random(out.shape[:2]) < 0.5
        # broadcast across channels
        for c in range(out.shape[-1]):
            out[..., c] = np.where(mask & salt, 255, out[..., c])
            out[..., c] = np.where(mask & ~salt, 0, out[..., c])
        return out
    if type_name == "brightness":
        delta = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}[severity]
        out = img.astype(np.float32) / 255.0
        out = np.clip(out + delta, 0, 1)
        return (out * 255).astype(np.uint8)
    if type_name == "contrast":
        c = {1: 0.75, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.15}[severity]
        out = img.astype(np.float32) / 255.0
        mean = out.mean()
        out = np.clip((out - mean) * c + mean, 0, 1)
        return (out * 255).astype(np.uint8)
    if type_name == "rotate":
        angle = {1: 3, 2: 12, 3: 27, 4: 42, 5: 57}[severity] * (1 if rng.random() > 0.5 else -1)
        # naive rotation using scipy fallback or simple skimage
        try:
            from scipy.ndimage import rotate
            out = rotate(img, angle, reshape=False, order=1, mode="nearest")
            return out.astype(np.uint8)
        except ImportError:
            raise RuntimeError("scipy required for rotation. pip install scipy")
    raise NotImplementedError(
        f"Image corruption '{type_name}' not implemented in this draft. "
        "Wire up the real implementation from code/corrupt/image/."
    )


def corrupt_image_dataset(images: np.ndarray, type_name: str, severity: int,
                          setting_key: str) -> Tuple[np.ndarray, List[dict]]:
    """Apply per-image corruption + log seeds/params per image."""
    out = np.empty_like(images)
    params_log = []
    for i in range(len(images)):
        seed = _seed_for(setting_key, i)
        out[i] = apply_image(images[i], type_name, severity, seed)
        sha = hashlib.sha256(out[i].tobytes()).hexdigest()
        params_log.append({
            "image_idx": i,
            "rng_seed": seed,
            "params": {"type": type_name, "severity": severity},
            "output_sha256": sha,
        })
    return out, params_log


# =========================================================
# Tabular corruptions
# =========================================================
def corrupt_tabular_dataset(df, type_name: str, severity: int,
                            target_columns: List[str],
                            conditioning_column: Optional[str],
                            setting_key: str):
    """Apply tabular corruption to df. Returns (corrupted_df, params_log).
    For real release, replace with imports from code/corrupt/tabular/.
    """
    import pandas as pd
    out = df.copy()
    params_log = []
    n = len(out)

    if type_name == "missing_mcar":
        p = {1: 0.05, 2: 0.15, 3: 0.25, 4: 0.40, 5: 0.50}[severity]
        for i in range(n):
            seed = _seed_for(setting_key, i)
            rng = np.random.default_rng(seed)
            row_hit = rng.random() < p
            cells_dropped = []
            if row_hit:
                for col in target_columns:
                    out.loc[out.index[i], col] = np.nan
                    cells_dropped.append(col)
            params_log.append({
                "row_idx": i, "rng_seed": seed,
                "params": {"type": type_name, "severity": severity,
                           "row_hit": row_hit, "cells_dropped": cells_dropped},
            })
        return out, params_log

    if type_name == "missing_mar":
        if not conditioning_column:
            raise ValueError("MAR requires a conditioning column.")
        # condition: rows with the rarer value of conditioning_column get higher drop
        rare_val = df[conditioning_column].value_counts().idxmin()
        base = {1: 0.05, 2: 0.15, 3: 0.25, 4: 0.40, 5: 0.50}[severity]
        for i in range(n):
            seed = _seed_for(setting_key, i)
            rng = np.random.default_rng(seed)
            is_rare = df[conditioning_column].iloc[i] == rare_val
            p = base * (1.5 if is_rare else 0.75)
            row_hit = rng.random() < p
            cells_dropped = []
            if row_hit:
                for col in target_columns:
                    out.loc[out.index[i], col] = np.nan
                    cells_dropped.append(col)
            params_log.append({
                "row_idx": i, "rng_seed": seed,
                "params": {"type": type_name, "severity": severity,
                           "conditioning_column": conditioning_column,
                           "is_rare_group": bool(is_rare),
                           "row_hit": row_hit, "cells_dropped": cells_dropped},
            })
        return out, params_log

    if type_name == "missing_mnar":
        base = {1: 0.05, 2: 0.15, 3: 0.25, 4: 0.40, 5: 0.50}[severity]
        # for numeric cols: higher quantile -> higher drop prob
        # for categorical: rarer category -> higher drop prob
        for i in range(n):
            seed = _seed_for(setting_key, i)
            rng = np.random.default_rng(seed)
            cells_dropped = []
            for col in target_columns:
                if df[col].dtype.kind in ("i", "f"):
                    rank = (df[col].rank(pct=True).iloc[i] or 0.5)
                    p = base * (0.5 + rank)   # 0.5x base for lowest, 1.5x for highest
                else:
                    freq = df[col].value_counts(normalize=True)
                    val = df[col].iloc[i]
                    rarity = 1.0 - freq.get(val, 0.0)
                    p = base * (0.5 + rarity)
                if rng.random() < p:
                    out.loc[out.index[i], col] = np.nan
                    cells_dropped.append(col)
            params_log.append({
                "row_idx": i, "rng_seed": seed,
                "params": {"type": type_name, "severity": severity,
                           "cells_dropped": cells_dropped},
            })
        return out, params_log

    if type_name == "gaussian_noise":
        sigma = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.5, 5: 5.0}[severity]
        for i in range(n):
            seed = _seed_for(setting_key, i)
            rng = np.random.default_rng(seed)
            applied_cols = []
            for col in target_columns:
                if df[col].dtype.kind not in ("i", "f"):
                    continue
                v = float(df[col].iloc[i])
                out.loc[out.index[i], col] = v + rng.normal(0, sigma)
                applied_cols.append(col)
            params_log.append({
                "row_idx": i, "rng_seed": seed,
                "params": {"type": type_name, "severity": severity,
                           "sigma": sigma, "applied_cols": applied_cols},
            })
        return out, params_log

    if type_name == "scaling":
        factor = {1: 10, 2: 50, 3: 100, 4: 500, 5: 1000}[severity]
        for i in range(n):
            seed = _seed_for(setting_key, i)
            applied_cols = []
            for col in target_columns:
                if df[col].dtype.kind not in ("i", "f"):
                    continue
                v = float(df[col].iloc[i])
                out.loc[out.index[i], col] = v * factor
                applied_cols.append(col)
            params_log.append({
                "row_idx": i, "rng_seed": seed,
                "params": {"type": type_name, "severity": severity,
                           "factor": factor, "applied_cols": applied_cols},
            })
        return out, params_log

    raise NotImplementedError(
        f"Tabular corruption '{type_name}' not implemented in this draft. "
        "Wire up the real implementation from code/corrupt/tabular/."
    )
