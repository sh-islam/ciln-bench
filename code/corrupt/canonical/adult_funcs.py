"""Adult tabular corruption functions, refactored for full reproducibility.

Every function takes:
  df_row: a dict-like view of one row of the (cleaned) Adult dataframe
  severity: 1, 3, or 5
  rng: numpy.random.Generator (per-row, seeded deterministically by the caller)

Every function returns (corrupted_row_dict, params_dict).

Corruption taxonomy (from Jenga, Schelter et al. EDBT 2021):
  1. missing_mcar  - missing values, completely at random
  2. missing_mar   - missing values, conditional on a partner column's value
  3. missing_mnar  - missing values, conditional on this column's own value
  4. scaling       - multiply numeric cells by a scale factor (10/100/1000)
  5. gaussian_noise - add gaussian noise to numeric cells

We deliberately omit Jenga's "swapped values" and "encoding errors" corruptions
(see tabular_training_recipe.md for the design decision).

Severity -> fraction r of rows corrupted (per Jenga's `fraction` parameter):
  sev 1 = 0.05 (5%)
  sev 3 = 0.25 (25%)
  sev 5 = 0.50 (50%)

Note: this design corrupts at the *row* level. A row is "corrupted" with
probability r based on its rng seed; if corrupted, every eligible column on
that row gets hit (bundled application across columns; see recipe doc).

Reproducibility: caller derives a per-row seed via SHA256 of
(master_seed, dataset, corruption, severity, row_idx); no global np.random
state is used.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# Column groups for Adult (matches journal_edition/tabular/common.py)
CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
NUMERICAL_COLS = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]
ALL_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# Severity -> fraction-of-rows-corrupted
SEVERITY_FRACTION = {1: 0.05, 3: 0.25, 5: 0.50}

# Severity -> scaling multiplier (for scaling corruption)
SEVERITY_SCALE_FACTOR = {1: 10, 3: 100, 5: 1000}

# Severity -> Gaussian noise std (for gaussian_noise corruption)
# Jenga draws std uniformly from [2, 5]; we map sev to fixed values in that range
SEVERITY_NOISE_STD = {1: 2.0, 3: 3.5, 5: 5.0}


def _row_is_corrupted(rng: np.random.Generator, severity: int) -> bool:
    """Caller decides per-row whether this row is in the corrupted fraction."""
    return rng.random() < SEVERITY_FRACTION[severity]


def missing_mcar(row: dict, severity: int, rng: np.random.Generator,
                 dataset_stats: dict = None) -> Tuple[dict, dict]:
    """Missing completely at random: with prob r, every eligible column on this
    row gets set to NaN (or pandas NA for category)."""
    if not _row_is_corrupted(rng, severity):
        return dict(row), {"corrupted": False}
    out = dict(row)
    for c in ALL_COLS:
        if c in out:
            out[c] = np.nan
    return out, {"corrupted": True, "cols_set_to_nan": ALL_COLS}


MAR_PARTNER_COL = "sex"  # observed conditioning column, preserved on corrupted rows
MAR_MULT_HIGH = 1.5      # softer than the original 2.0 so sev5 never hits 100% per-group
MAR_MULT_LOW = 0.75      # paired with HIGH so aggregate ~= base rate (asymmetry preserved)

def missing_mar(row: dict, severity: int, rng: np.random.Generator,
                dataset_stats: dict = None) -> Tuple[dict, dict]:
    """Missing at random (textbook Rubin 1976): missingness probability depends
    only on an *observed* partner column, AND that partner column remains
    observable on corrupted rows so downstream models can condition on it.

    Implementation:
      - Use `sex` (MAR_PARTNER_COL) as the partner column.
      - Female rows -> corrupted with prob 1.5*r
      - Male rows   -> corrupted with prob 0.75*r
      - When corrupted: NaN every column EXCEPT `sex`. Preserving the partner
        column is what makes this textbook MAR at *observation* time, not just
        at generation time.

    Why 1.5x / 0.75x instead of the original 2x / 0.5x:
      - At sev5 (base_r=0.5), 2*0.5=1.0 capped at 1.0 means 100% of Females
        get corrupted — leaves no observable Female rows, makes the MAR
        signal degenerate at the top severity.
      - 1.5*0.5=0.75 caps Female corruption at 75%, so at least 25% of
        Females remain observable at sev5. Male side: 0.75*0.5=0.375.
      - 2:1 contrast ratio between groups is preserved (1.5/0.75 = 2.0),
        so the MAR signal strength is the same; only the absolute
        probability is lower.
      - With test split ~67% Male / 33% Female, the aggregate corruption
        rate ((10064*0.75r + 5011*1.5r)/15075) ≈ r exactly (the asymmetry
        almost cancels because the Male share is slightly above the Female
        share's complement). Empirical aggregate stays ~5/25/50% per
        severity, matching the other corruptions.
    """
    base_r = SEVERITY_FRACTION[severity]
    sex_val = str(row.get(MAR_PARTNER_COL, "")).strip()
    if sex_val == "Female":
        p = min(1.0, MAR_MULT_HIGH * base_r)
    else:
        p = MAR_MULT_LOW * base_r
    if rng.random() >= p:
        return dict(row), {"corrupted": False, "sex": sex_val, "p_used": p}
    out = dict(row)
    cols_set = []
    for c in ALL_COLS:
        if c == MAR_PARTNER_COL:
            continue
        if c in out:
            out[c] = np.nan
            cols_set.append(c)
    return out, {"corrupted": True, "sex": sex_val, "p_used": p,
                 "cols_set_to_nan": cols_set, "preserved_col": MAR_PARTNER_COL}


def missing_mnar(row: dict, severity: int, rng: np.random.Generator,
                 dataset_stats: dict = None) -> Tuple[dict, dict]:
    """Missing not at random: probability depends on this column's own value.
    Implementation: corrupt each column independently with prob proportional to
    how "extreme" the value is in that column.
      - For numerical cols: higher quantile values are more likely to be missing
        (mimics top-coding / privacy redaction).
      - For categorical cols: rarer categories are more likely to be missing.
    `dataset_stats` must supply numerical quantiles and categorical frequencies.
    """
    base_r = SEVERITY_FRACTION[severity]
    out = dict(row)
    cols_set = []
    for c in NUMERICAL_COLS:
        if c in out and dataset_stats is not None:
            v = out[c]
            # Quantile rank in [0,1] of this value
            q = dataset_stats.get(f"{c}__quantile_rank", {}).get(v, 0.5)
            p = base_r * (0.5 + q)  # bias toward higher-quantile values
            if rng.random() < p:
                out[c] = np.nan
                cols_set.append(c)
    for c in CATEGORICAL_COLS:
        if c in out and dataset_stats is not None:
            v = str(out[c])
            freq = dataset_stats.get(f"{c}__freq", {}).get(v, 0.5)
            p = base_r * (1.5 - freq)  # rarer (lower freq) -> more likely missing
            if rng.random() < p:
                out[c] = np.nan
                cols_set.append(c)
    corrupted = len(cols_set) > 0
    return out, {"corrupted": corrupted, "cols_set_to_nan": cols_set}


def scaling(row: dict, severity: int, rng: np.random.Generator,
            dataset_stats: dict = None) -> Tuple[dict, dict]:
    """Multiply numerical cells by a fixed factor (10/100/1000 per severity).
    Applied at the row level: if row is corrupted, all numerical columns are
    scaled."""
    if not _row_is_corrupted(rng, severity):
        return dict(row), {"corrupted": False}
    factor = SEVERITY_SCALE_FACTOR[severity]
    out = dict(row)
    for c in NUMERICAL_COLS:
        if c in out and pd.notna(out[c]):
            out[c] = out[c] * factor
    return out, {"corrupted": True, "factor": factor, "cols_scaled": NUMERICAL_COLS}


def gaussian_noise(row: dict, severity: int, rng: np.random.Generator,
                   dataset_stats: dict = None) -> Tuple[dict, dict]:
    """Add Gaussian noise to numerical cells. Std scales with severity."""
    if not _row_is_corrupted(rng, severity):
        return dict(row), {"corrupted": False}
    std = SEVERITY_NOISE_STD[severity]
    out = dict(row)
    for c in NUMERICAL_COLS:
        if c in out and pd.notna(out[c]):
            out[c] = float(out[c]) + float(rng.normal(loc=0.0, scale=std))
    return out, {"corrupted": True, "noise_std": std, "cols_noised": NUMERICAL_COLS}


CORRUPTION_FUNCTIONS = {
    "missing_mcar":    missing_mcar,
    "missing_mar":     missing_mar,
    "missing_mnar":    missing_mnar,
    "scaling":         scaling,
    "gaussian_noise":  gaussian_noise,
}
