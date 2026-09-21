"""Public tabular corruption interface.

A generic, dataset-agnostic re-implementation of the tabular corruption
pipeline used to generate the released Adult benchmark. To use it on a new
tabular dataset, supply the column groups (numeric vs categorical) and any
dataset-level statistics required by the MNAR corruption.

Same severity grid as the released benchmark:

* ``severity=1`` → 5 %  of rows corrupted
* ``severity=3`` → 25 %
* ``severity=5`` → 50 %

Five corruption types:

* ``missing_mcar`` — Missing Completely At Random.
* ``missing_mar``  — Missing At Random conditional on an observed partner
  column.
* ``missing_mnar`` — Missing Not At Random; drop probability depends on the
  cell's own value (high-quantile numeric or low-frequency categorical).
* ``scaling``      — Multiply numeric cells by a fixed factor.
* ``gaussian_noise`` — Add Gaussian noise (fixed σ per severity) to numeric
  cells.

Identity guarantee
------------------
For the Adult column setup used in the paper, the output of every function
here is bit-identical to the canonical ``adult_funcs.py`` for the same
``(row, rng, severity)`` inputs. This is checked in
``tests/check_equivalence.py``.

Usage
-----

>>> from code.corrupt.public.tabular_corruptions import (
...     TabularCorruptionConfig, missing_mar, gaussian_noise
... )
>>> cfg = TabularCorruptionConfig(
...     numeric_cols=["age", "income"],
...     categorical_cols=["sex", "education"],
...     mar_partner_col="sex",
...     mar_high_group="Female",
... )
>>> out, params = missing_mar({"age": 30, "income": 50000, "sex": "Female",
...                            "education": "BS"},
...                           severity=3, rng=rng, config=cfg)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Same constants as canonical/adult_funcs.py
SEVERITY_FRACTION: Dict[int, float] = {1: 0.05, 3: 0.25, 5: 0.50}
SEVERITY_SCALE_FACTOR: Dict[int, int] = {1: 10, 3: 100, 5: 1000}
SEVERITY_NOISE_STD: Dict[int, float] = {1: 2.0, 3: 3.5, 5: 5.0}

# MAR-specific multipliers (paper uses 1.5x for the "high-risk" group and
# 0.75x for the "low-risk" group; preserves a 2:1 contrast ratio while
# preventing 100% corruption at severity 5).
MAR_MULT_HIGH = 1.5
MAR_MULT_LOW  = 0.75


@dataclass
class TabularCorruptionConfig:
    """All dataset-specific configuration the corruption functions need."""
    numeric_cols: List[str]
    categorical_cols: List[str]
    mar_partner_col: Optional[str] = None
    mar_high_group: Optional[str] = None
    # For MNAR: maps "{col}__quantile_rank" -> {value: rank_in_[0,1]} (numeric)
    # and "{col}__freq" -> {value: relative_frequency} (categorical)
    dataset_stats: Dict[str, Dict[Any, float]] = field(default_factory=dict)

    @property
    def all_cols(self) -> List[str]:
        return self.numeric_cols + self.categorical_cols


# ---------------------------------------------------------------------------
# Row-level helpers
# ---------------------------------------------------------------------------

def _row_is_corrupted(rng: np.random.Generator, severity: int) -> bool:
    """A row is "in the corrupted fraction" with probability ``SEVERITY_FRACTION[severity]``."""
    return rng.random() < SEVERITY_FRACTION[severity]


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------

def missing_mcar(
    row: dict, severity: int, rng: np.random.Generator,
    config: TabularCorruptionConfig,
) -> Tuple[dict, dict]:
    """Missing Completely At Random.

    With probability ``r = SEVERITY_FRACTION[severity]``, every eligible cell
    on this row is set to NaN.
    """
    if not _row_is_corrupted(rng, severity):
        return dict(row), {"corrupted": False}
    out = dict(row)
    cols_set = []
    for c in config.all_cols:
        if c in out:
            out[c] = np.nan
            cols_set.append(c)
    return out, {"corrupted": True, "cols_set_to_nan": cols_set}


def missing_mar(
    row: dict, severity: int, rng: np.random.Generator,
    config: TabularCorruptionConfig,
) -> Tuple[dict, dict]:
    """Missing At Random conditional on an observed partner column.

    The partner column (``config.mar_partner_col``) is preserved on corrupted
    rows so downstream models can still condition on it. Rows whose partner
    value equals ``config.mar_high_group`` get a higher drop probability
    (1.5×base) than other rows (0.75×base).
    """
    if not config.mar_partner_col or not config.mar_high_group:
        raise ValueError("MAR requires 'mar_partner_col' and 'mar_high_group' "
                         "in TabularCorruptionConfig.")
    base_r = SEVERITY_FRACTION[severity]
    partner_val = str(row.get(config.mar_partner_col, "")).strip()
    if partner_val == config.mar_high_group:
        p = min(1.0, MAR_MULT_HIGH * base_r)
    else:
        p = MAR_MULT_LOW * base_r
    if rng.random() >= p:
        return dict(row), {"corrupted": False, "partner": partner_val, "p_used": p}
    out = dict(row)
    cols_set = []
    for c in config.all_cols:
        if c == config.mar_partner_col:
            continue
        if c in out:
            out[c] = np.nan
            cols_set.append(c)
    return out, {
        "corrupted": True, "partner": partner_val, "p_used": p,
        "cols_set_to_nan": cols_set, "preserved_col": config.mar_partner_col,
    }


def missing_mnar(
    row: dict, severity: int, rng: np.random.Generator,
    config: TabularCorruptionConfig,
) -> Tuple[dict, dict]:
    """Missing Not At Random; drop probability depends on the cell's own value.

    * For numeric columns, higher-quantile values are more likely to be
      missing (mimics top-coding / privacy redaction).
    * For categorical columns, rarer categories are more likely to be missing.

    Requires ``config.dataset_stats`` with ``{col}__quantile_rank`` mapping
    for each numeric column and ``{col}__freq`` for each categorical column.
    """
    base_r = SEVERITY_FRACTION[severity]
    out = dict(row)
    cols_set: List[str] = []
    for c in config.numeric_cols:
        if c not in out:
            continue
        q = config.dataset_stats.get(f"{c}__quantile_rank", {}).get(out[c], 0.5)
        p = base_r * (0.5 + q)  # bias toward higher-quantile values
        if rng.random() < p:
            out[c] = np.nan
            cols_set.append(c)
    for c in config.categorical_cols:
        if c not in out:
            continue
        freq = config.dataset_stats.get(f"{c}__freq", {}).get(str(out[c]), 0.5)
        p = base_r * (1.5 - freq)  # rarer categories -> higher drop prob
        if rng.random() < p:
            out[c] = np.nan
            cols_set.append(c)
    return out, {"corrupted": len(cols_set) > 0, "cols_set_to_nan": cols_set}


def scaling(
    row: dict, severity: int, rng: np.random.Generator,
    config: TabularCorruptionConfig,
) -> Tuple[dict, dict]:
    """Multiply every numeric cell on a corrupted row by a fixed factor."""
    if not _row_is_corrupted(rng, severity):
        return dict(row), {"corrupted": False}
    factor = SEVERITY_SCALE_FACTOR[severity]
    out = dict(row)
    for c in config.numeric_cols:
        if c in out and pd.notna(out[c]):
            out[c] = out[c] * factor
    return out, {"corrupted": True, "factor": factor,
                 "cols_scaled": list(config.numeric_cols)}


def gaussian_noise(
    row: dict, severity: int, rng: np.random.Generator,
    config: TabularCorruptionConfig,
) -> Tuple[dict, dict]:
    """Add Gaussian noise with severity-dependent σ to every numeric cell on a corrupted row."""
    if not _row_is_corrupted(rng, severity):
        return dict(row), {"corrupted": False}
    std = SEVERITY_NOISE_STD[severity]
    out = dict(row)
    for c in config.numeric_cols:
        if c in out and pd.notna(out[c]):
            out[c] = float(out[c]) + float(rng.normal(loc=0.0, scale=std))
    return out, {"corrupted": True, "noise_std": std,
                 "cols_noised": list(config.numeric_cols)}


CORRUPTION_FUNCTIONS = {
    "missing_mcar":    missing_mcar,
    "missing_mar":     missing_mar,
    "missing_mnar":    missing_mnar,
    "scaling":         scaling,
    "gaussian_noise":  gaussian_noise,
}


# ---------------------------------------------------------------------------
# Adult-specific convenience (matches the canonical pipeline 1:1)
# ---------------------------------------------------------------------------

ADULT_CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
ADULT_NUMERIC_COLS = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]


def adult_config(dataset_stats: Optional[Dict[str, Dict[Any, float]]] = None) -> TabularCorruptionConfig:
    """Return a TabularCorruptionConfig matching the released Adult pipeline.

    Pass ``dataset_stats`` if you intend to call ``missing_mnar``.
    """
    return TabularCorruptionConfig(
        numeric_cols=ADULT_NUMERIC_COLS,
        categorical_cols=ADULT_CATEGORICAL_COLS,
        mar_partner_col="sex",
        mar_high_group="Female",
        dataset_stats=dataset_stats or {},
    )
