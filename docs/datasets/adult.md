# CILN-Bench: Adult

## At a glance

- **Base data**: UCI Adult Income (48,842 rows, 14 features, binary target).
- **Subset corrupted**: 13,566-row *noisy-label train* (NLT) split.
- **Corruption families**: Missingness, Numeric perturbation. 5 corruption types, 3 severities each, **15 settings** total.
- **Voter pool**: 5 voters. XGBoost, CatBoost, RTDL-MLP, FT-Transformer, TabPFN.
- **Headline metrics** (per setting): noise rate δ, VDV (reported as `nth` in the paper). Pre-computed numbers ship with the repo under `results/`.

## Corruptions

| Type            | Description                                              |
| --------------- | -------------------------------------------------------- |
| missing_mcar    | Drop eligible cells uniformly at random.                 |
| missing_mar     | Drop conditional on `sex` (Female = 1.5× base rate).     |
| missing_mnar    | Drop conditional on the cell's own value.                |
| scaling         | Multiply numeric cells by a fixed factor (×10/100/1000). |
| gaussian_noise  | Add zero-mean Gaussian noise to numeric cells.           |

Severity controls the corrupted-row fraction: sev 1 hits 5% of rows, sev 3 hits 25%, sev 5 hits 50%. The corruption types and severity scheme follow Schelter et al. (EDBT 2021), "Jenga".

The original Adult-specific code lives in [`code/corrupt/canonical/adult_funcs.py`](../../code/corrupt/canonical/adult_funcs.py). The public wrapper [`code/corrupt/public/tabular_corruptions.py`](../../code/corrupt/public/tabular_corruptions.py) is dataset-agnostic. Use `adult_config()` to reproduce the Adult pipeline, or pass your own `TabularCorruptionConfig` to corrupt any tabular dataset.
