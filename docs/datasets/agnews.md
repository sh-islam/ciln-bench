# CILN-Bench: AG-News

## Overview

- **Base data**: AG-News (120,000 training articles, 4 balanced classes: World, Sports, Business, Sci/Tech).
- **Subset corrupted**: 50,000-article *noisy-label train* (NLT) split. Splits are deterministic (`code/splits/build_splits_text.py`) and the index files ship with the HF release.
- **Corruption families**: Character, Structural. 2 corruption types, 3 severities each, **6 settings** total.
- **Voter pool**: 4 voters. fastText, DistilBERT-cased, RoBERTa-base, and sentence-transformer `all-MiniLM-L6-v2` used zero-shot with class-prompt embeddings.
- **Headline metrics** (per setting): noise rate, NTH (stored as `vdv` in the JSON). Pre-computed numbers ship under `results/agnews/`.

## Corruptions

| Family     | Type             | Severity parameter                                    | Source |
| ---------- | ---------------- | ----------------------------------------------------- | ------ |
| Character  | butter_fingers   | fraction of characters replaced by a keyboard-adjacent key: 0.05 / 0.15 / 0.30 | NL-Augmenter |
| Structural | front_truncation | leading fraction of words deleted: 0.20 / 0.50 / 0.75 | ours, written in NL-Augmenter's format |

Word-level corruptions (synonym substitution, word swap/deletion/shuffle, keyword removal) and surface-form corruptions (random casing, diacritic substitution) are implemented in `code/corrupt/canonical/text/text_funcs.py` but excluded from the release: they stay under the 2.5% noise-rate floor on AG-News.

Noise rate ranges from 13.0% to 46.6% across the 6 released settings.

## Files per setting (HF release)

```
settings/<setting>/
├── texts.npy              # (50000,) corrupted article text
├── labels.npy             # (50000,) int64 ground-truth labels
├── softmax_fasttext.npy   # (50000, 4) float32 per-voter softmax
├── softmax_distilbert.npy
├── softmax_roberta.npy
├── softmax_sbert.npy
├── avg_softmax.npy        # (50000, 4) mean over voters
├── manifest.json
└── params.jsonl           # per-row seed + corruption parameters
clean_start/               # clean-NLT softmaxes, labels, clean_correct_mask.npy (CILN-C)
splits/                    # CLT / CLV / NLT / NLV index files into the AG-News train split
sampled_labels/            # the sampled training labels used for the downstream runs (one file per seed)
```

Clean article text is not re-distributed. Recover it from the public AG-News train split with `splits/NLT_indices.npy`.

## Reproducing

```bash
# Corrupt the NLT split (all released settings)
python code/corrupt/canonical/text/produce_corrupted_text.py
# Run the four voters on the corrupted settings / on the clean split
python code/infer/run_inference_text.py
python code/infer/run_clean_inference_text.py
# NTH per setting -> results/agnews/nth_ciln_s.json
python code/analyze/canonical/compute_nth_text.py
```

Downstream (CE and Co-Teaching, DistilBERT-base learner): `code/downstream/agnews/`, per-seed results in `results/downstream/agnews/`.
