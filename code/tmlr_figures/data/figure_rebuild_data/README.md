# figure_rebuild_data

CSV exports for rebuilding paper figures from CILNBench release + experiment
outputs. All numbers reproduce the pipelines described in the paper and the
canonical scripts. Nothing was retrained or re-inferred.

## FILE 1: tv_cifar_all_settings.csv

90 rows (45 CIFAR settings x 2 variants).

**Columns**: corruption, family, severity, variant (S|C), tv_mean, tv_ci95_half.

**Source for CILN-S rows (variant=S)**:
`results/cifar10/tv_ciln_s.json`
Produced by `journal_edition/idn_2x2/tv_vs_cifar10h.py` — per-image
bincount of 4 CIFAR voters on the test-set corrupted images, then TV to
per-image CIFAR-10H distribution, bootstrapped with 50 upsampling
resamples for CI95. Matches the thesis's characterisation table exactly.

**CILN-C rows (variant=C)**: N/A in this CSV. The CILN-C TV values are in `results/cifar10/tv_ciln_c.json`, computed on the test-split rows that pass the clean-start mask (`code/analyze/canonical/tv_clean_start.py`).

## FILE 2: dm_losses_<setting>_<scenario>_<epoch>.csv

12 files: 3 settings (frost_sev1, contrast_sev5, glass_blur_sev3) x
2 scenarios (C = clean-img, N = noisy-img) x 2 epochs (warmup=10,
final=99). Seed 0, CILN-S variant.

**Columns**: sample_idx (0..22499), loss, is_noisy (0/1).

**Source**:
`journal_edition/robustness_transfer/results_v4/dividemix/ours_v2_{clean|noisy}__<setting>__seed0__persample.npz`.
Each NPZ has a (7, 22500) `losses1` array logged at epochs
[5, 10, 20, 30, 50, 75, 99]; we ship the rows at ep=10 (warmup) and
ep=99 (final). `is_noisy` is the ground-truth mask
(sampled_label != true_label) from the same NPZ.

## FILE 3: sorted_class_frequencies.csv + label_entropy.csv

Sorted class-frequency profiles for 4 (dataset, corruption) combinations
at 3 conditions each (clean labels, severity 1 sampled labels, severity
5 sampled labels).

Combinations:
- cifar10 pixelate
- cifar10 gaussian_noise (as `gaussian-noise` in CSV)
- mnist rotate
- mnist brightness

**sorted_class_frequencies.csv columns**: dataset, corruption,
condition, rank (1..10, class with highest frequency = rank 1),
class (name at that rank), freq (fraction 0..1).

**label_entropy.csv columns**: dataset, corruption, condition,
entropy_bits (Shannon entropy in bits of the distribution).

**Source**: `journal_edition/output_seed0/{dataset}/{corr}/severity_{sev}/noisy_label_train/`.
Sampled labels use the PL-IDN protocol (one voter uniformly per image,
seeded RNG with seed=0), taking that voter's argmax on the corrupted
image. Clean condition = true labels of the NLT pool. Same protocol as
the label-distribution figures.

## FILE 4: agnews_flip_distributions.csv

Voter-majority wrong-class prediction distribution on AG-News at
severity 5 for butter-fingers and front-truncation. Variant: **CILN-S
(full NLT)** — CILN-C is not defined for AG-News (no clean-correct mask
in that pipeline).

**Columns**: corruption, true_class, pred_class (!= true_class), pct
(percentage of that true class's misclassified rows landing on
pred_class). Rows per (corruption, true_class) sum to 100.

**Source**: `ciln_text/settings/{corr}_sev5/`. Majority vote = argmax of
per-image bincount over 4 AG-News voter argmaxes (fasttext, distilbert,
roberta, sbert). "Misclassified" = row where majority != true_class.

## Provenance notes

- Nothing was retrained or re-inferred. All numbers extracted from
  existing NPZ / JSON / softmax files in the original workspace.
- No approximations. Numbers are exact to the precision listed in the
  columns (typically 6 decimals).
- CILN-C TV rows are not in `tv_cifar_all_settings.csv`. See `results/cifar10/tv_ciln_c.json`.
