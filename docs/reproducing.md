# Reproducing the paper

Everything below assumes a downloaded release, e.g. `./ciln-bench-cifar10/settings/<setting>/...` from the Hugging Face repos linked in the main README, and `pip install -r requirements.txt` (Python 3.10+. NumPy, pandas and SciPy for the analyses. OpenCV, scikit-image, Wand and Pillow for image corruptions. PyTorch, timm, open_clip, transformers and fastText for voters and learners).

## canonical/ vs public/

`code/*/canonical/` is the exact code that produced the release. It hardcodes a few paths to the original research workspace. Edit the `Path(...)` constants at the top of a file to point at a downloaded release. `code/*/public/` is the same logic rewritten clean, with no hardcoded paths. `tests/check_equivalence.py` proves `public/` matches `canonical/` bit-for-bit. Text operators exist in `canonical/` only.

## Benchmark construction

`code/splits/` builds the clean-label-train / noisy-label-train / valid / test splits (index files also ship with each dataset under `splits/`). `code/corrupt/` applies a corruption setting to the noisy-label-train split. `code/train/` trains the voters, or download them from `sh-islam/ciln-bench-voters`. `code/infer/` runs the voters (`eval_image_voters.py`, `eval_tabular_voters.py`, `run_inference_text.py`), assembles the release layout (`assemble_release.py`) and derives the clean-start mask (`build_clean_start_mask.py`). `code/infer/voter_adapters.py` reproduces the released softmaxes bit-for-bit from the published checkpoints. `code/infer/test_set/` is the CIFAR-10 test-split corruption + inference used for the CIFAR-10H comparison.

Every corruption uses a seeded RNG and logs the seed per row in `params.jsonl`, together with the sampled parameters and a SHA-256 of the corrupted input.

## NTH

```bash
python examples/reproduce_vdv.py --data-root ./ciln-bench-cifar10/settings
```

NTH is stored under its working name in the JSON files: `m2_frobenius` (CIFAR-10, MNIST, Adult) and `vdv` (AG-News). Same quantity. The script asserts bit-identity against `results/<dataset>/nth_ciln_s.json`. The clean-start values are in `nth_ciln_c.json`. `code/analyze/canonical/compute_idn_2x2.py` also computes NTH for the three PL-IDN releases at their native rater count.

## TV vs CIFAR-10H

```bash
python examples/reproduce_tv.py --data-root ./ciln-bench-cifar10/settings --cifar10h cifar10h-counts.npy
```

Per setting, the CIFAR-10H reference is upsampled to the setting's noise rate, subsampled to four votes per image, and per-image TV is averaged over 50 bootstrap resamples (`code/analyze/canonical/tv_vs_cifar10h.py`). `tv_clean_start.py` gives the clean-start values, and `tv_full_reference_check.py` recomputes TV against the full CIFAR-10H distribution without the subsample. Every ordering in the paper survives (`results/cifar10/tv_full_reference_check_report.txt`).

## Downstream learning

`code/downstream/{cifar10,mnist,adult,agnews}/` hold the CE, Co-Teaching and DivideMix trainers with the recipes of the paper's Appendix C (ResNet-20, LeNet-5, a 3x256 MLP, DistilBERT-base). Training labels draw one voter uniformly per row and use its arg-max (`code/downstream/common/sample_labels.py`). On CIFAR-10, `prepare_pairs_v4.py` draws one fixed label set per condition. The labels used are released under `sampled_labels/` in each dataset repo.

Every reported accuracy is the mean over three seeds of `best_test_acc` in the per-seed JSONs under `results/downstream/`. `code/downstream/cifar10/analyze_table2.py` and `code/downstream/common/aggregate.py` rebuild the tables. The PL-IDN rows are our re-runs of the downloaded PL-IDN release under the same learner (`code/downstream/plidn/`, `results/downstream/plidn/`). `code/analyze/canonical/attractor_purity_table.py` (majority-vote labels) and `dm_selection_diagnostics.py` produce the diagnostics table and the DivideMix loss analysis.

## Figures

`code/tmlr_figures/` contains the scripts that produced the paper figures and their input data (Fig. 2 reads the released softmaxes directly. See `code/tmlr_figures/data/fig02_severity_distributions/DATA_ON_HF.md`).

## Corrupt your own data

```bash
python noisify_dataset.py
```

Interactive CLI over the same corruption code. Input contract: [usage.md](usage.md).
