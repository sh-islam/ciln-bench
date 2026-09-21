# CILN-Bench: CIFAR-10

## Overview

- **Base data**: CIFAR-10 (60,000 32×32 RGB images, 10 classes).
- **Subset corrupted**: 22,500-image *noisy-label train* (NLT) split, plus the 10,000-image test split for the TV-vs-CIFAR-10H comparison.
- **Corruption families**: Noise, Blur, Weather, Digital. 15 corruption types, 3 severities each, **45 settings** total. Noise rate ranges from 8.3% to 75.0%.
- **Voter pool**: 4 voters. ResNet-20, WRN-28-10, DeiT3-Small, CLIP ViT-B/32.
- **Headline metrics** (per setting): noise rate, NTH (stored as `m2_frobenius` in the JSON), TV distance to CIFAR-10H. Pre-computed numbers ship under `results/cifar10/`.

## Corruptions

| Family     | Types                                                |
| ---------- | ---------------------------------------------------- |
| Noise      | gaussian_noise, shot_noise, impulse_noise            |
| Blur       | defocus_blur, glass_blur, motion_blur, zoom_blur     |
| Weather    | fog, frost, snow, brightness                         |
| Digital    | contrast, elastic_transform, jpeg_compression, pixelate |

These are the standard CIFAR-C corruptions from Hendrycks & Dietterich (2019). We use their original code, kept verbatim, in [`code/corrupt/canonical/cifar_funcs.py`](../../code/corrupt/canonical/cifar_funcs.py).

## TV Distance

`results/cifar10/tv_ciln_s.json` holds the per-setting mean per-image TV distance of the benchmark vote-share vectors (computed on the corrupted CIFAR-10 test split, `settings/<setting>/test/`) against three references, each at the setting's noise rate:

- **CIFAR-10H upsampled.** Human labels upsampled so the overall error rate matches the setting's noise rate, then subsampled to 4 votes per image so both sides share the same quantisation (PL-IDN's protocol).
- **Symmetric flipping.** Independent per-voter symmetric noise.
- **CCN.** Per-class transition matrix estimated from the benchmark itself.

`results/cifar10/tv_ciln_c.json` holds the clean-start (CILN-C) values, and `results/cifar10/tv_full_reference_check.csv` recomputes TV against the full CIFAR-10H distribution without the 4-vote subsample (every ordering reported in the paper is unchanged. See the accompanying report).

Clean implementation: [`code/analyze/public/tv.py`](../../code/analyze/public/tv.py). Reproduction script: [`examples/reproduce_tv.py`](../../examples/reproduce_tv.py).

## Reproducing the headline numbers

```bash
# Recompute NTH from voter softmaxes (bit-identical to released JSON)
python examples/reproduce_vdv.py --data-root ./ciln-bench-cifar10/settings

# Recompute TV against CIFAR-10H
python examples/reproduce_tv.py --data-root ./ciln-bench-cifar10/settings \
                                --cifar10h cifar10h-counts.npy
```

Downstream (CE, Co-Teaching, DivideMix. ResNet-20 learner) and the PL-IDN comparison: `code/downstream/cifar10/`, `code/downstream/plidn/`, per-seed results in `results/downstream/`.
