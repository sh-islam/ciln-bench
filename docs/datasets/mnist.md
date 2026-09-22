# CILN-Bench: MNIST

## Overview

- **Base data**: MNIST (60,000 training + 10,000 test, 28×28 grayscale digits).
- **Subset corrupted**: 27,000-image *noisy-label train* (NLT) split.
- **Corruption families**: Noise, Blur, Weather, Geometric, Structural. 15 corruption types from MNIST-C span 35 candidate settings. After the 2.5% noise-rate floor, **29 settings are released**.
- **Voter pool**: 4 voters. LeNet-5, MLP, ResNet-20, DeiT3-Small.
- **Headline metrics** (per setting): noise rate, NTH (stored as `m2_frobenius` in the JSON). Pre-computed numbers ship under `results/mnist/`.

## Corruptions

| Family     | Types                                          |
| ---------- | ---------------------------------------------- |
| Noise      | shot_noise, impulse_noise, spatter             |
| Blur       | glass_blur, motion_blur                        |
| Weather    | brightness, fog                                |
| Geometric  | rotate, shear, translate, scale                |
| Structural | canny_edges, dotted_line, stripe, zigzag       |

Candidate rule: every continuous corruption contributes severities 1, 3, 5, except `scale`, whose noise rate does not rise with severity and contributes only its noisiest setting. The four Structural corruptions are binary operators (severity-invariant by construction) and contribute one setting each, shipped as `_sev1`. That gives 35 candidates. The six below the 2.5% floor are dropped:

`dotted_line_sev1` (2.4%), `motion_blur_sev1` (1.8%), `rotate_sev1` (1.6%), `shear_sev1` (1.6%), `shot_noise_sev1` (1.3%), `shot_noise_sev3` (1.8%).

Released (29): brightness 1/3/5, canny_edges 1, fog 1/3/5, glass_blur 1/3/5, impulse_noise 1/3/5, motion_blur 3/5, rotate 3/5, scale 1, shear 3/5, shot_noise 5, spatter 1/3/5, stripe 1, translate 1/3/5, zigzag 1. Noise rate ranges from 2.5% to 71.5%. The full list is in `results/released_settings.json`.

We follow Mu & Gilmer (2019), the MNIST-C reference. The corruption code lives in `code/corrupt/canonical/mnist_funcs.py`.

## Reproducing the headline numbers

```bash
# Recompute NTH from voter softmaxes (bit-identical to released JSON)
python examples/reproduce_vdv.py --data-root ./ciln-bench-mnist/settings
```

Downstream (CE, Co-Teaching, DivideMix. LeNet-5 learner, Adam, 100 epochs): `code/downstream/mnist/`, per-seed results in `results/downstream/mnist/`.
