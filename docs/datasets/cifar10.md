# CILN-Bench: CIFAR-10

## At a glance

- **Base data**: CIFAR-10 (60,000 32×32 RGB images, 10 classes).
- **Subset corrupted**: 22,500-image *noisy-label train* (NLT) split, plus the 10,000-image test split for the TV-vs-CIFAR-10H comparison.
- **Corruption families**: Noise, Blur, Weather, Geometric, Digital. 15 corruption types, 3 severities each, **45 settings** total.
- **Voter pool**: 4 voters. ResNet-20, WRN-28-10, DeiT3-Small, CLIP ViT-B/32.
- **Headline metrics** (per setting): noise rate δ, VDV (reported as `nth` in the paper). Pre-computed numbers ship with the repo under `results/`.

## Corruptions

| Family     | Types                                                |
| ---------- | ---------------------------------------------------- |
| Noise      | gaussian_noise, shot_noise, impulse_noise            |
| Blur       | defocus_blur, glass_blur, motion_blur, zoom_blur     |
| Weather    | fog, frost, snow                                     |
| Geometric  | elastic_transform                                    |
| Digital    | brightness, contrast, jpeg_compression, pixelate     |

These are the standard CIFAR-C corruptions from Hendrycks & Dietterich (2019). We use their original code, kept verbatim, in [`code/corrupt/canonical/cifar_funcs.py`](../../code/corrupt/canonical/cifar_funcs.py).

## TV vs CIFAR-10H

`results/tv_vs_cifar10h_v2.json` holds the per-setting mean per-image TV distance of the benchmark soft labels against three references, each at matched noise rate δ:

- **CIFAR-10H upsampled.** Human labels scaled so the overall error rate matches δ.
- **Symmetric flipping.** Independent per-voter symmetric noise.
- **CCN.** Per-class transition matrix estimated from the benchmark itself.

Clean implementation: [`code/analyze/public/tv.py`](../../code/analyze/public/tv.py). Reproduction script: [`examples/reproduce_tv.py`](../../examples/reproduce_tv.py).

## Reproducing the headline numbers

```bash
# Recompute VDV from voter softmaxes (bit-identical to released JSON)
python examples/reproduce_vdv.py --data-root ./ciln-bench-cifar10/settings

# Recompute TV against CIFAR-10H
python examples/reproduce_tv.py --data-root ./ciln-bench-cifar10/settings \
                                --cifar10h cifar10h-counts.npy
```
