# CILN-Bench: MNIST

## At a glance

- **Base data**: MNIST (60,000 training + 10,000 test, 28×28 grayscale digits).
- **Subset corrupted**: 27,000-image *noisy-label train* (NLT) split.
- **Corruption families**: Noise, Blur, Geometric, Weather, Digital, Structural. 14 corruption types span 37 candidate settings; after a low-impact filter, **30 settings are released**.
- **Voter pool**: 4 voters. LeNet-5, MLP, ResNet-20, DeiT3-Small.
- **Headline metrics** (per setting): noise rate δ, VDV (reported as `nth` in the paper). Pre-computed numbers ship with the repo under `results/`.

## Corruptions

| Family     | Types                                          |
| ---------- | ---------------------------------------------- |
| Noise      | shot_noise, impulse_noise, spatter             |
| Blur       | glass_blur, motion_blur                        |
| Geometric  | rotate, shear, translate, scale                |
| Weather    | fog                                            |
| Digital    | brightness                                     |
| Structural | canny_edges, dotted_line, stripe, zigzag       |

Structural corruptions are binary (they either apply or they don't), so they contribute one setting each rather than three severity levels.

We follow Mu & Gilmer (2019), the MNIST-C reference. Their original code lives in [`code/corrupt/canonical/mnist_funcs.py`](../../code/corrupt/canonical/mnist_funcs.py).

## Reproducing the headline numbers

```bash
# Recompute VDV from voter softmaxes (bit-identical to released JSON)
python examples/reproduce_vdv.py --data-root ./ciln-bench-mnist/settings
```
