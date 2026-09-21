# Data for Figure 2: tv_pca_2d

This bundle contains the minimal data needed to regenerate the PCA scatter plot
of per-setting average noisy-label distributions on CIFAR-10.

## What the figure shows

Each point is a single setting -- 45 CILN settings, 3 PL-IDN released datasets
(low / medium / high), plus CIFAR-10H as the human-labelling reference.

The position of each point is the first two principal components of its
10-dimensional mean noisy-label distribution. Points close to the CIFAR-10H
marker are noise distributions that most resemble real human-labelling
ambiguity on CIFAR-10.

## Files

- `mean_distributions.csv` -- 49 rows. Columns: `name`, `super_family`,
  `pipeline`, `p0` ... `p9`. Each row is one setting's mean noisy-label
  distribution: `p_c` = fraction of (image, voter) pairs in that setting whose
  noisy label was class `c`. Each row sums to ~1.0.

- `pca_components.csv` -- 2 rows. The PC1 and PC2 directions in the original
  10-dim basis.

- `pca_explained_variance.csv` -- 2 rows. Fraction of variance captured by each
  component. Top-2 capture ~84%.

- `pca_projections.csv` -- 49 rows. Each setting's (PC1, PC2) coordinate -- the
  raw data of what is scattered in the plot.

- `reproduce.py` -- Standalone script that reads `mean_distributions.csv`,
  re-fits the PCA, and regenerates the scatter plot. No proprietary data
  required.

## How the underlying vectors were computed

- **CILN (45 rows)**: For each (corruption family, severity), our 4-voter pool
  (ResNet-20, WRN-28-10, DeiT3-Small, CLIP ViT-B/32 zero-shot) produces an
  argmax label for each test image. The setting's mean vector is the average
  over the 10,000 corrupted CIFAR-10 test images of the per-image vote-share
  vector.

- **PL-IDN (3 rows)**: Gu et al. 2022's released TFRecord files contain 10
  noisy labels per image (one per rater model). For each level, the mean
  vector is the average over all (image, rater) pairs of the one-hot noisy
  label distribution.

- **CIFAR-10H (1 row)**: Northcutt 2021's human-labelling counts on the
  CIFAR-10 test set. ~50 labels per image. The mean vector is the average
  over all 10,000 images of the normalized per-image vote distribution.

## Reproducing the plot

```bash
pip install numpy scikit-learn matplotlib
python reproduce.py
```

This will write `tv_pca_2d_reproduced.pdf` -- the same scatter plot as the
paper's Figure 2.
