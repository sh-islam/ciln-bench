# Fig 5 - tv_pca_2d_top10 (CILN settings closest to CIFAR-10H)

## Input
`data/pca_input.npz` -- precomputed 10-dim per-setting average vote distribution
for: 45 CILN CIFAR settings, 3 PL-IDN tiers (low/med/high), and CIFAR-10H.
(We precomputed the PL-IDN vectors from Gu et al.'s tfrecord release so you do
not need those large files.)

## Run
```
python plot.py
```

Produces `fig05_tv_pca_top10.pdf`. Needs `adjustText` (see `requirements.txt`).
