# Fig 4 - tv_bars_4panel (TV to upsampled CIFAR-10H)

## Input
`data/tv_vs_cifar10h_v2.json` -- precomputed per-setting mean TV distance to
upsampled CIFAR-10H, computed using the protocol of Gu et al. 2022 on the
CILN v2 voter pool.

## Run
```
python plot.py
```

Produces `fig04_tv_bars.pdf`. The dashed lines mark PL-IDN's published TV at
low/medium tiers; PL-IDN (high)=0.742 is noted in the caption but omitted from
the y-range so the CILN bars stay readable.
