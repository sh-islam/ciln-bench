# Fig 1 - severity_bars_4x2 (CIFAR-10 VDV)

## Input
`data/v2_all_native.json` -- precomputed FRV/VDV per (corruption, severity) setting,
derived from the CILN voter pool (4 voters: ResNet-20, WRN-28-10, DeiT3-Small, CLIP-B/32).

## Run
```
python plot.py
```

Produces `fig01_severity_bars_vdv.pdf` (compare with expected_output.pdf).
