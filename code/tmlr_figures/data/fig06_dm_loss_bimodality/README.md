# Fig 6 - dm_loss_bimodality (per-sample loss distributions)

## Input
`data/ours_v2_noisy__<setting>__seed{0,1,2}__persample.npz` for three CILN settings:
frost_sev1, contrast_sev5, glass_blur_sev3. Each NPZ contains the per-sample loss
trajectories from an ERM training run logged every epoch, plus an `is_noisy` mask.

## Run
```
python plot.py
```

Produces `fig06_dm_loss_bimodality.pdf` (6 panels: 3 settings x 2 phases).
