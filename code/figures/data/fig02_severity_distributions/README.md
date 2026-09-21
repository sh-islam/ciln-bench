# Fig 2 - severity distributions and sorted-frequency lines

## Input
`data/` contains, for each dataset/corruption/severity:
- `labels.npy` -- ground-truth class labels
- `softmax_<voter>.npy` -- per-voter softmax over classes

The voter pools are listed in `plot.py` (`V_CIFAR`, `V_MNIST`, `V_ADULT`).
For each setting, one voter is sampled uniformly per example with a fixed RNG seed.
The sampled argmax label is used to compute class shares. Legends report Shannon entropy (`H`) of each class-share distribution in bits.

## Run
```bash
python plot.py
```

## Output
The script writes all PDFs to `pdfs/`, with two families per dataset/corruption panel:
- `dist_<dataset>_<corruption>.pdf` -- class-labeled share bar plot for clean, level 1, level 3, and level 5
- `kde_<dataset>_<corruption>.pdf` -- smoothed filled sorted class-frequency line plot for clean, level 1, and level 5, with the largest frequencies on the right
