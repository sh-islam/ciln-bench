The raw inputs for Fig 2 (per-setting `labels.npy` + `softmax_<voter>.npy`, plus the
clean-split softmaxes) are the released voter outputs themselves and are not
duplicated here. Download the HF datasets (see the top-level README) and point
`fig02_dists.py` / `fig08_appendix.py` at a folder laid out as
`<data>/<dataset>/<corruption>/severity_<l>/noisy_label_train/` and
`<data>/clean/<dataset>/noisylabeltrain_clean/`.
