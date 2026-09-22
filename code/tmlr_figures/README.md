# CILN TMLR figures — reproduction bundle v3 (current as of 2026-09-22)

`bash run_all.sh` regenerates every figure PDF of the paper into `out/`,
pixel-identical to the paper repo's `figs/pub/` at bundle time.

One folder per figure, each with its script(s) plus `plot_style_ciln.py`,
the shared style module. All fonts, rc params, and widths come from
wiseodd's `pub-ready-plots` (pin `pub-ready-plots==1.5`,
`prp.get_mpl_rcParams(layout=prp.Layout.TMLR, ...)` in
`plot_style_ciln.py` and `fig05_pca.py`).

Layout standards encoded in the scripts:
- Figs 2, 3, 4, 7, 8, 9: one row of three panels
  (`make_grid2(3, ncols=3, width_frac=0.88)`), text printed at 1.16x
  each PDF's trimmed width in the paper.
- Fig 5: 3.08x1.82in canvas, all annotations at the 7pt standard.
- Fig 6: 3 settings x (warmup, final), headers at top, aspect 0.21.
- Fig 1: recomposed from `rasters/` with labels extracted from
  `current_pdfs/` (values only, no `pred:` prefix), fitted single size.

Requirements: python3 with numpy scipy matplotlib pillow pypdf
pub-ready-plots==1.5
