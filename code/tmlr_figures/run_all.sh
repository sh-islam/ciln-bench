#!/usr/bin/env bash
# Regenerates every figure of the CILN TMLR paper into out/, matching figs/pub exactly.
# Requires: numpy scipy matplotlib pillow pypdf pub-ready-plots==1.5
set -e; cd "$(dirname "$0")"; OUT="${1:-out}"; mkdir -p "$OUT"; D=$(pwd)/data; O=$(pwd)/"$OUT"
(cd figure2_dists_figure7_kde  && python3 fig02_dists.py "$D/softmaxes" "$O")   # Fig 2 dist_row (1x3), Fig 7 kde_row (1x3)
(cd figure3_nth_figure4_tv     && python3 fig_rows.py "$D/v2_all_native.json" "$D/figure_rebuild_data/tv_cifar_all_settings.csv" "$O")  # Fig 3 nth_row, Fig 4 tv_row (1x3)
(cd figure5_pca                && python3 fig05_pca.py "$D/tv_pca_2d_data" "$O")   # Fig 5 (3.08x1.82 canvas, 7pt annotations)
(cd figure6_dm                 && python3 fig06_dm.py "$D/figure_rebuild_data" "$O")  # Fig 6 (3x2, top headers, aspect 0.21)
(cd figure8_9_appendix         && python3 fig08_appendix.py "$D/softmaxes" "$D/softmaxes" "$O")  # Figs 8, 9 (1x3)
(cd figure1_severity_examples  && python3 fig01_examples.py rasters current_pdfs "$O/fig01")  # 12 Fig 1 panels, no pred prefix
echo "All figures written to $OUT/"
