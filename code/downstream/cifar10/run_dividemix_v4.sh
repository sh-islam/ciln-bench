#!/usr/bin/env bash
# DivideMix v4 runner. 10 CILN settings (8 matched + 2 dispersal) x 2 image
# conditions x 3 seeds = 60 runs total. Same recipe as Co-Teaching v4.
#
# GPU 1 = clean-img variants. GPU 2 = noisy-img variants. ~12-13h per GPU.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
LOG_DIR="$ROOT/logs/v4"
mkdir -p "$LOG_DIR"

GPU="${GPU:-1}"
SEEDS=(0 1 2)
SETTINGS=(
  "frost_sev1"
  "pixelate_sev1"
  "defocus_blur_sev1"
  "snow_sev3"
  "elastic_transform_sev3"
  "gaussian_noise_sev1"
  "contrast_sev5"
  "gaussian_noise_sev3"
  "pixelate_sev5"
  "glass_blur_sev3"
)

if [[ "$GPU" == "1" ]]; then
  VARIANT="clean"
elif [[ "$GPU" == "2" ]]; then
  VARIANT="noisy"
else
  echo "GPU must be 1 or 2 (got $GPU)" >&2; exit 1
fi

run_one() {
  local script="$1"; local cond="$2"; local seed="$3"
  local logf="$LOG_DIR/$(basename "$script" .py)__${cond}__seed${seed}.log"
  echo "[$(date +%H:%M:%S)] GPU$GPU  $script  cond=$cond  seed=$seed  -> $logf"
  python -u "$HERE/$script" --condition "$cond" --seed "$seed" --gpu "$GPU" \
    > "$logf" 2>&1
}

echo "=== GPU$GPU  DivideMix (10 settings x 3 seeds = 30 runs) ==="
for s in "${SETTINGS[@]}"; do
  cond="ours_v2_${VARIANT}__${s}"
  for sd in "${SEEDS[@]}"; do
    run_one "train_dividemix_v4.py" "$cond" "$sd"
  done
done

echo "[$(date +%H:%M:%S)] GPU$GPU  DONE."
