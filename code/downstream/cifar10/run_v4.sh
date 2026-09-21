#!/usr/bin/env bash
# V4 downstream runner: ERM + Co-teaching across 8 Table-2 settings,
# 2 image conditions, 3 seeds = 96 runs total.
#
# Splits work across two GPUs (sequential within each GPU). Each GPU gets
# half the (setting x image_source) conditions, all seeds, both methods.
#
# Usage (typical):
#   bash run_v4.sh             # launches GPU1 worker; GPU2 must be started separately
#   GPU=1 bash run_v4.sh       # only the GPU1 half (8 conds x 3 seeds x 2 methods = 24 ERM + 24 Co-T)
#   GPU=2 bash run_v4.sh       # only the GPU2 half
# Or use run_v4_launch.sh below to start both in parallel under screen.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
LOG_DIR="$ROOT/logs/v4"
mkdir -p "$LOG_DIR"

GPU="${GPU:-1}"
PHASE="${PHASE:-all}"   # all | erm | coteach
SEEDS=(0 1 2)

# Conditions assigned to GPU 1 (alphabetic first half) and GPU 2 (second half).
# 16 conditions total, 8 per GPU.
GPU1_CONDS=(
  "ours_v2_clean__contrast_sev5"
  "ours_v2_clean__defocus_blur_sev1"
  "ours_v2_clean__elastic_transform_sev3"
  "ours_v2_clean__frost_sev1"
  "ours_v2_clean__gaussian_noise_sev1"
  "ours_v2_clean__gaussian_noise_sev3"
  "ours_v2_clean__pixelate_sev1"
  "ours_v2_clean__snow_sev3"
)
GPU2_CONDS=(
  "ours_v2_noisy__contrast_sev5"
  "ours_v2_noisy__defocus_blur_sev1"
  "ours_v2_noisy__elastic_transform_sev3"
  "ours_v2_noisy__frost_sev1"
  "ours_v2_noisy__gaussian_noise_sev1"
  "ours_v2_noisy__gaussian_noise_sev3"
  "ours_v2_noisy__pixelate_sev1"
  "ours_v2_noisy__snow_sev3"
)

if [[ "$GPU" == "1" ]]; then
  CONDS=("${GPU1_CONDS[@]}")
elif [[ "$GPU" == "2" ]]; then
  CONDS=("${GPU2_CONDS[@]}")
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

if [[ "$PHASE" == "all" || "$PHASE" == "erm" ]]; then
  echo "=== GPU$GPU  Phase 1: ERM (${#CONDS[@]} conds x ${#SEEDS[@]} seeds = $((${#CONDS[@]} * ${#SEEDS[@]})) runs) ==="
  for cond in "${CONDS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      run_one "train_erm_v4.py" "$cond" "$sd"
    done
  done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "coteach" ]]; then
  echo "=== GPU$GPU  Phase 2: Co-teaching (${#CONDS[@]} conds x ${#SEEDS[@]} seeds = $((${#CONDS[@]} * ${#SEEDS[@]})) runs) ==="
  for cond in "${CONDS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      run_one "train_coteaching_v4.py" "$cond" "$sd"
    done
  done
fi

echo "[$(date +%H:%M:%S)] GPU$GPU  DONE."
