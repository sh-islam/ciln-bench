#!/usr/bin/env bash
# V4 CCP downstream runner: ERM + Co-teaching on the 3 high-noise matched-tier
# settings, both image_source variants, 3 seeds = 36 runs total.
#
# GPU split:
#   GPU=1: clean-img conds (3 settings x 3 seeds x 2 methods = 18 runs)
#   GPU=2: noisy-img conds (3 settings x 3 seeds x 2 methods = 18 runs)
# GPU 0 is left free.
#
# Usage:
#   GPU=1 bash run_v4_ccp.sh
#   GPU=2 bash run_v4_ccp.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
LOG_DIR="$ROOT/logs/v4_ccp"
mkdir -p "$LOG_DIR"

GPU="${GPU:-1}"
PHASE="${PHASE:-all}"
SEEDS=(0 1 2)

GPU1_CONDS=(
  "ours_v2_ccp_clean__contrast_sev5"
  "ours_v2_ccp_clean__gaussian_noise_sev3"
  "ours_v2_ccp_clean__shot_noise_sev5"
)
GPU2_CONDS=(
  "ours_v2_ccp_noisy__contrast_sev5"
  "ours_v2_ccp_noisy__gaussian_noise_sev3"
  "ours_v2_ccp_noisy__shot_noise_sev5"
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
