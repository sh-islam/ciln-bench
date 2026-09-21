#!/usr/bin/env bash
# Run ERM (fresh, with logging) -> Co-teaching (with logging) -> DivideMix on a GPU's
# half of the conditions sequentially. Designed to be run inside a screen session per GPU.
#
# Usage: GPU=1 CHUNK=A bash run_all_v3.sh
#        GPU=2 CHUNK=B bash run_all_v3.sh
set -e
cd "$(dirname "$0")"

GPU=${GPU:-1}
CHUNK=${CHUNK:-A}
LOGS_BASE=../logs/all_v3
mkdir -p $LOGS_BASE

if [ "$CHUNK" = "A" ]; then
  CONDS=(
    gu_low
    gu_medium
    gu_high
    ours__elastic_transform_sev3
    ours__brightness_sev5
  )
else
  CONDS=(
    ours__impulse_noise_sev3
    ours__fog_sev3
    ours__motion_blur_sev3
    ours__contrast_sev3
    ours__impulse_noise_sev5
  )
fi
SEEDS=(0 1 2)

echo "[$(date '+%F %H:%M:%S')] CHUNK=$CHUNK GPU=$GPU starting"
echo "  conditions: ${CONDS[@]}"

# Stage 1: ERM
echo "[$(date '+%F %H:%M:%S')] === STAGE 1: ERM ==="
for c in "${CONDS[@]}"; do
  for s in "${SEEDS[@]}"; do
    out=$LOGS_BASE/erm__${c}__seed${s}.log
    echo "[$(date '+%F %H:%M:%S')] erm $c s$s -> $out"
    python3 train_erm_v3.py --condition "$c" --seed "$s" --gpu "$GPU" --epochs 100 > $out 2>&1
  done
done

# Stage 2: Co-teaching
echo "[$(date '+%F %H:%M:%S')] === STAGE 2: Co-teaching ==="
for c in "${CONDS[@]}"; do
  for s in "${SEEDS[@]}"; do
    out=$LOGS_BASE/coteaching__${c}__seed${s}.log
    echo "[$(date '+%F %H:%M:%S')] coteaching $c s$s -> $out"
    python3 train_coteaching_v3.py --condition "$c" --seed "$s" --gpu "$GPU" --epochs 100 > $out 2>&1
  done
done

# Stage 3: DivideMix
echo "[$(date '+%F %H:%M:%S')] === STAGE 3: DivideMix ==="
for c in "${CONDS[@]}"; do
  for s in "${SEEDS[@]}"; do
    out=$LOGS_BASE/dividemix__${c}__seed${s}.log
    echo "[$(date '+%F %H:%M:%S')] dividemix $c s$s -> $out"
    python3 train_dividemix_v3.py --condition "$c" --seed "$s" --gpu "$GPU" --epochs 100 > $out 2>&1
  done
done

echo "[$(date '+%F %H:%M:%S')] CHUNK=$CHUNK GPU=$GPU ALL DONE"
