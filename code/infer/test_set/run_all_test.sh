#!/bin/bash
# Test-set pipeline:
#   Stage 1 (CPU): corrupt CIFAR-10 test for all 45 settings.
#   Stage 2a (GPU 1): v2-pool inference on corrupted test (4 voters).
#   Stage 2b (GPU 2): Gu-arch-pool inference on corrupted test (7 voters).
#
# Stages 2a and 2b run in PARALLEL (separate GPUs, won't compete).
# Use sentinels for resumability.
set -e
DIR=/path/to/ciln-workspace/journal_edition/test_inference
LOG=$DIR/logs/all.log
SENTDIR=$DIR/sentinels
mkdir -p "$SENTDIR" "$DIR/logs"

# ---- Stage 1: corrupt test (CPU, ~30 min) ----
if [ -f "$SENTDIR/stage1.done" ]; then
    echo "[$(date)] SKIP Stage 1 (sentinel exists)" | tee -a "$LOG"
else
    echo "[$(date)] START Stage 1: corrupt CIFAR-10 test" | tee -a "$LOG"
    bash "$DIR/stage1_corrupt_test.sh" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    echo "[$(date)] END Stage 1 rc=$rc" | tee -a "$LOG"
    if [ "$rc" -ne 0 ]; then exit $rc; fi
    touch "$SENTDIR/stage1.done"
fi

# ---- Stage 2a + 2b: inference in parallel on GPU 1 and GPU 2 ----
cd /path/to/ciln-workspace/journal_edition
source /path/to/ciln-workspace/venv/bin/activate

if [ -f "$SENTDIR/stage2a.done" ]; then
    echo "[$(date)] SKIP Stage 2a v2 (sentinel exists)" | tee -a "$LOG"
else
    echo "[$(date)] START Stage 2a v2 on GPU 1" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 python -u "$DIR/stage2_v2_test_infer.py" 2>&1 \
        > "$DIR/logs/stage2a_v2.log" &
    V2_PID=$!
    echo "  v2 PID=$V2_PID" | tee -a "$LOG"
fi

if [ -f "$SENTDIR/stage2b.done" ]; then
    echo "[$(date)] SKIP Stage 2b gu_arch (sentinel exists)" | tee -a "$LOG"
else
    echo "[$(date)] START Stage 2b gu_arch on GPU 2" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=2 python -u "$DIR/stage2_gu_arch_test_infer.py" 2>&1 \
        > "$DIR/logs/stage2b_gu_arch.log" &
    GU_PID=$!
    echo "  gu_arch PID=$GU_PID" | tee -a "$LOG"
fi

# Wait for both
RC2A=0; RC2B=0
if [ -n "$V2_PID" ]; then
    wait $V2_PID; RC2A=$?
    echo "[$(date)] v2 finished rc=$RC2A" | tee -a "$LOG"
    [ $RC2A -eq 0 ] && touch "$SENTDIR/stage2a.done"
fi
if [ -n "$GU_PID" ]; then
    wait $GU_PID; RC2B=$?
    echo "[$(date)] gu_arch finished rc=$RC2B" | tee -a "$LOG"
    [ $RC2B -eq 0 ] && touch "$SENTDIR/stage2b.done"
fi

if [ $RC2A -ne 0 ] || [ $RC2B -ne 0 ]; then
    echo "[$(date)] FAILED -- 2a rc=$RC2A 2b rc=$RC2B" | tee -a "$LOG"
    exit 1
fi

touch "$SENTDIR/all.done"
echo "[$(date)] ============ TEST PIPELINE DONE ============" | tee -a "$LOG"
