#!/bin/bash
# Stage 1: corrupt CIFAR-10 test set for all 45 settings.
# Writes to output_seed0/cifar10/<corruption>/severity_<S>/test/{images.npy,labels.npy,manifest.json}
set -e
DIR=/path/to/ciln-workspace/journal_edition/test_inference
LOG=$DIR/logs/stage1_corrupt.log

cd /path/to/ciln-workspace/journal_edition/corruptions
source /path/to/ciln-workspace/venv/bin/activate

echo "[$(date)] Stage 1 start: corrupt CIFAR-10 test (45 settings)" | tee -a "$LOG"
python -u produce_corrupted.py --datasets cifar10 --splits test --resume 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[$(date)] Stage 1 end rc=$RC" | tee -a "$LOG"
exit $RC
