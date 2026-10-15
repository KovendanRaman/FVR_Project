#!/bin/bash
#SBATCH --job-name=fvr-train
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/kraman/FVR_Project/train-%j.log

# ============================================================================
# Self-chaining FVR training.
#
# Each job trains EPOCHS_PER_JOB epochs (sized to fit under the 3h wall clock),
# then resubmits itself to continue, resuming from checkpoints/<RUN>/last.pth.
# The chain stops when train.py has reached --epochs (TARGET_EPOCHS).
#
# Launch the FIRST job with:
#     sbatch train_chain.sbatch
#
# Everything is resumable: model, optimizer, scheduler, epoch all restore from
# last.pth, so an outage or a killed job costs at most the current chunk.
# ============================================================================

set -uo pipefail

# ---- config (edit these) --------------------------------------------------
RUN="lolblur_p128"                 # name for this training run's checkpoint dir
TARGET_EPOCHS=200                  # total epochs you ultimately want
EPOCHS_PER_JOB=2                   # epochs per 3h job (bump up if kernel makes epochs fast)
PATCH=128
BATCH=8
DEPTH=4
LR=0.0001
BLURRY=~/FVR_Project/data/processed/train/blurry
SHARP=~/FVR_Project/data/processed/train/sharp
# ---------------------------------------------------------------------------

FVR=/home-mscluster/kraman/.conda/envs/fvr
IMG=/home-mscluster/kraman/anaconda.sif
CKPT_DIR=~/FVR_Project/checkpoints/$RUN
LAST=$CKPT_DIR/last.pth
SELF=~/FVR_Project/train_chain.sbatch
mkdir -p "$CKPT_DIR"

cd ~/FVR_Project

# --- figure out where we are: current completed epoch from last.pth (if any) --
CURRENT_EPOCH=$(apptainer exec --nv "$IMG" "$FVR/bin/python" - <<PY 2>/dev/null
import torch, os
p = os.path.expanduser("$LAST")
try:
    print(torch.load(p, map_location="cpu")["epoch"] + 1)
except Exception:
    print(0)
PY
)
CURRENT_EPOCH=${CURRENT_EPOCH:-0}
echo "=== chain step: completed epochs so far = $CURRENT_EPOCH / $TARGET_EPOCHS ==="

if [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCHS" ]; then
    echo "=== target reached ($TARGET_EPOCHS). Chain complete. ==="
    exit 0
fi

# this job will train up to this many total epochs, capped at the target
STOP_AT=$(( CURRENT_EPOCH + EPOCHS_PER_JOB ))
if [ "$STOP_AT" -gt "$TARGET_EPOCHS" ]; then STOP_AT=$TARGET_EPOCHS; fi
echo "=== this job trains epochs $CURRENT_EPOCH -> $STOP_AT ==="

# resume flag only if a checkpoint exists
RESUME_ARG=""
if [ -f "$LAST" ]; then RESUME_ARG="--resume $LAST"; fi

# --- train this chunk ------------------------------------------------------
# train.py's --epochs is the ABSOLUTE target; it will run from the resumed
# epoch up to STOP_AT this job.
srun apptainer exec --nv "$IMG" "$FVR/bin/python" train.py \
    --blurry-dir "$BLURRY" \
    --sharp-dir  "$SHARP" \
    --patch-size $PATCH \
    --batch-size $BATCH \
    --depth $DEPTH \
    --lr $LR \
    --epochs $STOP_AT \
    --out-dir "$CKPT_DIR" \
    $RESUME_ARG
TRAIN_RC=$?

echo "=== train.py exited with code $TRAIN_RC ==="

# --- resubmit successor if not done and training didn't hard-fail ----------
if [ "$TRAIN_RC" -ne 0 ]; then
    echo "!! training exited nonzero; NOT auto-resubmitting (inspect the log)."
    exit $TRAIN_RC
fi

# recompute completed epochs
NOW=$(apptainer exec --nv "$IMG" "$FVR/bin/python" - <<PY 2>/dev/null
import torch, os
try:
    print(torch.load(os.path.expanduser("$LAST"), map_location="cpu")["epoch"] + 1)
except Exception:
    print(0)
PY
)
NOW=${NOW:-0}
if [ "$NOW" -lt "$TARGET_EPOCHS" ]; then
    echo "=== $NOW/$TARGET_EPOCHS done; resubmitting next chain job ==="
    sbatch "$SELF"
else
    echo "=== $NOW/$TARGET_EPOCHS done; TARGET REACHED, chain ends. ==="
fi
