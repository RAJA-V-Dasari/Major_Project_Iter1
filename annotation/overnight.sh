#!/usr/bin/env bash
#
# Unattended bootstrap loop. Run it before bed; review in the morning.
#
#   ./overnight.sh 6
#
# Each round: rebuild the dataset from whatever labels exist, train,
# score on val, then pseudo-label the whole corpus with the new
# weights. Rounds get better only because a human corrected labels
# between them - the loop cannot improve on its own, and is not meant
# to. What it buys is that all the WAITING happens while you sleep,
# instead of between your review sessions.
#
# Everything stays on this machine: the pages are student answer
# scripts with names, USNs and marks on them.

set -euo pipefail

cd "$(dirname "$0")"

ROUNDS="${1:-4}"

PY=../.venv/bin/python

EPOCHS="${EPOCHS:-150}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-4}"

LOG="overnight.log"

echo "=== overnight run started $(date) ===" | tee -a "$LOG"
echo "rounds=$ROUNDS epochs=$EPOCHS imgsz=$IMGSZ" | tee -a "$LOG"

for round in $(seq 1 "$ROUNDS"); do

  echo "" | tee -a "$LOG"
  echo "--- round $round/$ROUNDS  $(date +%H:%M:%S) ---" | tee -a "$LOG"

  # Labels can change between rounds (that is the point), so the
  # dataset is rebuilt each time rather than reused.
  $PY label_helper.py apply >>"$LOG" 2>&1
  $PY build_dataset.py labels/instances_default.json --symlink >>"$LOG" 2>&1

  $PY train_layout.py \
      --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$BATCH" \
      --name "round${round}" >>"$LOG" 2>&1

  WEIGHTS="runs/round${round}/weights/best.pt"

  if [ ! -f "$WEIGHTS" ]; then
    echo "  ! training produced no weights - stopping" | tee -a "$LOG"
    exit 1
  fi

  echo "  scoring on val" | tee -a "$LOG"
  $PY evaluate.py --weights "$WEIGHTS" --imgsz "$IMGSZ" 2>&1 \
      | grep -E "mAP|^  [a-z_]+ " | tee -a "$LOG"

  # Seed guesses for the next batch of pages to review. This is what
  # makes the morning fast: correcting a guess beats assigning from
  # scratch.
  echo "  seeding next review batch" | tee -a "$LOG"
  $PY label_helper.py seed --weights "$WEIGHTS" --count 20 >>"$LOG" 2>&1 || true

done

echo "" | tee -a "$LOG"
echo "  pseudo-labelling the full corpus with the final weights" | tee -a "$LOG"

$PY pseudo_label.py \
    --weights "runs/round${ROUNDS}/weights/best.pt" \
    --imgsz "$IMGSZ" \
    --human labels/instances_default.json >>"$LOG" 2>&1 || true

echo "" | tee -a "$LOG"
echo "=== finished $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "In the morning:" | tee -a "$LOG"
echo "  1. python3 review_labels.py         # check what was labelled" | tee -a "$LOG"
echo "  2. correct assignments.json          # seeded guesses need review" | tee -a "$LOG"
echo "  3. cat preannotations/pseudo_review.txt   # pages the model doubted" | tee -a "$LOG"
