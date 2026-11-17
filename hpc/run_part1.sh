#!/bin/bash
# Part 1 runs. Usage (from the repo root, inside a GPU session):
#
#   sbatch hpc/gpu_job.sbatch part1 debug   # Q1 sanity check, ~8 min, want > 88%
#   sbatch hpc/gpu_job.sbatch part1 all     # everything below (~2-3 h on an L4)
# (or, inside a GPU Code Server session: bash hpc/run_part1.sh debug / bash hpc/start_bg.sh part1 all)
#
# Stages (each can also be run on its own):
#   full       Q1  train on full IMDB + eval            -> out_original.txt        (want > 91%)
#   transform  Q2  eval the Q1 model on transformed test -> out_transformed.txt     (want drop > 4 pts)
#   augment    Q3  train on augmented data + eval transformed -> out_augmented_transformed.txt
#   augeval    Q3  eval the augmented model on original test  -> out_augmented_original.txt
#   all        full, transform, augment, augeval in order
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$A4_ROOT/part1"
PY=.venv/bin/python
stage="${1:-}"
shift || true   # anything after the stage name is passed on to main.py

run() { echo; echo "===== $(date '+%F %T') :: python main.py $* ====="; $PY main.py "$@"; }

case "$stage" in
    debug)     run --train --eval --debug_train ;;
    full)      run --train --eval ;;
    transform) run --eval_transformed ;;
    augment)   run --train_augmented --eval_transformed "$@" ;;
    augeval)   run --eval --model_dir out_augmented ;;
    all)
        run --train --eval
        run --eval_transformed
        run --train_augmented --eval_transformed
        run --eval --model_dir out_augmented
        ;;
    *) echo "usage: $0 {debug|full|transform|augment|augeval|all}"; exit 1 ;;
esac

echo; echo "===== $(date '+%F %T') :: done. Output files:"; ls -l out_*.txt 2>/dev/null || true
