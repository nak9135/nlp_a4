#!/bin/bash
# Part 2 (T5 text-to-SQL) runs. Usage (from the repo root, inside a GPU session):
#
#   sbatch hpc/gpu_job.sbatch part2 debug                # tiny end-to-end check, ~2 min
#   sbatch hpc/gpu_job.sbatch part2 baseline             # fine-tune with the defaults in train_t5.py
#   sbatch hpc/gpu_job.sbatch part2 baseline --learning_rate 3e-4 --experiment_name lr3e-4
#
# Any extra flags after the stage name are passed straight to train_t5.py.
# Outputs: results/t5_ft_<name>_{dev,test}.sql, records/t5_ft_<name>_{dev,test}.pkl,
#          checkpoints/ft_experiments/<name>/{best,last}/
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$A4_ROOT/part2"
PY=.venv/bin/python
stage="${1:-}"; shift || true

run() { echo; echo "===== $(date '+%F %T') :: python train_t5.py $* ====="; $PY train_t5.py "$@"; }

case "$stage" in
    debug)    run --finetune --debug --max_n_epochs 1 --experiment_name debug "$@" ;;
    baseline) run --finetune --experiment_name experiment "$@" ;;
    scratch)
        run --experiment_name experiment_ec --learning_rate 1e-3 --max_n_epochs 60 --patience_epochs 10 "$@"
        # The PDF asks for t5_ft_experiment_ec_test.{sql,pkl} even though this model is trained from
        # scratch (the code names it t5_scr_...), so copy to the required submission names.
        cp results/t5_scr_experiment_ec_test.sql results/t5_ft_experiment_ec_test.sql
        cp records/t5_scr_experiment_ec_test.pkl records/t5_ft_experiment_ec_test.pkl
        echo "Copied extra-credit outputs to results/t5_ft_experiment_ec_test.sql and records/t5_ft_experiment_ec_test.pkl"
        ;;
    *) echo "usage: $0 {debug|baseline|scratch} [extra train_t5.py flags]"; exit 1 ;;
esac
