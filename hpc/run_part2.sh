#!/bin/bash
# Part 2 (T5 text-to-SQL) runs. Usage (from the repo root, inside a GPU session):
#
#   sbatch hpc/gpu_job.sbatch part2 debug                # tiny end-to-end check, ~2 min
#   sbatch hpc/gpu_job.sbatch part2 baseline             # fine-tune with the defaults in train_t5.py
#   sbatch hpc/gpu_job.sbatch part2 baseline --learning_rate 3e-4 --experiment_name lr3e-4
#   sbatch hpc/gpu_job.sbatch part2 scratch                # extra credit: T5-small from scratch
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
        # Extra credit: same t5-small architecture, random init, word-level SQL tokens.
        # Later flags override these defaults (argparse keeps the last value), e.g.
        #   scratch --learning_rate 1e-4 --experiment_name ec_lr1e-4
        name=experiment_ec
        prev=""
        for a in "$@"; do [ "$prev" = "--experiment_name" ] && name="$a"; prev="$a"; done
        # lr 1e-3 / 5e-4 without clipping diverged after warm-up (first Kaggle attempt), hence
        # gradient clipping and a lower peak learning rate.
        run --sql_tokenizer word --no_prefix --learning_rate 3e-4 --grad_clip 1.0 --max_n_epochs 80 \
            --num_warmup_epochs 2 --eval_every 5 --patience_epochs 4 --max_gen_length 256 \
            --experiment_name experiment_ec "$@"
        # The PDF asks for t5_ft_experiment_ec_test.{sql,pkl} even though this model is trained from
        # scratch (the code names it t5_scr_<name>_...), so also save a copy under the required names.
        cp "results/t5_scr_${name}_test.sql" "results/t5_ft_experiment_ec_test__from_${name}.sql"
        cp "records/t5_scr_${name}_test.pkl" "records/t5_ft_experiment_ec_test__from_${name}.pkl"
        echo "Extra-credit outputs copied to results/records t5_ft_experiment_ec_test__from_${name}.*"
        echo "(rename the chosen run's files to t5_ft_experiment_ec_test.sql / .pkl before submitting)"
        ;;
    *) echo "usage: $0 {debug|baseline|scratch} [extra train_t5.py flags]"; exit 1 ;;
esac
