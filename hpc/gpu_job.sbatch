#!/bin/bash
# Submit from the repo root (in a Jupyter terminal on the burst cluster):
#
#   sbatch hpc/gpu_job.sbatch part1 debug
#   sbatch hpc/gpu_job.sbatch part1 all
#   sbatch hpc/gpu_job.sbatch part2 debug
#   sbatch hpc/gpu_job.sbatch part2 baseline --learning_rate 3e-4 --experiment_name lr3e-4
#
# Defaults: 1 x L4 (g2-standard-12), 5 hours. Override on the command line, e.g. an A100:
#   sbatch --partition=c12m85-a100-1 --time=02:00:00 hpc/gpu_job.sbatch part2 baseline
#
# Watch:   squeue -u $USER            tail -f logs/a4_<jobid>.out
# Cancel:  scancel <jobid>
#
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --job-name=a4
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # sbatch copies this script elsewhere, so don't rely on $0
echo "Job $SLURM_JOB_ID on $(hostname) | partition $SLURM_JOB_PARTITION | args: $*"
nvidia-smi -L || true

part="$1"; shift
bash "hpc/run_${part}.sh" "$@"
