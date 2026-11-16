#!/bin/bash
# Run Part 1 / Part 2 stages on a Kaggle GPU notebook.
#   bash run_kaggle.sh part1 debug
#   bash run_kaggle.sh part1 all
#   bash run_kaggle.sh part2 baseline --learning_rate 3e-4 --experiment_name lr3e-4
#
# Kaggle layout:
#   /kaggle/input/<dataset>/...  read-only copy of this repo (uploaded as a zip dataset)
#   /tmp/nlp_a4                  working copy + uv venvs + caches (fast, not saved)
#   /kaggle/working/a4_outputs   results we keep (saved in the notebook's Output tab)
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${A4_WORK:-/tmp/nlp_a4}"
KEEP="${A4_KEEP:-/kaggle/working/a4_outputs}"
export A4_SCRATCH="${A4_SCRATCH:-/tmp/a4_cache}"

part="$1"; shift
stage="$1"

# 1. Writable working copy (once per session)
if [ ! -d "$WORK" ]; then
    echo ">> Copying code to $WORK"
    cp -r "$SRC" "$WORK"
    rm -rf "$WORK"/part*/.venv
fi
cd "$WORK"

# 2. uv + this part's venv (once per session, ~3-5 min)
command -v uv >/dev/null 2>&1 || pip install -q uv
if [ ! -x "$part/.venv/bin/python" ]; then
    bash hpc/setup_envs.sh "$part"
fi

# 3. Run the stage, logging to the kept output folder too
mkdir -p "$KEEP/logs"
log="$KEEP/logs/${part}_$(echo "$*" | tr ' /' '__' | tr -cd '[:alnum:]_.-' | cut -c1-80)_$(date +%m%d-%H%M).log"
echo ">> Running hpc/run_${part}.sh $*   (log: $log)"
set +e
bash "hpc/run_${part}.sh" "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e

# 4. Keep the small, important outputs (also on failure, so partial results survive)
echo ">> Copying outputs to $KEEP"
if [ "$part" = part1 ]; then
    mkdir -p "$KEEP/part1"
    cp -v part1/out_*.txt "$KEEP/part1/" 2>/dev/null || true
else
    mkdir -p "$KEEP/part2/results" "$KEEP/part2/records" "$KEEP/part2/checkpoints"
    cp -v part2/results/t5_*.sql "$KEEP/part2/results/" 2>/dev/null || true
    find part2/records -name 't5_*.pkl' -exec cp -v {} "$KEEP/part2/records/" \; 2>/dev/null || true
    # best checkpoint of each experiment (~240 MB each); skip the tiny debug run
    for d in part2/checkpoints/*_experiments/*/best; do
        [ -d "$d" ] || continue
        name="$(basename "$(dirname "$d")")"; kind="$(basename "$(dirname "$(dirname "$d")")")"
        [ "$name" = debug ] && continue
        mkdir -p "$KEEP/part2/checkpoints/$kind/$name"
        cp -r "$d" "$KEEP/part2/checkpoints/$kind/$name/"
        echo "kept checkpoint $kind/$name/best"
    done
fi
exit $status
