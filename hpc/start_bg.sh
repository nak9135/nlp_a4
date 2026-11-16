#!/bin/bash
# Start a run in the background so it keeps going if the browser tab or terminal closes.
#   bash hpc/start_bg.sh part1 all
#   bash hpc/start_bg.sh part2 baseline --learning_rate 3e-4 --experiment_name lr3e-4
# Watch it with:   tail -f logs/<logfile>        (Ctrl+C stops watching, not the run)
# Check if alive:  ps aux | grep -v grep | grep -E "main.py|train_t5.py"
set -euo pipefail
cd "$(dirname "$0")/.."
part="$1"; shift
mkdir -p logs
log="logs/${part}_$(echo "$*" | tr ' /' '__' | tr -cd '[:alnum:]_.-' | cut -c1-80)_$(date +%m%d-%H%M).log"
nohup bash "hpc/run_${part}.sh" "$@" > "$log" 2>&1 &
echo "Started PID $! -> $log"
echo "Follow it with:  tail -f $log"
