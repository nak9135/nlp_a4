#!/bin/bash
# One-time setup:   bash hpc/setup_envs.sh            (both parts)
#                   bash hpc/setup_envs.sh part1      (just one part)
# Installs uv (if missing) and builds part1/.venv and/or part2/.venv with Python 3.11.
# Runs fine on a CPU-only session (e.g. Jupyter on the `interactive` partition): installing
# packages needs no GPU, and the GPU jobs later reuse these venvs from the shared filesystem.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$A4_ROOT/logs"     # Slurm writes job logs here
parts="${*:-part1 part2}"

if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

for part in $parts; do
    echo ">> Building $part/.venv"
    cd "$A4_ROOT/$part"
    uv venv --python 3.11 .venv
    # Linux: the full requirements file works here (bitsandbytes included)
    uv pip install --python .venv/bin/python -r requirements.txt
done

if [[ " $parts " == *" part1 "* ]]; then
    echo ">> Downloading NLTK data for part 1"
    "$A4_ROOT/part1/.venv/bin/python" -c "import nltk, os; d=os.environ['NLTK_DATA']; nltk.download('wordnet', download_dir=d); nltk.download('punkt', download_dir=d)"
fi

echo ">> Checking imports (CUDA will say False on a CPU session; that's expected)"
for part in $parts; do
    "$A4_ROOT/$part/.venv/bin/python" -c "import torch, transformers; print('$part torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
done
echo ">> Done."
