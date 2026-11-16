# Source this in every new terminal:   source hpc/env.sh
# Keeps uv, Hugging Face downloads and caches on fast/large storage.
# NYU HPC: /scratch/$USER (default). Elsewhere (e.g. Kaggle) set A4_SCRATCH first.

SCRATCH_DIR="${A4_SCRATCH:-/scratch/$USER}"
mkdir -p "$SCRATCH_DIR"

export PATH="$HOME/.local/bin:$PATH"           # where the uv installer puts uv
export UV_CACHE_DIR="$SCRATCH_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/.local/uv-python"
export UV_LINK_MODE=copy                       # cache and venvs may sit on different filesystems
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export NLTK_DATA="$SCRATCH_DIR/.cache/nltk_data"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1                      # log lines show up immediately

# repo root = parent of this file's folder
A4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export A4_ROOT
