#!/usr/bin/env bash
# Regenerate paper_artifacts/ from the frozen protocol. CPU only.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
python3 -m scripts.reproduce_paper "$@"
