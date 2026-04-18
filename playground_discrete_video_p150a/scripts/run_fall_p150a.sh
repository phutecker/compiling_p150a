#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

. "${REPO_ROOT}/.venv_imglab/bin/activate"

export TT_METAL_HOME="${HOME}/.local/lib/tt-metal"
export PYTHONPATH="${REPO_ROOT}/playground_discrete_video/ttnn_runtime:${HOME}/.local/lib/tt-metal/tools:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${HOME}/.local/lib/tt-metal/.build/gcc/ttnn/Release:${HOME}/.local/lib/tt-metal/.build/gcc/tt_metal/Release:${HOME}/.local/lib/tt-metal/.build/gcc/lib:${HOME}/.local/lib/tt-metal/.build/gcc/tt_metal/third_party/umd/device/Release:${LD_LIBRARY_PATH:-}"

mkdir -p "${ROOT_DIR}/logs"

python "${SCRIPT_DIR}/render_fall_p150a.py" "$@" 2>&1 | tee "${ROOT_DIR}/logs/run_fall_p150a.log"
