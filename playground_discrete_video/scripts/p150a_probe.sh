#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

. "${ROOT_DIR}/../.venv_imglab/bin/activate"

export TT_METAL_HOME="${HOME}/.local/lib/tt-metal"
export LD_LIBRARY_PATH="${HOME}/.local/lib/tt-metal/.build/gcc/ttnn/Release:${HOME}/.local/lib/tt-metal/.build/gcc/tt_metal/Release:${HOME}/.local/lib/tt-metal/.build/gcc/lib:${HOME}/.local/lib/tt-metal/.build/gcc/tt_metal/third_party/umd/device/Release:${LD_LIBRARY_PATH:-}"

python "${SCRIPT_DIR}/p150a_probe.py"
