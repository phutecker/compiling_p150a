#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="${1:-output}"

mkdir -p "$OUTPUT_DIR"

find "$OUTPUT_DIR" -maxdepth 1 -type f ! -name '.gitkeep' -delete
echo "Cleared generated files from $OUTPUT_DIR"
