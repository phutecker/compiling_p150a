#!/usr/bin/env bash

set -euo pipefail

INPUT_DIR="${1:-input}"
OUTPUT_DIR="${2:-output}"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

shopt -s nullglob nocaseglob
files=("$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg "$INPUT_DIR"/*.heic)

if [ ${#files[@]} -eq 0 ]; then
  echo "No JPG/JPEG/HEIC files found in $INPUT_DIR"
  exit 0
fi

for src in "${files[@]}"; do
  base="$(basename "$src")"
  stem="${base%.*}"
  dst="$OUTPUT_DIR/$stem.png"
  echo "Converting $src -> $dst"
  if ! ffmpeg -y -i "$src" -frames:v 1 "$dst" >/dev/null 2>&1; then
    echo "Skipping $src (conversion failed)"
  fi
done
