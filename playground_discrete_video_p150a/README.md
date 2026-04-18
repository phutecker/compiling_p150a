# p150a Video Sandbox

This sandbox keeps the hardware-backed renderer separate from the original CPU playground.

Layout:

- `src/`: Python renderer implementation
- `scripts/`: shell/python entrypoints only
- `artifacts/videos/`: rendered MP4 outputs
- `artifacts/frames/`: optional PNG frame dumps for inspection
- `logs/`: runtime logs and per-run JSON summaries

External libraries are not vendored into this folder:

- Python environment: `/home/ttpatrick/compiling_p150a/.venv_imglab`
- Tenstorrent runtime: `~/.local/lib/tt-metal`

Default behavior writes MP4 directly. PNG frames are only written when `--save-frames` is passed.
