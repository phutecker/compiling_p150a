# Playground Discrete Video

This folder is a separate sandbox for experimenting with the image editor without touching the original `code/` tree.

What is here:

- `output/`: copied source PNGs with no numeric suffix plus generated still renders
- `frames/`: frame sequences for short videos
- `scripts/discrete_lab.py`: local Python recreation of the C++ renderer's main look
- `scripts/p150a_probe.sh`: sandboxed TT open-device smoke test
- `ttnn_runtime/`: non-invasive runtime shim so the local TT build can import cleanly from this repo

Quick start:

```bash
. /home/ttpatrick/compiling_p150a/.venv_imglab/bin/activate
python /home/ttpatrick/compiling_p150a/playground_discrete_video/scripts/discrete_lab.py --mode all
```

TT open-device probe:

```bash
/home/ttpatrick/compiling_p150a/playground_discrete_video/scripts/p150a_probe.sh
```

Notes:

- The original renderer in `code/` is a C++ OpenCV program.
- This sandbox keeps that code untouched and recreates the same overall style with local Python deps.
- The Tenstorrent `p150a` is visible on this machine, and the probe script can open and close it from Python.
- The renderer itself is still a CPU-side effect pipeline; it is not wired to TT kernels.
