#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    script_dir = pathlib.Path(__file__).resolve().parent
    root = script_dir.parent
    sys.path.insert(0, str(root / "ttnn_runtime"))
    sys.path.insert(0, str(pathlib.Path.home() / ".local/lib/tt-metal/tools"))

    import ttnn

    print(f"Imported ttnn from: {getattr(ttnn, '__file__', 'unknown')}")
    num_devices = ttnn.get_num_devices()
    print(f"Reported TT devices: {num_devices}")
    if num_devices < 1:
        print("No TT devices reported.")
        return 1

    device = ttnn.open_device(device_id=0)
    print(f"Opened device: {device}")
    ttnn.close_device(device)
    print("Closed device 0 cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
