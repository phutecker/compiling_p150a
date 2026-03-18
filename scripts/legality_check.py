#!/usr/bin/env python3

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    print("Checking local TT-Metalium API legality for batch2 variant...")

    api_root = pathlib.Path(
        "/home/ttpatrick/.local/share/containers/storage/overlay/"
        "9506a74f929155ed2bd932965de53bcc35a496e042b2fc590abfb5ae126947e3/"
        "diff/root/.cache/uv/archive-v0/ic5dQ7xXBxV5fehfYv7kr/ttnn/tt_metal/hw/inc/api/compute"
    )

    required = {
        "cb_api.h": ["cb_wait_front", "cb_pop_front", "cb_push_back", "cb_reserve_back"],
        "tile_move_copy.h": ["copy_tile"],
        "pack.h": ["pack_tile"],
        "eltwise_unary/exp.h": ["exp_tile", "exp_tile_init"],
    }

    ok = True
    for relative_path, symbols in required.items():
        full_path = api_root / relative_path
        print(f"Inspecting {full_path} ...")
        if not full_path.exists():
            print(f"ERROR: required header not found: {full_path}")
            ok = False
            continue
        text = read_text(full_path)
        for symbol in symbols:
            if symbol not in text:
                print(f"ERROR: symbol '{symbol}' not found in {full_path}")
                ok = False
            else:
                print(f"OK: found '{symbol}'")

    print("Checking local batch2 kernel source...")
    batch2_kernel = ROOT / "batch2/kernels/compute/eltwise_sfpu.cpp"
    text = read_text(batch2_kernel)
    for expected in [
        "cb_wait_front(tt::CBIndex::c_0, 2)",
        "copy_tile(tt::CBIndex::c_0, 0, 0)",
        "copy_tile(tt::CBIndex::c_0, 1, 1)",
        "ckernel::exp_tile(0)",
        "ckernel::exp_tile(1)",
        "pack_tile(0, tt::CBIndex::c_2)",
        "pack_tile(1, tt::CBIndex::c_2)",
        "cb_pop_front(tt::CBIndex::c_0, 2)",
    ]:
        if expected not in text:
            print(f"ERROR: batch2 kernel is missing expected operation: {expected}")
            ok = False
        else:
            print(f"OK: batch2 kernel contains: {expected}")

    if ok:
        print("PASS: legality check completed successfully.")
        return 0

    print("FAIL: legality check found issues.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
