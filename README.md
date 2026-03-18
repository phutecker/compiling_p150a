# Tiny TT-Metalium `eltwise_sfpu` Lab

This repo is a very small, auditable experiment scaffold based on the official TT-Metalium `eltwise_sfpu` example kernel.

It contains two variants:

- `baseline`: the original single-tile-at-a-time compute kernel behavior
- `batch2`: a conservative two-tiles-per-iteration variant

## Important Status

The P150A is present on the host and visible outside the Codex sandbox.

Observed during setup:

- host `tt-smi -ls` sees a `p150a` board
- host `lspci` sees Tenstorrent PCI device `1e52:b140`
- an earlier in-sandbox run falsely reported no device because the sandbox could not see the host device nodes

The current blocker is different:

- the local unpacked TT-Metal package contains the official example kernels and `libtt_metal.so`
- the repaired `tt-metalium` wrapper can now launch the packaged runtime non-interactively
- but the packaged install does **not** expose the public standalone host headers that a tiny custom TT-Metal host runner would normally need
- under `tt_metal/api/tt-metalium/`, only a few constants/buffer headers were present; key public host entry points such as `host_api.hpp`, `device.hpp`, `program.hpp`, and `tt_metal.hpp` were not found
- no packaged example host-side sources or exported CMake/pkg-config surface for a standalone custom-kernel build were found either

Because of that blocker, this repo records the experiment **honestly**:

- the official example kernel files were copied
- the `batch2` kernel was implemented conservatively using only APIs verified in the local headers
- the runner is noisy and explicit
- the runner verifies `tt-smi` and a `tt-metalium` `open_device` smoke test
- results are written even when execution is blocked
- no fake benchmark numbers are produced

There is also a separate runtime blocker right now:

- `tt-smi` can see the P150A, so this is not a simple "board missing" case
- the saved `tt-metalium` smoke log fails much earlier, while trying to allocate a TLB window during initial device interaction
- host diagnostics showed 1G hugepages were available (`free_hugepages: 3`, `nr_hugepages: 4`), so this does **not** look like simple hugepage exhaustion
- host diagnostics also showed a still-running `tt-metalium` Docker container holding `/dev/tenstorrent` and `/dev/hugepages-1G`, which makes a stale container/runtime mapping issue the most likely immediate cause

So the two blockers are distinct:

- no supported standalone host build surface for the copied custom kernels
- current device-open path appears to be contaminated by a stale `tt-metalium` container/runtime state

## Copied Files

The minimum example source files copied from the local TT-Metal checkout snapshot were:

- `ttnn/ttnn/cpp/ttnn/operations/examples/example/device/kernels/compute/eltwise_sfpu.cpp`
- `ttnn/ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/reader_unary.cpp`
- `ttnn/ttnn/cpp/ttnn/operations/examples/example/device/kernels/dataflow/writer_unary.cpp`

The local source snapshot used during setup was found under:

- `/home/ttpatrick/.local/share/containers/storage/overlay/9506a74f929155ed2bd932965de53bcc35a496e042b2fc590abfb5ae126947e3/diff/root/.cache/uv/archive-v0/ic5dQ7xXBxV5fehfYv7kr/ttnn`

## What `baseline` Means

`baseline` is the original official compute kernel structure:

- wait for 1 tile
- copy 1 tile to DST slot 0
- run SFPU op
- pack 1 tile
- pop 1 tile

## What `batch2` Means

`batch2` tries to process 2 tiles per loop iteration in a legal way.

What changed:

- wait for 2 input tiles
- reserve 2 output tiles
- copy input tile 0 to DST slot 0
- copy input tile 1 to DST slot 1
- run `exp_tile` on DST slots 0 and 1
- pack DST slots 0 and 1
- pop 2 input tiles
- push 2 output tiles

## Batch2 Legality Caveat

I did **not** invent new TT-Metalium APIs.

The `batch2` variant was based on local headers showing that the following are legal:

- `cb_wait_front(cb, 2)`
- `cb_reserve_back(cb, 2)`
- `copy_tile(in_cb, in_index, dst_index)` with explicit tile indices
- `pack_tile(dst_index, out_cb)`
- `cb_pop_front(cb, 2)`
- `cb_push_back(cb, 2)`
- `exp_tile(idst)` on an explicit DST tile index

This makes a real 2-tile-per-iteration compute-kernel variant plausible and consistent with the local API docs.

However, I could **not** execute it end-to-end from this repo because the packaged TT-Metalium install still does not expose a supported standalone host-side build surface for a tiny custom runner.

## How To Run

Run:

```bash
./scripts/run_all.sh
```

The script will:

- locate the local TT-Metal example snapshot
- print the copied files
- run legality checks
- check device visibility
- capture host runtime diagnostics to `results/logs/runtime_diagnostics.log`
- refuse to launch a fresh `tt-metalium` runtime if another `tt-metalium` container is already running and may still hold TLB mappings
- run a `tt-metalium` `open_device` smoke test
- check whether a supported standalone host build surface is actually present
- write logs to `results/logs/`
- write a summary CSV to `results/results.csv`

## Results

Results are stored in:

- `results/results.csv`
- `results/logs/`

When the environment is blocked, the CSV will contain explicit failure rows instead of fake benchmark data.

Current expected outcome on this machine:

- `runtime_busy` if a stale `tt-metalium` container is already running
- otherwise `ttnn_open_failed` if the runtime still cannot open the device
- otherwise `packaged_runtime_only` because no supported standalone custom-host build path was found

Useful manual checks:

```bash
tt-smi -ls
docker ps --filter ancestor=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc
tt-metalium -lc 'python3 /home/user/compiling_p150a/scripts/ttnn_smoke.py --mode open_device'
tt-metalium -lc 'python3 /home/user/compiling_p150a/scripts/ttnn_smoke.py --mode exp_probe'
```

Conservative manual cleanup, only if you know no real workload should still be running:

```bash
docker rm -f <stale_tt_metalium_container_id>
```

If cleanup does not resolve the issue, the next step should be a Tenstorrent-supported runtime reset workflow, not ad hoc local changes.
