# compiling_p150a

## Current status

We successfully moved off the packaged/container tt-metalium snapshot path and built the custom compare runner against the official local tt-metal source build in:

- `~/.local/lib/tt-metal`

The custom runner was integrated as an official local programming example and built successfully as:

- `~/.local/lib/tt-metal/.build/gcc/programming_examples/Debug/metal_example_compare_eltwise_sfpu`

The p150a is healthy and visible:
- `tt-smi -ls` works
- `/proc/driver/tenstorrent/0/pids` is empty when idle
- device open and kernel execution both work

## Important paths

Official tt-metal checkout:
- `~/.local/lib/tt-metal`

Experiment repo:
- `~/compiling_p150a`

Local variants:
- `baseline/kernels/compute/eltwise_sfpu.cpp`
- `baseline/kernels/dataflow/reader_unary.cpp`
- `baseline/kernels/dataflow/writer_unary.cpp`
- `batch2/kernels/compute/eltwise_sfpu.cpp`
- `batch2/kernels/dataflow/reader_unary.cpp`
- `batch2/kernels/dataflow/writer_unary.cpp`

Custom compare runner source:
- `compare/compare_eltwise_sfpu.cpp`

Integrated official-build example source:
- `~/.local/lib/tt-metal/tt_metal/programming_examples/compare_eltwise_sfpu/compare_eltwise_sfpu.cpp`

Integrated official-build example CMake:
- `~/.local/lib/tt-metal/tt_metal/programming_examples/compare_eltwise_sfpu/CMakeLists.txt`

Programming examples registry modified here:
- `~/.local/lib/tt-metal/tt_metal/programming_examples/CMakeLists.txt`

## Build command that worked

From `~/.local/lib/tt-metal`:

```bash
export TT_METAL_HOME="$HOME/.local/lib/tt-metal"

~/.local/opt/cmake-3.31.6-linux-x86_64/bin/cmake \
  -S . -B ./.build/gcc \
  --preset gcc \
  -DENABLE_DISTRIBUTED=OFF

~/.local/opt/cmake-3.31.6-linux-x86_64/bin/cmake \
  --build ./.build/gcc --target metal_example_compare_eltwise_sfpu -jEOF
