// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

namespace fs = std::filesystem;
using namespace tt;
using namespace tt::tt_metal;

namespace {

struct Options {
    fs::path repo_root;
    std::string variant = "both";
    uint32_t n_tiles = 64;
    uint32_t repeats = 5;
    uint32_t seed = 7;
};

struct VariantResult {
    std::string name;
    std::vector<double> runtimes_ms;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
};

fs::path executable_dir() {
    std::vector<char> buffer(4096, '\0');
    const auto len = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (len <= 0) {
        return fs::current_path();
    }
    buffer.at(static_cast<size_t>(len)) = '\0';
    return fs::path(buffer.data()).parent_path();
}

bool looks_like_repo_root(const fs::path& path) {
    return fs::exists(path / "baseline/kernels/compute/eltwise_sfpu.cpp") &&
           fs::exists(path / "batch2/kernels/compute/eltwise_sfpu.cpp");
}

std::optional<fs::path> find_repo_root_from(const fs::path& start) {
    if (start.empty()) {
        return std::nullopt;
    }

    fs::path current = fs::weakly_canonical(start);
    if (fs::is_regular_file(current)) {
        current = current.parent_path();
    }

    while (!current.empty()) {
        if (looks_like_repo_root(current)) {
            return current;
        }
        if (current == current.root_path()) {
            break;
        }
        current = current.parent_path();
    }

    return std::nullopt;
}

Options parse_args(int argc, char** argv) {
    Options opts;

    if (auto repo = find_repo_root_from(fs::current_path())) {
        opts.repo_root = *repo;
    } else if (auto repo = find_repo_root_from(executable_dir())) {
        opts.repo_root = *repo;
    } else {
        opts.repo_root = "/home/ttpatrick/compiling_p150a";
    }

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(name));
            }
            return argv[++i];
        };

        if (arg == "--repo-root") {
            opts.repo_root = require_value(arg);
        } else if (arg == "--variant") {
            opts.variant = require_value(arg);
        } else if (arg == "--n-tiles") {
            opts.n_tiles = static_cast<uint32_t>(std::stoul(require_value(arg)));
        } else if (arg == "--repeats") {
            opts.repeats = static_cast<uint32_t>(std::stoul(require_value(arg)));
        } else if (arg == "--seed") {
            opts.seed = static_cast<uint32_t>(std::stoul(require_value(arg)));
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }

    if (!looks_like_repo_root(opts.repo_root)) {
        throw std::runtime_error("repo root does not contain baseline/ and batch2/ kernels: " + opts.repo_root.string());
    }
    if (opts.variant != "baseline" && opts.variant != "batch2" && opts.variant != "both") {
        throw std::runtime_error("--variant must be one of: baseline, batch2, both");
    }
    if (opts.n_tiles == 0 || (opts.n_tiles % 2) != 0) {
        throw std::runtime_error("--n-tiles must be a positive even number so both variants process identical work");
    }
    if (opts.repeats == 0) {
        throw std::runtime_error("--repeats must be at least 1");
    }

    return opts;
}

std::vector<bfloat16> make_input(uint32_t n_tiles, uint32_t seed) {
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<bfloat16> data(n_tiles * elements_per_tile);
    for (auto& value : data) {
        value = bfloat16(dist(rng));
    }
    return data;
}

std::pair<float, float> compare_against_exp(std::span<const bfloat16> input, std::span<const bfloat16> output) {
    if (input.size() != output.size()) {
        throw std::runtime_error("input/output size mismatch during validation");
    }

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        const float expected = static_cast<float>(bfloat16(std::exp(static_cast<float>(input[i]))));
        const float got = static_cast<float>(output[i]);
        const float abs_err = std::abs(expected - got);
        const float rel_err = abs_err / std::max(std::abs(expected), 1.0e-8f);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
    return {max_abs_err, max_rel_err};
}

VariantResult run_variant(
    distributed::MeshDevice& mesh_device,
    distributed::MeshCommandQueue& cq,
    const fs::path& repo_root,
    const std::string& variant_name,
    std::span<const bfloat16> input,
    uint32_t n_tiles,
    uint32_t repeats) {
    const fs::path variant_root = repo_root / variant_name;
    const fs::path reader_path = variant_root / "kernels/dataflow/reader_unary.cpp";
    const fs::path writer_path = variant_root / "kernels/dataflow/writer_unary.cpp";
    const fs::path compute_path = variant_root / "kernels/compute/eltwise_sfpu.cpp";

    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;
    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t output_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t block_dim = 2;
    const uint32_t block_cnt = n_tiles / block_dim;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = tile_size_bytes * n_tiles,
    };

    VariantResult result{.name = variant_name};
    std::vector<bfloat16> latest_output;

    for (uint32_t run = 0; run < repeats + 1; ++run) {
        auto src_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, &mesh_device);
        auto dst_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, &mesh_device);

        Program program = CreateProgram();

        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(block_dim * tile_size_bytes, {{input_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(input_cb_index, tile_size_bytes));
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(block_dim * tile_size_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, tile_size_bytes));

        auto reader_kernel = CreateKernel(
            program,
            reader_path.string(),
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
            });

        auto writer_kernel = CreateKernel(
            program,
            writer_path.string(),
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            });

        std::map<std::string, std::string> compute_defines;
        if (variant_name == "baseline") {
            compute_defines = {
                {"SFPU_OP_EXP_INCLUDE", "1"},
                {"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"},
            };
        }

        CreateKernel(
            program,
            compute_path.string(),
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = false,
                .compile_args = {block_cnt, block_dim},
                .defines = compute_defines,
            });

        distributed::EnqueueWriteMeshBuffer(cq, src_buffer, std::vector<bfloat16>(input.begin(), input.end()), true);
        SetRuntimeArgs(program, reader_kernel, core, {src_buffer->address(), 0, 0, n_tiles});
        SetRuntimeArgs(program, writer_kernel, core, {dst_buffer->address(), 0, 0, n_tiles});

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));

        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        const auto t1 = std::chrono::steady_clock::now();

        latest_output.clear();
        distributed::EnqueueReadMeshBuffer(cq, latest_output, dst_buffer, true);

        const auto [max_abs_err, max_rel_err] = compare_against_exp(input, latest_output);
        result.max_abs_err = std::max(result.max_abs_err, max_abs_err);
        result.max_rel_err = std::max(result.max_rel_err, max_rel_err);

        if (run > 0) {
            result.runtimes_ms.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    return result;
}

double average_ms(std::span<const double> samples) {
    return std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
}

double min_ms(std::span<const double> samples) { return *std::min_element(samples.begin(), samples.end()); }

void print_result(const VariantResult& result) {
    std::cout << "variant=" << result.name
              << " repeats=" << result.runtimes_ms.size()
              << " avg_ms=" << average_ms(result.runtimes_ms)
              << " min_ms=" << min_ms(result.runtimes_ms)
              << " max_abs_err=" << result.max_abs_err
              << " max_rel_err=" << result.max_rel_err << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        std::cout << "repo_root=" << opts.repo_root << '\n';
        std::cout << "n_tiles=" << opts.n_tiles << " repeats=" << opts.repeats << " seed=" << opts.seed << '\n';

        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh_device->mesh_command_queue();
        const auto input = make_input(opts.n_tiles, opts.seed);

        std::vector<VariantResult> results;
        if (opts.variant == "baseline" || opts.variant == "both") {
            results.push_back(run_variant(*mesh_device, cq, opts.repo_root, "baseline", input, opts.n_tiles, opts.repeats));
        }
        if (opts.variant == "batch2" || opts.variant == "both") {
            results.push_back(run_variant(*mesh_device, cq, opts.repo_root, "batch2", input, opts.n_tiles, opts.repeats));
        }

        for (const auto& result : results) {
            print_result(result);
        }

        if (results.size() == 2) {
            const auto baseline_avg = average_ms(results[0].runtimes_ms);
            const auto batch2_avg = average_ms(results[1].runtimes_ms);
            std::cout << "speedup_vs_baseline=" << (baseline_avg / batch2_avg) << '\n';
        }

        if (!mesh_device->close()) {
            throw std::runtime_error("failed to close mesh device");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
