#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
FRAMES_DIR = ROOT / "frames"
VIDEO_PATH = ROOT / "fall_discrete_video.mp4"
VIDEO_GENTLE_PATH = ROOT / "fall_discrete_video_gentle_30fps.mp4"
CONTACT_SHEET_PATH = ROOT / "preview_grid.png"


@dataclass(frozen=True)
class RunSpec:
    label: str
    intensity: float
    drift: float


@dataclass(frozen=True)
class RandomBand:
    rms_low: float
    rms_high: float
    target_color: tuple[int, int, int]
    mix_scale: float
    passes: int
    mix_pattern: str
    shift_pattern_x: str
    shift_pattern_y: str
    shift_radius: int


BLEND_RECIPES = {
    "cross_stitch": [
        (-1, 0, 1 / 8),
        (1, 0, 1 / 8),
        (0, -1, 1 / 8),
        (0, 1, 1 / 8),
        (-1, -1, 1 / 16),
        (1, -1, 1 / 16),
        (-1, 1, 1 / 16),
        (1, 1, 1 / 16),
    ],
    "north_pull": [
        (0, -1, 1 / 4),
        (-1, -1, 1 / 8),
        (1, -1, 1 / 8),
        (-1, 0, 1 / 16),
        (1, 0, 1 / 16),
        (0, 1, 1 / 16),
    ],
    "scanline": [
        (-1, 0, 1 / 4),
        (1, 0, 1 / 4),
        (0, -1, 1 / 16),
        (0, 1, 1 / 16),
        (-2, 0, 1 / 16),
        (2, 0, 1 / 16),
    ],
    "halo": [
        (-1, -1, 1 / 8),
        (1, -1, 1 / 8),
        (-1, 1, 1 / 8),
        (1, 1, 1 / 8),
        (-1, 0, 1 / 16),
        (1, 0, 1 / 16),
        (0, -1, 1 / 16),
        (0, 1, 1 / 16),
    ],
    "stutter_step": [
        (1, 0, 1 / 4),
        (1, 1, 1 / 8),
        (0, 1, 1 / 8),
        (-1, 0, 1 / 16),
        (0, -1, 1 / 16),
        (-1, -1, 1 / 16),
    ],
}

FRACTION_PATTERNS = {
    "checker_ladder": (
        np.array([0.0, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8], dtype=np.float32),
        np.array(
            [
                [0, 2, 1, 3],
                [4, 1, 5, 2],
                [1, 3, 0, 4],
                [5, 2, 4, 1],
            ],
            dtype=np.int8,
        ),
    ),
    "corner_bloom": (
        np.array([0.0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6], dtype=np.float32),
        np.array(
            [
                [5, 3, 2, 4],
                [3, 1, 0, 2],
                [2, 0, 1, 3],
                [4, 2, 3, 5],
            ],
            dtype=np.int8,
        ),
    ),
    "scan_bars": (
        np.array([0.0, 1 / 5, 2 / 5, 3 / 5, 4 / 5], dtype=np.float32),
        np.array(
            [
                [1, 3, 4, 2, 0, 2],
                [1, 3, 4, 2, 0, 2],
                [0, 2, 3, 4, 1, 3],
                [0, 2, 3, 4, 1, 3],
            ],
            dtype=np.int8,
        ),
    ),
}

SHADING_PROFILES = {
    "default": 8,
    "moody": 10,
    "soft": 6,
    "electric": 9,
    "wash": 7,
}

AWKWARD_COLORS = [
    (250, 210, 15),
    (85, 40, 240),
    (145, 250, 35),
    (35, 205, 250),
    (245, 105, 215),
    (235, 235, 85),
    (115, 145, 245),
    (45, 255, 120),
]

SOURCE_NAME_RE = re.compile(r".*_\d+$")


def log(message: str) -> None:
    print(message, flush=True)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def source_pngs() -> list[Path]:
    files = []
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        stem = path.stem
        if "_mix_" in stem:
            continue
        if SOURCE_NAME_RE.match(stem):
            continue
        files.append(path)
    return files


def load_rgb(path: Path, max_dim: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="RGB").save(path)


def pattern_map(name: str, height: int, width: int, x_offset: int, y_offset: int) -> np.ndarray:
    levels, grid = FRACTION_PATTERNS[name]
    row_index = (np.arange(height) + y_offset) % grid.shape[0]
    col_index = (np.arange(width) + x_offset) % grid.shape[1]
    lookup = grid[np.ix_(row_index, col_index)]
    return levels[lookup]


def random_pattern_name(rng: random.Random) -> str:
    return rng.choice(tuple(FRACTION_PATTERNS.keys()))


def random_shading_pair(rng: random.Random) -> tuple[str, str]:
    first = rng.choice(tuple(SHADING_PROFILES.keys()))
    choices = [name for name in SHADING_PROFILES if name != first]
    second = rng.choice(choices)
    return first, second


def make_random_bands(run: RunSpec, rng: random.Random) -> list[RandomBand]:
    bands = []
    band_count = 3 + rng.randrange(2)
    for index in range(band_count):
        center = 25.0 + rng.randrange(390)
        width = 28.0 + rng.randrange(70)
        rms_low = clamp(center - width * (0.45 + 0.15 * index), 0.0, 441.0)
        rms_high = clamp(center + width * (0.55 + 0.10 * index), 0.0, 441.0)
        bands.append(
            RandomBand(
                rms_low=min(rms_low, rms_high),
                rms_high=max(rms_low, rms_high),
                target_color=rng.choice(AWKWARD_COLORS),
                mix_scale=clamp((0.30 + 0.12 * index) * run.intensity, 0.10, 0.95),
                passes=1 + rng.randrange(2),
                mix_pattern=random_pattern_name(rng),
                shift_pattern_x=random_pattern_name(rng),
                shift_pattern_y=random_pattern_name(rng),
                shift_radius=max(2, int(round((2.0 + index) * run.drift))),
            )
        )
    return bands


def process_recipe_iteration(image: np.ndarray, recipe_name: str) -> None:
    temp = image.copy()
    height, width, _ = image.shape
    blended = np.zeros_like(temp, dtype=np.float32)
    carried = np.ones((height, width), dtype=np.float32)

    for dx, dy, weight in BLEND_RECIPES[recipe_name]:
        if dx >= 0:
            dst_x = slice(0, width - dx)
            src_x = slice(dx, width)
        else:
            dst_x = slice(-dx, width)
            src_x = slice(0, width + dx)
        if dy >= 0:
            dst_y = slice(0, height - dy)
            src_y = slice(dy, height)
        else:
            dst_y = slice(-dy, height)
            src_y = slice(0, height + dy)
        blended[dst_y, dst_x] += temp[src_y, src_x] * weight
        carried[dst_y, dst_x] -= weight

    image[:] = blended + temp * np.maximum(carried, 0.0)[..., None]


def apply_blend_burst(image: np.ndarray, recipe_names: list[str]) -> None:
    for recipe_name in recipe_names:
        process_recipe_iteration(image, recipe_name)


def blend_cycle(length: int) -> list[str]:
    recipe_order = ["cross_stitch", "halo", "scanline", "north_pull", "stutter_step"]
    return [recipe_order[index % len(recipe_order)] for index in range(length)]


def shift_colorize(
    image: np.ndarray,
    band: RandomBand,
    mix_pattern_name: str,
    shift_pattern_x_name: str,
    shift_pattern_y_name: str,
    num_it: int,
    x_offset: int,
    y_offset: int,
) -> None:
    height, width, _ = image.shape
    grid_y, grid_x = np.indices((height, width), dtype=np.int32)
    target = np.array(band.target_color, dtype=np.float32)

    for step in range(num_it):
        temp = image.copy()
        rms_map = np.sqrt(np.sum(image * image, axis=2))
        mask = (rms_map >= band.rms_low) & (rms_map <= band.rms_high)
        if not np.any(mask):
            continue

        mix = np.clip(
            pattern_map(mix_pattern_name, height, width, x_offset + step, y_offset + step) * band.mix_scale,
            0.0,
            1.0,
        )
        shift_x = np.rint(
            (pattern_map(shift_pattern_x_name, height, width, x_offset + step, y_offset + step) - 0.5)
            * 2.0
            * band.shift_radius
        ).astype(np.int32)
        shift_y = np.rint(
            (pattern_map(shift_pattern_y_name, height, width, x_offset + step, y_offset + step) - 0.5)
            * 2.0
            * band.shift_radius
        ).astype(np.int32)

        sx = np.clip(grid_x + shift_x, 0, width - 1)
        sy = np.clip(grid_y + shift_y, 0, height - 1)
        shifted = temp[sy, sx]
        moved = shifted + (target - shifted) * mix[..., None]
        image[mask] = moved[mask]


def apply_shading_pass(image: np.ndarray, shading: str, run: RunSpec, phase_seed: int, rng: random.Random) -> None:
    num_it = max(1, int(round(SHADING_PROFILES[shading] * run.intensity)))
    drift_step = max(1, int(round(run.drift)))
    bands = make_random_bands(run, rng)

    for index in range(num_it):
        x_phase = phase_seed + index * drift_step
        y_phase = phase_seed + int(round(index * run.drift))

        if shading == "moody":
            for band_index, band in enumerate(bands):
                boosted = RandomBand(
                    band.rms_low,
                    band.rms_high,
                    band.target_color,
                    clamp(band.mix_scale * 1.05, 0.0, 1.0),
                    band.passes,
                    band.mix_pattern,
                    band.shift_pattern_x,
                    band.shift_pattern_y,
                    band.shift_radius + 1,
                )
                shift_colorize(
                    image,
                    boosted,
                    boosted.mix_pattern,
                    boosted.shift_pattern_x,
                    boosted.shift_pattern_y,
                    boosted.passes,
                    x_phase,
                    y_phase * (band_index + 1),
                )
            apply_blend_burst(image, ["north_pull", "halo", "cross_stitch", "north_pull", "halo"])
        elif shading == "soft":
            for band_index, band in enumerate(bands):
                softened = RandomBand(
                    band.rms_low,
                    band.rms_high,
                    band.target_color,
                    clamp(band.mix_scale * 0.70, 0.0, 1.0),
                    1,
                    band.mix_pattern,
                    band.shift_pattern_x,
                    band.shift_pattern_y,
                    max(1, band.shift_radius - 1),
                )
                shift_colorize(
                    image,
                    softened,
                    softened.mix_pattern,
                    softened.shift_pattern_y,
                    softened.shift_pattern_x,
                    softened.passes,
                    x_phase,
                    y_phase + band_index,
                )
            apply_blend_burst(image, ["cross_stitch", "halo", "cross_stitch", "halo"])
        elif shading == "electric":
            for band_index, band in enumerate(bands):
                charged = RandomBand(
                    band.rms_low,
                    band.rms_high,
                    band.target_color,
                    clamp(band.mix_scale * 1.20, 0.0, 1.0),
                    band.passes + 1,
                    "scan_bars",
                    band.shift_pattern_x,
                    band.shift_pattern_y,
                    band.shift_radius + 2,
                )
                shift_colorize(
                    image,
                    charged,
                    charged.mix_pattern,
                    charged.shift_pattern_x,
                    charged.shift_pattern_y,
                    charged.passes,
                    x_phase * (band_index + 1),
                    y_phase,
                )
            apply_blend_burst(image, ["scanline", "stutter_step", "cross_stitch", "scanline", "stutter_step", "cross_stitch"])
        elif shading == "wash":
            for band_index, band in enumerate(bands):
                washed = RandomBand(
                    band.rms_low,
                    band.rms_high,
                    band.target_color,
                    clamp(band.mix_scale * 0.85, 0.0, 1.0),
                    band.passes,
                    "corner_bloom",
                    band.shift_pattern_x,
                    band.shift_pattern_y,
                    band.shift_radius,
                )
                shift_colorize(
                    image,
                    washed,
                    washed.mix_pattern,
                    washed.shift_pattern_y,
                    washed.shift_pattern_x,
                    washed.passes,
                    x_phase,
                    y_phase + band_index * 2,
                )
            apply_blend_burst(image, ["halo", "cross_stitch", "halo", "cross_stitch", "north_pull"])
        else:
            for band_index, band in enumerate(bands):
                shift_colorize(
                    image,
                    band,
                    band.mix_pattern,
                    band.shift_pattern_x,
                    band.shift_pattern_y,
                    band.passes,
                    x_phase + band_index,
                    y_phase * (band_index + 1),
                )
            if index % 2 == 0:
                apply_blend_burst(image, ["cross_stitch", "north_pull", "halo", "cross_stitch", "scanline"])
            else:
                apply_blend_burst(image, ["scanline", "stutter_step", "cross_stitch", "halo", "stutter_step"])


def render_variant(source_path: Path, run: RunSpec, max_dim: int, seed: int, prefix: str | None = None) -> Path:
    rng = random.Random(seed)
    image = load_rgb(source_path, max_dim=max_dim)
    first_shading, second_shading = random_shading_pair(rng)
    apply_shading_pass(image, first_shading, run, rng.randrange(19), rng)
    apply_shading_pass(image, second_shading, run, rng.randrange(7, 38), rng)

    name_prefix = prefix if prefix is not None else source_path.stem
    output_name = (
        f"{name_prefix}_{run.label}_mix_{first_shading}_{second_shading}"
        f"_i{int(round(run.intensity * 100))}_d{int(round(run.drift * 100))}.png"
    )
    output_path = OUTPUT_DIR / output_name
    save_rgb(image, output_path)
    return output_path


def interpolate_color(color_a: tuple[int, int, int], color_b: tuple[int, int, int], mix: float) -> tuple[int, int, int]:
    return tuple(int(round(color_a[channel] * (1.0 - mix) + color_b[channel] * mix)) for channel in range(3))


def make_video_bands(base_image: np.ndarray) -> list[tuple[float, float]]:
    rms_map = np.sqrt(np.sum(base_image * base_image, axis=2))
    quantiles = [0.18, 0.38, 0.58, 0.78]
    centers = [float(np.quantile(rms_map, q)) for q in quantiles]
    widths = [28.0, 34.0, 40.0, 48.0]
    bands = []
    for center, width in zip(centers, widths):
        bands.append((clamp(center - width, 0.0, 441.0), clamp(center + width, 0.0, 441.0)))
    return bands


def make_video_band_set(frame_index: int, total_frames: int, base_image: np.ndarray) -> list[RandomBand]:
    base_bands = make_video_bands(base_image)
    band_count = 3 if (frame_index // 18) % 2 == 0 else 4
    frame_phase = frame_index / max(total_frames - 1, 1)
    palette_phase = 0.5 - 0.5 * math.cos(frame_phase * math.pi)
    color_a = (45, 255, 120)
    color_b = (245, 105, 215)
    color_c = (35, 205, 250)
    color_d = (235, 235, 85)
    colors = [
        interpolate_color(color_a, color_b, palette_phase * 0.35),
        interpolate_color(color_c, color_d, palette_phase * 0.30),
        interpolate_color(color_b, color_c, palette_phase * 0.25),
        interpolate_color(color_d, color_a, palette_phase * 0.20),
    ]
    patterns = ["checker_ladder", "corner_bloom", "scan_bars", "checker_ladder"]

    bands = []
    for index in range(band_count):
        rms_low, rms_high = base_bands[index]
        drift_radius = 2 + (frame_index // 24)
        bands.append(
            RandomBand(
                rms_low=rms_low,
                rms_high=rms_high,
                target_color=colors[index],
                mix_scale=0.15,
                passes=1,
                mix_pattern=patterns[index],
                shift_pattern_x=patterns[(index + 1) % len(patterns)],
                shift_pattern_y=patterns[(index + 2) % len(patterns)],
                shift_radius=drift_radius,
            )
        )
    return bands


def render_gentle_frame(source_path: Path, frame_index: int, total_frames: int, max_dim: int) -> np.ndarray:
    image = load_rgb(source_path, max_dim=max_dim)
    height, width, _ = image.shape
    bands = make_video_band_set(frame_index, total_frames, image)

    for iteration in range(2):
        phase_step = frame_index // 3 + iteration * 2
        for band_index, band in enumerate(bands):
            shift_colorize(
                image,
                band,
                band.mix_pattern,
                band.shift_pattern_x,
                band.shift_pattern_y,
                num_it=1,
                x_offset=phase_step + band_index,
                y_offset=phase_step // 2 + band_index * 2,
            )
        apply_blend_burst(image, blend_cycle(10))

    vignette_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    vignette_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    vignette = 1.0 - 0.08 * (vignette_y[:, None] ** 2 + vignette_x[None, :] ** 2)
    image *= np.clip(vignette, 0.88, 1.0)[..., None]
    return np.clip(image, 0, 255)


def make_preview_grid(paths: list[Path], destination: Path) -> None:
    if not paths:
        return

    thumbs = []
    for path in paths[:9]:
        image = Image.open(path).convert("RGB")
        image.thumbnail((440, 440), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (440, 440), (18, 18, 18))
        offset = ((440 - image.width) // 2, (440 - image.height) // 2)
        canvas.paste(image, offset)
        thumbs.append(canvas)

    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    grid = Image.new("RGB", (cols * 440, rows * 440), (10, 10, 10))
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * 440
        y = (index // cols) * 440
        grid.paste(thumb, (x, y))
    grid.save(destination)


def create_stills() -> list[Path]:
    curated_runs = [
        RunSpec("basement_glow", 1.05, 0.45),
        RunSpec("toxic_poster", 1.30, 0.90),
    ]
    generated = []
    for source_path in source_pngs():
        for run_index, run in enumerate(curated_runs):
            seed = abs(hash((source_path.stem, run.label, run_index))) % (2**32)
            log(f"rendering still: {source_path.name} -> {run.label}")
            generated.append(render_variant(source_path, run, max_dim=1280, seed=seed))
    make_preview_grid(generated, CONTACT_SHEET_PATH)
    return generated


def video_runs(frame_count: int) -> list[RunSpec]:
    runs = []
    for frame in range(frame_count):
        phase = frame / max(frame_count - 1, 1)
        intensity = 0.82 + 0.48 * math.sin(phase * math.pi)
        drift = 0.38 + 1.05 * phase
        runs.append(RunSpec(f"frame_{frame:03d}", clamp(intensity, 0.5, 1.45), clamp(drift, 0.3, 1.6)))
    return runs


def create_video() -> Path:
    source_path = OUTPUT_DIR / "fall.png"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing video source image: {source_path}")

    frame_paths = []
    for old_frame in FRAMES_DIR.glob("fall_seq_*.png"):
        old_frame.unlink()

    for frame_index, run in enumerate(video_runs(frame_count=20)):
        seed = 1337 + frame_index * 97
        log(f"rendering video frame {frame_index + 1}/20")
        rendered = render_variant(source_path, run, max_dim=960, seed=seed, prefix=f"fall_seq_{frame_index:03d}")
        frame_path = FRAMES_DIR / f"fall_seq_{frame_index:03d}.png"
        if frame_path.exists():
            frame_path.unlink()
        rendered.replace(frame_path)
        frame_paths.append(frame_path)

    writer = imageio.get_writer(VIDEO_PATH, fps=12, codec="libx264", quality=7, pixelformat="yuv420p")
    try:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()
    return VIDEO_PATH


def create_gentle_video(frame_count: int = 90, fps: int = 30, max_dim: int = 720) -> Path:
    source_path = OUTPUT_DIR / "fall.png"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing video source image: {source_path}")

    for old_frame in FRAMES_DIR.glob("fall_gentle_*.png"):
        old_frame.unlink()

    frame_paths = []
    for frame_index in range(frame_count):
        log(f"rendering gentle video frame {frame_index + 1}/{frame_count}")
        image = render_gentle_frame(source_path, frame_index, frame_count, max_dim=max_dim)
        frame_path = FRAMES_DIR / f"fall_gentle_{frame_index:03d}.png"
        save_rgb(image, frame_path)
        frame_paths.append(frame_path)

    writer = imageio.get_writer(VIDEO_GENTLE_PATH, fps=fps, codec="libx264", quality=7, pixelformat="yuv420p")
    try:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()
    return VIDEO_GENTLE_PATH


def probe_p150a() -> str:
    try:
        result = subprocess.run(
            ["tt-smi", "-ls"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "tt-smi not found"
    text = result.stdout or ""
    if "p150a" in text:
        return "p150a visible via tt-smi"
    return "tt-smi ran, but p150a was not detected"


def main() -> int:
    parser = argparse.ArgumentParser(description="Separate playground for discrete stills and a short video.")
    parser.add_argument("--mode", choices=["stills", "video", "video_gentle", "all"], default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    log(probe_p150a())

    if args.mode in {"stills", "all"}:
        stills = create_stills()
        log(f"generated {len(stills)} still renders")

    if args.mode in {"video", "all"}:
        video_path = create_video()
        log(f"wrote video: {video_path}")

    if args.mode in {"video_gentle", "all"}:
        video_path = create_gentle_video()
        log(f"wrote gentle video: {video_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
