#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_INPUT = REPO_ROOT / "playground_discrete_video" / "output" / "fall.png"

ARTIFACTS_DIR = ROOT / "artifacts"
VIDEOS_DIR = ARTIFACTS_DIR / "videos"
FRAMES_ROOT = ARTIFACTS_DIR / "frames"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(REPO_ROOT / "playground_discrete_video" / "ttnn_runtime"))
sys.path.insert(0, str(Path.home() / ".local/lib/tt-metal" / "tools"))

import ttnn  # noqa: E402


@dataclass(frozen=True)
class BandSpec:
    low: float
    high: float
    color: tuple[float, float, float]


BLEND_RECIPES: dict[str, list[tuple[int, int, float]]] = {
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

BLEND_SEQUENCE = [
    "cross_stitch",
    "halo",
    "scanline",
    "north_pull",
    "stutter_step",
    "cross_stitch",
    "halo",
    "scanline",
    "north_pull",
    "stutter_step",
]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_image(path: Path, max_dim: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def pad_image_to_tt(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width, _ = image.shape
    padded_height = int(math.ceil(height / 32.0) * 32)
    padded_width = int(math.ceil(width / 32.0) * 32)
    padded = np.zeros((1, padded_height, padded_width, 32), dtype=np.float32)
    padded[0, :height, :width, :3] = image
    return padded, height, width


def unpad_image_from_tt(image: torch.Tensor, height: int, width: int) -> np.ndarray:
    rgb = image[0, :height, :width, :3].float().cpu().numpy()
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def compute_base_bands(base_image: np.ndarray) -> list[tuple[float, float]]:
    rms = np.sqrt(np.sum(np.square(base_image), axis=2))
    quantiles = [0.18, 0.38, 0.58, 0.78]
    widths = [28.0, 34.0, 40.0, 48.0]
    bands = []
    for quantile, width in zip(quantiles, widths):
        center = float(np.quantile(rms, quantile))
        bands.append((max(0.0, center - width), min(441.0, center + width)))
    return bands


def interpolate_color(color_a: tuple[int, int, int], color_b: tuple[int, int, int], mix: float) -> tuple[float, float, float]:
    return tuple(color_a[index] * (1.0 - mix) + color_b[index] * mix for index in range(3))


def make_frame_bands(frame_index: int, total_frames: int, base_bands: list[tuple[float, float]]) -> list[BandSpec]:
    phase = frame_index / max(total_frames - 1, 1)
    curve = 0.5 - 0.5 * math.cos(phase * math.pi)
    color_a = (45, 255, 120)
    color_b = (245, 105, 215)
    color_c = (35, 205, 250)
    color_d = (235, 235, 85)
    colors = [
        interpolate_color(color_a, color_b, curve * 0.35),
        interpolate_color(color_c, color_d, curve * 0.30),
        interpolate_color(color_b, color_c, curve * 0.25),
        interpolate_color(color_d, color_a, curve * 0.20),
    ]
    band_count = 3 if (frame_index // 18) % 2 == 0 else 4
    return [
        BandSpec(low=base_bands[index][0], high=base_bands[index][1], color=colors[index])
        for index in range(band_count)
    ]


def make_target_tensor(height: int, width: int, color: tuple[float, float, float]) -> torch.Tensor:
    target = torch.zeros((1, height, width, 32), dtype=torch.bfloat16)
    target[..., 0] = color[0]
    target[..., 1] = color[1]
    target[..., 2] = color[2]
    return target


def compute_rms_mask(frame: ttnn.Tensor, low: float, high: float, padded_height: int, padded_width: int) -> ttnn.Tensor:
    rms = ttnn.sqrt(ttnn.sum(ttnn.multiply(frame, frame), dim=3))
    rms = ttnn.reshape(rms, [1, padded_height, padded_width, 1])
    rms = ttnn.repeat(rms, ttnn.Shape([1, 1, 1, 32]))
    return ttnn.multiply(ttnn.ge(rms, low), ttnn.le(rms, high))


def apply_color_pull(
    frame: ttnn.Tensor,
    band: BandSpec,
    target_tensor: ttnn.Tensor,
    padded_height: int,
    padded_width: int,
) -> ttnn.Tensor:
    mask = compute_rms_mask(frame, band.low, band.high, padded_height, padded_width)
    mixed = ttnn.add(ttnn.multiply(frame, 0.85), ttnn.multiply(target_tensor, 0.15))
    return ttnn.where(mask, mixed, frame)


def apply_blend_recipe(frame: ttnn.Tensor, recipe_name: str) -> ttnn.Tensor:
    blended = ttnn.multiply(frame, 0.0)
    carried = 1.0
    for dx, dy, weight in BLEND_RECIPES[recipe_name]:
        shifted = ttnn.roll(frame, shifts=[dy, dx], dim=[1, 2])
        blended = ttnn.add(blended, ttnn.multiply(shifted, weight))
        carried -= weight
    return ttnn.add(blended, ttnn.multiply(frame, max(carried, 0.0)))


def render_frame_on_device(
    base_tensor: ttnn.Tensor,
    target_tensors: list[ttnn.Tensor],
    bands: list[BandSpec],
    padded_height: int,
    padded_width: int,
) -> ttnn.Tensor:
    frame = base_tensor
    for _ in range(2):
        for band, target_tensor in zip(bands, target_tensors):
            frame = apply_color_pull(frame, band, target_tensor, padded_height, padded_width)
        for recipe_name in BLEND_SEQUENCE:
            frame = apply_blend_recipe(frame, recipe_name)
    return frame


def save_frame(image: np.ndarray, output_path: Path) -> None:
    Image.fromarray(image, mode="RGB").save(output_path)


class Mp4Writer:
    def __init__(self, output_path: Path, width: int, height: int, fps: int):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("MP4 writer stdin is not available")
        self.process.stdin.write(frame.tobytes())

    def close(self) -> str:
        stderr_bytes = b""
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stderr is not None:
            stderr_bytes = self.process.stderr.read()
        return_code = self.process.wait()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}: {stderr_text}")
        return stderr_text


def summarise_run(
    performance: list[dict[str, float]],
    args: argparse.Namespace,
    video_path: Path,
    frames_dir: Path | None,
    original_height: int,
    original_width: int,
    padded_height: int,
    padded_width: int,
) -> dict[str, object]:
    device_times = [entry["device_compute_ms"] for entry in performance] or [0.0]
    total_frame_times = [entry["frame_total_ms"] for entry in performance] or [0.0]
    upload_times = [entry["upload_ms"] for entry in performance] or [0.0]
    readback_times = [entry["readback_ms"] for entry in performance] or [0.0]
    encode_times = [entry["encode_ms"] for entry in performance] or [0.0]
    steady = performance[1:] if len(performance) > 1 else performance
    steady_total = [entry["frame_total_ms"] for entry in steady] or [0.0]
    steady_device = [entry["device_compute_ms"] for entry in steady] or [0.0]

    return {
        "input": str(args.input),
        "video": str(video_path),
        "frames_dir": str(frames_dir) if frames_dir is not None else None,
        "logs_dir": str(LOG_DIR),
        "frame_count": args.frames,
        "target_fps": args.fps,
        "save_frames": args.save_frames,
        "device_compute_avg_ms": float(np.mean(device_times)),
        "device_compute_max_ms": float(np.max(device_times)),
        "upload_avg_ms": float(np.mean(upload_times)),
        "readback_avg_ms": float(np.mean(readback_times)),
        "encode_avg_ms": float(np.mean(encode_times)),
        "end_to_end_avg_ms": float(np.mean(total_frame_times)),
        "end_to_end_fps": 1000.0 / float(np.mean(total_frame_times)),
        "steady_state_device_avg_ms": float(np.mean(steady_device)),
        "steady_state_end_to_end_avg_ms": float(np.mean(steady_total)),
        "steady_state_fps": 1000.0 / float(np.mean(steady_total)),
        "device_images_per_second": 1000.0 / float(np.mean(device_times)),
        "target_realtime_ratio": (1000.0 / float(np.mean(total_frame_times))) / float(args.fps),
        "original_height": original_height,
        "original_width": original_width,
        "padded_height": padded_height,
        "padded_width": padded_width,
        "ttnn_import": ttnn.__file__,
        "device_count": ttnn.get_num_devices(),
        "on_device_ops": [
            "rms band mask reduction",
            "15 percent color pull",
            "10 blend passes per iteration",
            "2 iterations per frame",
        ],
        "host_ops": [
            "image load and resize",
            "frame schedule generation",
            "tensor upload",
            "tensor readback",
            "optional png write",
            "mp4 encode",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware-backed p150a image-to-video renderer.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-dim", type=int, default=256)
    parser.add_argument("--prefix", default="fall_p150a")
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    image = load_image(args.input, max_dim=args.max_dim)
    base_bands = compute_base_bands(image)
    padded, original_height, original_width = pad_image_to_tt(image)
    padded_height = padded.shape[1]
    padded_width = padded.shape[2]

    frames_dir = FRAMES_ROOT / args.prefix if args.save_frames else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for stale_frame in frames_dir.glob("*.png"):
            stale_frame.unlink()

    video_path = VIDEOS_DIR / f"{args.prefix}_{args.frames:03d}f_{args.fps:02d}fps.mp4"

    log(f"Using input: {args.input}")
    log(f"Original image shape: {image.shape}")
    log(f"Padded TT tensor shape: {padded.shape}")
    log(f"Artifacts root: {ARTIFACTS_DIR}")
    log(f"Video target: {video_path}")
    log(f"Frames enabled: {args.save_frames}")
    log(f"ttnn import: {ttnn.__file__}")
    log(f"Reported TT devices: {ttnn.get_num_devices()}")

    run_start = time.perf_counter()
    performance: list[dict[str, float]] = []
    ffmpeg_log = ""

    device = ttnn.open_device(device_id=0)
    try:
        writer = Mp4Writer(video_path, original_width, original_height, args.fps)
        try:
            base_host = torch.from_numpy(padded).to(dtype=torch.bfloat16)
            base_tensor = ttnn.from_torch(base_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            log(f"Opened TT device: {device}")

            for frame_index in range(args.frames):
                frame_start = time.perf_counter()
                bands = make_frame_bands(frame_index, args.frames, base_bands)
                target_tensors = []

                upload_start = time.perf_counter()
                for band in bands:
                    target_host = make_target_tensor(padded_height, padded_width, band.color)
                    target_tensors.append(
                        ttnn.from_torch(target_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    )
                upload_end = time.perf_counter()

                compute_start = time.perf_counter()
                frame_tensor = render_frame_on_device(base_tensor, target_tensors, bands, padded_height, padded_width)
                ttnn.synchronize_device(device)
                compute_end = time.perf_counter()

                readback_start = time.perf_counter()
                frame_torch = ttnn.to_torch(ttnn.from_device(frame_tensor))
                output_image = unpad_image_from_tt(frame_torch, original_height, original_width)
                readback_end = time.perf_counter()

                save_start = time.perf_counter()
                if frames_dir is not None:
                    save_frame(output_image, frames_dir / f"{args.prefix}_{frame_index:03d}.png")
                save_end = time.perf_counter()

                encode_start = time.perf_counter()
                writer.write(output_image)
                encode_end = time.perf_counter()

                frame_end = time.perf_counter()
                frame_total_ms = (frame_end - frame_start) * 1000.0
                frame_stats = {
                    "frame_index": frame_index,
                    "band_count": len(bands),
                    "upload_ms": (upload_end - upload_start) * 1000.0,
                    "device_compute_ms": (compute_end - compute_start) * 1000.0,
                    "readback_ms": (readback_end - readback_start) * 1000.0,
                    "save_frame_ms": (save_end - save_start) * 1000.0,
                    "encode_ms": (encode_end - encode_start) * 1000.0,
                    "frame_total_ms": frame_total_ms,
                    "frame_fps": 1000.0 / frame_total_ms,
                    "device_ips": 1000.0 / max((compute_end - compute_start) * 1000.0, 1e-9),
                }
                performance.append(frame_stats)
                log(
                    "frame "
                    f"{frame_index:03d}: bands={frame_stats['band_count']} "
                    f"upload_ms={frame_stats['upload_ms']:.2f} "
                    f"device_ms={frame_stats['device_compute_ms']:.2f} "
                    f"readback_ms={frame_stats['readback_ms']:.2f} "
                    f"encode_ms={frame_stats['encode_ms']:.2f} "
                    f"total_ms={frame_stats['frame_total_ms']:.2f} "
                    f"ips={frame_stats['frame_fps']:.2f}"
                )
        finally:
            ffmpeg_log = writer.close()
    finally:
        ttnn.close_device(device)

    run_end = time.perf_counter()
    summary = summarise_run(
        performance=performance,
        args=args,
        video_path=video_path,
        frames_dir=frames_dir,
        original_height=original_height,
        original_width=original_width,
        padded_height=padded_height,
        padded_width=padded_width,
    )
    summary["run_wall_ms"] = (run_end - run_start) * 1000.0
    summary["ffmpeg_log"] = ffmpeg_log

    summary_path = LOG_DIR / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps({"summary": summary, "per_frame": performance}, indent=2))

    log(f"Wrote summary: {summary_path}")
    log(f"Wrote video: {video_path}")
    log(
        "Performance summary: "
        f"device_avg_ms={summary['device_compute_avg_ms']:.2f} "
        f"steady_device_avg_ms={summary['steady_state_device_avg_ms']:.2f} "
        f"end_to_end_avg_ms={summary['end_to_end_avg_ms']:.2f} "
        f"end_to_end_fps={summary['end_to_end_fps']:.2f} "
        f"device_ips={summary['device_images_per_second']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
