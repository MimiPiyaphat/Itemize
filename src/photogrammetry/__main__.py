import argparse
import json
import math
import re
import struct
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .images import DEFAULT_BLUR_THRESHOLD, MODES, prepare_images
from .reconstruction import PipelineError, find_tool, reconstruct, run_logged
from .video import extract_frames

DEFAULT_MAX_IMAGE_SIZE = 1600


def export_model(blender: Path, mesh: Path, output: Path) -> Path:
    pending = output / "model.pending.glb"
    model = output / "model.glb"
    script = Path(__file__).with_name("export_glb.py").resolve()
    try:
        run_logged(
            blender, "export_glb", output / "logs" / "export_glb.log",
            "--background", "--factory-startup", "--python-exit-code", "1", "--python", script,
            "--", mesh, pending,
        )
        if not pending.is_file():
            raise PipelineError("export_glb", "Blender did not produce a GLB. See logs/export_glb.log.")
        with pending.open("rb") as handle:
            header = handle.read(12)
        if len(header) != 12 or struct.unpack("<4sII", header) != (
            b"glTF", 2, pending.stat().st_size,
        ) or pending.stat().st_size < 20:
            raise PipelineError("export_glb", "Blender produced an invalid GLB. See logs/export_glb.log.")
        pending.replace(model)
    finally:
        pending.unlink(missing_ok=True)
    return model


def arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an editable 3D mesh from overlapping views.")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--images", type=Path, help="Directory containing JPG or PNG photos.")
    sources.add_argument("--video", type=Path, help="Short MP4 clip of a stationary object.")
    parser.add_argument("--output", required=True, type=Path, help="New directory for this run.")
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD)
    parser.add_argument("--video-interval", type=float, default=0.5, help="Seconds between sampled frames.")
    parser.add_argument("--colmap", type=Path, help="Path to the COLMAP executable.")
    parser.add_argument("--blender", type=Path, help="Path to the Blender executable.")
    parser.add_argument("--max-image-size", type=int, default=DEFAULT_MAX_IMAGE_SIZE)
    parser.add_argument("--cpu-features", action="store_true", help="Use CPU matching. Dense processing still needs CUDA.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and score images without 3D tools.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.blur_threshold) or args.blur_threshold < 0:
        parser.error("--blur-threshold must be finite and nonnegative.")
    if not math.isfinite(args.video_interval) or args.video_interval <= 0:
        parser.error("--video-interval must be finite and positive.")
    if args.max_image_size < 1:
        parser.error("--max-image-size must be positive.")
    if args.images is not None and not args.images.is_dir():
        parser.error("--images must point to an existing directory.")
    if args.video is not None and (args.video.suffix.lower() != ".mp4" or not args.video.is_file()):
        parser.error("--video must point to an existing MP4 file.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    output = args.output.resolve()
    try:
        output.mkdir(parents=True)
    except OSError as exc:
        print(f"Cannot create a new run folder: {exc}", file=sys.stderr)
        return 1

    source = (args.images or args.video).resolve()
    started = perf_counter()
    stage = "setup"
    result = None
    counts = {"original": 0, "accepted": 0, "registered": None}
    report = {
        "status": "failed",
        "source": str(source),
        "counts": counts,
        "settings": {
            "mode": args.mode,
            "blur_threshold": args.blur_threshold,
            "video_interval_seconds": args.video_interval if args.video else None,
            "max_image_size": args.max_image_size,
            "cpu_features": args.cpu_features,
            "prepare_only": args.prepare_only,
        },
        "tools": {},
        "images": [],
        "artifacts": {},
    }
    try:
        if not args.prepare_only:
            colmap = find_tool("colmap", args.colmap)
            blender = find_tool("blender", args.blender)
            for name, tool, argument in (
                ("colmap", colmap, "-h"),
                ("blender", blender, "--version"),
            ):
                version_log = output / "logs" / f"{name}_version.log"
                banner = run_logged(tool, "setup", version_log, argument)
                version = re.search(rf"\b{name}\s+v?(\d+(?:\.\d+)+[^\s]*)", banner, re.IGNORECASE)
                report["tools"][name] = {
                    "path": str(tool),
                    "version": version.group(1) if version else None,
                    "version_log": str(version_log.relative_to(output)),
                }

        if args.video:
            stage = "extract_frames"
            source = output / "frames"
            frames = extract_frames(
                args.video.resolve(), source,
                interval_seconds=args.video_interval, blur_threshold=args.blur_threshold,
            )
            counts["original"] = len(frames)
            report["frames"] = [asdict(frame) for frame in frames]
        stage = "prepare_images"
        images = output / "images"
        records = prepare_images(source, images, mode=args.mode, blur_threshold=args.blur_threshold)
        report["images"] = [asdict(record) for record in records]
        if not args.video:
            counts["original"] = len(records)
        counts["accepted"] = sum(record.prepared is not None for record in records)
        if counts["accepted"] == 0:
            raise PipelineError(stage, "No usable images. Check image formats and the blur threshold.")
        if args.prepare_only:
            report["status"] = "prepared"
        else:
            if counts["accepted"] < 3:
                raise PipelineError(stage, "Provide at least three accepted photos from different views.")
            stage = "reconstruction"
            result = reconstruct(
                colmap, images, output, max_image_size=args.max_image_size, use_gpu=not args.cpu_features,
            )
            stage = "export_glb"
            model = export_model(blender, result.mesh, output)
            report["artifacts"]["glb"] = str(model.relative_to(output))
            report["status"] = "success"
    except (PipelineError, OSError, ValueError) as exc:
        if isinstance(exc, PipelineError):
            stage = exc.stage
            if exc.result is not None:
                result = exc.result
        report["failed_stage"] = stage
        report["error"] = str(exc)
        print(f"{stage}: {exc}", file=sys.stderr)
    finally:
        if result is not None:
            counts["registered"] = result.registered_images
            for name in ("sparse_model", "sparse_cloud", "dense_cloud", "mesh"):
                artifact = getattr(result, name)
                if artifact is not None:
                    report["artifacts"][name] = str(artifact.relative_to(output))
        report["elapsed_seconds"] = round(perf_counter() - started, 3)
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {report_path}")
    return int(report["status"] == "failed")


if __name__ == "__main__":
    sys.exit(main())
