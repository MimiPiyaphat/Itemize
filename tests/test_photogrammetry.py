import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.photogrammetry.images import prepare_images
from src.photogrammetry.reconstruction import PipelineError, reconstruct
from src.photogrammetry.video import extract_frames


def test_prepare_images_reports_corruption_and_keeps_filter_selection(tmp_path):
    source = tmp_path / "original photos"
    source.mkdir()
    pixels = np.full((96, 160, 3), 80, dtype=np.uint8)
    pixels[24:72, 40:120] = 150
    image = Image.fromarray(pixels)
    orientation = Image.Exif()
    orientation[274] = 6
    original = source / "sharp.JPG"
    image.save(original, exif=orientation)
    image.filter(ImageFilter.GaussianBlur(7)).save(source / "blur.png")
    (source / "broken.jpg").write_bytes(b"broken")
    originals = {file.name: file.read_bytes() for file in source.iterdir()}

    baseline, filtered, enhanced = (
        prepare_images(source, tmp_path / mode, mode=mode, blur_threshold=2)
        for mode in ("baseline", "filter", "filter-clahe")
    )
    scores = {record.source: record.sharpness for record in baseline}
    assert scores["sharp.JPG"] > 2 > scores["blur.png"]
    assert {record.status for record in filtered} == {"accepted", "blurry", "unreadable"}
    for records in (filtered, enhanced):
        assert [record.source for record in records if record.prepared] == ["sharp.JPG"]
    first = next(record.prepared for record in filtered if record.prepared)
    second = next(record.prepared for record in enhanced if record.prepared)
    with Image.open(tmp_path / "filter" / first) as normal, Image.open(
        tmp_path / "filter-clahe" / second
    ) as adjusted:
        assert normal.size == adjusted.size == (96, 160)
        assert normal.tobytes() != adjusted.tobytes()
    assert {file.name: file.read_bytes() for file in source.iterdir()} == originals


@pytest.mark.parametrize("outcome", ["success", "mesh_failure", "empty_cloud", "too_few_views"])
def test_reconstruction_selects_largest_model_and_keeps_point_cloud(tmp_path, monkeypatch, outcome):
    workspace = tmp_path / "run folder"
    workspace.mkdir()
    images = tmp_path / "photos here"
    images.mkdir()
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        stage = args[1]

        def arg(name):
            return Path(args[args.index(name) + 1])

        output = b""
        if stage == "feature_extractor":
            arg("--database_path").touch()
        if stage == "mapper":
            for model in ("0", "1"):
                directory = arg("--output_path") / model
                directory.mkdir()
                (directory / "images.bin").touch()
        if stage == "model_analyzer":
            registered = 10 if arg("--path").name == "1" else 5
            if outcome == "too_few_views":
                registered = 2 if arg("--path").name == "1" else 1
            total = 12 if arg("--path").name == "1" else 15
            output = (
                f"I00 model.cc] Images: {total}\n"
                f"I00 model.cc] Registered images: {registered}\n"
            ).encode()
        if stage in {"model_converter", "stereo_fusion", "poisson_mesher"}:
            if stage == "poisson_mesher" and outcome == "mesh_failure":
                return subprocess.CompletedProcess(args, 1, b"failed to create mesh")
            vertices = 0 if stage == "stereo_fusion" and outcome == "empty_cloud" else 1
            arg("--output_path").write_bytes(
                f"ply\nformat ascii 1.0\nelement vertex {vertices}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n".encode()
                + b"0 0 0\n" * vertices
            )
        if stage == "image_undistorter":
            (arg("--output_path") / "stereo").mkdir()
        if stage == "patch_match_stereo":
            depth = arg("--workspace_path") / "stereo" / "depth_maps"
            depth.mkdir()
            (depth / "sample.png.geometric.bin").write_bytes(b"depth")
        return subprocess.CompletedProcess(args, 0, output)

    monkeypatch.setattr(subprocess, "run", run)
    if outcome == "too_few_views":
        with pytest.raises(PipelineError) as caught:
            reconstruct(Path("colmap"), images, workspace, max_image_size=1600, use_gpu=True)
        assert caught.value.stage == "mapper"
        assert caught.value.result.registered_images == 2
        return
    if outcome == "empty_cloud":
        with pytest.raises(PipelineError) as caught:
            reconstruct(Path("colmap"), images, workspace, max_image_size=1600, use_gpu=True)
        assert caught.value.stage == "stereo_fusion"
        assert caught.value.result.sparse_cloud.is_file()
        assert caught.value.result.dense_cloud is None
        assert not any(args[1] == "poisson_mesher" for args in commands)
        return
    if outcome == "mesh_failure":
        with pytest.raises(PipelineError) as caught:
            reconstruct(Path("colmap"), images, workspace, max_image_size=1600, use_gpu=True)
        assert caught.value.stage == "poisson_mesher"
        assert caught.value.result.dense_cloud.is_file()
        assert caught.value.result.mesh is None
    else:
        result = reconstruct(Path("colmap"), images, workspace, max_image_size=1600, use_gpu=True)
        assert result.registered_images == 10
        assert result.sparse_model.name == "1"
        assert result.mesh.is_file()
    assert str(images) in next(args for args in commands if args[1] == "mapper")
    assert str(workspace / "sparse" / "1") in next(args for args in commands if args[1] == "image_undistorter")
    assert (workspace / "logs" / "poisson_mesher.log").is_file()


def test_cli_prepare_only_and_existing_output(tmp_path):
    source = tmp_path / "photos"
    source.mkdir()
    Image.new("RGB", (32, 32)).save(source / "one.png")
    output = tmp_path / "output"
    args = [
        sys.executable, "-m", "src.photogrammetry",
        "--images", str(source), "--output", str(output), "--prepare-only",
    ]
    completed = subprocess.run(args, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "prepared"
    assert report["counts"] == {"original": 1, "accepted": 1, "registered": None}
    assert subprocess.run(args, capture_output=True).returncode != 0
    assert report == json.loads((output / "report.json").read_text(encoding="utf-8"))


def test_cli_tracks_failed_glb_and_preserves_mesh(tmp_path, monkeypatch):
    from src.photogrammetry import __main__ as cli
    from src.photogrammetry.reconstruction import ReconstructionResult

    source = tmp_path / "photos"
    source.mkdir()
    for number in range(3):
        Image.new("RGB", (32, 32)).save(source / f"{number}.png")
    output = tmp_path / "bad export"
    monkeypatch.setattr(cli, "find_tool", lambda name, path: Path(name))

    def fake_colmap(tool, images, workspace, **kwargs):
        assert len(list(images.glob("*.jpg"))) == 3
        mesh = workspace / "mesh.ply"
        mesh.write_bytes(b"ply")
        return ReconstructionResult(registered_images=3, mesh=mesh)

    def fake_run(args, **kwargs):
        if "--python" in args:
            Path(args[-1]).write_bytes(b"invalid GLB")
        return subprocess.CompletedProcess(args, 0, b"test tool version")

    monkeypatch.setattr(cli, "reconstruct", fake_colmap)
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cli.main(["--images", str(source), "--output", str(output)]) != 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["failed_stage"] == "export_glb"
    assert report["counts"]["registered"] == 3
    assert (output / report["artifacts"]["mesh"]).is_file()
    assert not (output / "model.glb").exists()


def test_video_rejects_unreadable_clip(tmp_path):
    bad_video = tmp_path / "unknown.mp4"
    bad_video.write_bytes(b"bad")
    with pytest.raises(ValueError, match="video"):
        extract_frames(bad_video, tmp_path / "frames", interval_seconds=0.25)


def test_video_keeps_source_times_and_rejects_blurry_and_repeated_frames(tmp_path):
    clip = tmp_path / "test.mp4"
    texture = np.random.default_rng(42).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    subsequent = np.roll(texture, 40, axis=1)
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 4, (128, 96))
    assert writer.isOpened()
    try:
        for image in (texture, texture, np.zeros_like(texture), subsequent):
            writer.write(image)
    finally:
        writer.release()
    frames = extract_frames(clip, tmp_path / "frames", interval_seconds=0.25)
    assert [(frame.time_seconds, frame.status) for frame in frames] == [
        (0.0, "accepted"), (0.25, "duplicate"), (0.5, "blurry"), (0.75, "accepted"),
    ]
    assert len(list((tmp_path / "frames").glob("*.png"))) == 2
