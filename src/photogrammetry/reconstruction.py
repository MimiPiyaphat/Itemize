import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReconstructionResult:
    registered_images: int | None = None
    sparse_model: Path | None = None
    sparse_cloud: Path | None = None
    dense_cloud: Path | None = None
    mesh: Path | None = None


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.result: ReconstructionResult | None = None


def find_tool(name: str, requested: Path | None) -> Path:
    if requested is None:
        executable = shutil.which(f"{name}.exe" if os.name == "nt" else name)
        if executable is None:
            raise PipelineError("setup", f"{name} was not found. Supply --{name} with its executable path.")
        return Path(executable).resolve()
    path = requested.resolve()
    if not path.is_file() or (os.name == "nt" and path.suffix.lower() != ".exe"):
        raise PipelineError("setup", f"Use the {name} executable path: {path}")
    return path


def run_logged(tool: Path, stage: str, log: Path, *args: str | Path) -> str:
    command = [str(tool), *map(str, args)]
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{stage}]", flush=True)
    env = os.environ.copy()
    if os.name == "nt":
        env["PATH"] = os.pathsep.join((str(tool.parent), str(tool.parent.parent), env.get("PATH", "")))
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    except OSError as exc:
        log.write_text(f"{command!r}\n{exc}\n", encoding="utf-8")
        raise PipelineError(stage, f"Could not start {tool.name}. See {log}.") from exc
    output = process.stdout.decode("utf-8", errors="replace")
    log.write_text(f"{command!r}\n{output}", encoding="utf-8")
    if process.returncode:
        raise PipelineError(stage, f"{tool.name} exited with code {process.returncode}. See {log}.")
    return output


def ply_has_vertices(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(4096).split(b"end_header", 1)
    if len(header) != 2 or not header[0].splitlines() or header[0].splitlines()[0] != b"ply":
        return False
    vertices = re.search(rb"(?m)^element vertex (\d+)\r?$", header[0])
    return vertices is not None and int(vertices.group(1)) > 0


def reconstruct(
    colmap: Path, images: Path, workspace: Path, *, max_image_size: int, use_gpu: bool
) -> ReconstructionResult:
    result = ReconstructionResult()
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    dense = workspace / "dense"
    sparse.mkdir(parents=True)
    dense.mkdir()
    gpu = str(int(use_gpu))

    def run(stage: str, *args: str | Path, suffix: str = "") -> str:
        return run_logged(colmap, stage, workspace / "logs" / f"{stage}{suffix}.log", stage, *args)

    def require_file(path: Path, stage: str) -> Path:
        if not path.is_file() or path.stat().st_size == 0:
            raise PipelineError(stage, f"COLMAP did not produce {path.name}. See the stage log.")
        return path

    try:
        run(
            "feature_extractor", "--database_path", database, "--image_path", images,
            "--FeatureExtraction.use_gpu", gpu, "--FeatureExtraction.max_image_size", str(max_image_size),
        )
        if not database.is_file():
            raise PipelineError("feature_extractor", "COLMAP did not produce the feature database.")
        run("exhaustive_matcher", "--database_path", database, "--FeatureMatching.use_gpu", gpu)
        run("mapper", "--database_path", database, "--image_path", images, "--output_path", sparse)

        candidates = []
        for folder in sorted(sparse.iterdir()):
            if not (folder / "images.bin").is_file():
                continue
            stats = run("model_analyzer", "--path", folder, suffix=f"_{folder.name}")
            match = re.search(r"\bRegistered\s+images:\s*(\d+)", stats, re.IGNORECASE)
            if match is None:
                match = re.search(r"\bImages:\s*(\d+)", stats, re.IGNORECASE)
            if match is None:
                raise PipelineError("model_analyzer", f"Could not read registered image count for {folder.name}.")
            candidates.append((int(match.group(1)), folder))
        result.registered_images, result.sparse_model = max(
            candidates, key=lambda item: item[0], default=(0, None)
        )
        if result.registered_images < 3:
            raise PipelineError("mapper", "Too few registered views. Try sharper photos with more overlap.")

        cloud = workspace / "sparse.ply"
        run("model_converter", "--input_path", result.sparse_model, "--output_path", cloud, "--output_type", "PLY")
        result.sparse_cloud = require_file(cloud, "model_converter")
        run(
            "image_undistorter", "--image_path", images, "--input_path", result.sparse_model,
            "--output_path", dense, "--output_type", "COLMAP", "--max_image_size", str(max_image_size),
        )
        if not (dense / "stereo").is_dir():
            raise PipelineError("image_undistorter", "COLMAP did not prepare the dense workspace.")
        run("patch_match_stereo", "--workspace_path", dense, "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true")
        if not any((dense / "stereo" / "depth_maps").glob("*.geometric.bin")):
            raise PipelineError("patch_match_stereo", "No depth maps were produced. Check the CUDA log.")

        fused = dense / "fused.ply"
        run("stereo_fusion", "--workspace_path", dense, "--workspace_format", "COLMAP",
            "--input_type", "geometric", "--output_path", fused)
        require_file(fused, "stereo_fusion")
        if not ply_has_vertices(fused):
            raise PipelineError("stereo_fusion", "COLMAP produced no usable dense points. See stereo_fusion.log.")
        result.dense_cloud = fused
        mesh = workspace / "mesh.ply"
        run("poisson_mesher", "--input_path", fused, "--output_path", mesh)
        result.mesh = require_file(mesh, "poisson_mesher")
    except PipelineError as exc:
        exc.result = result
        raise
    return result
