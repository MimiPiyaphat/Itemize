from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODES = ("baseline", "filter", "filter-clahe")
DEFAULT_BLUR_THRESHOLD = 30.0
SHARPNESS_MAX_SIZE = 960


@dataclass(frozen=True)
class ImageRecord:
    source: str
    prepared: str | None
    sharpness: float | None
    status: str
    detail: str | None = None


def sharpness_score(gray: np.ndarray) -> float:
    height, width = gray.shape
    scale = min(1.0, SHARPNESS_MAX_SIZE / max(width, height))
    if scale < 1:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def prepare_images(
    source: Path, destination: Path, *, mode: str, blur_threshold: float
) -> list[ImageRecord]:
    if mode not in MODES:
        raise ValueError(f"Unknown image mode: {mode}")
    paths = sorted(
        (path for path in source.iterdir() if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES),
        key=lambda path: (path.name.casefold(), path.name),
    )
    destination.mkdir(parents=True)
    records = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as original:
                oriented = ImageOps.exif_transpose(original)
                exif = oriented.getexif()
                rgb = np.asarray(oriented.convert("RGB"))
        except (OSError, ValueError, Image.DecompressionBombError, UnidentifiedImageError) as exc:
            records.append(ImageRecord(path.name, None, None, "unreadable", str(exc)))
            continue

        score = sharpness_score(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if mode != "baseline" and score < blur_threshold:
            records.append(ImageRecord(path.name, None, score, "blurry"))
            continue
        if mode == "filter-clahe":
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        name = f"view_{index:04d}.jpg"
        Image.fromarray(rgb).save(destination / name, quality=95, subsampling=0, exif=exif)
        records.append(ImageRecord(path.name, name, score, "accepted"))
    return records
