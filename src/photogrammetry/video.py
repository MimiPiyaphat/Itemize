import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .images import DEFAULT_BLUR_THRESHOLD, sharpness_score


@dataclass(frozen=True)
class VideoFrame:
    time_seconds: float
    filename: str | None
    sharpness: float
    status: str


def extract_frames(
    source: Path,
    destination: Path,
    *,
    interval_seconds: float,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> list[VideoFrame]:
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("The video interval must be a positive number.")
    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Could not open the video or read its frame rate: {source}")
        destination.mkdir(parents=True)
        records = []
        frame_index = 0
        next_time = 0.0
        last_time = -1.0
        last_thumbnail = None
        while True:
            ok, image = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp) or timestamp <= last_time:
                timestamp = frame_index / fps
            last_time = timestamp
            frame_index += 1
            if timestamp + 1e-6 < next_time:
                continue
            next_time = (math.floor((timestamp + 1e-6) / interval_seconds) + 1) * interval_seconds

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            score = sharpness_score(gray)
            thumbnail = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            name = None
            if score < blur_threshold:
                status = "blurry"
            elif last_thumbnail is not None and np.mean(cv2.absdiff(thumbnail, last_thumbnail)) < 3:
                status = "duplicate"
            else:
                status = "accepted"
                name = f"frame_{frame_index - 1:06d}.png"
                written, encoded = cv2.imencode(".png", image)
                if not written:
                    raise ValueError(f"Could not encode video frame {frame_index - 1}.")
                (destination / name).write_bytes(encoded.tobytes())
                last_thumbnail = thumbnail
            records.append(VideoFrame(round(timestamp, 6), name, score, status))
        if not records:
            raise ValueError(f"No decodable frames in the video: {source}")
        return records
    finally:
        capture.release()
