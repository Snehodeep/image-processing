"""Collect the sharpest frames of a short burst from a camera, a video file or image files."""
from __future__ import annotations

import time
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from .select import BestFrames

MAX_FAILED_READS = 30
MAX_REQUEST = (10000, 10000)   # drivers answer an oversized request with their largest mode

Resolution = Union[str, Tuple[int, int], None]


def _open_camera(camera: Union[int, str], resolution: Resolution) -> cv2.VideoCapture:
    if isinstance(resolution, str) and resolution != "max":
        raise ValueError(f"resolution must be 'max', None or (width, height), not {resolution!r}")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera!r}")
    if resolution is None:
        return cap
    w, h = MAX_REQUEST if resolution == "max" else resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frame = cap.read()
    if ok and frame is not None:
        return cap
    # A few drivers refuse a mode they do not have instead of picking the nearest: use the default
    cap.release()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera!r}")
    return cap


def capture_burst(camera: Union[int, str] = 0, seconds: float = 2.0, keep: int = 5, *,
                  resolution: Resolution = "max", warmup: float = 1.0) -> List[np.ndarray]:
    """Record `seconds` of video from `camera` and return the `keep` sharpest frames, sharpest first.

    Works with any source OpenCV can open: a camera index (USB webcam, laptop camera, document
    camera) or a stream URL (for example a phone running an IP-camera app).

    Small print is readable only with enough camera pixels, so `resolution="max"` asks the camera
    for its largest mode. Pass (width, height) to ask for a specific mode, or None to keep the
    camera's default. The first `warmup` seconds are dropped while auto exposure and focus settle.
    Hold the subject still: frames that moved are left out when they are merged.
    """
    cap = _open_camera(camera, resolution)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # always read the newest frame
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        best = BestFrames(keep)
        failed = 0
        warm_end = time.monotonic() + warmup
        end = warm_end + seconds
        while time.monotonic() < end:
            ok, frame = cap.read()
            if not ok or frame is None:
                failed += 1
                if failed > MAX_FAILED_READS:
                    break
                continue
            failed = 0
            if time.monotonic() >= warm_end:
                best.add(frame)
    finally:
        cap.release()

    if not len(best):
        raise RuntimeError(f"camera {camera!r} returned no frames")
    return best.frames


def burst_from_video(path: str, keep: int = 5, *, start: float = 0.0,
                     seconds: Optional[float] = None) -> List[np.ndarray]:
    """The `keep` sharpest frames of a video file, sharpest first.

    Reads from `start` seconds for `seconds` seconds (None = to the end). Use a short stretch where
    the subject is still, such as the 2 s after it is set down.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path!r}")
    try:
        # Seek and stop by frame number: some OpenCV builds report CAP_PROP_POS_MSEC wrongly after a seek
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps > 0:
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
            remaining = None if seconds is None else max(1, round(seconds * fps))
        else:
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
            remaining = None
        stop_ms = None if seconds is None or fps > 0 else (start + seconds) * 1000.0
        best = BestFrames(keep)
        while remaining is None or remaining > 0:
            if stop_ms is not None and cap.get(cv2.CAP_PROP_POS_MSEC) >= stop_ms:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            best.add(frame)
            if remaining is not None:
                remaining -= 1
    finally:
        cap.release()

    if not len(best):
        raise RuntimeError(f"no frames read from {path!r}")
    return best.frames


def burst_from_files(paths: Iterable[str], keep: int = 5) -> List[np.ndarray]:
    """The `keep` sharpest of several still images of the same scene, sharpest first."""
    best = BestFrames(keep)
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"could not read image {path!r}")
        best.add(frame)
    if not len(best):
        raise ValueError("no image files given")
    return best.frames
