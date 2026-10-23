"""Multi-frame noise reduction: align each frame to a reference and average."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .sharpness import to_gray

Box = Tuple[int, int, int, int]  # x, y, w, h

MIN_REGION = 32  # regions smaller than this on either side are returned unmerged


@dataclass
class FrameShift:
    """How one frame lined up against the reference."""
    dx: float
    dy: float
    response: float     # phase-correlation peak, 0..1; low = the content does not match
    used: bool
    reason: str = ""    # why it was left out: "size", "low_response", "moved", "region_too_small"


@dataclass
class MergeResult:
    image: np.ndarray                 # the averaged region, same dtype as the input
    used: int                         # frames averaged, the reference included
    box: Box                          # the region, clipped to the frame
    shifts: List[FrameShift] = field(default_factory=list)  # one per frame after the reference


def _clip_box(box: Optional[Sequence[int]], shape: Tuple[int, ...]) -> Box:
    H, W = shape[:2]
    if box is None:
        return 0, 0, W, H
    x, y, w, h = (int(round(v)) for v in box)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box {tuple(box)} lies outside the {W}x{H} frame")
    return x1, y1, x2 - x1, y2 - y1


def _even_dft_size(n: int) -> int:
    """Largest even n' <= n that the FFT needs no padding for. cv2.phaseCorrelate pads to
    getOptimalDFTSize, and when that size is odd its peak is off by half a pixel (two identical
    images report a 0.5 px shift), so shifts are measured on a window of this size instead."""
    m = n - (n % 2)
    while m > 2 and cv2.getOptimalDFTSize(m) != m:
        m -= 2
    return m


def _to_dtype(acc: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.floor(acc + 0.5), info.min, info.max).astype(dtype)
    return acc.astype(dtype)


def merge(frames: Sequence[np.ndarray], box: Optional[Sequence[int]] = None, *,
          max_shift: float = 0.02, min_response: float = 0.2) -> MergeResult:
    """Average the same region of several frames after aligning each one to `frames[0]`.

    Sensor noise is random from frame to frame while the scene is not, so averaging N aligned
    frames cuts noise by about sqrt(N) and leaves edges sharp. `frames[0]` is the reference: pass
    the sharpest frame first (`BestFrames` and `enhance` do this for you).

    Each frame's region is aligned by phase correlation, which measures a sub-pixel translation.
    A frame is left out when its correlation peak is below `min_response` (different content, such
    as a hand in view) or when it moved more than `max_shift` times the region's long edge (the
    object was still being placed). Frames of a different size are skipped.

    `box` is (x, y, w, h) in frame pixels; None means the whole frame.
    """
    if len(frames) == 0:
        raise ValueError("merge() needs at least one frame")
    ref_frame = frames[0]
    x, y, w, h = _clip_box(box, ref_frame.shape)
    ref = ref_frame[y:y + h, x:x + w]

    if w < MIN_REGION or h < MIN_REGION:
        shifts = [FrameShift(0.0, 0.0, 0.0, False, "region_too_small") for _ in frames[1:]]
        return MergeResult(ref.copy(), 1, (x, y, w, h), shifts)

    # Shifts are measured on a centred window sized so phaseCorrelate does not pad (see _even_dft_size)
    cw, ch = _even_dft_size(w), _even_dft_size(h)
    ox, oy = (w - cw) // 2, (h - ch) // 2
    ref_g = to_gray(ref)[oy:oy + ch, ox:ox + cw].astype(np.float32)
    window = cv2.createHanningWindow((cw, ch), cv2.CV_32F)
    limit = max_shift * max(w, h)
    acc = ref.astype(np.float32)
    used = 1
    shifts: List[FrameShift] = []
    for frame in frames[1:]:
        if frame.shape != ref_frame.shape:
            shifts.append(FrameShift(0.0, 0.0, 0.0, False, "size"))
            continue
        crop = frame[y:y + h, x:x + w]
        crop_g = to_gray(crop)[oy:oy + ch, ox:ox + cw].astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(ref_g, crop_g, window)
        if response < min_response:
            shifts.append(FrameShift(dx, dy, response, False, "low_response"))
            continue
        if abs(dx) > limit or abs(dy) > limit:
            shifts.append(FrameShift(dx, dy, response, False, "moved"))
            continue
        if abs(dx) >= 0.25 or abs(dy) >= 0.25:
            # Cubic keeps print edges sharper than linear when moving by a fraction of a pixel
            m = np.float32([[1, 0, -dx], [0, 1, -dy]])
            crop = cv2.warpAffine(crop, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        acc += crop
        used += 1
        shifts.append(FrameShift(dx, dy, response, True))

    return MergeResult(_to_dtype(acc / used, ref.dtype), used, (x, y, w, h), shifts)
