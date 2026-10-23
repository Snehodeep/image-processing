"""Focus / motion-blur score used to rank burst frames."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """Single-channel copy of a BGR, BGRA or already grey image."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.shape[2] == 1:
        return img[..., 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian on a half-size grey copy. Higher is sharper.

    Edges give strong Laplacian responses, so a focused, still frame scores high and a frame with
    motion blur or focus hunting scores low. Halving first keeps it fast on 4K frames and lets
    pixel-level sensor noise count for less than real edges.
    """
    gray = to_gray(img)
    if min(gray.shape[:2]) >= 8:
        gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
