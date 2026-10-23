"""Find a printed paper document (label, receipt, page) in a camera frame."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h

DETECT_EDGE = 1280              # detection runs on a copy with this long edge, so thresholds do not
                                # depend on camera resolution; the box is scaled back to the full frame
PAPER_LEVELS = (200, 180, 160, 140, 120, 100)   # brightness levels tried for the paper
PAPER_MAX_SAT = 100             # paper is grey/white; cardboard, wood and tape are saturated
STROKE_CONTRAST = 28            # dark print against the paper around it
MIN_PAPER_AREA = 0.004          # paper region size, as a fraction of the frame
MAX_PAPER_AREA = 0.60
MIN_TEXT_DENSITY = 0.03         # sparser text than this is damped (large mostly-blank regions)
MIN_TEXT_PX = 300               # text-line pixels (at DETECT_EDGE) needed to accept a document


def _rect(w: float, h: float) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, round(w)), max(1, round(h))))


def _detect_copy(img: np.ndarray, edge: int) -> Tuple[np.ndarray, float]:
    f = min(1.0, edge / float(max(img.shape[:2])))
    if f >= 1.0:
        return img, 1.0
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f


def find_document(img: np.ndarray, *, pad: float = 0.03, min_text_px: int = MIN_TEXT_PX) -> Optional[Box]:
    """Box (x, y, w, h) of the bright, low-saturation paper region holding the most printed text
    lines, in full-resolution frame pixels, or None when no such region is found.

    Built for a light label or page with dark print, seen against a darker or coloured background
    (a parcel, a desk). A bright blank wall is not picked: it has paper-like colour but no text
    lines. `img` is BGR; `pad` adds that fraction of the box on each side.
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("find_document() needs a BGR colour image")
    img = img[..., :3]
    H, W = img.shape[:2]
    small, f = _detect_copy(img, DETECT_EDGE)
    h, w = small.shape[:2]
    u = max(h, w) / float(DETECT_EDGE)
    k = max(3, round(9 * u)) | 1

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Closing paints the dark print over, so the paper reads as one bright surface
    bright = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_CLOSE, _rect(k, k))
    low_sat = cv2.morphologyEx(hsv[..., 1], cv2.MORPH_OPEN, _rect(k, k)) < PAPER_MAX_SAT
    paper_gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, _rect(k, k))
    strokes = ((cv2.subtract(paper_gray, gray) > STROKE_CONTRAST) & low_sat).astype(np.uint8)

    # Letters -> words/lines; keep only line-shaped runs so specks, edges and glints do not count as text
    runs = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, _rect(6 * u, 1))
    _, lab, st, _ = cv2.connectedComponentsWithStats(runs)
    bw, bh, ba = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    is_line = (bw >= 3 * bh) & (bw >= 8 * u) & (bh <= 30 * u) & (ba >= 0.3 * bw * bh)
    is_line[0] = False
    text = is_line[lab].astype(np.uint8)

    best, best_score = None, 0.0
    for level in PAPER_LEVELS:
        paper = ((bright > level) & low_sat).astype(np.uint8)
        paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, _rect(15 * u, 15 * u))   # drop thin glints
        n, plab, pst, _ = cv2.connectedComponentsWithStats(paper)
        for i in range(1, n):
            x, y, cw, ch, area = (int(v) for v in pst[i])
            if not (MIN_PAPER_AREA * h * w <= area <= MAX_PAPER_AREA * h * w):
                continue
            text_px = int(text[y:y + ch, x:x + cw][plab[y:y + ch, x:x + cw] == i].sum())
            score = text_px * min(1.0, (text_px / area) / MIN_TEXT_DENSITY)
            if text_px >= min_text_px and score > best_score:
                best, best_score = (x, y, cw, ch), score

    if best is None:
        return None

    x, y, cw, ch = best
    px, py = cw * pad, ch * pad
    x1, y1 = max(0, int((x - px) / f)), max(0, int((y - py) / f))
    x2, y2 = min(W, int(np.ceil((x + cw + px) / f))), min(H, int(np.ceil((y + ch + py) / f)))
    return x1, y1, x2 - x1, y2 - y1
