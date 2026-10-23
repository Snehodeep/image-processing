"""Find a printed paper document (bill, receipt, payslip, label, form) in a camera frame."""
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
MIN_TEXT_PX = 300               # text-line pixels (at DETECT_EDGE), after that damping, needed to
                                # accept a document; real pages score 10x or more above it


def _rect(w: float, h: float) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, round(w)), max(1, round(h))))


def _detect_copy(img: np.ndarray, edge: int) -> Tuple[np.ndarray, float]:
    f = min(1.0, edge / float(max(img.shape[:2])))
    if f >= 1.0:
        return img, 1.0
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), f


def _text_lines(runs: np.ndarray, u: float) -> np.ndarray:
    """Mask of the line-shaped components of `runs` (joined letters): long, thin and solid.

    Length and thickness come from each component's second moments along its own axis, not from its
    upright bounding box, so the lines of a tilted page still count."""
    n, lab, st, cent = cv2.connectedComponentsWithStats(runs)
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    count = np.maximum(np.bincount(ids, minlength=n), 1).astype(np.float64)
    dx, dy = xs - cent[ids, 0], ys - cent[ids, 1]
    sxx = np.bincount(ids, dx * dx, n) / count
    syy = np.bincount(ids, dy * dy, n) / count
    sxy = np.bincount(ids, dx * dy, n) / count
    half, spread = (sxx + syy) / 2, np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    # A solid bar of length L and thickness t has variances L^2/12 and t^2/12 along its axes
    length = np.sqrt(12 * (half + spread))
    thick = np.sqrt(12 * np.maximum(half - spread, 0) + 1)
    area = st[:, cv2.CC_STAT_AREA]
    is_line = (length >= 3 * thick) & (length >= 8 * u) & (thick <= 30 * u) & (area >= 0.3 * length * thick)
    is_line[0] = False
    return is_line[lab].astype(np.uint8)


MAX_OUTLINE_GROWTH = 1.6        # an outline may add up to this factor to the paper box's area


def _outline(bgr: np.ndarray, box: Box, u: float) -> Optional[Box]:
    """The document's outer edge around the paper box, when one is clearly visible. Parts of a
    document that are not paper-coloured (a coloured header, a logo band, a photo) are left out of
    the paper region; the edge against the background brings them back in."""
    x, y, w, h = box
    # Edges per colour channel: a blue header on a dark desk differs in colour more than in grey
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    edges = np.zeros(bgr.shape[:2], np.uint8)
    for ch in cv2.split(blurred):
        edges |= cv2.Canny(ch, 30, 90)
    edges = cv2.dilate(edges, _rect(3 * u, 3 * u))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    slack = 4 * u
    best = None
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        contains = (cx <= x + slack and cy <= y + slack and cx + cw >= x + w - slack and cy + ch >= y + h - slack)
        if contains and cw * ch <= MAX_OUTLINE_GROWTH * w * h and (best is None or cw * ch < best[2] * best[3]):
            best = (cx, cy, cw, ch)
    if best is None:
        return None
    # Never shrink: the outline only adds what lies outside the paper box
    x1, y1 = min(x, best[0]), min(y, best[1])
    x2, y2 = max(x + w, best[0] + best[2]), max(y + h, best[1] + best[3])
    return x1, y1, x2 - x1, y2 - y1


def find_document(img: np.ndarray, *, pad: float = 0.03, min_text_px: int = MIN_TEXT_PX) -> Optional[Box]:
    """Box (x, y, w, h) of the bright, low-saturation paper region holding the most printed text
    lines, in full-resolution frame pixels, or None when no such region is found.

    Works for bills, receipts, invoices, payslips, labels and forms: light paper with dark print, at
    any angle, against a background that differs in brightness or colour (a desk, a table, a parcel).
    The box reaches the page's outer edge, so a coloured header or a logo band stays in. A bright
    blank wall is not picked: it has paper-like colour but no text lines. `img` is BGR; `pad` adds
    that fraction of the box on each side.
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

    # Letters -> words/lines; keep only line-shaped runs so specks, edges and glints do not count as text.
    # Letters are joined along both axes, so a page lying sideways is found too.
    text = np.zeros_like(strokes)
    for kernel in (_rect(6 * u, 1), _rect(1, 6 * u)):
        text |= _text_lines(cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, kernel), u)

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
            # Sparse "text" on a big bright area is a wall with a few edges on it, not a document
            score = text_px * min(1.0, (text_px / area) / MIN_TEXT_DENSITY)
            if score >= min_text_px and score > best_score:
                best, best_score = (x, y, cw, ch), score

    if best is None:
        return None

    x, y, cw, ch = _outline(small, best, u) or best
    px, py = cw * pad, ch * pad
    x1, y1 = max(0, int((x - px) / f)), max(0, int((y - py) / f))
    x2, y2 = min(W, int(np.ceil((x + cw + px) / f))), min(H, int(np.ceil((y + ch + py) / f)))
    return x1, y1, x2 - x1, y2 - y1
