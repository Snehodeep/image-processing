"""One call from a burst of frames to one clean image."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .document import find_document
from .merge import Box, FrameShift, merge
from .sharpness import sharpness

Region = Union[str, Sequence[int], None]


@dataclass
class EnhanceResult:
    image: np.ndarray                 # the merged region at full resolution
    reference: np.ndarray             # the same region from the sharpest single frame, for comparison
    box: Box                          # the region in the input frames (x, y, w, h)
    used: int                         # frames averaged, the reference included
    total: int                        # frames given
    document_found: Optional[bool] = None   # region="auto" only: whether a document was found
    shifts: List[FrameShift] = field(default_factory=list)


def enhance(frames: Sequence[np.ndarray], region: Region = "auto", *, sort: bool = True,
            max_shift: float = 0.02, min_response: float = 0.2) -> EnhanceResult:
    """Pick the sharpest frame as the reference, align the others to it and average them.

    `region`:
      - "auto": find a printed document (bill, receipt, payslip, label) in the sharpest frame and merge only
        that box; when none is found, the whole frame is merged.
      - "full" or None: the whole frame.
      - (x, y, w, h): that box, in frame pixels.

    `sort=False` keeps the given order and uses `frames[0]` as the reference.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("enhance() needs at least one frame")
    if sort and len(frames) > 1:
        frames = [f for _, f in sorted(((sharpness(f), f) for f in frames), key=lambda sf: sf[0], reverse=True)]

    found: Optional[bool] = None
    if isinstance(region, str):
        if region == "auto":
            box = find_document(frames[0]) if frames[0].ndim == 3 else None
            found = box is not None
        elif region == "full":
            box = None
        else:
            raise ValueError(f"region must be 'auto', 'full', None or (x, y, w, h), not {region!r}")
    else:
        box = None if region is None else tuple(region)

    result = merge(frames, box, max_shift=max_shift, min_response=min_response)
    x, y, w, h = result.box
    reference = frames[0][y:y + h, x:x + w].copy()
    return EnhanceResult(image=result.image, reference=reference, box=result.box, used=result.used,
                         total=len(frames), document_found=found, shifts=result.shifts)
