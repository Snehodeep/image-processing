"""burstclean: turn a burst of noisy camera frames into one clean image.

Keep the sharpest frames of a short burst, align them to the sharpest one by phase correlation,
and average them. Sensor noise falls by about sqrt(N) while edges and print stay sharp.
"""
from .capture import burst_from_files, burst_from_video, capture_burst
from .document import find_document
from .merge import FrameShift, MergeResult, merge
from .pipeline import EnhanceResult, enhance
from .select import BestFrames
from .sharpness import sharpness

__version__ = "0.1.0"

__all__ = [
    "BestFrames",
    "EnhanceResult",
    "FrameShift",
    "MergeResult",
    "burst_from_files",
    "burst_from_video",
    "capture_burst",
    "enhance",
    "find_document",
    "merge",
    "sharpness",
]
