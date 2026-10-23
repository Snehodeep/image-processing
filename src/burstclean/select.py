"""Keep the sharpest few frames of a stream without holding the whole stream in memory."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .sharpness import sharpness


class BestFrames:
    """Streams frames in and keeps only the `keep` sharpest, sharpest first.

    Two seconds of video is about 60 frames (1.5 GB at 4K); scoring each frame as it arrives and
    dropping the rest keeps memory at `keep` frames.

        best = BestFrames(keep=5)
        for frame in frames:
            best.add(frame)
        frames = best.frames
    """

    def __init__(self, keep: int = 5):
        if keep < 1:
            raise ValueError("keep must be at least 1")
        self.keep = keep
        self.seen = 0
        self._best: List[Tuple[float, np.ndarray]] = []

    def add(self, frame: np.ndarray) -> float:
        """Score `frame`, keep it if it is among the sharpest so far, and return its score."""
        score = sharpness(frame)
        self.seen += 1
        self._best.append((score, frame))
        self._best.sort(key=lambda sf: sf[0], reverse=True)
        del self._best[self.keep:]
        return score

    @property
    def frames(self) -> List[np.ndarray]:
        """The kept frames, sharpest first."""
        return [frame for _, frame in self._best]

    @property
    def scores(self) -> List[float]:
        return [score for score, _ in self._best]

    def __len__(self) -> int:
        return len(self._best)
