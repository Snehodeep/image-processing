import cv2
import numpy as np
import pytest

from burstclean import BestFrames, sharpness
from _scenes import desk_frame


def test_sharpness_prefers_sharp_frame():
    img, _, _ = desk_frame(0.5)
    blurred = cv2.GaussianBlur(img, (0, 0), 4)
    assert sharpness(img) > sharpness(blurred)


def test_sharpness_accepts_grey_and_tiny_images():
    img, _, _ = desk_frame(0.25)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert sharpness(gray) > 0
    assert sharpness(np.zeros((4, 4), np.uint8)) == 0.0


def test_best_frames_keeps_the_sharpest_in_order():
    img, _, _ = desk_frame(0.25)
    blurs = [0, 6, 1, 4, 2, 8]                      # sigma 0 = the sharp original
    best = BestFrames(keep=3)
    for s in blurs:
        best.add(img if s == 0 else cv2.GaussianBlur(img, (0, 0), s))
    assert len(best) == 3 and best.seen == 6
    assert best.scores == sorted(best.scores, reverse=True)
    assert np.array_equal(best.frames[0], img)
    expected = sorted((sharpness(img if s == 0 else cv2.GaussianBlur(img, (0, 0), s)) for s in blurs),
                      reverse=True)[:3]
    assert best.scores == pytest.approx(expected)


def test_best_frames_rejects_zero_keep():
    with pytest.raises(ValueError):
        BestFrames(keep=0)
