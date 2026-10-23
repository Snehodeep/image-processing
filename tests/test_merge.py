import numpy as np
import pytest

from burstclean import find_document, merge
from _scenes import camera_view, desk_frame, noisy


def _err(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def test_averaging_reduces_noise_and_drops_moved_frame():
    img, _, _ = desk_frame()
    frames = [noisy(img, 12, seed) for seed in range(4)]
    moved = np.roll(img, 400, axis=1)                         # parcel moved during the burst
    box = find_document(frames[0])
    result = merge(frames + [moved], box)
    assert result.used == 4
    assert [s.used for s in result.shifts] == [True, True, True, False]
    assert result.shifts[-1].reason in ("moved", "low_response")
    x, y, w, h = box
    clean = img[y:y + h, x:x + w]
    assert _err(result.image, clean) < _err(frames[0][y:y + h, x:x + w], clean) * 0.7


def test_shifted_frames_are_realigned():
    scene, _, _ = desk_frame()                               # camera sees it at 960x540
    box = (340, 190, 360, 230)
    moves = [(3.0, -2.0), (-1.5, 2.5), (0.75, 1.25), (0.5, 0.0)]
    ref_frame = camera_view(scene)
    frames = [ref_frame] + [camera_view(scene, dx, dy) for dx, dy in moves]
    result = merge(frames, box)
    assert result.used == 5
    for s, (dx, dy) in zip(result.shifts, moves):            # sub-pixel, to about a quarter pixel
        assert s.dx == pytest.approx(dx, abs=0.3) and s.dy == pytest.approx(dy, abs=0.3)
    x, y, w, h = box
    ref = ref_frame[y:y + h, x:x + w]
    unaligned = np.mean([f[y:y + h, x:x + w].astype(np.float32) for f in frames], axis=0)
    assert _err(result.image, ref) < _err(unaligned, ref) * 0.5


@pytest.mark.parametrize("width", [371, 372, 375, 401])
def test_identical_frames_report_no_shift_for_any_region_size(width):
    # cv2.phaseCorrelate alone reports 0.5 px here when the padded FFT size is odd (372 -> 375)
    img, _, _ = desk_frame(0.25)
    box = (314, 131, width, 288)
    result = merge([img, img.copy()], box)
    s = result.shifts[0]
    assert abs(s.dx) < 0.05 and abs(s.dy) < 0.05
    x, y, w, h = result.box
    assert np.array_equal(result.image, img[y:y + h, x:x + w])


def test_whole_frame_when_no_box_and_grey_input():
    img, _, _ = desk_frame(0.25)
    gray = img[..., 1].copy()
    frames = [noisy(gray, 10, s) for s in range(3)]
    result = merge(frames)
    assert result.box == (0, 0, gray.shape[1], gray.shape[0])
    assert result.image.shape == gray.shape and result.image.dtype == np.uint8
    assert result.used == 3


def test_other_size_frames_skipped_and_uint16_kept():
    img = (np.random.default_rng(0).random((200, 300)) * 60000).astype(np.uint16)
    result = merge([img, img.copy(), img[:100]])
    assert result.used == 2
    assert result.shifts[1].reason == "size"
    assert result.image.dtype == np.uint16
    assert np.array_equal(result.image, img)


def test_small_region_or_single_frame_is_returned_unmerged():
    img, _, _ = desk_frame(0.25)
    assert merge([img, img], (10, 10, 20, 20)).used == 1
    single = merge([img], (10, 10, 100, 100))
    assert single.used == 1 and np.array_equal(single.image, img[10:110, 10:110])


def test_box_is_clipped_and_outside_box_rejected():
    img, _, _ = desk_frame(0.25)
    h, w = img.shape[:2]
    assert merge([img], (w - 50, h - 50, 200, 200)).box == (w - 50, h - 50, 50, 50)
    with pytest.raises(ValueError):
        merge([img], (w + 10, 0, 50, 50))
    with pytest.raises(ValueError):
        merge([])
