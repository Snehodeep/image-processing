import cv2
import numpy as np
import pytest

from burstclean import enhance, find_document
from _scenes import CARDBOARD, desk_frame, inside, noisy


def test_find_document_picks_text_not_bright_wall():
    img, label, sticker = desk_frame()
    box = find_document(img)
    assert box is not None
    assert inside(label, box), box
    assert inside(sticker, box), box          # barcode sticker touching the label is kept
    assert box[1] > 300                        # wall excluded


def test_find_document_none_on_blank_scene():
    img = np.full((2160, 3840, 3), CARDBOARD, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (3839, 600), (225, 228, 230), -1)
    assert find_document(img) is None


def test_enhance_auto_crops_document_at_full_resolution():
    img, label, _ = desk_frame()
    frames = [noisy(img, 10, s) for s in range(3)]
    result = enhance(frames)
    assert result.document_found is True
    assert inside(label, result.box)
    x, y, w, h = result.box
    assert result.image.shape[:2] == (h, w) == result.reference.shape[:2]
    assert result.used == 3 and result.total == 3


def test_enhance_auto_falls_back_to_whole_frame():
    img = np.full((540, 960, 3), CARDBOARD, dtype=np.uint8)
    result = enhance([img, img.copy()])
    assert result.document_found is False
    assert result.box == (0, 0, 960, 540)


def test_enhance_full_and_explicit_box():
    img, _, _ = desk_frame(0.25)
    assert enhance([img, img], "full").box == (0, 0, img.shape[1], img.shape[0])
    assert enhance([img, img], None).document_found is None
    assert enhance([img, img], (100, 50, 200, 120)).box == (100, 50, 200, 120)
    with pytest.raises(ValueError):
        enhance([img], "middle")


def test_enhance_uses_sharpest_frame_as_reference():
    img, _, _ = desk_frame(0.5)
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    result = enhance([blurred, img], "full")
    assert np.array_equal(result.reference, img)
    in_order = enhance([blurred, img], "full", sort=False)
    assert np.array_equal(in_order.reference, blurred)
