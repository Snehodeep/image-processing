import cv2
import numpy as np
import pytest

from burstclean import enhance, find_document
from _scenes import (CARDBOARD, CLOTH, DARK_DESK, GREY_DESK, WOOD, desk_frame, document, inside, noisy,
                     on_desk)

DOCUMENTS = {
    "thermal receipt": lambda: on_desk(document(300, 900, 0.55, margin=15), WOOD),
    "payslip filling most of the frame": lambda: on_desk(document(1000, 1000, 0.9, table=True), DARK_DESK),
    "invoice with a coloured header": lambda: on_desk(document(700, 900, 0.8, header=(160, 70, 20)), DARK_DESK),
    "bill lying sideways": lambda: on_desk(document(700, 500, 0.8), WOOD, angle=90),
    "bill tilted 5 degrees": lambda: on_desk(document(700, 500, 0.8), WOOD, angle=5),
    "bill tilted 20 degrees": lambda: on_desk(document(700, 500, 0.8), WOOD, angle=-20),
    "bill tilted 45 degrees": lambda: on_desk(document(700, 500, 0.8), WOOD, angle=45),
    "page on a light grey desk": lambda: on_desk(document(700, 900, 0.8, table=True), GREY_DESK),
    "small bill far from the camera": lambda: on_desk(document(240, 300, 0.3, margin=10), CLOTH, center=(600, 400)),
}


@pytest.mark.parametrize("name", DOCUMENTS)
def test_find_document_covers_whole_document(name):
    img, (x1, y1, x2, y2) = DOCUMENTS[name]()
    box = find_document(img)
    assert box is not None
    tol = 3                                    # anti-aliased edge pixels
    assert inside((x1 + tol, y1 + tol, x2 - tol, y2 - tol), box), box
    assert box[2] * box[3] <= 1.3 * (x2 - x1) * (y2 - y1), box     # no runaway growth


def test_find_document_picks_text_not_bright_wall():
    img, label, sticker = desk_frame()
    box = find_document(img)
    assert box is not None
    assert inside(label, box), box
    assert inside(sticker, box), box          # barcode sticker touching the label is kept
    assert box[1] > 300                        # wall excluded


def test_find_document_none_on_wall_with_scattered_edges():
    # A room: a big bright wall with a door frame, a shelf and a few thin dark marks. They are
    # line-shaped, but far too sparse for print, so this must not be taken for a document.
    img = np.full((720, 1280, 3), (70, 80, 95), np.uint8)
    cv2.rectangle(img, (100, 0), (1100, 480), (225, 228, 230), -1)                  # wall
    rng = np.random.default_rng(3)
    for _ in range(10):
        x, y = int(rng.integers(150, 1050)), int(rng.integers(30, 430))
        if rng.random() < 0.5:
            cv2.line(img, (x, y), (x, y + int(rng.integers(25, 60))), (60, 60, 60), 2)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(25, 60)), y), (60, 60, 60), 2)
    assert find_document(img) is None


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
