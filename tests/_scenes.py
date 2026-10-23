"""Synthetic camera scenes shared by the tests."""
import cv2
import numpy as np

CARDBOARD = (40, 95, 150)    # saturated brown, BGR
PAPER = (235, 235, 235)


def desk_frame(scale: float = 1.0):
    """A parcel on a desk at 4K (scaled by `scale`): a bright text-free wall above a parcel whose
    label has printed text and a barcode sticker touching its bottom edge.
    Returns the frame, the label box and the sticker box as (x1, y1, x2, y2)."""
    img = np.full((2160, 3840, 3), CARDBOARD, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (3839, 300), (225, 228, 230), -1)              # wall: bright, no text
    label = (1500, 850, 2800, 1650)
    cv2.rectangle(img, label[:2], label[2:], PAPER, -1)
    for i, line in enumerate(["Order Number - #ORD10042", "To: Customer Name Here",
                              "Near College Main Road", "560001 Bengaluru KA India",
                              "Phone: 0000000000", "From: Example Books"]):
        cv2.putText(img, line, (1580, 960 + i * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 3)
    sticker = (1950, 1600, 2400, 1800)
    cv2.rectangle(img, sticker[:2], sticker[2:], PAPER, -1)
    for x in range(1980, 2370, 14):
        cv2.rectangle(img, (x, 1620), (x + 6, 1720), (10, 10, 10), -1)
    cv2.putText(img, "EX000000001IN", (2010, 1780), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        label = tuple(int(v * scale) for v in label)
        sticker = tuple(int(v * scale) for v in sticker)
    return img, label, sticker


def noisy(img: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def camera_view(scene: np.ndarray, dx: float = 0.0, dy: float = 0.0, factor: int = 4) -> np.ndarray:
    """What a camera with 1/`factor` of the scene's resolution sees when the scene moves by (dx, dy)
    camera pixels: an exact shift on the fine scene, then pixel binning. Unlike interpolating the
    small image, this is a true sub-pixel move. dx, dy must be multiples of 1/factor."""
    h, w = scene.shape[:2]
    m = np.float32([[1, 0, round(dx * factor)], [0, 1, round(dy * factor)]])
    moved = cv2.warpAffine(scene, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(moved, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def inside(inner, box) -> bool:
    x, y, w, h = box
    return x <= inner[0] and y <= inner[1] and inner[2] <= x + w and inner[3] <= y + h
