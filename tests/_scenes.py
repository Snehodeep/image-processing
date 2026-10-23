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


WOOD, DARK_DESK, GREY_DESK, CLOTH = (60, 105, 160), (45, 45, 50), (175, 178, 180), (120, 60, 160)
_LINES = ["Basic salary      45,000.00", "HRA               18,000.00", "Total due     Rs 1,240.50",
          "Item  Qty  Rate  Amount", "Account no 0000 1111 2222", "Date 12/03/2026  Ref 88821"]


def document(w, h, size, *, margin=40, table=False, header=None):
    """A printed page w x h: lines of text, optional table rules and a coloured header band."""
    doc = np.full((h, w, 3), 240, np.uint8)
    step = int(size * 42)
    y = int(size * 40)
    if header:
        cv2.rectangle(doc, (0, 0), (w, int(h * 0.1)), header, -1)
        cv2.putText(doc, "INVOICE", (margin, int(h * 0.07)), cv2.FONT_HERSHEY_SIMPLEX, size * 1.6,
                    (255, 255, 255), max(2, int(size * 3)))
        y = int(h * 0.1) + step
    i = 0
    while y < h - step // 2:
        cv2.putText(doc, f"{i + 1:02d} {_LINES[i % len(_LINES)]}", (margin, y), cv2.FONT_HERSHEY_SIMPLEX, size, (25, 25, 25),
                    max(1, int(size * 2)))
        if table:
            cv2.line(doc, (margin - 10, y + step // 3), (w - 30, y + step // 3), (90, 90, 90), 1)
        y += step
        i += 1
    if table:
        cv2.line(doc, (w // 2, int(h * 0.12)), (w // 2, y - step // 2), (90, 90, 90), 1)
    return doc


def on_desk(doc, desk, center=(960, 540), angle=0.0, size=(1920, 1080)):
    """`doc` lying on a plain desk colour, rotated by `angle` degrees. Returns the frame and the
    document's bounding box as (x1, y1, x2, y2)."""
    W, H = size
    h, w = doc.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[0, 2] += center[0] - w / 2
    m[1, 2] += center[1] - h / 2
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (W, H)) > 127
    frame = np.full((H, W, 3), desk, np.uint8)
    frame[mask] = cv2.warpAffine(doc, m, (W, H))[mask]
    corners = cv2.transform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], m)[0]
    x1, y1 = np.floor(corners.min(0)).astype(int)
    x2, y2 = np.ceil(corners.max(0)).astype(int)
    return frame, (int(x1), int(y1), int(x2), int(y2))
