"""How much cleaner does a document get as more frames are merged?

Simulates a cheap webcam looking at three kinds of document (a payslip, a shop receipt and a
parcel label): the camera has a quarter of the scene's resolution, the camera or desk shakes by up
to a few pixels between frames, the sensor adds noise and the stream is MJPEG-compressed.
Compares, against the noise-free view of the document:
  - a single frame,
  - burstclean (aligned, then averaged),
  - a plain average of the same frames without alignment.

    python examples/benchmark.py                 # print the table
    python examples/benchmark.py --save docs     # also write a before/after picture
"""
import argparse
import os

import cv2
import numpy as np

import burstclean as bc

FACTOR = 4          # scene pixels per camera pixel: a 3840x2160 scene seen at 960x540


def page(w, h, lines, size, *, table=False):
    """A printed page with dark text lines and, optionally, table rules."""
    doc = np.full((h, w, 3), 238, np.uint8)
    step = int(size * 58)
    for i, line in enumerate(lines):
        y = int(size * 60) + i * step
        cv2.putText(doc, line, (int(size * 30), y), cv2.FONT_HERSHEY_SIMPLEX, size, (25, 25, 25), max(2, int(size * 2)))
        if table:
            cv2.line(doc, (int(size * 20), y + step // 3), (w - int(size * 20), y + step // 3), (110, 110, 110), 3)
    return doc


def on_desk(doc, desk):
    img = np.full((2160, 3840, 3), desk, np.uint8)
    h, w = doc.shape[:2]
    y, x = (2160 - h) // 2, (3840 - w) // 2
    img[y:y + h, x:x + w] = doc
    return img


def documents():
    payslip = page(1700, 1500, ["PAYSLIP  March 2026", "Employee  A. Kumar   ID 10042", "Basic salary     45,000.00",
                                "HRA              18,000.00", "Provident fund   -5,400.00", "Net pay          57,600.00",
                                "Account  0000 1111 2222"], 2.4, table=True)
    receipt = page(760, 1900, ["CITY MART", "12/03/2026 18:42", "Rice 5kg    410.00", "Milk 1L      56.00",
                               "Bread        45.00", "Eggs x12     84.00", "TOTAL       595.00", "UPI   Ref 88821",
                               "Thank you"], 1.7)
    label = page(1400, 1080, ["SPEED POST  EX000000001IN", "To: A. Kumar", "12 College Main Road",
                              "Bengaluru 560001  Karnataka", "Phone 90000 00000   COD Rs 450", "Wt 0.75 kg"], 2.6)
    return {"payslip": on_desk(payslip, (45, 45, 50)),            # dark desk
            "receipt": on_desk(receipt, (60, 105, 160)),          # wooden table
            "parcel label": on_desk(label, (40, 95, 150))}        # cardboard


def camera(sc, dx, dy, rng, sigma, jpeg):
    h, w = sc.shape[:2]
    m = np.float32([[1, 0, round(dx * FACTOR)], [0, 1, round(dy * FACTOR)]])
    view = cv2.warpAffine(sc, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    view = cv2.resize(view, (w // FACTOR, h // FACTOR), interpolation=cv2.INTER_AREA)
    if sigma:
        view = np.clip(view + rng.normal(0, sigma, view.shape), 0, 255).astype(np.uint8)
    if jpeg:
        _, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        view = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return view


def psnr(a, b) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def burst(sc, shake, n, rng, sigma, jpeg):
    """n frames; frame 0 is the reference position, the others shake by up to `shake` px."""
    steps = np.arange(-shake, shake + 1e-9, 1.0 / FACTOR) if shake else np.array([0.0])
    moves = [(0.0, 0.0)] + [(rng.choice(steps), rng.choice(steps)) for _ in range(n - 1)]
    return [camera(sc, dx, dy, rng, sigma, jpeg) for dx, dy in moves]


def save_picture(frames, directory):
    r = bc.enhance(frames, "auto", sort=False)
    _, _, w, h = r.box

    def zoom(im, caption):
        im = cv2.resize(im[h // 30:h // 30 + h * 2 // 5, w // 20:w // 20 + w * 3 // 5], None,
                        fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        bar = np.full((44, im.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, caption, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2, cv2.LINE_AA)
        return np.vstack([bar, im])

    a, b = zoom(r.reference, "1 frame"), zoom(r.image, f"{r.used} frames, burstclean")
    gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
    os.makedirs(directory, exist_ok=True)
    out = os.path.join(directory, "before_after.png")
    cv2.imwrite(out, np.hstack([a, gap, b]))
    print(f"wrote {out} (left: single frame, right: {r.used} frames merged)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=10.0, help="sensor noise, grey levels (default 10)")
    p.add_argument("--jpeg", type=int, default=75, help="stream JPEG quality, 0 = off (default 75)")
    p.add_argument("--shakes", type=float, nargs="+", default=[0.0, 0.5, 2.0],
                   help="largest shake between frames, px, one column each (default 0 0.5 2)")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--save", metavar="DIR", help="write before_after.png (single frame vs. 8 merged) here")
    args = p.parse_args()

    counts = (1, 2, 3, 5, 8)
    shake_max = max(args.shakes)
    docs = documents()
    single = []
    aligned = {(s, n): [] for s in args.shakes for n in counts}
    plain = {n: [] for n in counts}           # plain average at the largest shake
    per_doc = {name: [] for name in docs}     # fixed camera, 5 frames: gain over one frame
    for name, sc in docs.items():
        clean = camera(sc, 0, 0, None, 0, 0)
        for t in range(args.trials):
            for shake in args.shakes:
                frames = burst(sc, shake, max(counts), np.random.default_rng(t), args.sigma, args.jpeg)
                for n in counts:
                    r = bc.enhance(frames[:n], "auto", sort=False)       # frame 0 is the reference
                    assert r.document_found, f"{name}: document not found"
                    x, y, w, h = r.box
                    truth = clean[y:y + h, x:x + w]
                    aligned[shake, n].append(psnr(r.image, truth))
                    if n == 1 and shake == args.shakes[0]:
                        single.append(psnr(r.reference, truth))
                        first = single[-1]
                    if n == 5 and shake == 0.0:
                        per_doc[name].append(aligned[shake, n][-1] - first)
                    if shake == shake_max:
                        avg = np.mean([f[y:y + h, x:x + w].astype(np.float32) for f in frames[:n]], axis=0)
                        plain[n].append(psnr(np.clip(avg + 0.5, 0, 255).astype(np.uint8), truth))
                if args.save and name == "payslip" and t == 0 and shake == args.shakes[0]:
                    save_picture(frames, args.save)

    print(f"\nDocument region, camera 960x540, noise sigma {args.sigma:g}, JPEG {args.jpeg or 'off'}, "
          f"{len(docs)} documents x {args.trials} trials. PSNR against the noise-free view in dB, higher "
          f"is better. Single frame: {np.mean(single):.1f} dB.\n")
    head = " | ".join(f"burstclean, shake {s:g} px" for s in args.shakes)
    print(f"| Frames | {head} | Plain average, shake {shake_max:g} px |")
    print("| --- |" + " --- |" * (len(args.shakes) + 1))
    for n in counts:
        cells = " | ".join(f"{np.mean(aligned[s, n]):.1f}" for s in args.shakes)
        print(f"| {n} | {cells} | {np.mean(plain[n]):.1f} |")
    if 0.0 in args.shakes:
        print("\nFixed camera, 5 frames, gain over one frame: "
              + ", ".join(f"{name} +{np.mean(g):.1f} dB" for name, g in per_doc.items()))


if __name__ == "__main__":
    main()
