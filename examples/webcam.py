"""Capture a 2-second burst from a webcam and save the clean image next to a single frame.

    python examples/webcam.py              # camera 0
    python examples/webcam.py 1 --full     # camera 1, merge the whole frame instead of finding a document

Hold the label or page still under the camera while it records.
"""
import argparse

import cv2
import numpy as np

import burstclean as bc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("camera", nargs="?", type=int, default=0)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--full", action="store_true", help="merge the whole frame")
    args = p.parse_args()

    print(f"Recording {args.seconds:g} s from camera {args.camera}, hold still...")
    frames = bc.capture_burst(args.camera, args.seconds)
    h, w = frames[0].shape[:2]
    print(f"Camera gave {w}x{h}; kept the {len(frames)} sharpest frames")

    result = bc.enhance(frames, "full" if args.full else "auto", sort=False)
    x, y, rw, rh = result.box
    if result.document_found is False:
        print("No document found, merged the whole frame")
    print(f"Merged {result.used} of {result.total} frames, region {rw}x{rh} at ({x}, {y})")
    for i, s in enumerate(result.shifts, start=2):
        if not s.used:
            print(f"  sharpest #{i} left out: {s.reason}")

    cv2.imwrite("burst_single.png", result.reference)
    cv2.imwrite("burst_clean.png", result.image)
    gap = np.full((rh, 12) + result.image.shape[2:], 255, result.image.dtype)
    cv2.imwrite("burst_compare.png", np.hstack([result.reference, gap, result.image]))
    print("Wrote burst_single.png, burst_clean.png and burst_compare.png (left: single, right: merged)")


if __name__ == "__main__":
    main()
