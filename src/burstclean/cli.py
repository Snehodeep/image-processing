"""Command line: burstclean camera | video | files ... -o out.png"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import cv2

from . import __version__
from .capture import burst_from_files, burst_from_video, capture_burst
from .pipeline import enhance


def _region(value: str):
    if value in ("auto", "full"):
        return value
    try:
        x, y, w, h = (int(v) for v in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("region must be auto, full or x,y,w,h") from None
    return x, y, w, h


def _resolution(value: str):
    if value == "max":
        return "max"
    if value == "default":
        return None
    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("resolution must be max, default or WxH such as 1920x1080") from None
    return w, h


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-o", "--output", required=True, help="where to write the enhanced image (.png or .jpg)")
    common.add_argument("--keep", type=int, default=5, help="sharpest frames to keep and merge (default 5)")
    common.add_argument("--region", type=_region, default="auto",
                        help="auto (find a document), full, or x,y,w,h (default auto)")
    common.add_argument("--reference", metavar="PATH",
                        help="also write the same region from the sharpest single frame, for comparison")
    common.add_argument("--max-shift", type=float, default=0.02,
                        help="largest shift merged, as a fraction of the region's long edge (default 0.02)")
    common.add_argument("--min-response", type=float, default=0.2,
                        help="smallest phase-correlation peak merged, 0..1 (default 0.2)")

    p = argparse.ArgumentParser(prog="burstclean",
                                description="Turn a burst of noisy camera frames into one clean image.")
    p.add_argument("--version", action="version", version=f"burstclean {__version__}")
    sub = p.add_subparsers(dest="source", required=True)

    cam = sub.add_parser("camera", parents=[common], help="record a short burst from any camera or stream")
    cam.add_argument("--camera", default="0", help="camera index or stream URL (default 0)")
    cam.add_argument("--seconds", type=float, default=2.0, help="burst length (default 2)")
    cam.add_argument("--resolution", type=_resolution, default="max",
                     help="max (the camera's largest mode), default (the camera's own default) or WxH "
                          "such as 1920x1080 (default max)")
    cam.add_argument("--warmup", type=float, default=1.0, help="seconds dropped while exposure settles (default 1)")

    vid = sub.add_parser("video", parents=[common], help="use the sharpest frames of a video file")
    vid.add_argument("path")
    vid.add_argument("--start", type=float, default=0.0, help="start time in seconds (default 0)")
    vid.add_argument("--seconds", type=float, default=None, help="length to read (default: to the end)")

    files = sub.add_parser("files", parents=[common], help="merge several photos of the same scene")
    files.add_argument("paths", nargs="+")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.source == "camera":
            camera = int(args.camera) if args.camera.isdigit() else args.camera
            frames = capture_burst(camera, args.seconds, args.keep, resolution=args.resolution, warmup=args.warmup)
        elif args.source == "video":
            frames = burst_from_video(args.path, args.keep, start=args.start, seconds=args.seconds)
        else:
            frames = burst_from_files(args.paths, args.keep)
        result = enhance(frames, args.region, sort=False,
                         max_shift=args.max_shift, min_response=args.min_response)
    except (RuntimeError, ValueError) as e:
        print(f"burstclean: {e}", file=sys.stderr)
        return 1

    if not cv2.imwrite(args.output, result.image):
        print(f"burstclean: could not write {args.output}", file=sys.stderr)
        return 1
    if args.reference and not cv2.imwrite(args.reference, result.reference):
        print(f"burstclean: could not write {args.reference}", file=sys.stderr)
        return 1

    x, y, w, h = result.box
    where = {True: "document found", False: "no document found, whole frame", None: "region"}[result.document_found]
    print(f"merged {result.used} of {result.total} frames · {w}x{h} at ({x}, {y}) · {where} -> {args.output}")
    for i, s in enumerate(result.shifts, start=2):
        if not s.used:
            print(f"  sharpest #{i} left out: {s.reason} (shift {s.dx:+.1f}, {s.dy:+.1f} px, "
                  f"response {s.response:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
