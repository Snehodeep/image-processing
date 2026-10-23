# burstclean

> Turn 2 seconds of video from a cheap camera into one clean, readable photo of a bill, receipt, payslip or label.

[![PyPI](https://img.shields.io/pypi/v/burstclean)](https://pypi.org/project/burstclean/)
[![Python](https://img.shields.io/pypi/pyversions/burstclean)](https://pypi.org/project/burstclean/)
[![CI](https://github.com/Snehodeep/image-processing/actions/workflows/ci.yml/badge.svg)](https://github.com/Snehodeep/image-processing/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Snehodeep/image-processing/blob/main/LICENSE)

burstclean records a short burst, keeps the sharpest frames, lines them up and averages them. Grain and compression speckle fade, and the print stays sharp.

![One webcam frame next to 8 frames merged by burstclean](https://raw.githubusercontent.com/Snehodeep/image-processing/main/docs/before_after.png)

```bash
pip install burstclean
```

```python
import cv2, burstclean as bc
result = bc.enhance(bc.capture_burst(camera=0))   # record 2 s, find the page, align, average
cv2.imwrite("clean.png", result.image)
```

- **Any camera:** USB webcam, laptop camera, document camera, phone stream, video file or a few photos.
- **Any printed page:** bills, receipts, invoices, payslips, labels and forms, upright, tilted or sideways.
- **Measured gain:** +3.8 to +5.8 dB with 5 frames on a fixed camera, +0.5 to +2.9 dB when it shakes ([benchmark](https://github.com/Snehodeep/image-processing#results)).
- **Small:** only NumPy and OpenCV; about 0.1 s for 5 frames at 1080p.

## Why

Bills, receipts, invoices, payslips, parcel labels and forms are often read through whatever camera is at hand: a cheap USB webcam, a laptop camera, a phone. Their small sensors make every indoor frame grainy, and the stream is compressed on top. Small print such as amounts, account numbers, dates and tracking numbers gets hard to read, for people and for OCR or vision models alike. A single snapshot is also a gamble: it may land on a moment of focus hunting or a shaky hand.

A short video gives you many looks at the same page. Sensor noise is random from frame to frame, but the page stays the same. Aligning N frames and averaging them cuts the noise by about √N, and the edges stay sharp. Phone cameras do the same thing in their night and HDR modes. burstclean does it with plain OpenCV, so you get a better input for OCR without buying a better camera.

It works with any source OpenCV can open:

- a USB webcam
- a laptop's built-in camera
- a document camera
- a phone streaming over Wi-Fi (an IP-camera app gives you a URL)
- a recorded video, or a handful of photos

## Install

```bash
pip install burstclean
```

For the latest code from GitHub: `pip install git+https://github.com/Snehodeep/image-processing.git`. It needs Python 3.9 or newer. It installs `numpy` and `opencv-python-headless`.

## Quick start

```python
import cv2
import burstclean as bc

frames = bc.capture_burst(camera=0, seconds=2)   # the 5 sharpest frames, sharpest first
result = bc.enhance(frames)                      # finds the document, aligns, averages

cv2.imwrite("clean.png", result.image)           # merged, at full camera resolution
cv2.imwrite("single.png", result.reference)      # same region from the sharpest single frame
print(f"merged {result.used} of {result.total} frames, region {result.box}")
```

You can use other frame sources:

```python
frames = bc.capture_burst("http://192.168.1.20:8080/video")        # a phone running an IP-camera app
frames = bc.burst_from_video("clip.mp4", start=3.0, seconds=2)      # a still stretch of a video
frames = bc.burst_from_files(["a.jpg", "b.jpg", "c.jpg"])            # several photos of the same page

best = bc.BestFrames(keep=5)          # your own capture loop: only the 5 sharpest are kept in memory
for frame in my_frames():
    best.add(frame)
result = bc.enhance(best.frames)
```

From the command line:

```bash
burstclean camera -o clean.png --reference single.png      # 2 s from camera 0
burstclean camera --camera 1 --resolution 1920x1080 -o clean.png
burstclean video clip.mp4 --start 3 --seconds 2 -o clean.png
burstclean files a.jpg b.jpg c.jpg -o clean.png --region full
```

The command prints how many frames were merged and why any were left out. In this run of five photos of a payslip, one was taken after the page had been nudged 70 px:

```
$ burstclean files shot*.jpg -o payslip.png
merged 4 of 5 frames · 748x962 at (586, 59) · document found -> payslip.png
  sharpest #3 left out: moved (shift +70.0, +3.0 px, response 0.58)
```

## How it works

1. **Capture a burst.** Record about 2 s of video. burstclean asks the camera for its largest mode, because small print is only readable with enough pixels. A 720p webcam gives 720p and a 4K camera gives 4K. The first second is dropped while auto exposure and focus settle.
2. **Keep the sharpest frames.** Each frame is scored by the variance of the Laplacian on a half-size grey copy. Sharp, still frames score high; motion blur and focus hunting score low. Only the top 5 are kept, so even 2 s of 4K (about 1.5 GB) is never held in memory.
3. **Find the document.** By default (`region="auto"`), burstclean looks in the sharpest frame for a printed page:
    - a bright, low-saturation paper area holding dark text lines at any angle, so a page that is tilted or lying sideways is found too;
    - extended to the page's outer edge, so a coloured header, a logo band or a photo on the page stays in.

    Only that box is merged, cut at full camera resolution. When no document is found, the whole frame is used. You can also pass `"full"` or your own `(x, y, w, h)`.
4. **Align and average.** Each other frame's region is aligned to the sharpest one by phase correlation, which measures the shift to about a quarter of a pixel. The frame is moved back with cubic interpolation, then everything is averaged. A frame is left out when:
    - its correlation peak is below 0.2, meaning the content is different (a hand in view), or
    - it moved more than 2% of the region, meaning the page was still being placed.

## Results

[`examples/benchmark.py`](https://github.com/Snehodeep/image-processing/blob/main/examples/benchmark.py) simulates a cheap webcam looking at three documents: a payslip, a shop receipt and a parcel label.

- 960×540 camera
- sensor noise σ = 10 grey levels
- MJPEG quality 75
- 5 trials per document

It measures each result against the noise-free view as PSNR, in dB; higher is better. A single frame scores **31.4 dB** on average.

| Frames | burstclean, fixed camera | burstclean, shake ≤ 0.5 px | burstclean, shake ≤ 2 px | Plain average, shake ≤ 2 px |
| --- | --- | --- | --- | --- |
| 1 | 31.4 | 31.4 | 31.4 | 31.4 |
| 2 | 33.7 | 32.1 | 32.9 | 23.8 |
| 3 | 34.8 | 32.3 | 34.1 | 22.0 |
| 5 | 35.9 | 31.9 | 34.0 | 20.4 |
| 8 | 36.8 | 32.2 | 34.3 | 21.1 |

- **A fixed camera gains the most:** +4.5 dB with 5 frames and +5.4 dB with 8. At +4.5 dB, the squared error falls to about a third of a single frame's. By document, 5 frames give: payslip +5.8 dB, receipt +4.1 dB, parcel label +3.8 dB. A camera on a stand or a phone propped over a desk is the best setup.
- **When the frames shake**, each one has to be moved by a fraction of a pixel, and that interpolation softens it a little. The gain is smaller (+0.5 to +2.9 dB) but still positive.
- **Alignment is what makes averaging work.** Averaging the same shaky frames without it smears the print and loses 8 to 11 dB.

Run it yourself with `python examples/benchmark.py`. Add `--save docs` to write the before/after picture.

## API

| Name | What it does |
| --- | --- |
| `capture_burst(camera=0, seconds=2.0, keep=5, *, resolution="max", warmup=1.0)` | Records from a camera index or stream URL; returns the `keep` sharpest frames, sharpest first |
| `burst_from_video(path, keep=5, *, start=0.0, seconds=None)` | Same, from a video file |
| `burst_from_files(paths, keep=5)` | Same, from still images |
| `BestFrames(keep=5)` | `.add(frame)` scores and keeps the sharpest; `.frames`, `.scores`, `.seen` |
| `enhance(frames, region="auto", *, sort=True, max_shift=0.02, min_response=0.2)` | The full pipeline. Returns `EnhanceResult` with `image`, `reference`, `box`, `used`, `total`, `document_found`, `shifts` |
| `merge(frames, box=None, *, max_shift=0.02, min_response=0.2)` | Aligns and averages one region against `frames[0]`. Returns `MergeResult` with `image`, `used`, `box`, `shifts` |
| `find_document(img, *, pad=0.03)` | `(x, y, w, h)` of the printed page in a BGR frame, or `None` |
| `sharpness(img)` | Laplacian-variance focus score |

`shifts` holds one `FrameShift(dx, dy, response, used, reason)` per frame after the reference, so you can see what was dropped and why. Any image dtype works: `merge` keeps uint8, uint16 or float, grey or colour.

## Tuning

| Setting | Default | When to change it |
| --- | --- | --- |
| `keep` | 5 | Raise it for a fixed camera in dim light: more frames, less noise. Lower it if memory or time is tight. |
| `seconds` | 2.0 | Raise it if the camera delivers few frames per second. Lower it if the page is not held still that long. |
| `resolution` | `"max"` | Pass `(width, height)` for a specific mode, for example when the largest mode is slow to stream, or `None` for the camera's own default. |
| `max_shift` | 0.02 | Raise it for a handheld camera whose frames are dropped as "moved". Lower it if you see doubled edges. |
| `min_response` | 0.2 | Raise it if frames with something else in view are getting merged. Lower it if good frames are dropped as "low_response". |

## Limits

- **Translation only.** Rotation, zoom and perspective change between frames are not modelled. They are small for a camera on a stand or a propped phone. Frames where they are large fail the shift or response check and are left out.
- **The page must be still** for the burst. Moving parts inside the region blur.
- **It needs some detail to align on.** In a nearly featureless region (a blank wall) under heavy noise, the correlation peak is weak. Frames are then left out as `low_response`, so less noise is removed, but the result is never worse than the single sharpest frame. Printed text gives plenty to align on.
- **The document finder is a heuristic.** It expects light paper with dark print, at any angle, against a background that differs in brightness or colour. The box is axis-aligned, so a tilted page comes with some background. For coloured paper, or a white page on a white desk, it may find nothing and merge the whole frame instead. Pass the region yourself if you know it.
- **Speed (Apple M1, 5 frames, document region):** 63 ms at 720p, 100 ms at 1080p, 0.35 s at 4K. Merging the whole frame takes 90 ms, 0.3 s and 1.1 s. Scoring sharpness takes about 8 ms per 4K frame.
- **No sharpening or contrast changes.** The output is the camera's own image with the noise averaged out, which is usually what OCR and vision models read best.

A note for anyone using `cv2.phaseCorrelate` directly: it pads the input to an "optimal" FFT size, and when that size is odd (for example 372 → 375), it reports a 0.5 px shift between two identical images. burstclean measures shifts on a centred window whose size needs no padding, so this does not happen.

## Development

```bash
git clone https://github.com/Snehodeep/image-processing.git
cd image-processing
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](https://github.com/Snehodeep/image-processing/blob/main/LICENSE).
