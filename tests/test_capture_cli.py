import cv2
import numpy as np
import pytest

from burstclean import burst_from_files, burst_from_video, capture_burst, sharpness
from burstclean.cli import main
from _scenes import desk_frame, noisy


@pytest.fixture
def video(tmp_path):
    """A 20-frame MJPG clip of the desk scene where every 4th frame is sharp and the rest blurred."""
    img, _, _ = desk_frame(0.25)
    h, w = img.shape[:2]
    path = str(tmp_path / "burst.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    assert out.isOpened()
    for i in range(20):
        out.write(img if i % 4 == 0 else cv2.GaussianBlur(img, (0, 0), 3))
    out.release()
    return path


def test_burst_from_video_keeps_sharpest_first(video):
    frames = burst_from_video(video, keep=3)
    assert len(frames) == 3
    scores = [sharpness(f) for f in frames]
    assert scores == sorted(scores, reverse=True)
    blurred_score = sharpness(cv2.GaussianBlur(desk_frame(0.25)[0], (0, 0), 3))
    assert min(scores) > blurred_score * 1.5


def test_burst_from_video_window(video):
    best = burst_from_video(video, keep=50, start=1.0, seconds=0.5)      # frames 10-14 at 10 fps
    assert len(best) == 5


def test_capture_burst_reads_a_stream_source(video):
    frames = capture_burst(video, seconds=5, keep=4, resolution=None, warmup=0)
    assert 1 <= len(frames) <= 4


@pytest.mark.parametrize("resolution", ["max", (640, 480)])
def test_capture_burst_requested_resolution_falls_back(video, resolution):
    # A file ignores the size request, like a camera without that mode: frames still come through
    frames = capture_burst(video, seconds=5, keep=2, resolution=resolution, warmup=0)
    assert len(frames) == 2


def test_capture_burst_bad_source_and_resolution(video):
    with pytest.raises(RuntimeError):
        capture_burst("/no/such/file.avi", seconds=0.1, warmup=0)
    with pytest.raises(ValueError):
        capture_burst(video, resolution="4k")


def test_cli_files_writes_output_and_reference(tmp_path, capsys):
    img, _, _ = desk_frame(0.5)
    paths = []
    for s in range(3):
        p = tmp_path / f"shot{s}.png"
        cv2.imwrite(str(p), noisy(img, 10, s))
        paths.append(str(p))
    out, ref = tmp_path / "out.png", tmp_path / "ref.png"
    assert main(["files", *paths, "-o", str(out), "--reference", str(ref)]) == 0
    merged, single = cv2.imread(str(out)), cv2.imread(str(ref))
    assert merged is not None and merged.shape == single.shape
    assert "merged 3 of 3 frames" in capsys.readouterr().out
    assert burst_from_files(paths, keep=2).__len__() == 2


def test_cli_reports_errors(tmp_path, capsys):
    assert main(["files", str(tmp_path / "missing.png"), "-o", str(tmp_path / "o.png")]) == 1
    assert "could not read image" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["files", "a.png", "-o", "o.png", "--region", "1,2"])
