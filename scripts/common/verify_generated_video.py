import argparse
import hashlib
import os

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--expected-frames", type=int, required=True)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    count = 0
    black = 0
    hashes = set()
    means = []
    stds = []
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        black += int(frame.max() == 0)
        hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
        means.append(float(frame.mean()))
        stds.append(float(frame.std()))

    valid = (
        count == args.expected_frames
        and black == 0
        and len(hashes) == count
        and width == 832
        and height == 464
        and abs(fps - 16.0) < 1e-6
    )
    print(
        "VIDEO_VERIFY"
        f" path={args.video} valid={valid} frames={count}"
        f" expected={args.expected_frames} unique={len(hashes)} black={black}"
        f" size={width}x{height} fps={fps:.6g} bytes={os.path.getsize(args.video)}"
        f" mean_min={min(means):.6f} mean_max={max(means):.6f}"
        f" std_min={min(stds):.6f} std_max={max(stds):.6f}",
        flush=True,
    )
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
