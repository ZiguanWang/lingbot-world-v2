#!/usr/bin/env python3
"""Exercise a running interactive server and validate returned frame chunks."""

import argparse
import hashlib
import json
import urllib.request

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:7860/step")
    p.add_argument("--output", default="interactive_mvp_verify.mp4")
    args = p.parse_args()
    actions = [
        {"keys": ["w"], "mouse_dx": 18, "mouse_dy": -5},
        {"keys": ["w", "d"], "mouse_dx": -22, "mouse_dy": 8},
        {"keys": ["a"], "mouse_dx": 12, "mouse_dy": 0},
        {"keys": [], "mouse_dx": -16, "mouse_dy": -4},
    ]
    all_frames, timings, counts = [], [], []
    for action in actions:
        request = urllib.request.Request(
            args.url, data=json.dumps(action).encode(), method="POST",
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        chunk = []
        for encoded in result["frames"]:
            raw = __import__("base64").b64decode(encoded)
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Could not decode returned JPEG")
            chunk.append(frame)
        counts.append(len(chunk)); timings.append(result["timing"]); all_frames += chunk

    height, width = all_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), 16, (width, height))
    hashes, black = set(), 0
    for frame in all_frames:
        if frame.shape[:2] != (height, width):
            raise RuntimeError("Returned frame dimensions changed within session")
        hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
        black += int(frame.max() == 0)
        writer.write(frame)
    writer.release()
    valid = (
        counts == [16, 16, 16, 16] and len(hashes) == len(all_frames)
        and black == 0 and min(t["frames"] for t in timings) == 16)
    fps = [t["frames"] / t["total_s"] for t in timings]
    print(json.dumps({
        "valid": valid, "chunks": counts, "frames": len(all_frames),
        "unique": len(hashes), "black": black, "size": [width, height],
        "fps": fps, "timings": timings, "output": args.output,
    }, ensure_ascii=False))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
