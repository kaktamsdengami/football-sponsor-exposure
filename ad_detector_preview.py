"""
Quick preview of the ad-region detector.

Samples a few frames across the video, draws the detected bounding boxes
(green = board, orange = jersey) and saves images to inspect. Used to tune the
prompts / confidence threshold before wiring into the pipeline.

Usage:
    python ad_detector_preview.py
    python ad_detector_preview.py input_videos/pub_2.mp4 20
"""
import sys
import os
import cv2
from brand_reader.ad_detector import AdRegionDetector

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "input_videos/ucl.mp4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
OUT_DIR = "output_videos/ad_detector_preview"

COLORS = {"board": (0, 220, 160), "jersey": (60, 140, 255)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(total * (i + 1) / (N + 1)) for i in range(N)]

    det = AdRegionDetector()
    print(f"{VIDEO}: {total} frames -- preview on {N} frames")

    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        regions = det.detect(frame)
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            c = COLORS.get(r["type"], (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f'{r["type"]} {r["score"]:.2f}',
                        (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, 1, cv2.LINE_AA)

        path = os.path.join(OUT_DIR, f"frame_{k:02d}_idx{fi}.jpg")
        cv2.imwrite(path, frame)
        print(f"  {path}  --  {len(regions)} regions")

    cap.release()
    print(f"\nImages in {OUT_DIR}/")


if __name__ == "__main__":
    main()
