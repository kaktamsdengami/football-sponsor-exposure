"""
Preview of the advertising band detector (pitch-boundary following).

For each sampled frame, writes:
  - frame_XX_annot.jpg: the frame with the pitch boundary (blue) and tiles (green)
  - crops/frame_XX_tYY.jpg: each tile, cropped

This is for debugging the boundary and the band height. For what actually gets
sent to the LLM, see band_reflow_preview.py.

Usage:
    python board_regions_preview.py
    python board_regions_preview.py input_videos/pub_2.mp4 16
"""
import sys
import os
import cv2
import numpy as np
from brand_reader.board_regions import pitch_boundary, board_band_tiles

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "input_videos/ucl.mp4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
OUT_DIR = "output_videos/board_band_preview"
CROP_DIR = os.path.join(OUT_DIR, "crops")


def main():
    os.makedirs(CROP_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(total * (i + 1) / (N + 1)) for i in range(N)]

    print(f"{VIDEO}: {total} frames -- preview on {N} frames")
    n_ok = 0

    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        annot = frame.copy()

        yb = pitch_boundary(frame)
        if yb is None:
            print(f"  frame_{k:02d} (idx {fi}): no pitch detected")
            cv2.putText(annot, "NO PITCH", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUT_DIR, f"frame_{k:02d}_annot.jpg"), annot)
            continue

        pts = np.stack([np.arange(w), yb.astype(np.int32)], axis=1)
        cv2.polylines(annot, [pts], False, (255, 120, 0), 2, cv2.LINE_AA)

        tiles = board_band_tiles(frame)
        for t, tile in enumerate(tiles):
            x1, y1, x2, y2 = tile["bbox"]
            cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 220, 160), 2)
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                cv2.imwrite(os.path.join(CROP_DIR, f"frame_{k:02d}_t{t:02d}.jpg"), crop)

        if tiles:
            n_ok += 1
        bh = f"{tiles[0]['band_h']:.0f}px" if tiles else "-"
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{k:02d}_annot.jpg"), annot)
        print(f"  frame_{k:02d} (idx {fi}): {len(tiles):2d} tiles, band height ~{bh}")

    cap.release()
    print(f"\nFrames with a band: {n_ok}/{N}")
    print(f"Annotated: {OUT_DIR}/   Crops: {CROP_DIR}/")


if __name__ == "__main__":
    main()
