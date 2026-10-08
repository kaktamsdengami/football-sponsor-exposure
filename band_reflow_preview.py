"""
Preview of the rectified + reflowed band -- what will be sent to the LLM.

For each sampled frame, writes ONE image holding all the far-touchline
advertising, folded into a few rows. Also checks the reprojection: a box drawn on
the reflowed image is mapped back into the original frame.

Usage:
    python band_reflow_preview.py
    python band_reflow_preview.py input_videos/pub_2.mp4 8
"""
import sys
import os
import cv2
from brand_reader.board_regions import band_image, reflow_box_to_frame

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "input_videos/ucl.mp4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
OUT_DIR = "output_videos/band_reflow_preview"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(total * (i + 1) / (N + 1)) for i in range(N)]

    print(f"{VIDEO}: {total} frames -- preview on {N} frames")
    got = 0

    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        res = band_image(frame)
        if res is None:
            print(f"  frame_{k:02d} (idx {fi}): no usable boundary")
            continue

        img, meta = res
        got += 1
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{k:02d}.jpg"), img)

        # reprojection check: left half of the first row
        r0 = meta["layout"]["rows"][0]
        box = (0, r0["y0"], r0["w"] // 2, r0["y0"] + r0["h"])
        fb = reflow_box_to_frame(meta["rect"], meta["layout"], box)
        print(f"  frame_{k:02d} (idx {fi}): {img.shape[1]}x{img.shape[0]} "
              f"H={meta['rect']['H']:.0f}px  -> frame{fb}")

    cap.release()
    print(f"\nBands obtained: {got}/{N}")
    print(f"Images: {OUT_DIR}/")


if __name__ == "__main__":
    main()
