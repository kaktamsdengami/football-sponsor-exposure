"""
Build a frame set to hand-label for the ad-board / logo OBB detector.

    python pilot/build_label_set.py input_videos/MATCH.mp4 [more.mp4 ...] \
        --every 4 --max 250 --out datasets/board_v1

For each kept frame it writes:
    <out>/images/<video>_<frame>.jpg
    <out>/labels/<video>_<frame>.txt   pre-labels from pilot/models/board_yolo11s_ee.pt
                                       in YOLO-OBB format (class cx-normalised 4 corners)

Then correct the pre-labels in a labeller (see pilot/LABELING.md) and re-run
training. The pre-labeller is the weak Swedish-league model -- it exists only to
save you drawing the easy perimeter tiles; you still fix every frame.

Frame selection:
  - uniform sample every --every seconds
  - near-duplicate frames skipped (mean-abs-diff on a 32x32 grey thumb) so you
    do not label 3 near-identical frames of the same still camera
  - --max caps the total; frames are spread across the whole video, not the
    first N seconds
"""

import argparse
import os
import sys

import cv2
import numpy as np

MODEL = os.path.join(os.path.dirname(__file__), "models", "board_yolo11s_ee.pt")


def thumb(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--every", type=float, default=4.0, help="seconds between samples")
    ap.add_argument("--max", type=int, default=250, help="cap on kept frames (total)")
    ap.add_argument("--dup-thr", type=float, default=6.0,
                    help="mean abs grey diff below which a frame is a near-duplicate")
    ap.add_argument("--out", default="datasets/board_v1")
    ap.add_argument("--no-prelabel", action="store_true")
    ap.add_argument("--conf", type=float, default=0.20)
    args = ap.parse_args()

    idir = os.path.join(args.out, "images")
    ldir = os.path.join(args.out, "labels")
    os.makedirs(idir, exist_ok=True)
    os.makedirs(ldir, exist_ok=True)

    model = None
    if not args.no_prelabel and os.path.exists(MODEL):
        from ultralytics import YOLO
        model = YOLO(MODEL)
        print(f"pre-labelling with {MODEL}")
    elif not args.no_prelabel:
        print(f"(no {MODEL} -- frames will have empty label files)")

    # First pass over all videos: collect candidate (video, frame_idx) uniformly.
    cands = []
    metas = {}
    for v in args.videos:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print(f"skip (cannot open): {v}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        step = max(1, int(round(args.every * fps)))
        metas[v] = (fps, n)
        for f in range(0, n, step):
            cands.append((v, f))
    if not cands:
        sys.exit("no frames")

    # Spread the cap across all candidates rather than truncating.
    if len(cands) > args.max:
        idx = np.linspace(0, len(cands) - 1, args.max).round().astype(int)
        cands = [cands[i] for i in sorted(set(idx))]
    by_video = {}
    for v, f in cands:
        by_video.setdefault(v, []).append(f)

    kept = 0
    for v, frames in by_video.items():
        cap = cv2.VideoCapture(v)
        base = os.path.splitext(os.path.basename(v))[0]
        want = set(frames)
        last = max(frames)
        prev_t = None
        i = 0
        while i <= last:
            if not cap.grab():
                break
            if i in want:
                ok, fr = cap.retrieve()
                if ok:
                    t = thumb(fr)
                    if prev_t is None or np.abs(t - prev_t).mean() >= args.dup_thr:
                        prev_t = t
                        name = f"{base}_{i:07d}"
                        cv2.imwrite(os.path.join(idir, name + ".jpg"), fr,
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                        lbl = os.path.join(ldir, name + ".txt")
                        lines = []
                        if model is not None:
                            r = model.predict(fr, conf=args.conf, imgsz=1280,
                                              verbose=False)[0]
                            if r.obb is not None and len(r.obb):
                                H, W = fr.shape[:2]
                                for poly in r.obb.xyxyxyxy.cpu().numpy():
                                    p = (poly / [W, H]).reshape(-1)
                                    lines.append("0 " + " ".join(f"{x:.6f}" for x in p))
                        open(lbl, "w").write("\n".join(lines))
                        kept += 1
                        if kept % 25 == 0:
                            print(f"  {kept} frames")
            i += 1
        cap.release()

    yaml = f"""# hand-labelled ad-board / logo regions, one class
path: {os.path.abspath(args.out)}
train: images        # split later, or point train/val at subfolders
val: images
names:
  0: logo
"""
    open(os.path.join(args.out, "data.yaml"), "w").write(yaml)
    print(f"\n{kept} frames -> {args.out}")
    print("next: correct labels (pilot/LABELING.md), then split + train")


if __name__ == "__main__":
    main()
