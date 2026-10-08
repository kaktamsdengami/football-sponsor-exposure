"""
Does the tracker work on footage it was never tuned on -- and does its
confidence number mean anything?

No annotations needed. The test is a ROUND TRIP: track a box forward N seconds,
then start a fresh track from where it ended and go back the same distance. It
should land where it started. However far it misses by is drift, in pixels, and
that is free ground truth on any video of any sport.

What actually matters is the second column of the result. Everything downstream
trusts `conf`, so the question is not only "how much does it drift" but "does a
high confidence reliably mean low drift". A tracker that fails loudly is usable.
One that fails while reporting 0.95 is not.

Boxes are placed automatically on parts of the frame that have something to
track, in two shapes: wide and flat like a perimeter board, and small and square
like a jersey logo. Those behave very differently and are reported separately.

    python tracker_check.py
    python tracker_check.py input_videos/nhl.mp4 --span 2.0
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

from brand_reader.shot_types import cut_points, frame_signature, shots_from_cuts
from brand_reader.tracking import track_polygon

OUT_DIR = "output_videos/tracker_check"
SHAPES = {"board": (360, 60), "jersey": (90, 90)}    # w, h in source pixels


def _find_shots(path, seg_fps=6.0):
    """Shot spans as (f_start, f_end), so no track ever crosses a cut."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / seg_fps)))
    sigs, idxs, i = [], [], 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            sigs.append(frame_signature(fr))
            idxs.append(i)
        i += 1
    cap.release()
    cuts, _ = cut_points(sigs)
    return [(idxs[a], idxs[b - 1]) for a, b in shots_from_cuts(len(sigs), cuts)], fps


def _place_boxes(frame, shape, n=3, min_feats=25):
    """
    Boxes on parts of the frame that actually have texture.

    A logo has corners to track; bare grass or plain ice does not. Placing boxes
    on featureless regions would measure nothing but the obvious.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, 900, 0.01, 8)
    if pts is None:
        return []
    pts = pts.reshape(-1, 2)
    bw, bh = shape
    gx, gy = 8, 6
    counts = {}
    for x, y in pts:
        counts[(int(x / (w / gx)), int(y / (h / gy)))] = \
            counts.get((int(x / (w / gx)), int(y / (h / gy))), 0) + 1
    best = sorted(counts.items(), key=lambda kv: -kv[1])[:n * 3]
    out = []
    for (cx, cy), c in best:
        if c < min_feats or len(out) >= n:
            continue
        x = int((cx + 0.5) * (w / gx) - bw / 2)
        y = int((cy + 0.5) * (h / gy) - bh / 2)
        x, y = max(0, min(x, w - bw - 1)), max(0, min(y, h - bh - 1))
        if any(abs(x - o[0][0]) < bw and abs(y - o[0][1]) < bh for o in out):
            continue
        out.append([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
    return out


def _round_trip(cap, f0, poly, f_end, step, span_f):
    """
    Track out to `span_f` frames, then back. Returns (drift_px, conf, reached_f)
    or None if the outbound leg never got anywhere.
    """
    out = track_polygon(cap, f0, poly, f0, min(f_end, f0 + span_f), step)
    far = [s for s in out if s["frame"] > f0]
    if not far:
        return None
    end = max(far, key=lambda s: s["frame"])
    back = track_polygon(cap, end["frame"], end["poly"], f0, end["frame"], step)
    home = [s for s in back if s["frame"] == f0]
    if not home:
        return None
    a = np.array(poly, np.float64)
    b = np.array(home[0]["poly"], np.float64)
    drift = float(np.mean(np.linalg.norm(a - b, axis=1)))
    return drift, end["conf"], end["frame"] - f0


def check(path, span_s=2.0, sample_fps=4.0, per_shot=2, min_shot_s=1.5):
    shots, fps = _find_shots(path)
    step = max(1, int(round(fps / sample_fps)))
    span_f = int(round(span_s * fps))
    cap = cv2.VideoCapture(path)
    rows = []
    usable = [s for s in shots if (s[1] - s[0]) / fps >= min_shot_s]
    print(f"\n{os.path.basename(path)}: {len(shots)} shots, "
          f"{len(usable)} long enough (>= {min_shot_s}s)")

    for f_start, f_end in usable:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        ok, frame = cap.read()
        if not ok:
            continue
        for shape_name, shape in SHAPES.items():
            for poly in _place_boxes(frame, shape, n=per_shot):
                r = _round_trip(cap, f_start, poly, f_end, step, span_f)
                if r is None:
                    rows.append({"video": os.path.basename(path), "shape": shape_name,
                                 "f0": f_start, "drift_px": "", "conf": "",
                                 "reached_s": 0.0, "result": "no track"})
                    continue
                drift, conf, reached = r
                rows.append({"video": os.path.basename(path), "shape": shape_name,
                             "f0": f_start, "drift_px": round(drift, 2),
                             "conf": round(conf, 3),
                             "reached_s": round(reached / fps, 2), "result": "ok"})
    cap.release()
    return rows


def report(rows):
    ok = [r for r in rows if r["result"] == "ok"]
    dead = len(rows) - len(ok)
    if not ok:
        print("  no successful tracks")
        return
    for shape in SHAPES:
        sub = [r for r in ok if r["shape"] == shape]
        if not sub:
            continue
        d = np.array([r["drift_px"] for r in sub])
        s = np.array([r["reached_s"] for r in sub])
        print(f"  {shape:7s} n={len(sub):3d}  reached {np.median(s):4.2f}s  "
              f"drift median {np.median(d):6.2f}px  p90 {np.percentile(d, 90):7.2f}px  "
              f"max {d.max():8.2f}px")
    print(f"  tracks that never started: {dead}")

    # The question that matters: does a high conf actually mean low drift?
    print("  confidence vs drift:")
    for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)):
        sub = [r for r in ok if lo <= r["conf"] < hi]
        if not sub:
            continue
        d = np.array([r["drift_px"] for r in sub])
        print(f"    conf {lo:.2f}-{hi:.2f}  n={len(sub):3d}  "
              f"median {np.median(d):6.2f}px  p90 {np.percentile(d, 90):7.2f}px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", default=None)
    ap.add_argument("--span", type=float, default=2.0, help="seconds out before turning back")
    ap.add_argument("--sample-fps", type=float, default=4.0)
    args = ap.parse_args()

    vids = args.videos or [f"input_videos/{v}" for v in
                           ("ucl1.mp4", "ucl.mp4", "pub_2.mp4", "pub_3.mp4", "nhl.mp4")]
    vids = [v for v in vids if os.path.exists(v)]
    if not vids:
        sys.exit("no videos found")
    os.makedirs(OUT_DIR, exist_ok=True)

    allrows = []
    for v in vids:
        rows = check(v, span_s=args.span, sample_fps=args.sample_fps)
        report(rows)
        allrows += rows

    print("\n=== ALL VIDEOS ===")
    report(allrows)
    path = os.path.join(OUT_DIR, "round_trip.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
