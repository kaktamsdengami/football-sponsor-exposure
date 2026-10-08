"""
Turn the identified strips into per-brand numbers.

Four measures, the same four the old `main.py` baseline produced, so the two can
be held side by side:

    time        seconds the brand was on screen
    surface     how much of the picture it occupied
    distance    how far from the centre of frame it sat
    brightness  how bright it was, on its own and against the whole frame

Where the geometry comes from. A creative group knows which frames it covers;
each of those frames has a detection record carrying the band's geometry as
scalars, from which the region is rebuilt without decoding the video again. If
`identify` located the brand's own text inside the strip, that box is used. If
it did not -- a name that came from the vision model or from a human reading,
neither of which returns coordinates -- the brand inherits the WHOLE band. That
overstates a single sponsor sharing a board with others, so the `area_source`
column says which happened and the summary reports the split.

Brightness needs pixels, so it costs one sequential pass over the sampled frames
(no band extraction, just a decode and two means). Relative brightness is the
region's luminance over the frame's, which is what makes it comparable between a
night match and an afternoon one.

NOT a media value. These are raw measurements with no weighting applied: no
premium for a central placement, no discount for clutter or for a replay. That
weighting is a product decision and belongs in the report stage, on top of this.
"""

import csv
import json
import math
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from brand_reader.board_regions import band_polygon, rect_from_scalars
from .store import RunStore
from .timeline import poly_area, poly_bbox, poly_centroid


def _band_poly(det):
    """The whole advertising band in frame coordinates, or None."""
    try:
        rect = rect_from_scalars(det["coef"], det["span"], det["fit_med"],
                                 det["local_clamp"], det["H"], det["over"],
                                 det.get("segments"), det.get("knots"),
                                 det.get("off_shift", 0))
        return band_polygon(rect).tolist()
    except Exception:
        return None


def _luma(frame):
    """Rec.601 luminance. cv2 gives BGR, so the weights are reversed."""
    f = frame.astype(np.float32)
    return 0.299 * f[:, :, 2] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 0]


def _region_mean(lum, polys):
    """
    Mean luminance across ALL of a brand's panels in one frame, or None if they
    land off-frame. A brand repeated along the board is measured over every
    instance, not just the first one found.
    """
    h, w = lum.shape
    if not polys:
        return None
    pts_all = [p for poly in polys for p in poly]
    x1, y1, x2, y2 = poly_bbox(pts_all)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    for poly in polys:
        cv2.fillPoly(mask, [np.array(poly, np.int32) - [x1, y1]], 255)
    vals = lum[y1:y2, x1:x2][mask > 0]
    return float(vals.mean()) if vals.size else None


def _brand_polys(b, band_poly, fr=None):
    """
    (polys, source) for one brand on ONE frame `fr`.

    A perimeter repeats its advertisers, so a brand can hold several boxes on a
    single frame. All of them count towards surface -- four `mastercard` panels
    are four panels of ink, even though they are one brand for the purposes of
    TIME.

    `identify` (OCR) stores boxes per frame in `boxes_by_frame`, because the
    camera pans and the brand moves -- a box read on one frame is not where the
    brand is on the next. When that map is present we use only this frame's
    boxes, and fall straight to the whole band if this frame has none (rather
    than borrowing another frame's box). Vision-model / human readings carry no
    boxes at all and always fall back to the whole band, which over-states the
    surface; the caller records that.
    """
    bbf = b.get("boxes_by_frame")
    if bbf is not None:
        polys = [p for p in bbf.get(str(fr), []) if p] if fr is not None else []
        if polys:
            return polys, "brand_box"
        return ([band_poly], "whole_band") if band_poly else ([], "none")
    polys = b.get("polys")
    if polys is None:                       # records written before this change
        polys = [b["poly"]] if b.get("poly") else []
    polys = [p for p in polys if p]
    if polys:
        return polys, "brand_box"
    return ([band_poly], "whole_band") if band_poly else ([], "none")


def _brightness_pass(video, frames_needed, regions, progress=True):
    """
    One sequential walk of the video: for every needed frame, the mean luminance
    of the whole picture and of each region asked for on that frame.

    Sequential rather than seeking per frame -- seeking thousands of times is
    far slower than decoding straight through and skipping.
    """
    out = {}
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return out
    want = set(frames_needed)
    last = max(want) if want else -1
    i = 0
    done = 0
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, frame = cap.retrieve()
            if ok:
                lum = _luma(frame)
                frame_mean = float(lum.mean())
                per = {}
                for key, poly in regions.get(i, []):
                    m = _region_mean(lum, poly)
                    if m is not None:
                        per[key] = m
                out[i] = (frame_mean, per)
                done += 1
                if progress and done % 200 == 0:
                    print(f"  brightness: {done}/{len(want)} frames")
        i += 1
    cap.release()
    return out


def run_aggregate(run_dir, video=None, min_seconds=0.0, brightness=True,
                  progress=True):
    """Write exposure_by_brand.csv and exposure_detail.csv. Returns (store, stats)."""
    out_root, name = os.path.split(os.path.normpath(run_dir))
    store = RunStore(out_root, name, create=False)

    cre_path = os.path.join(store.dir, "creatives.jsonl")
    if not os.path.exists(cre_path):
        raise RuntimeError(f"No creatives.jsonl in {run_dir} -- run `dedup` "
                           "and `identify` first.")
    creatives = [json.loads(l) for l in open(cre_path, encoding="utf-8") if l.strip()]

    manifest = store.read_manifest()
    info = manifest.get("video", {})
    video = video or info.get("path")
    fps = float(info.get("fps") or 25.0)
    W, H = int(info.get("width") or 0), int(info.get("height") or 0)
    duration = float(info.get("frames") or 0) / fps
    step = manifest.get("stats", {}).get("detect", {}).get("step", 0)
    frame_s = (step / fps) if step else 0.25
    half_diag = math.hypot(W, H) / 2.0 or 1.0
    frame_area = float(W * H) or 1.0

    dets = {d["frame"]: d for d in store.read_detections() if d.get("band")}

    print(f"Aggregate: {store.dir}")
    print(f"Video    : {W}x{H} @ {fps:.1f} fps, {duration:.0f}s")
    print(f"Groups   : {len(creatives)}, each sampled frame = {frame_s:.2f}s")

    # --- one row per (brand, frame) ---------------------------------------
    rows = []
    regions = defaultdict(list)
    for c in creatives:
        band = None
        for b in c.get("brands") or []:
            # A brand is credited the frames it was ACTUALLY read on (OCR), not
            # every frame of the camera view. Group-level readings (vision model
            # / human) have no per-frame data, so they fall back to the whole
            # view's frames -- an upper bound, flagged below via `time_basis`.
            b_frames = b.get("frames") or c["frames"]
            basis = b.get("time_basis", "group")
            for fr in b_frames:
                det = dets.get(fr)
                if det is None:
                    continue
                band = _band_poly(det)
                polys, src = _brand_polys(b, band, fr)
                if not polys:
                    continue
                # Surface is the sum over every panel this brand occupies;
                # position is their area-weighted centre. One row per
                # (brand, frame), so TIME is unaffected by the repetition.
                areas = [poly_area(p) for p in polys]
                total_area = float(sum(areas))
                cents = [poly_centroid(p) for p in polys]
                cx = sum(a * c0[0] for a, c0 in zip(areas, cents)) / max(total_area, 1e-9)
                cy = sum(a * c0[1] for a, c0 in zip(areas, cents)) / max(total_area, 1e-9)
                key = (b["name"], fr)
                regions[fr].append((key, polys))
                rows.append({
                    "brand": b["name"],
                    "paying": b.get("paying", True),
                    "creative": c["creative"],
                    "frame": fr,
                    "t": round(fr / fps, 3),
                    "seconds": frame_s,
                    "n_panels": len(polys),
                    "area_px": round(total_area, 1),
                    "area_pct": round(100.0 * total_area / frame_area, 4),
                    "dist_center": round(math.hypot(cx - W / 2.0, cy - H / 2.0)
                                         / half_diag, 4),
                    "area_source": src,
                    "time_basis": basis,
                    "identified_by": b.get("source", "?"),
                    "brightness": "",
                    "brightness_rel": "",
                })

    if not rows:
        print("  no identified brands -- run `identify` first")
        return store, {"brands": 0}

    # --- brightness -------------------------------------------------------
    if brightness and video and os.path.exists(video):
        t0 = time.perf_counter()
        print(f"  measuring brightness over {len(regions)} frames...")
        lum = _brightness_pass(video, regions.keys(), regions, progress)
        for r in rows:
            got = lum.get(r["frame"])
            if not got:
                continue
            frame_mean, per = got
            v = per.get((r["brand"], r["frame"]))
            if v is None:
                continue
            r["brightness"] = round(v, 1)
            r["brightness_rel"] = round(v / max(frame_mean, 1e-6), 3)
        print(f"  brightness pass: {time.perf_counter() - t0:.0f}s")
    elif brightness:
        print(f"  (skipping brightness -- video not found at {video})")

    # --- per-brand summary -------------------------------------------------
    by = defaultdict(list)
    for r in rows:
        by[r["brand"]].append(r)

    def _mean(vals, nd=4):
        vals = [v for v in vals if v != "" and v is not None]
        return round(float(np.mean(vals)), nd) if vals else ""

    summary = []
    for brand, rs in by.items():
        secs = len({r["frame"] for r in rs}) * frame_s
        whole = sum(1 for r in rs if r["area_source"] == "whole_band")
        bases = {r.get("time_basis", "group") for r in rs}
        time_basis = bases.pop() if len(bases) == 1 else "mixed"
        # `group` / `mixed` means at least some of this time was credited from a
        # whole-view reading with no per-frame evidence -- an upper bound, since
        # other advertisers shared that view. `per_frame` time is measured.
        upper_bound = time_basis != "per_frame"
        summary.append({
            "brand": brand,
            "paying": rs[0]["paying"],
            "exposure_s": round(secs, 2),
            "time_basis": time_basis,
            "time_is_upper_bound": upper_bound,
            "pct_of_video": round(100.0 * secs / max(duration, 1e-9), 2),
            "n_frames": len({r["frame"] for r in rs}),
            "mean_panels": _mean([r["n_panels"] for r in rs], 2),
            "mean_area_px": _mean([r["area_px"] for r in rs], 0),
            "mean_area_pct": _mean([r["area_pct"] for r in rs], 3),
            "mean_dist_center": _mean([r["dist_center"] for r in rs], 3),
            "mean_brightness": _mean([r["brightness"] for r in rs], 1),
            "mean_brightness_rel": _mean([r["brightness_rel"] for r in rs], 3),
            "first_seen_s": round(min(r["t"] for r in rs), 2),
            "last_seen_s": round(max(r["t"] for r in rs), 2),
            # How much of this brand's area is the whole board rather than its
            # own box -- high means the surface figure is an over-estimate.
            "area_from_whole_band_pct": round(100.0 * whole / len(rs), 1),
            "identified_by": ",".join(sorted({r["identified_by"] for r in rs})),
        })
    summary = [s for s in summary if s["exposure_s"] >= min_seconds]
    summary.sort(key=lambda s: (-s["paying"], -s["exposure_s"]))

    sum_path = os.path.join(store.dir, "exposure_by_brand.csv")
    with open(sum_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    det_path = os.path.join(store.dir, "exposure_detail.csv")
    with open(det_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["t"], r["brand"])))

    paying = [s for s in summary if s["paying"]]
    stats = {
        "brands": len(summary),
        "paying_brands": len(paying),
        "total_exposure_s": round(sum(s["exposure_s"] for s in summary), 1),
        "paying_exposure_s": round(sum(s["exposure_s"] for s in paying), 1),
        "measured_time_brands": sum(1 for s in summary
                                    if not s["time_is_upper_bound"]),
        "upper_bound_time_brands": sum(1 for s in summary
                                       if s["time_is_upper_bound"]),
        "detail_rows": len(rows),
    }
    store.write_manifest({"min_seconds": min_seconds}, None, stats, stage="aggregate")

    hdr = (f"\n  {'brand':<32} {'time':>7} {'%vid':>6} {'surface%':>9} "
           f"{'dist':>6} {'bright':>7} {'rel':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    def _num(v, spec):
        """Blank cells stay blank -- `--no-brightness` leaves these empty."""
        return format(v, spec) if isinstance(v, (int, float)) else str(v).rjust(
            int(spec.split(".")[0] or 0))

    for s in summary:
        flags = ("" if s["paying"] else "  (not paying)") + \
                ("  [time: upper bound]" if s["time_is_upper_bound"] else "")
        print(f"  {s['brand']:<32} {s['exposure_s']:6.1f}s {s['pct_of_video']:5.1f}% "
              f"{_num(s['mean_area_pct'], '8.3f')}% "
              f"{_num(s['mean_dist_center'], '6.3f')} "
              f"{_num(s['mean_brightness'], '7.1f')} "
              f"{_num(s['mean_brightness_rel'], '5.2f')}{flags}")
    print(f"\n  paying sponsors: {stats['paying_exposure_s']:.1f}s of "
          f"{duration:.0f}s of video")
    ub = [s for s in summary if s["time_is_upper_bound"]]
    if ub:
        print(f"  NOTE: time is an UPPER BOUND for: "
              + ", ".join(s["brand"] for s in ub[:8]))
        print("        (named from a whole-view reading, no per-frame evidence; "
              "other advertisers shared that view. Run OCR/`--llm` per frame "
              "to measure it.)")
    over = [s for s in summary if s["area_from_whole_band_pct"] > 50]
    if over:
        print(f"  NOTE: surface is the whole board (over-stated) for: "
              + ", ".join(s["brand"] for s in over[:6]))
    print(f"\n  Wrote {sum_path}")
    print(f"        {det_path}  ({len(rows)} rows)")
    return store, stats
