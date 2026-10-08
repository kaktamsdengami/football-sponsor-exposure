"""
Build the exposure timeline: turn each annotated polygon into a measurement that
spans the frames around it.

A person outlines a board on one frame. That one outline says nothing about how
long the board stayed on screen or how its size changed while the camera moved --
and those two numbers are the report. So every annotated polygon is tracked
outward through its own shot, and each tracked frame becomes an exposure record.

Tracking never crosses a shot boundary: a cut replaces the whole picture, so
there is nothing to follow across it.

Where a track stops, it stops. The gap it leaves shows up in the coverage ledger
as an honest unmeasured second, which is better than a confident wrong number.
"""

import json
import os
from collections import Counter

import cv2

from brand_reader.board_regions import band_polygon, rect_from_scalars
from brand_reader.tracking import track_polygon
from .store import RunStore
from .timeline import exposure_record


def run_expose(run_dir, video=None, sample_fps=4.0, min_conf=0.35,
               min_auto_conf=0.55, progress=True):
    """Write exposure.jsonl for a run. Returns (store, stats)."""
    out_root, name = os.path.split(os.path.normpath(run_dir))
    store = RunStore(out_root, name, create=False)
    if not os.path.exists(store.shots_path):
        raise RuntimeError(f"No shots.jsonl in {run_dir} -- run `segment` first.")

    manifest = store.read_manifest()
    info = manifest.get("video", {})
    video = video or info.get("path")
    fps = float(info.get("fps") or 25.0)
    frame_wh = (int(info.get("width") or 0), int(info.get("height") or 0))
    step = max(1, int(round(fps / max(sample_fps, 0.01))))

    shots = {s["shot"]: s for s in store.read_shots()}
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")

    anchors = [(a, b) for a in store.read_annotations() for b in a["boxes"]]
    print(f"Expose   : {store.dir}")
    print(f"Video    : {video}  ({frame_wh[0]}x{frame_wh[1]} @ {fps:.1f} fps)")
    print(f"Anchors  : {len(anchors)} annotated polygons, sampling every {step} "
          f"frames (~{fps / step:.1f}/s)")

    records = []
    reach = []
    for n, (a, box) in enumerate(anchors, 1):
        shot = shots.get(a["shot"])
        if shot is None:
            continue
        samples = track_polygon(cap, a["frame"], box["poly"],
                                shot["f_start"], shot["f_end"], step,
                                min_conf=min_conf)
        if not samples:
            continue
        span_f = samples[-1]["frame"] - samples[0]["frame"]
        reach.append((span_f / fps, shot["dur_s"], box["brand"]))
        for s in samples:
            is_anchor = s["frame"] == a["frame"]
            records.append(exposure_record(
                t=s["frame"] / fps, frame=s["frame"], shot=a["shot"],
                brand=box["brand"], surface=box["surface"],
                poly=s["poly"], frame_wh=frame_wh,
                source="human" if is_anchor else "tracked_human",
                conf=s["conf"], anchor_frame=a["frame"],
                anchor_age_s=abs(s["frame"] - a["frame"]) / fps,
                fb_error_px=s["fb_err"],
            ))
        if progress:
            print(f"  [{n:3d}/{len(anchors)}] shot {a['shot']:2d} f{a['frame']:<5d} "
                  f"{box['brand']:<22s} {len(samples):3d} frames = "
                  f"{span_f / fps:5.2f}s of {shot['dur_s']:.1f}s")

    # --- what detect found, as exposure records ----------------------------
    # detect locates the board but cannot say whose it is, so these carry
    # brand=None until `identify` fills it in. They still contribute geometry
    # (where, how big) and, through the ledger, coverage.
    n_auto = 0
    for d in store.read_detections():
        if d.get("band") is None or float(d.get("conf", 0.0)) < min_auto_conf:
            continue
        try:
            rect = rect_from_scalars(d["coef"], d["span"], d["fit_med"],
                                     d["local_clamp"], d["H"], d["over"],
                                     d.get("segments"), d.get("knots"))
            poly = band_polygon(rect).tolist()
        except Exception:
            continue
        n_auto += 1
        records.append(exposure_record(
            t=d["t"], frame=d["frame"], shot=d.get("shot"),
            brand=None, surface="perimeter_led",
            poly=poly, frame_wh=frame_wh,
            source="auto", conf=float(d.get("conf", 0.0)),
            anchor_frame=d["frame"], anchor_age_s=0.0,
        ))
    if n_auto:
        print(f"  + {n_auto} automatic board reads from detect "
              f"(brand unknown until `identify`)")

    # Clutter: how many distinct brands share each measured frame. Needed by the
    # weighting later -- a logo alone on screen is worth more than one of six.
    per_frame = {}
    for r in records:
        if r["brand"] is not None:
            per_frame.setdefault(r["frame"], set()).add(r["brand"])
    for r in records:
        r["n_brands_in_frame"] = len(per_frame.get(r["frame"], ()))

    cap.release()
    records.sort(key=lambda r: (r["frame"], r["brand"] or ""))
    path = os.path.join(store.dir, "exposure.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_brand = Counter(r["brand"] or "(unidentified)" for r in records)
    by_surface = Counter(r["surface"] for r in records)
    tracked = sum(1 for r in records if r["source"].startswith("tracked"))
    covered_s = len({r["frame"] for r in records}) * step / fps
    stats = {
        "anchors": len(anchors),
        "records": len(records),
        "tracked_records": tracked,
        "auto_records": n_auto,
        "frames_with_exposure": len(per_frame),
        "approx_covered_s": round(covered_s, 2),
        "by_brand": dict(by_brand),
        "by_surface": dict(by_surface),
        "sample_fps": round(fps / step, 2),
    }

    print(f"\n  {len(records)} exposure records "
          f"({tracked} from tracking, {len(records) - tracked} anchors)")
    if reach:
        got = sum(r[0] for r in reach)
        want = sum(r[1] for r in reach)
        print(f"  tracking reach: {got:.1f}s of {want:.1f}s of shot time "
              f"({100 * got / max(want, 1e-9):.0f} %)")
    print(f"  brands  : " + "  ".join(f"{k}={v}" for k, v in by_brand.most_common()))
    print(f"  surfaces: " + "  ".join(f"{k}={v}" for k, v in by_surface.most_common()))
    print(f"  Wrote {path}")
    return store, stats
