"""
Group the band strips that show the same thing, so later stages do the work once.

`detect` writes one strip per sampled frame -- 235 for a 3-minute clip, and
thousands for a full match. Most of them are near-duplicates: the camera holds
still and the same stretch of perimeter is captured over and over. Reading each
one separately, whether by OCR or by a vision model, is waste.

WHAT A GROUP IS NOT. A group is not "one advert". The perimeter carries
different adverts at different physical positions, so when the camera pans, the
strip genuinely shows a different set of brands even though nothing rotated.
Trying to make one group per advert would need the strip cut into panels first.
A group here means only "these strips look the same, read one of them", and
`identify` is expected to return SEVERAL brands from a single strip.

MATCHING. Strips are compared with a colour histogram in Lab, which ignores
where things sit horizontally, plus a row-mean colour profile that keeps the
vertical structure (board rows versus crowd rows). Both survive the horizontal
shift a panning camera introduces, which a direct pixel comparison would not.

Clustering is greedy against each group's representative, in time order, so a
view that comes back later rejoins its original group instead of starting a new
one.
"""

import json
import os
import time

import cv2
import numpy as np

from .store import RunStore

# Below this combined distance, two strips are treated as the same view.
# Measured on clip_b.mp4: within-view pairs sit near 0.09, genuinely different
# views at 0.8+. The gap is wide, so the exact value is not delicate.
DIST_THR = 0.30

_SIG_SIZE = (256, 96)


def _signature(path):
    """(colour histogram, row-mean profile) or None. Both shift-invariant."""
    im = cv2.imread(path)
    if im is None:
        return None
    im = cv2.resize(im, _SIG_SIZE, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [6, 8, 8], [0, 256] * 3)
    hist = cv2.normalize(hist, hist).flatten()
    rows = lab.reshape(_SIG_SIZE[1], -1, 3).mean(axis=1) / 255.0
    return hist, rows.astype(np.float32)


def _distance(a, b):
    """0 = identical. Colour mix dominates; the row profile breaks ties."""
    h = float(cv2.compareHist(a[0], b[0], cv2.HISTCMP_BHATTACHARYYA))
    r = float(np.abs(a[1] - b[1]).mean())
    return 0.75 * h + 0.25 * min(r * 4.0, 1.0)


def _spans(times, max_gap):
    """Contiguous time ranges from a sorted list of sample times."""
    out = []
    for t in times:
        if out and t - out[-1][1] <= max_gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(round(a, 2), round(b, 2)) for a, b in out]


def run_dedup(run_dir, min_conf=0.55, thr=DIST_THR, max_gap=1.5, progress=True):
    """Write creatives.jsonl for a run. Returns (store, stats)."""
    out_root, name = os.path.split(os.path.normpath(run_dir))
    store = RunStore(out_root, name, create=False)
    manifest = store.read_manifest()
    fps = float(manifest.get("video", {}).get("fps") or 25.0)
    step = manifest.get("stats", {}).get("detect", {}).get("step", 0)
    frame_s = (step / fps) if step else 0.25

    strips = sorted((d for d in store.read_detections()
                     if d.get("band") and float(d.get("conf", 0)) >= min_conf),
                    key=lambda d: d["t"])
    print(f"Dedup    : {store.dir}")
    print(f"Strips   : {len(strips)} with band confidence >= {min_conf}")
    if not strips:
        print("  nothing to group -- run `detect` first")
        return store, {"strips": 0, "creatives": 0}

    t0 = time.perf_counter()
    groups = []          # each: {"sig", "members": [detection records]}
    for n, d in enumerate(strips, 1):
        sig = _signature(os.path.join(store.dir, d["band"]))
        if sig is None:
            continue
        best_i, best_d = None, 1e9
        for i, g in enumerate(groups):
            dist = _distance(sig, g["sig"])
            if dist < best_d:
                best_i, best_d = i, dist
        if best_i is not None and best_d < thr:
            groups[best_i]["members"].append(d)
            # keep the clearest strip as the group's representative
            if d["conf"] > groups[best_i]["rep"]["conf"]:
                groups[best_i]["rep"], groups[best_i]["sig"] = d, sig
        else:
            groups.append({"sig": sig, "rep": d, "members": [d]})
        if progress and n % 50 == 0:
            print(f"  {n}/{len(strips)} strips -> {len(groups)} groups")

    creatives = []
    for i, g in enumerate(sorted(groups, key=lambda g: -len(g["members"]))):
        ts = sorted(m["t"] for m in g["members"])
        creatives.append({
            "creative": i,
            "n_frames": len(g["members"]),
            "seconds": round(len(g["members"]) * frame_s, 2),
            "t_first": round(ts[0], 2),
            "t_last": round(ts[-1], 2),
            "spans": _spans(ts, max_gap),
            "rep_frame": g["rep"]["frame"],
            "rep_band": g["rep"]["band"],
            "rep_conf": g["rep"]["conf"],
            "frames": [m["frame"] for m in g["members"]],
            # filled by `identify`
            "brands": None,
        })

    path = os.path.join(store.dir, "creatives.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for c in creatives:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    total_s = sum(c["seconds"] for c in creatives)
    stats = {
        "strips": len(strips),
        "creatives": len(creatives),
        "reduction": round(len(strips) / max(len(creatives), 1), 1),
        "board_seconds": round(total_s, 1),
        "seconds": round(elapsed, 1),
    }
    store.write_manifest({"min_conf": min_conf, "thr": thr}, None, stats, stage="dedup")

    print(f"\n  {len(strips)} strips -> {len(creatives)} groups "
          f"({stats['reduction']}x less work for identify)")
    print(f"  {total_s:.0f}s of board time covered")
    print(f"\n  largest groups:")
    for c in creatives[:10]:
        sp = ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in c["spans"][:4])
        more = "..." if len(c["spans"]) > 4 else ""
        print(f"    #{c['creative']:<3} {c['n_frames']:4d} frames  {c['seconds']:6.1f}s  "
              f"conf {c['rep_conf']:.2f}  seen at {sp}{more}")
    print(f"\n  Wrote {path}")
    return store, stats
