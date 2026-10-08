"""
Segment each camera view's timeline into board STATES -- stretches where the
perimeter LED is holding one creative.

WHY. `identify` credits board time in proportion to how readable a logo is, not
whether the advertiser is present. A stylised mark (Lay's sunburst, a PS5 glyph,
a tight FedEx wordmark) is read on only a handful of the frames it is actually
up, and the rest of that stretch is credited to nobody. But an LED board holds a
creative for about five seconds. If the same view's frames are grouped by
content into states, one lucky read anywhere in a state can be carried across
the whole state -- and a state nobody could read becomes an explicit gap instead
of silence.

SCOPE. States are cut WITHIN one `dedup` creative group only (one camera view).
No carry across a cut or a pan: that would lean on the whole perimeter being
synchronised, which is not established on every broadcaster's footage.

METHOD. Reuse `dedup._signature` / `dedup._distance` (Lab colour histogram + row
-mean profile, both shift-invariant) between time-consecutive frames of a view.
A large jump means the creative changed. The change threshold is not hardcoded
-- it is read from the run's own distribution of consecutive distances, which is
bimodal when the LED rotates (same creative ~0, creative changed ~large) and
unimodal when it does not (a static perimeter), in which case we fall back to
`dedup.DIST_THR`.
"""

import json
import os
import time

import numpy as np

from .dedup import DIST_THR, _distance, _signature
from .store import RunStore

# Consecutive frames more than this far apart in time are not compared -- the
# camera view was interrupted, so a state boundary is forced regardless of how
# similar the two strips look. Matches dedup's span-stitching gap.
MAX_GAP_S = 1.5

# The widest empty band in the sorted distances, as a fraction of their whole
# spread, below which we do not believe there are two clusters. A rotating LED
# gives "same creative" ~0 and "creative changed" ~large with nothing between;
# a static perimeter is one blob with only sampling noise for gaps.
MIN_SEPARATION = 0.30

# Each side of the split must hold at least this share of the distances, so a
# single far outlier cannot masquerade as a second cluster.
MIN_CLUSTER_FRAC = 0.10

# Need at least this many consecutive distances before calibration is meaningful.
MIN_DISTANCES = 20


def split_threshold(distances):
    """
    (threshold, separation) for a 1-D set of non-negative distances.

    The distances are bimodal when the LED rotates -- "same creative" near zero,
    "creative changed" large, and a clear empty band between the two. The
    threshold is the midpoint of the WIDEST such band; `separation` is that
    band's width as a fraction of the whole spread, in [0, 1]. A caller treats a
    small `separation` as "one blob, no real split".
    """
    d = np.sort(np.asarray([x for x in distances if x is not None], dtype=np.float64))
    if d.size < 2 or d[-1] <= d[0]:
        return None, 0.0
    spread = float(d[-1] - d[0])
    gaps = np.diff(d)
    # only consider a gap a real valley if both sides carry enough mass
    lo = int(np.ceil(MIN_CLUSTER_FRAC * d.size))
    hi = int(np.floor((1.0 - MIN_CLUSTER_FRAC) * d.size))
    if hi <= lo:
        return None, 0.0
    k = lo + int(np.argmax(gaps[lo:hi]))
    thr = float((d[k] + d[k + 1]) / 2.0)
    separation = float(gaps[k] / spread) if spread > 0 else 0.0
    return thr, separation


def calibrate_threshold(distances, override=None):
    """
    (threshold, source) for the run.

    `override` wins. Otherwise the valley between the two clusters, if the
    distances are cleanly bimodal. Otherwise `dedup.DIST_THR` -- a static
    perimeter has no creative changes to find and forcing a split would only
    fragment one long state into noise.
    """
    if override is not None:
        return float(override), "override"
    thr, sep = split_threshold(distances)
    n = len([d for d in distances if d is not None])
    if thr is not None and n >= MIN_DISTANCES and sep >= MIN_SEPARATION:
        return thr, "valley"
    return float(DIST_THR), "fallback"


def segment_states(frames, times, shots, sigs, thr, max_gap=MAX_GAP_S):
    """
    Split one view's frames into states. Inputs are parallel lists already
    ordered by time. Returns a list of states, each a list of positional
    indices into the input lists.

    A boundary is forced when the previous frame is more than `max_gap` away in
    time, when the shot id changes, or when the strip-to-strip distance exceeds
    `thr`.
    """
    if not frames:
        return []
    states = [[0]]
    for i in range(1, len(frames)):
        gap = times[i] - times[i - 1]
        shot_changed = shots[i] is not None and shots[i - 1] is not None \
            and shots[i] != shots[i - 1]
        if gap > max_gap or shot_changed:
            states.append([i])
            continue
        a, b = sigs[i - 1], sigs[i]
        if a is None or b is None or _distance(a, b) > thr:
            states.append([i])
        else:
            states[-1].append(i)
    return states


def carry_within_states(frames_by_brand, frame_to_state, state_frames):
    """
    Expand each brand's frame list to cover the whole of every state it was
    read on. Pure, so the selftest can exercise it without a video.

    `frames_by_brand`  {brand -> iterable of frames it was literally read on}
    `frame_to_state`   {frame -> state key}
    `state_frames`     {state key -> sorted list of that state's frames}

    Returns {brand -> sorted list of frames after carry}. A frame with no known
    state (e.g. not sampled) is kept as-is and carries nothing.
    """
    out = {}
    for brand, read_frames in frames_by_brand.items():
        hit_states = {frame_to_state[f] for f in read_frames if f in frame_to_state}
        carried = set(read_frames)
        for s in hit_states:
            carried.update(state_frames.get(s, ()))
        out[brand] = sorted(carried)
    return out


def run_states(run_dir, change_thr=None, max_gap=MAX_GAP_S, progress=True):
    """Write states.jsonl for a run. Returns (store, stats)."""
    out_root, name = os.path.split(os.path.normpath(run_dir))
    store = RunStore(out_root, name, create=False)

    cre_path = os.path.join(store.dir, "creatives.jsonl")
    if not os.path.exists(cre_path):
        raise RuntimeError(f"No creatives.jsonl in {run_dir} -- run `dedup` first.")
    creatives = [json.loads(l) for l in open(cre_path, encoding="utf-8") if l.strip()]

    manifest = store.read_manifest()
    fps = float(manifest.get("video", {}).get("fps") or 25.0)
    step = manifest.get("stats", {}).get("detect", {}).get("step", 0)
    frame_s = (step / fps) if step else 0.25

    dets = {d["frame"]: d for d in store.read_detections() if d.get("band")}

    print(f"States   : {store.dir}")
    print(f"Groups   : {len(creatives)} from dedup")

    t0 = time.perf_counter()

    # Pass 1: per creative, order frames by time, sign each strip, measure every
    # consecutive distance. Signatures are kept so the states can be cut once the
    # threshold is known without re-reading the images.
    views = []          # [{creative, frames, times, shots, sigs, dists}]
    all_dists = []
    for c in creatives:
        rows = []
        for fr in c.get("frames") or []:
            d = dets.get(fr)
            if d is None:
                continue
            rows.append((fr, float(d["t"]), d.get("shot")))
        rows.sort(key=lambda r: r[1])
        if not rows:
            continue
        frames = [r[0] for r in rows]
        times = [r[1] for r in rows]
        vshots = [r[2] for r in rows]
        sigs = [_signature(os.path.join(store.dir, f"bands/{fr:07d}.jpg"))
                for fr in frames]
        dists = []
        for i in range(1, len(frames)):
            if times[i] - times[i - 1] > max_gap \
                    or sigs[i] is None or sigs[i - 1] is None:
                dists.append(None)
            else:
                dd = _distance(sigs[i - 1], sigs[i])
                dists.append(dd)
                all_dists.append(dd)
        views.append({"creative": c["creative"], "frames": frames, "times": times,
                      "shots": vshots, "sigs": sigs, "dists": dists})
        if progress and len(views) % 25 == 0:
            print(f"  signed {len(views)}/{len(creatives)} groups")

    thr, thr_source = calibrate_threshold(all_dists, override=change_thr)
    _, separation = split_threshold(all_dists)
    darr = np.asarray([d for d in all_dists if d is not None], dtype=float)
    print(f"Threshold: {thr:.3f}  ({thr_source}"
          + (f", cluster separation {separation:.2f}" if thr_source != "override" else "")
          + ")")

    # Pass 2: cut states with the calibrated threshold.
    records = []
    sid = 0
    for v in views:
        groups = segment_states(v["frames"], v["times"], v["shots"], v["sigs"],
                                thr, max_gap)
        for g in groups:
            g_frames = [v["frames"][i] for i in g]
            g_times = [v["times"][i] for i in g]
            g_dists = [v["dists"][i - 1] for i in g[1:]
                       if v["dists"][i - 1] is not None]
            records.append({
                "state": sid,
                "creative": v["creative"],
                "shot": v["shots"][g[0]],
                "t_first": round(g_times[0], 2),
                "t_last": round(g_times[-1], 2),
                "n_frames": len(g_frames),
                "seconds": round(len(g_frames) * frame_s, 2),
                "frames": g_frames,
                "mean_consec_dist": round(float(np.mean(g_dists)), 4) if g_dists else 0.0,
                # filled by identify
                "status": "pending",
            })
            sid += 1

    path = os.path.join(store.dir, "states.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    per_cre = [sum(1 for r in records if r["creative"] == v["creative"])
               for v in views]
    stats = {
        "creatives": len(views),
        "states": len(records),
        "mean_states_per_creative": round(float(np.mean(per_cre)), 2) if per_cre else 0.0,
        "max_states_per_creative": max(per_cre) if per_cre else 0,
        "change_thr": round(thr, 4),
        "thr_source": thr_source,
        "cluster_separation": round(separation, 3),
        "consec_distances": len(darr),
        "dist_p10": round(float(np.percentile(darr, 10)), 4) if darr.size else None,
        "dist_p50": round(float(np.percentile(darr, 50)), 4) if darr.size else None,
        "dist_p90": round(float(np.percentile(darr, 90)), 4) if darr.size else None,
        "board_seconds": round(sum(r["seconds"] for r in records), 1),
        "seconds": round(elapsed, 1),
    }
    store.write_manifest({"change_thr": change_thr, "max_gap": max_gap},
                         None, stats, stage="states")

    print(f"\n  {len(views)} groups -> {len(records)} states "
          f"({stats['mean_states_per_creative']} per group, "
          f"max {stats['max_states_per_creative']})")
    print(f"  consecutive-distance spread: p10 {stats['dist_p10']}  "
          f"p50 {stats['dist_p50']}  p90 {stats['dist_p90']}")
    multi = sorted((v for v in views
                    if sum(1 for r in records if r["creative"] == v["creative"]) > 1),
                   key=lambda v: -sum(1 for r in records if r["creative"] == v["creative"]))
    if multi:
        print(f"\n  most-segmented views (LED rotating under a held camera):")
        for v in multi[:8]:
            n = sum(1 for r in records if r["creative"] == v["creative"])
            print(f"    #{v['creative']:<3} {len(v['frames']):4d} frames -> {n} states")
    print(f"\n  Wrote {path}")
    return store, stats
