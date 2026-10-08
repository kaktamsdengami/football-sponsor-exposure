"""
The per-frame exposure timeline and the coverage ledger.

Two record types, both written into a run directory.

`exposure.jsonl` -- one record per (sample, surface instance). This is what
`aggregate` reads; it never reads shot records. Every number stays traceable to
the evidence it came from through `source` + `anchor_frame`, so a suspicious
total can always be walked back to the frame a human or a fit actually saw.

`coverage.jsonl` -- a partition of the WHOLE match into spans, each saying
whether that interval was measured and, when it was not, why. The invariant is
that the spans tile `[0, duration)` exactly: no gaps, no overlaps.

The ledger exists because of a specific failure. A 9.3s shot opened on 2s of
unreadable overhead footage and settled into 7s of a large, clearly visible
board. Routing sent the human only the bad opening; the automatic pass never ran
on the rest. Seven seconds of large exposure scored zero -- and nothing in the
system knew those seconds existed. Forcing every second to be accounted for
turns that from a silent zero into a loud line with a reason attached.

Shot records are deliberately NOT the unit of measurement. A shot is one camera
take; the camera pans and zooms inside it, so one label and one sampled area
cannot represent it. Shots survive only as context on a timeline record.
"""

import math

# Where a measurement came from. Kept on every exposure record because the
# confidence you should place in an area depends on how it was obtained.
SOURCES = (
    "human",          # a person annotated this exact frame
    "auto",           # geometric fit succeeded on this exact frame
    "tracked_human",  # propagated from a human anchor by the tracker
    "tracked_auto",   # propagated from an automatic anchor
)

# Coverage statuses. Anything that is not "uncovered" asserts that the interval
# was actually looked at -- including a human who looked and found nothing.
STATUSES = ("human", "auto", "tracked", "uncovered")


# ---------------------------------------------------------------------------
# Polygon geometry (canonical definitions -- annotate.py imports these)
# ---------------------------------------------------------------------------

def rect_poly(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def poly_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def poly_area(poly):
    """Shoelace. The true area of a slanted board, not its bounding box."""
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2.0


def poly_centroid(poly):
    return (sum(p[0] for p in poly) / len(poly),
            sum(p[1] for p in poly) / len(poly))


def centrality(poly, w, h):
    """
    1.0 at the centre of frame, 0.0 at a corner.

    Sponsorship valuation weights central placement far above peripheral, so
    this is a first-class field rather than something derived later.
    """
    cx, cy = poly_centroid(poly)
    d = math.hypot(cx - w / 2.0, cy - h / 2.0)
    return max(0.0, 1.0 - d / (math.hypot(w, h) / 2.0))


def exposure_record(t, frame, shot, brand, surface, poly, frame_wh,
                    source, conf, anchor_frame=None, anchor_age_s=None,
                    occluded_frac=0.0, clarity=1.0, n_brands_in_frame=1,
                    fb_error_px=None):
    """
    Build one exposure row with its derived fields filled in.

    `occluded_frac` is the share of the polygon hidden by something in front of
    it -- a player crossing the board. Without it exposure is overstated exactly
    when players cluster near the touchline, which is when it matters most.
    """
    w, h = frame_wh
    area = poly_area(poly)
    area_frac = area / float(max(w * h, 1))
    occ = min(max(float(occluded_frac), 0.0), 1.0)
    return {
        "t": round(float(t), 3),
        "frame": int(frame),
        "shot": None if shot is None else int(shot),
        "brand": brand,
        "surface": surface,
        "poly": [[int(p[0]), int(p[1])] for p in poly],
        "area_px": round(area, 1),
        "area_frac": round(area_frac, 6),
        "occluded_frac": round(occ, 3),
        "visible_area_frac": round(area_frac * (1.0 - occ), 6),
        "centrality": round(centrality(poly, w, h), 3),
        "clarity": round(float(clarity), 3),
        "n_brands_in_frame": int(n_brands_in_frame),
        "source": source,
        "anchor_frame": None if anchor_frame is None else int(anchor_frame),
        "anchor_age_s": None if anchor_age_s is None else round(float(anchor_age_s), 3),
        "conf": round(float(conf), 3),
        "fb_error_px": None if fb_error_px is None else round(float(fb_error_px), 2),
    }


# ---------------------------------------------------------------------------
# Span algebra -- the coverage ledger is built out of these four operations
# ---------------------------------------------------------------------------

def clamp_spans(spans, t0, t1):
    """Trim to [t0, t1) and drop anything empty."""
    out = []
    for s in spans:
        a, b = max(s[0], t0), min(s[1], t1)
        if b > a:
            out.append((a, b) + tuple(s[2:]))
    return out


def union(intervals):
    """Merge overlapping/touching (start, end) pairs. Ignores any payload."""
    iv = sorted((a, b) for a, b, *_ in intervals if b > a)
    if not iv:
        return []
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def complement(covered, t0, t1):
    """Intervals of [t0, t1) not inside `covered`."""
    gaps, cur = [], t0
    for a, b in union(covered):
        if a > cur:
            gaps.append((cur, min(a, t1)))
        cur = max(cur, b)
        if cur >= t1:
            break
    if cur < t1:
        gaps.append((cur, t1))
    return [(a, b) for a, b in gaps if b > a]


def merge_adjacent(spans, key=("status", "reason", "shot"), eps=1e-6):
    """
    Coalesce neighbouring spans that agree on `key`. Keeps the ledger readable:
    without it a 9-second gap sampled at 4 Hz prints as 36 rows.
    """
    out = []
    for s in sorted(spans, key=lambda r: r["t_start"]):
        prev = out[-1] if out else None
        same = prev and all(prev.get(k) == s.get(k) for k in key)
        if same and s["t_start"] - prev["t_end"] <= eps:
            prev["t_end"] = max(prev["t_end"], s["t_end"])
            prev["dur_s"] = round(prev["t_end"] - prev["t_start"], 3)
        else:
            out.append(dict(s))
    return out


def validate_partition(spans, t0, t1, eps=1e-3):
    """
    Raise unless `spans` tile [t0, t1) exactly. This is the whole point of the
    ledger: if a second of the match is not accounted for, that is a bug, not a
    rounding detail.
    """
    if not spans:
        raise ValueError("empty coverage ledger")
    ordered = sorted(spans, key=lambda r: r["t_start"])
    cur = t0
    for s in ordered:
        if s["t_start"] < cur - eps:
            raise ValueError(f"coverage overlap at t={s['t_start']:.3f} (prev ends {cur:.3f})")
        if s["t_start"] > cur + eps:
            raise ValueError(f"coverage gap {cur:.3f}-{s['t_start']:.3f}s")
        cur = max(cur, s["t_end"])
    if abs(cur - t1) > eps:
        raise ValueError(f"coverage ends at {cur:.3f}s, match is {t1:.3f}s")
    return True


def coverage_span(t_start, t_end, status, reason, shot=None, source=None, conf=None):
    return {
        "t_start": round(float(t_start), 3),
        "t_end": round(float(t_end), 3),
        "dur_s": round(float(t_end) - float(t_start), 3),
        "shot": None if shot is None else int(shot),
        "status": status,
        "source": source,
        "conf": None if conf is None else round(float(conf), 3),
        "reason": reason,
    }
