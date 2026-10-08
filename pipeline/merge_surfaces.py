"""
Merge per-frame exposure from several surface runs into one per-brand table.

WHY. A brand's on-screen time is the UNION of the frames it was visible in, not
the sum over surfaces. A brand on the pitch-level LED and on the board line
above it in the same frame is one second of exposure, not two. Summing the
per-surface CSVs double-counts exactly the brands that appear most often, which
is the worst place to be wrong.

Each input is a run directory that has been through `aggregate` (or `scan`), so
it has an exposure_detail.csv with one row per (brand, frame, surface-instance).

Surfaces are unioned as TIME INTERVALS, not frame indices. `detect` samples at
4/s and `scan` at 1/s, so a shared frame index means different things in each
and "same frame" is not a question you can ask across them. Each row instead
covers [t, t + its own run's sample period), and a brand's exposure is the
measure of the union of its intervals. That is also what makes partial overlap
come out right: a scan sample covering one second and two detect samples inside
it contribute one second between them, not three.

    python -m pipeline.merge_surfaces output_videos/merged \
        led=output_videos/s5_base line2=output_videos/s5_L2

Writes <out>/exposure_by_brand.csv (same schema score.py and report read) and
<out>/exposure_detail.csv with a `surface` column naming which run each row
came from.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

DETAIL = "exposure_detail.csv"
BY_BRAND = "exposure_by_brand.csv"


def _read_detail(run_dir):
    path = os.path.join(run_dir, DETAIL)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- run `aggregate` on {run_dir} first.")
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _sample_period(run_dir):
    """
    (seconds each sampled row stands for, manifest).

    `detect` and `scan` both record their step, under their own stage key.
    Reading it from the manifest rather than assuming keeps a run that was
    sampled at a different rate from being silently mis-scaled.
    """
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    fps = float(m["video"]["fps"])
    stats = m.get("stats", {})
    for stage in ("detect", "scan"):
        st = stats.get(stage)
        if st and st.get("step"):
            return int(st["step"]) / fps, m
    raise SystemExit(f"{run_dir}: manifest has no detect or scan step -- "
                     f"cannot tell how much time each row covers.")


def _union_seconds(intervals):
    """Measure of the union of [start, end) intervals."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s)


def merge(out_dir, surfaces):
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    per_brand_spans = defaultdict(list)           # brand -> [(t0, t1)]
    per_brand_rows = defaultdict(list)
    paying = {}
    manifest = None
    periods = {}

    for label, run_dir in surfaces:
        spf, m = _sample_period(run_dir)
        periods[label] = spf
        manifest = manifest or m
        for r in _read_detail(run_dir):
            r["surface"] = label
            r["covers_s"] = spf
            brand = r["brand"]
            if not brand:
                continue
            paying[brand] = str(r.get("paying", "True")).strip().lower() in ("true", "1", "yes")
            t = float(r["t"])
            per_brand_spans[brand].append((t, t + spf))
            per_brand_rows[brand].append(r)
            rows.append(r)

    # Per-brand aggregate. Time is the union of frames; the size/position
    # figures stay means over every surface instance, because a brand shown on
    # two lines at once really does occupy both areas.
    duration = float(manifest["video"]["frames"]) / float(manifest["video"]["fps"])
    out_brand = []
    for brand, spans in per_brand_spans.items():
        rs = per_brand_rows[brand]
        secs = _union_seconds(spans)
        ts = sorted(float(r["t"]) for r in rs)
        n_surfaces = len({r["surface"] for r in rs})
        out_brand.append({
            "brand": brand,
            "paying": paying[brand],
            "exposure_s": round(secs, 2),
            "time_basis": "time_union",
            "time_is_upper_bound": False,
            "pct_of_video": round(100 * secs / duration, 2) if duration else 0.0,
            "n_frames": len({(r["surface"], r["frame"]) for r in rs}),
            "n_surfaces": n_surfaces,
            "surfaces": "+".join(sorted({r["surface"] for r in rs})),
            "mean_panels": round(_mean(rs, "n_panels"), 2),
            "mean_area_px": round(_mean(rs, "area_px"), 0),
            "mean_area_pct": round(_mean(rs, "area_pct"), 3),
            "mean_dist_center": round(_mean(rs, "dist_center"), 3),
            "mean_brightness": round(_mean(rs, "brightness"), 1),
            "mean_brightness_rel": round(_mean(rs, "brightness_rel"), 3),
            "first_seen_s": round(ts[0], 2),
            "last_seen_s": round(ts[-1], 2),
            "area_from_whole_band_pct": round(
                100 * sum(1 for r in rs if r.get("area_source") == "whole_band") / len(rs), 1),
            "identified_by": "+".join(sorted({r.get("identified_by", "?") for r in rs})),
        })
    out_brand.sort(key=lambda r: (not r["paying"], -r["exposure_s"]))

    with open(os.path.join(out_dir, BY_BRAND), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_brand[0].keys()))
        w.writeheader()
        w.writerows(out_brand)
    with open(os.path.join(out_dir, DETAIL), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Carry a manifest so downstream stages (report, score) can find the video.
    manifest = dict(manifest)
    manifest["merged_from"] = {label: run for label, run in surfaces}
    manifest.setdefault("stats", {})["merge_surfaces"] = {
        "surfaces": len(surfaces),
        "brands": len(out_brand),
        "paying_brands": sum(1 for r in out_brand if r["paying"]),
        "rows": len(rows),
        "sample_period_s": {k: round(v, 4) for k, v in periods.items()},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    paid = [r for r in out_brand if r["paying"]]
    print(f"\n  merged {len(surfaces)} surface(s) -> {len(out_brand)} brands "
          f"({len(paid)} paying)")
    print(f"  {'brand':<30} {'time':>8}  surfaces")
    for r in out_brand:
        tag = "" if r["paying"] else "   (not paying)"
        print(f"  {r['brand']:<30} {r['exposure_s']:7.1f}s  {r['surfaces']}{tag}")
    print(f"\n  Wrote {os.path.join(out_dir, BY_BRAND)}")
    return out_brand


def _mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("out_dir")
    ap.add_argument("surfaces", nargs="+", metavar="LABEL=RUN_DIR")
    a = ap.parse_args(argv)
    pairs = []
    for s in a.surfaces:
        if "=" not in s:
            raise SystemExit(f"expected LABEL=RUN_DIR, got {s!r}")
        label, run = s.split("=", 1)
        pairs.append((label, run))
    merge(a.out_dir, pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
