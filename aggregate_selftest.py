"""
Check the aggregate arithmetic on synthetic data -- no video needed.

    python aggregate_selftest.py

The distinction it guards is easy to get wrong in either direction. A perimeter
repeats its advertisers: four `mastercard` panels in one shot are

    ONE brand for TIME       -- the brand was on screen for one frame's worth,
                                not four
    FOUR panels for SURFACE  -- four panels of ink is what the viewer saw

Getting the first wrong inflates a sponsor's seconds; getting the second wrong
shrinks its surface to a quarter. Both are silent, and neither shows up without
a case where the two answers differ.

Fixtures are a 1000x1000 frame with three brands on the same three frames:
REPEATED (4 panels), SINGLE (1 identical panel), NOBOX (no coordinates at all,
so it must fall back to the whole band and be flagged as over-stated).
"""

import csv
import json
import math
import os
import shutil
import tempfile

from pipeline.aggregate import run_aggregate

W = H = 1000
PANEL = (100, 50)                       # 5000 px each


def _rect(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _fixture(d):
    os.makedirs(os.path.join(d, "bands"), exist_ok=True)
    json.dump({"run": "selftest",
               "video": {"path": "none.mp4", "fps": 25.0, "frames": 250,
                         "width": W, "height": H},
               "stats": {"detect": {"step": 5}}},      # 0.2s per sampled frame
              open(os.path.join(d, "manifest.json"), "w"))

    with open(os.path.join(d, "detections.jsonl"), "w") as f:
        for fr in (0, 5, 10):
            f.write(json.dumps({
                "frame": fr, "t": fr / 25.0, "band": f"bands/{fr}.jpg",
                "coef": [0.0, 500.0], "span": [100, 900], "fit_med": 500.0,
                "local_clamp": [0.8, 1.25], "H": 50, "over": 17, "conf": 1.0,
            }) + "\n")

    pw, ph = PANEL
    with open(os.path.join(d, "creatives.jsonl"), "w") as f:
        f.write(json.dumps({
            "creative": 0, "n_frames": 3, "seconds": 0.6, "frames": [0, 5, 10],
            "rep_frame": 0, "rep_band": "bands/0.jpg", "rep_conf": 1.0,
            "spans": [], "t_first": 0.0, "t_last": 0.4,
            "brands": [
                {"name": "REPEATED", "source": "ocr",
                 "instances": 4,
                 "polys": [_rect(x, 470, pw, ph) for x in (100, 300, 500, 700)]},
                {"name": "SINGLE", "source": "ocr",
                 "instances": 1, "polys": [_rect(100, 470, pw, ph)]},
                {"name": "NOBOX", "source": "reading",
                 "instances": 1, "polys": []},
            ]}) + "\n")


def main():
    d = os.path.join(tempfile.gettempdir(), "aggregate_selftest")
    shutil.rmtree(d, ignore_errors=True)
    _fixture(d)
    run_aggregate(d, brightness=False, progress=False)

    rows = {r["brand"]: r for r in csv.DictReader(
        open(os.path.join(d, "exposure_by_brand.csv"), encoding="utf-8-sig"))}
    one_panel = PANEL[0] * PANEL[1]
    # area-weighted centre of the four panels sits at x=450, y=495
    want_dist = round(math.hypot(450 - W / 2, 495 - H / 2)
                      / (math.hypot(W, H) / 2), 3)

    checks = [
        ("time is not multiplied by repetition",
         float(rows["REPEATED"]["exposure_s"]), 0.6),
        ("a single-panel brand gets the same time",
         float(rows["SINGLE"]["exposure_s"]), 0.6),
        ("surface sums every panel",
         float(rows["REPEATED"]["mean_area_px"]), one_panel * 4),
        ("a single panel stays one panel",
         float(rows["SINGLE"]["mean_area_px"]), one_panel),
        ("repeated surface is exactly 4x single",
         float(rows["REPEATED"]["mean_area_px"])
         / float(rows["SINGLE"]["mean_area_px"]), 4.0),
        ("panel count is reported",
         float(rows["REPEATED"]["mean_panels"]), 4.0),
        ("position is the area-weighted centre",
         float(rows["REPEATED"]["mean_dist_center"]), want_dist),
        ("a brand with no box falls back to the whole band",
         float(rows["NOBOX"]["area_from_whole_band_pct"]), 100.0),
    ]

    print("\n--- CHECKS ---")
    ok = True
    for label, got, want in checks:
        good = abs(got - want) < 1e-3
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label}: {got} (expected {want})")
    print("\n  " + ("ALL PASS" if ok else "SOME FAILED"))
    shutil.rmtree(d, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
