"""
Compare several `scan` runs against the hand count, per brand, over a common window.

`score.py` scores one finished run. This answers the question you actually ask
while tuning: "did that flag help, and WHICH brands did it help?" -- and it does
it on partial runs, so a 120-frame probe can be judged without waiting for a
full pass.

The per-brand column is the point. The headline number hid that upscaling lifts
small text and does nothing at all for low-contrast text; only the split showed
that two brands needed different fixes.

    python compare_scans.py output_videos/s5scan output_videos/s5up
    python compare_scans.py base=output_videos/s5scan clahe=output_videos/s5clahe

Runs are compared over the SHORTEST window any of them covers, so a probe and a
full run are still comparable.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

GT = "ground_truth/pilot_match_5min_observations.json"
SPONSORS = "pilot/sponsors.json"


def load_scan(run_dir):
    """
    (brand -> {frame: t}, last t covered). Works on a partially written file.

    Times are kept per frame, not just frame indices, because runs are compared
    over the SHORTEST window any of them reached -- and a full run must then be
    cut down to that window too. Counting all of the longer run's frames against
    a truth restricted to the short window flatters it wildly (it put one brand at
    139s inside a 119s window).
    """
    path = os.path.join(run_dir, "scan.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- is that a scan run?")
    brands, tmax = defaultdict(dict), 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue            # a run still in flight ends mid-line
            t = float(r["t"])
            tmax = max(tmax, t)
            for h in r.get("hits") or []:
                brands[h["brand"]][r["frame"]] = t
    return brands, tmax


def clip_to(brands, tmax):
    """Drop frames past the common window."""
    return {b: {f for f, t in fr.items() if t <= tmax + 1e-9}
            for b, fr in brands.items()}


def load_truth(path, tmax, every_default=5.0):
    with open(path, encoding="utf-8") as f:
        obs = json.load(f)
    every = float(obs.get("every_s", every_default))
    gt = defaultdict(float)
    for s in obs["samples"]:
        if float(s["t"]) <= tmax:
            for b in s["brands"]:
                gt[b] += every
    return gt


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("runs", nargs="+", metavar="[LABEL=]RUN_DIR")
    ap.add_argument("--truth", default=GT)
    ap.add_argument("--sponsors", default=SPONSORS)
    ap.add_argument("--all-brands", action="store_true",
                    help="include club/stadium/league items, not just advertisers")
    a = ap.parse_args(argv)

    labels, dirs = [], []
    for r in a.runs:
        label, d = (r.split("=", 1) if "=" in r else (os.path.basename(r), r))
        labels.append(label)
        dirs.append(d)

    raw = [load_scan(d) for d in dirs]
    tmax = min(t for _, t in raw)
    # Cut every run down to the common window before counting anything.
    loaded = [(clip_to(br, tmax), t) for br, t in raw]
    fps_note = f"common window 0-{tmax:.0f}s"

    with open(a.sponsors, encoding="utf-8") as f:
        paying = {s["name"]: bool(s.get("paying", True))
                  for s in json.load(f)["sponsors"]}
    gt = load_truth(a.truth, tmax)

    # A scan sample stands for one sampling interval; frames are counted, and
    # each is worth (frame step / fps). Read that from the run manifest when it
    # is there, else assume the 1/s default the probes use.
    def spf(run_dir):
        p = os.path.join(run_dir, "manifest.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            st = m.get("stats", {}).get("scan", {})
            if st.get("step") and m.get("video", {}).get("fps"):
                return st["step"] / float(m["video"]["fps"])
        return 1.0

    periods = [spf(d) for d in dirs]

    keys = [b for b in sorted(set(gt) | {k for br, _ in loaded for k in br},
                              key=lambda x: -gt.get(x, 0.0))
            if a.all_brands or paying.get(b, True)]

    print(f"\n{fps_note}   truth: {a.truth}")
    head = "  " + f"{'brand':<26} {'hand':>6}" + "".join(f"{l:>9}" for l in labels)
    print("\n" + head)
    print("  " + "-" * (len(head) - 2))
    for b in keys:
        cells = "".join(f"{len(br.get(b, ())) * p:8.0f}s"
                        for (br, _), p in zip(loaded, periods))
        print(f"  {b:<26} {gt.get(b, 0.0):5.0f}s{cells}")

    print()
    total = sum(v for k, v in gt.items() if a.all_brands or paying.get(k, True))
    for label, (br, _), p in zip(labels, loaded, periods):
        got = {k: len(v) * p for k, v in br.items()
               if a.all_brands or paying.get(k, True)}
        cap = sum(min(gt.get(k, 0.0), got.get(k, 0.0)) for k in gt
                  if a.all_brands or paying.get(k, True))
        scored = [k for k in gt if a.all_brands or paying.get(k, True)]
        mae = (sum(abs(got.get(k, 0.0) - gt[k]) for k in scored) / len(scored)
               if scored else 0.0)
        missed = [k for k in scored if gt[k] > 0 and not got.get(k)]
        print(f"  {label:<12} captured {100 * cap / total if total else 0:5.1f}%   "
              f"MAE {mae:5.1f}s   missed {len(missed)}"
              + (f"  ({', '.join(missed[:4])})" if missed else ""))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
