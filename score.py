"""
Score a run's per-brand exposure against a hand count.

WHY THIS EXISTS. Three passes over the same 5-minute clip produced three
different answers for the same sponsor (Mastercard: 1.1s, then 18.7s, then
6.3s) and nothing in the repo could say which was closest. Without a target,
every "fix" to the measurement is a guess that moves numbers in an unknown
direction. This turns that into an arithmetic question.

WHAT IT MEASURES. Two failure modes matter to a client report, and they are not
the same:

  MISSED   a sponsor who was on the boards and got zero seconds. This is the
           one that loses the contract -- the club shows the report to the
           advertiser and the advertiser's own brand is not on it.
  ERROR    a sponsor who is on the report with the wrong number.

So the summary reports them separately rather than folding both into one
average. `captured` is the share of real board-seconds the run actually
credited to the right brand, counting neither over-credit nor under-credit as
success: sum(min(predicted, truth)) / sum(truth).

The hand count is approximate and says so. Treat a brand as healthy when it is
within roughly a fifth of the target, not when it matches to the second.

    python score.py output_videos/test5min
    python score.py output_videos/test5min --truth ground_truth/5min_ucl_final.json
"""

import argparse
import csv
import json
import os
import sys

TOL = 0.20          # within this fraction of the hand count reads as "ok"


def _norm(s):
    """Fold spelling differences so Lay's / LAYS / lays are one brand."""
    return "".join(c for c in s.upper() if c.isalnum())


def load_truth(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, {_norm(k): (k, float(v)) for k, v in data["brands"].items()}


def load_run(run_dir):
    """
    {normalised brand -> (name, seconds, time_basis)} for the PAYING brands only.

    Competition and broadcaster branding share the boards but are not
    advertisers, and the hand count deliberately leaves them out, so counting
    them here would score the run against something it was never asked for.
    """
    path = os.path.join(run_dir, "exposure_by_brand.csv")
    if not os.path.exists(path):
        raise SystemExit(f"No exposure_by_brand.csv in {run_dir} -- run `aggregate` first.")
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("paying", "True")).strip().lower() not in ("true", "1", "yes"):
                continue
            out[_norm(row["brand"])] = (row["brand"],
                                        float(row["exposure_s"]),
                                        row.get("time_basis", "?"))
    return out


def score(run_dir, truth_path):
    data, truth = load_truth(truth_path)
    run = load_run(run_dir)

    rows = []
    for key, (name, want) in sorted(truth.items(), key=lambda kv: -kv[1][1]):
        got, basis = 0.0, "-"
        if key in run:
            _, got, basis = run[key]
        err = got - want
        if want == 0.0:
            # A partner the hand count looked for and did not find. Reporting
            # zero is the right answer; inventing seconds for them is not.
            verdict = "ok" if got == 0.0 else "over"
        elif got == 0.0:
            verdict = "MISSED"
        elif abs(err) <= TOL * want:
            verdict = "ok"
        else:
            verdict = "under" if err < 0 else "over"
        rows.append((name, want, got, err, verdict, basis))

    extra = [(n, s, b) for k, (n, s, b) in run.items() if k not in truth]

    want_total = sum(r[1] for r in rows)
    got_total = sum(r[2] for r in rows)
    captured = sum(min(r[1], r[2]) for r in rows) / want_total if want_total else 0.0
    missed = [r for r in rows if r[4] == "MISSED"]
    okish = [r for r in rows if r[4] == "ok"]
    mae = sum(abs(r[3]) for r in rows) / len(rows) if rows else 0.0

    print(f"\nRun    : {run_dir}")
    print(f"Truth  : {truth_path}  ({data.get('method', '?')})")
    print(f"\n  {'brand':<24} {'hand':>7} {'run':>8} {'error':>8}   {'':<7} basis")
    print("  " + "-" * 66)
    for name, want, got, err, verdict, basis in rows:
        print(f"  {name:<24} {want:6.1f}s {got:7.1f}s {err:+7.1f}s   "
              f"{verdict:<7} {basis}")
    if extra:
        print("\n  reported but not in the hand count:")
        for name, secs, basis in sorted(extra, key=lambda e: -e[1]):
            print(f"    {name:<22} {secs:7.1f}s   {basis}")

    print(f"\n  captured   {100 * captured:5.1f}%  of the {want_total:.0f}s "
          f"the hand count says was on the boards")
    print(f"  totals     hand {want_total:.0f}s   run {got_total:.0f}s "
          f"({got_total - want_total:+.0f}s)")
    print(f"  per-brand  {len(okish)}/{len(rows)} within {int(100*TOL)}%, "
          f"mean absolute error {mae:.1f}s")
    if missed:
        print(f"  MISSED     {len(missed)} sponsor(s) got zero seconds: "
              + ", ".join(r[0] for r in missed))
    print()
    return {"captured": captured, "mae": mae, "missed": [r[0] for r in missed],
            "ok": len(okish), "brands": len(rows)}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:                                    # brand names may be Cyrillic
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--truth", default="ground_truth/pilot_match_5min.json")
    a = ap.parse_args(argv)
    score(a.run_dir, a.truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
