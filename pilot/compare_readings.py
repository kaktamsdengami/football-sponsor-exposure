"""
Compare one reader's brand readings against another's, group by group.

Used to decide whether a cheaper/faster model can replace the one that produced
the readings we trust. `score.py` answers "how close are the final seconds";
this answers the question you need FIRST: does this reader see the same things
on the same frames? A model can land plausible totals while being wrong
everywhere, if its misses and its false positives happen to cancel.

Reports, per brand:

    recall      of the groups where the reference saw it, how many did this
                reader also see. Low recall = brands going missing = the
                failure that loses a contract.
    precision   of the groups where this reader claimed it, how many did the
                reference agree with. Low precision = invented exposure.

Only groups BOTH readers covered are compared, so a partial run scores fairly
on what it actually did.

    python pilot/compare_readings.py output_videos/s5v/haiku_readings.jsonl \
        --ref output_videos/s5v/readings.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def _load_any(path):
    """{gid -> set(brand)} from either readings shape."""
    out = {}
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                names = set()
                for b in r.get("brands") or []:
                    # accept both ["NAME"] and [{"name": "NAME", ...}]
                    names.add(b if isinstance(b, str) else b.get("name", ""))
                out[int(r["gid"])] = {n for n in names if n}
        return out
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        if k.startswith("_"):
            continue
        names = {b if isinstance(b, str) else b.get("name", "") for b in v}
        out[int(k)] = {n for n in names if n}
    return out


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("candidate")
    ap.add_argument("--ref", required=True, help="the readings to trust")
    ap.add_argument("--seconds-per-group", type=float, default=1.0,
                    help="only used to express disagreement in seconds")
    a = ap.parse_args(argv)

    cand, ref = _load_any(a.candidate), _load_any(a.ref)
    shared = sorted(set(cand) & set(ref))
    if not shared:
        raise SystemExit("No group ids in common -- did the reader record `gid`?")

    tp, fn, fp = defaultdict(int), defaultdict(int), defaultdict(int)
    exact = 0
    for g in shared:
        c, r = cand[g], ref[g]
        exact += (c == r)
        for b in r & c:
            tp[b] += 1
        for b in r - c:
            fn[b] += 1
        for b in c - r:
            fp[b] += 1

    brands = sorted(set(tp) | set(fn) | set(fp),
                    key=lambda b: -(tp[b] + fn[b]))
    print(f"\ncandidate : {a.candidate}")
    print(f"reference : {a.ref}")
    print(f"compared  : {len(shared)} groups both read "
          f"({len(cand)} vs {len(ref)} total)")
    print(f"identical : {exact}/{len(shared)} groups matched exactly "
          f"({100*exact/len(shared):.0f}%)\n")
    print(f"  {'brand':<28} {'ref':>4} {'hit':>4} {'miss':>5} {'extra':>6} "
          f"{'recall':>7} {'prec':>6}")
    print("  " + "-" * 66)
    for b in brands:
        n_ref = tp[b] + fn[b]
        rec = tp[b] / n_ref if n_ref else float("nan")
        prec = tp[b] / (tp[b] + fp[b]) if (tp[b] + fp[b]) else float("nan")
        flag = ""
        if n_ref >= 3 and rec < 0.6:
            flag = "  <- MISSING IT"
        elif fp[b] >= 3 and (prec != prec or prec < 0.6):
            flag = "  <- INVENTING IT"
        print(f"  {b:<28} {n_ref:4d} {tp[b]:4d} {fn[b]:5d} {fp[b]:6d} "
              f"{rec:7.2f} {prec:6.2f}{flag}")

    t_tp = sum(tp.values()); t_fn = sum(fn.values()); t_fp = sum(fp.values())
    print(f"\n  overall recall    {t_tp/(t_tp+t_fn) if t_tp+t_fn else 0:.2f}"
          f"   ({t_fn} sightings missed)")
    print(f"  overall precision {t_tp/(t_tp+t_fp) if t_tp+t_fp else 0:.2f}"
          f"   ({t_fp} sightings invented)")
    print("\n  Recall is the one that matters: a missed brand is a sponsor "
          "reading zero\n  on their own report. Judge the candidate on that "
          "first.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
