"""
Plain-text shot timeline for a segmented run, plus the moments the pipeline is
not sure about.

    python shot_report.py output_videos/clip_b

Two sections:

  TIMELINE   every second of the video, which kind of shot it is
  UNCERTAIN  time ranges the pipeline flags as shaky, with the reason -- so you
             can hold them next to your own list of "moments I'd double-check"

Writes the same text to <run>/shot_report.txt.
"""

import json
import os
import sys


def mmss(t):
    return f"{int(t) // 60}:{t % 60:04.1f}"


def load(run_dir):
    p = os.path.join(run_dir, "shots.jsonl")
    if not os.path.exists(p):
        sys.exit(f"no shots.jsonl in {run_dir} -- run `pipeline.cli segment` first")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- what "uncertain" means here ------------------------------------------
# Each check returns a short reason string, or None.

def _reasons(s):
    out = []
    if s["dur_s"] < 1.0:
        out.append(f"very short ({s['dur_s']:.2f}s) - likely an over-cut or a transition")
    if s["purity"] < 0.70:
        out.append(f"shot type unstable (frames agreed only {s['purity']*100:.0f}%)")
    if s.get("partial"):
        out.append("partial - some frames read fine, some do not")
    bc = s.get("band_conf")
    if bc is not None and 0.35 <= bc < 0.60:
        out.append(f"board read borderline (confidence {bc:.2f})")
    if s["cut_severity"] and s["cut_severity"] < 1.15 and s["shot"] > 0:
        out.append(f"cut into this shot was faint (severity {s['cut_severity']:.2f}) "
                   f"- may be the same shot as the previous one")
    return out


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "output_videos/clip_b"
    shots = load(run_dir)
    end = shots[-1]["t_end"]

    lines = []
    lines.append(f"SHOT TIMELINE  --  {os.path.basename(run_dir)}  ({mmss(end)} total, "
                 f"{len(shots)} shots)")
    lines.append("=" * 74)
    lines.append("")
    lines.append(f"{'#':>3}  {'from':>7} {'to':>7} {'len':>6}   {'kind':<10} {'handled by':<8}  note")
    lines.append("-" * 74)

    ROUTE = {"auto": "AUTO", "review": "you", "partial": "part"}
    for s in shots:
        note = ""
        if s["route"] == "auto":
            note = "board read ok, no check needed"
        elif s.get("partial"):
            note = "mostly ok; a few frames need you"
        else:
            note = s["reason"]
        lines.append(f"{s['shot']:>3}  {mmss(s['t_start']):>7} {mmss(s['t_end']):>7} "
                     f"{s['dur_s']:>5.1f}s   {s['type']:<10} {ROUTE[s['route']]:<8}  {note}")

    # --- uncertain section ---
    flagged = [(s, _reasons(s)) for s in shots]
    flagged = [(s, r) for s, r in flagged if r]

    lines.append("")
    lines.append("")
    lines.append(f"UNCERTAIN MOMENTS  --  {len(flagged)} of {len(shots)} shots")
    lines.append("=" * 74)
    lines.append("(check these in the app; compare with the ones you'd flag yourself)")
    lines.append("")
    if not flagged:
        lines.append("  none")
    for s, rs in flagged:
        lines.append(f"  {mmss(s['t_start'])} - {mmss(s['t_end'])}   "
                     f"shot {s['shot']} ({s['type']}, {s['dur_s']:.1f}s)")
        for r in rs:
            lines.append(f"      - {r}")
        lines.append("")

    # --- fast-cut clusters: 3+ short shots back to back ---
    runs, cur = [], []
    for s in shots:
        if s["dur_s"] < 1.5:
            cur.append(s)
        else:
            if len(cur) >= 3:
                runs.append(cur)
            cur = []
    if len(cur) >= 3:
        runs.append(cur)
    if runs:
        lines.append("")
        lines.append("FAST-CUT CLUSTERS  (many short shots in a row - segmentation is "
                     "least reliable here)")
        lines.append("=" * 74)
        for r in runs:
            lines.append(f"  {mmss(r[0]['t_start'])} - {mmss(r[-1]['t_end'])}   "
                         f"{len(r)} shots in {r[-1]['t_end']-r[0]['t_start']:.1f}s "
                         f"(shots {r[0]['shot']}-{r[-1]['shot']})")

    # --- summary ---
    by_type = {}
    for s in shots:
        by_type[s["type"]] = by_type.get(s["type"], 0.0) + s["dur_s"]
    lines.append("")
    lines.append("TIME BY SHOT KIND")
    lines.append("=" * 74)
    for k, v in sorted(by_type.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<12} {v:6.1f}s  ({100*v/end:4.1f}%)")
    auto_s = sum(s["dur_s"] for s in shots if s["route"] == "auto")
    lines.append("")
    lines.append(f"  handled automatically : {auto_s:.1f}s ({100*auto_s/end:.0f}%)")
    lines.append(f"  needs your review     : {end-auto_s:.1f}s ({100*(end-auto_s)/end:.0f}%)")

    text = "\n".join(lines)
    print(text)
    out = os.path.join(run_dir, "shot_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n(saved to {out})")


if __name__ == "__main__":
    main()
