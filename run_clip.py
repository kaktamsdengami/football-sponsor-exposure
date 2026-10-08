"""
One broadcast video in, one per-advertiser exposure report out. Unattended.

This is the whole product as a single command. It runs every stage in order,
reads both far-touchline advertising lines, merges them into one per-brand
table, writes the coverage ledger, and -- when a hand count exists for the clip
-- prints the score so the run's accuracy is visible in the run itself rather
than in someone's memory.

    python run_clip.py --video input_videos/pilot_match_5min.mp4 --run-name s5

Nothing here needs a human. `annotate.py` still exists for the reviewed path,
but a run of this script is complete on its own.

Per-line work goes to <run>_L1 / <run>_L2 and the merged answer to <run>; the
per-line directories are kept because that is where the band images and the
identify diagnostics live when a number needs explaining.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable


def sh(args, label):
    """Run a stage, echo it, and stop the run if it fails."""
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}")
    t0 = time.perf_counter()
    r = subprocess.run([PY] + args, text=True)
    if r.returncode != 0:
        raise SystemExit(f"\n{label} failed (exit {r.returncode}) -- stopping.")
    print(f"  [{label} took {time.perf_counter() - t0:.0f}s]")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-root", default="output_videos")
    ap.add_argument("--sponsors", default="pilot/sponsors.json",
                    help="the club's advertiser list")
    ap.add_argument("--lines", default="1,2",
                    help="which touchline advertising lines to read (comma "
                         "separated). 1 = pitch-level LED, 2 = the line above it. "
                         "Empty string runs the scan only")
    ap.add_argument("--scan", action="store_true",
                    help="also run the geometry-free whole-frame reader and "
                         "merge it in. Catches the framings the touchline model "
                         "cannot fit -- goal cam, corner cam, dugout hoarding")
    ap.add_argument("--scan-fps", type=float, default=1.0)
    ap.add_argument("--scan-upscale", type=float, default=1.0,
                    help="magnify before OCR; helps small far-side board text")
    ap.add_argument("--sample-fps", type=float, default=4.0)
    ap.add_argument("--ocr-every", type=int, default=1,
                    help="OCR every Nth band frame per group. 1 is right for a "
                         "5-minute clip; raise it for a full match")
    ap.add_argument("--min-unknown-seconds", type=float, default=4.0,
                    help="board time an unlisted but readable brand must hold "
                         "before it is reported as an advertiser nobody listed")
    ap.add_argument("--truth", default=None,
                    help="hand count to score against; defaults to "
                         "ground_truth/<video stem>.json when that file exists")
    ap.add_argument("--report-config", default="pilot/report_config.json")
    ap.add_argument("--debug-overlay", action="store_true",
                    help="keep the outlined frames, so a number can be looked at")
    ap.add_argument("--skip-segment", action="store_true",
                    help="reuse shots.jsonl from a previous run of this name")
    a = ap.parse_args(argv)

    lines = [int(x) for x in a.lines.split(",") if x.strip()]
    base = os.path.join(a.out_root, a.run_name)
    line_dirs = {L: f"{base}_L{L}" for L in lines}
    t_start = time.perf_counter()

    print(f"video      {a.video}")
    print(f"run        {base}")
    print(f"lines      {lines}")
    print(f"sponsors   {a.sponsors}")

    # --- pass 0: shots. Cut once and share across the lines; the shot
    # boundaries do not depend on which board row we then read.
    shots_src = os.path.join(base, "shots.jsonl")
    if lines and not (a.skip_segment and os.path.exists(shots_src)):
        sh(["-m", "pipeline.cli", "segment", "--video", a.video,
            "--out-root", a.out_root, "--run-name", a.run_name], "segment")

    # --- per line: detect -> dedup -> identify
    for L in lines:
        d = line_dirs[L]
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(shots_src, os.path.join(d, "shots.jsonl"))
        shutil.copyfile(a.sponsors, os.path.join(d, "sponsors.json"))

        cmd = ["-m", "pipeline.cli", "detect", "--video", a.video,
               "--out-root", a.out_root, "--run-name", f"{a.run_name}_L{L}",
               "--sample-fps", str(a.sample_fps), "--band-line", str(L)]
        if a.debug_overlay:
            cmd.append("--debug-overlay")
        sh(cmd, f"detect line {L}")
        sh(["-m", "pipeline.cli", "dedup", d], f"dedup line {L}")
        sh(["-m", "pipeline.cli", "identify", d,
            "--sponsors", os.path.join(d, "sponsors.json"),
            "--ocr-every", str(a.ocr_every),
            "--min-unknown-seconds", str(a.min_unknown_seconds)],
           f"identify line {L}")
        sh(["-m", "pipeline.cli", "aggregate", d], f"aggregate line {L}")

    # --- geometry-free pass over the whole frame
    scan_dir = f"{base}_scan"
    if a.scan:
        os.makedirs(scan_dir, exist_ok=True)
        shutil.copyfile(a.sponsors, os.path.join(scan_dir, "sponsors.json"))
        sh(["-m", "pipeline.cli", "scan", scan_dir, "--video", a.video,
            "--sponsors", os.path.join(scan_dir, "sponsors.json"),
            "--scan-fps", str(a.scan_fps),
            "--upscale", str(a.scan_upscale)], "scan (whole frame)")

    # --- merge every surface into the one table the client sees
    labels = {1: "led", 2: "line2"}
    parts = [f"{labels.get(L, f'line{L}')}={line_dirs[L]}" for L in lines]
    if a.scan:
        parts.append(f"scan={scan_dir}")
    if not parts:
        raise SystemExit("Nothing to merge: pass --lines and/or --scan.")
    sh(["-m", "pipeline.merge_surfaces", base] + parts, "merge surfaces")

    # --- honesty: which seconds were measured, and which were not.
    # One ledger per surface, because each answers a different question: the
    # band path says how much of the match it could fit a touchline to, the
    # scan path says how much of it named an advertiser and how much was read
    # and came back blank.
    if lines:
        sh(["-m", "pipeline.cli", "coverage", line_dirs[lines[0]]],
           "coverage (band path)")
    if a.scan:
        sh(["-m", "pipeline.cli", "coverage", scan_dir], "coverage (scan path)")

    # --- client report, when a config for it exists
    if os.path.exists(a.report_config):
        shutil.copyfile(a.report_config, os.path.join(base, "report_config.json"))
        try:
            sh(["-m", "pipeline.cli", "report", base,
                "--config", os.path.join(base, "report_config.json")], "report")
        except SystemExit as e:
            print(f"  report skipped: {e}")

    # --- score. A run that cannot say how accurate it is has not finished.
    truth = a.truth
    if truth is None:
        stem = os.path.splitext(os.path.basename(a.video))[0]
        guess = os.path.join("ground_truth", f"{stem}.json")
        truth = guess if os.path.exists(guess) else None
    if truth:
        sh(["score.py", base, "--truth", truth], "score")
    else:
        print("\n  No hand count for this video -- accuracy is UNVERIFIED.\n"
              "  Build one with ground_truth/build_samples.py + "
              "aggregate_truth.py before quoting these numbers to a client.")

    print(f"\n  whole run: {time.perf_counter() - t_start:.0f}s -> {base}")
    _summarise(base)
    return 0


def _summarise(base):
    path = os.path.join(base, "manifest.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    st = m.get("stats", {}).get("merge_surfaces", {})
    if st:
        print(f"  {st.get('paying_brands', '?')} paying advertisers across "
              f"{st.get('surfaces', '?')} surface(s) -> "
              f"{os.path.join(base, 'exposure_by_brand.csv')}")


if __name__ == "__main__":
    raise SystemExit(main())
