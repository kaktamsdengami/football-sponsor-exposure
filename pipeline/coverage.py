"""
Build the coverage ledger for a run: which seconds of the match were measured,
and for the rest, why not.

Every second of the match ends up in exactly one span. A span is "measured" when
something actually looked at that interval:

    human    a person annotated a frame there -- INCLUDING when they marked it
             empty. Looking and finding nothing is a measurement; not looking is
             a gap, and the two must never be confused.
    auto     the geometric fit ran there and its band confidence held up
    scan     the whole frame was read with no geometry and named an advertiser
    tracked  propagated from an anchor by the tracker (not built yet)

`scan_blank` is its own status and not a gap: the whole frame WAS read and no
advertiser came out of it. That is two different situations the pipeline cannot
tell apart -- nothing was on offer (a close-up, a beauty shot: 11 of the 60
hand-count samples are exactly this) or a board was there and could not be read.
Calling it "measured" would overclaim and calling it "uncovered" would blame a
stage that did its job, so it is reported separately and the report must not
fold it into either.

Everything else is "uncovered", with a reason naming which stage should have
covered it. That is the point: a gap has an owner.

What one observation is assumed to prove is a knob, not a fact. Until the
tracker exists, a single annotated frame proves presence at that instant and
almost nothing about the seconds around it, so `--obs-window` defaults small and
deliberately makes the ledger look bad. It is supposed to look bad -- that is
the measurement gap the tracker is being built to close.
"""

import json
import os

from .store import RunStore
from .timeline import (
    clamp_spans, coverage_span, merge_adjacent, validate_partition,
)

# Confidence at or above which an automatic band read counts as a measurement.
AUTO_CONF_THR = 0.55

# Which evidence wins when two observations claim the same instant. A person who
# looked beats a fit that ran; a fit beats a whole-frame read; any of those beat
# a propagated guess; and all of them beat a frame that was read and yielded
# nothing.
_PRIORITY = {"human": 5, "auto": 4, "scan": 3, "tracked": 2, "scan_blank": 1}


def _shot_at(shots, t):
    for s in shots:
        if s["t_start"] <= t <= s["t_end"]:
            return s
    return None


# detect reads these shot kinds regardless of how segment routed the shot.
_DETECT_TYPES = ("wide_play", "medium")


def _gap_reason(shot, ran_detect, ran_scan=False):
    """Name what owes this interval a measurement, and why it is missing."""
    if shot is None:
        if ran_scan:
            # A scan-only run has no shots at all; saying "segmentation gap"
            # would blame a stage nobody asked to run.
            return "not sampled by scan (between samples, or past --max-frames)"
        return "outside every detected shot (segmentation gap)"
    if shot["type"] in _DETECT_TYPES:
        if not ran_detect:
            return f"{shot['type']} shot: detect has not run over this span"
        # detect did run here, so the band was read and rejected, not skipped --
        # an unreadable moment. A human anchor here would still add a lot.
        return (f"{shot['type']} shot: board read too poorly to count automatically "
                "-- needs a human")
    return f"{shot['type']} shot: no human anchor at this instant"


def run_coverage(run_dir, obs_window=0.5, auto_conf=AUTO_CONF_THR, top=12):
    """Build coverage.jsonl. Returns (store, stats)."""
    out_root, name = os.path.split(os.path.normpath(run_dir))
    store = RunStore(out_root, name, create=False)
    has_shots = os.path.exists(store.shots_path)

    manifest = store.read_manifest()
    info = manifest.get("video", {})
    fps = float(info.get("fps") or 25.0)
    duration = float(info.get("frames") or 0) / fps
    if duration <= 0:
        raise RuntimeError(f"{run_dir}: manifest has no video length -- "
                           f"run detect or scan first.")
    shots = store.read_shots() if has_shots else []
    hw = obs_window / 2.0

    # Board states (optional -- `states` + `identify`). A state marked `unread`
    # is a stretch where the geometric fit DID measure a board but no advertiser
    # could be identified on it. That is measured-but-not-named: a real gap in
    # the report, and one the old ledger could not see because "auto" only ever
    # meant "the fit ran".
    unread_ranges = []
    states_path = os.path.join(store.dir, "states.jsonl")
    ran_states = os.path.exists(states_path)
    if ran_states:
        for line in open(states_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("status") == "unread":
                unread_ranges.append((float(s["t_first"]), float(s["t_last"])))

    def _advertiser_tag(status, t_mid):
        if not ran_states or status != "auto":
            return None
        return ("unread" if any(a <= t_mid <= b for a, b in unread_ranges)
                else "named")

    measured = []      # (t_start, t_end, status, source, conf, shot)

    # --- human observations -------------------------------------------------
    n_annot = n_empty = 0
    for a in store.read_annotations():
        n_annot += 1
        n_empty += bool(a.get("empty"))
        t = float(a["t"])
        measured.append((t - hw, t + hw, "human", "human", 1.0, a.get("shot")))

    # --- automatic observations --------------------------------------------
    n_det = n_det_ok = 0
    det_hw = hw
    dets = list(store.read_detections())
    if dets:
        # Each detection stands for half a sampling interval either side.
        step_s = manifest.get("stats", {}).get("step", 0) / fps
        det_hw = (step_s / 2.0) if step_s else hw
        for d in dets:
            n_det += 1
            if d.get("band") is None or float(d.get("conf", 0.0)) < auto_conf:
                continue
            n_det_ok += 1
            t = float(d["t"])
            measured.append((t - det_hw, t + det_hw, "auto", "auto",
                             float(d.get("conf", 0.0)), None))

    # --- whole-frame scan observations --------------------------------------
    # scan.jsonl has one record per sampled frame, whether or not it named
    # anything. A frame that was read and yielded nothing is NOT a gap: the work
    # was done. It gets its own status so the report cannot quietly count it as
    # measured advertising time.
    n_scan = n_scan_named = 0
    scan_step = manifest.get("stats", {}).get("scan", {}).get("step", 0)
    scan_hw = (scan_step / fps / 2.0) if scan_step else hw
    scan_path = os.path.join(store.dir, "scan.jsonl")
    llm_path = os.path.join(store.dir, "scan_llm.jsonl")
    ran_scan = os.path.exists(scan_path) or os.path.exists(llm_path)

    if os.path.exists(scan_path):
        with open(scan_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n_scan += 1
                t = float(r["t"])
                hits = r.get("hits") or []
                if hits:
                    n_scan_named += 1
                    conf = max(float(h.get("ocr_conf", 0.0)) for h in hits)
                    measured.append((t - scan_hw, t + scan_hw, "scan", "scan",
                                     conf, None))
                else:
                    measured.append((t - scan_hw, t + scan_hw, "scan_blank",
                                     "scan", 0.0, None))

    # The vision path (API or readings) records one row per GROUP, each standing
    # for the frames it represents -- not one row per frame. Without this the
    # ledger called a fully-read run 100% uncovered, which is the exact failure
    # the coverage stage exists to prevent, aimed at itself.
    if os.path.exists(llm_path):
        with open(llm_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                members = int(r.get("members", 1))
                n_scan += members
                t0 = float(r["rep_t"])
                span = members * scan_hw * 2.0
                if r.get("brands"):
                    n_scan_named += members
                    measured.append((t0, t0 + span, "scan", "vision", 1.0, None))
                else:
                    measured.append((t0, t0 + span, "scan_blank", "vision",
                                     0.0, None))

    # --- tracked observations ----------------------------------------------
    # exposure.jsonl holds one record per tracked frame. Each stands for half a
    # sampling interval either side, the same way a detection does.
    n_track = 0
    exp_path = os.path.join(store.dir, "exposure.jsonl")
    if os.path.exists(exp_path):
        seen = {}
        with open(exp_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r["source"].startswith("tracked"):
                    continue
                # Several brands can be tracked on the same frame; the frame is
                # measured once, at the best confidence available on it.
                k = r["frame"]
                if r["conf"] > seen.get(k, (None, -1.0))[1]:
                    seen[k] = (r, r["conf"])
        frames = sorted(seen)
        thw = (frames[1] - frames[0]) / fps / 2.0 if len(frames) > 1 else hw
        for k in frames:
            r = seen[k][0]
            n_track += 1
            measured.append((r["t"] - thw, r["t"] + thw, "tracked", r["source"],
                             r["conf"], r["shot"]))

    measured = clamp_spans(measured, 0.0, duration)

    # --- assemble the partition --------------------------------------------
    # Observation windows overlap freely (two annotated frames a third of a
    # second apart both claim half a second), so the ledger is built on
    # elementary intervals between all endpoints. Every point then belongs to
    # exactly one span by construction, and where observations collide the
    # stronger evidence wins.
    # Snap every boundary to a real frame time. Without this, two observation
    # windows that touch produce sub-frame slivers that clutter the ledger with
    # zero-length "gaps" nobody can act on.
    def snap(x):
        return round(round(x * fps) / fps, 6)

    edges = sorted({0.0, round(duration, 6)}
                   | {snap(x) for a, b, *_ in measured for x in (a, b)})
    spans = []
    min_span = 0.5 / fps          # anything shorter than half a frame is noise
    for a, b in zip(edges, edges[1:]):
        if b - a < min_span:
            continue
        mid = (a + b) / 2.0
        here = [m for m in measured if m[0] <= mid <= m[1]]
        if here:
            best = max(here, key=lambda m: (_PRIORITY.get(m[2], 0), m[4] or 0.0))
            sp = coverage_span(a, b, best[2], "measured",
                               shot=best[5], source=best[3], conf=best[4])
            sp["advertiser"] = _advertiser_tag(best[2], mid)
            spans.append(sp)
        else:
            shot = _shot_at(shots, mid)
            sp = coverage_span(a, b, "uncovered",
                               _gap_reason(shot, bool(dets), ran_scan),
                               shot=shot["shot"] if shot else None)
            sp["advertiser"] = None
            spans.append(sp)

    spans = merge_adjacent(spans, key=("status", "reason", "shot", "advertiser"))
    validate_partition(spans, 0.0, duration)

    with open(store.dir + "/coverage.jsonl", "w", encoding="utf-8") as f:
        for s in spans:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    by_status = {}
    for s in spans:
        by_status[s["status"]] = by_status.get(s["status"], 0.0) + s["dur_s"]
    covered = (duration - by_status.get("uncovered", 0.0)
               - by_status.get("scan_blank", 0.0))
    gaps = sorted((s for s in spans if s["status"] == "uncovered"),
                  key=lambda s: -s["dur_s"])

    unread_spans = sorted((s for s in spans if s.get("advertiser") == "unread"),
                          key=lambda s: -s["dur_s"])
    unread_board_s = sum(s["dur_s"] for s in unread_spans)
    named_board_s = sum(s["dur_s"] for s in spans if s.get("advertiser") == "named")

    stats = {
        "duration_s": round(duration, 2),
        "measured_s": round(covered, 2),
        "measured_share": round(covered / duration, 4) if duration else 0.0,
        "by_status": {k: round(v, 2) for k, v in by_status.items()},
        "spans": len(spans),
        "gaps": len(gaps),
        "largest_gap_s": round(gaps[0]["dur_s"], 2) if gaps else 0.0,
        "annotations": n_annot,
        "annotations_empty": n_empty,
        "detections": n_det,
        "detections_used": n_det_ok,
        "scan_frames": n_scan,
        "scan_frames_named": n_scan_named,
        "tracked_frames": n_track,
        "obs_window_s": obs_window,
    }
    if ran_states:
        stats["named_board_s"] = round(named_board_s, 2)
        stats["unread_board_s"] = round(unread_board_s, 2)
        stats["unread_board_spans"] = len(unread_spans)

    print(f"Coverage : {store.dir}")
    print(f"  match         {duration:8.2f}s")
    print(f"  MEASURED      {covered:8.2f}s  ({100 * covered / max(duration, 1e-9):3.0f} %)")
    for k in ("human", "auto", "scan", "tracked"):
        if by_status.get(k):
            print(f"    {k:<10s}  {by_status[k]:8.2f}s")
    if by_status.get("scan_blank"):
        blank = by_status["scan_blank"]
        print(f"  READ, NOTHING FOUND {blank:6.2f}s  "
              f"({100 * blank / max(duration, 1e-9):3.0f} %)")
        print( "    the whole frame was read and named no advertiser. Either "
               "nothing was on")
        print( "    offer (close-up, beauty shot) or a board was there and could "
               "not be read.")
        print( "    Do NOT report this as measured advertising time.")
    print(f"  uncovered     {by_status.get('uncovered', 0.0):8.2f}s  "
          f"({100 * by_status.get('uncovered', 0.0) / max(duration, 1e-9):3.0f} %)"
          f"  in {len(gaps)} gaps")
    if ran_states:
        print(f"  of the measured time, board on screen but NO advertiser "
              f"identified: {unread_board_s:8.2f}s  in {len(unread_spans)} spans")
    if not dets:
        print("  (detect has not run on this run -- no automatic coverage)")
    if not n_track and not os.path.exists(exp_path):
        print("  (no exposure.jsonl -- run `expose` to fold in detect + tracking)")
    print(f"\n  Largest gaps:")
    for g in gaps[:top]:
        sh = "  -" if g["shot"] is None else f"{g['shot']:3d}"
        print(f"    {g['t_start']:6.2f}-{g['t_end']:6.2f}s {g['dur_s']:6.2f}s  "
              f"shot {sh}  {g['reason']}")
    if unread_spans:
        print(f"\n  Largest measured-but-unnamed spans (board visible, advertiser "
              f"unread):")
        for g in unread_spans[:top]:
            sh = "  -" if g["shot"] is None else f"{g['shot']:3d}"
            print(f"    {g['t_start']:6.2f}-{g['t_end']:6.2f}s {g['dur_s']:6.2f}s  "
                  f"shot {sh}")
    # Persist the ledger's numbers. Without this the stats existed only on
    # stdout, so `report` -- which reads them from the manifest -- printed
    # "coverage ledger unavailable" on every report it has ever produced, and the
    # honesty section the whole stage exists to feed was always empty.
    store.write_manifest({"obs_window": obs_window, "auto_conf": auto_conf},
                         manifest.get("video"), stats, stage="coverage")

    print(f"\n  Ledger: {store.dir}/coverage.jsonl  ({len(spans)} spans)")
    return store, stats
