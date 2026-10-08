"""
PASS 0 -- segmentation: video -> shots.jsonl.

One cheap pass (everything on a 480px downscale, no network, no full-res band
work) that cuts the match into shots, labels each one, and makes the routing
decision the rest of the pipeline hangs off:

    route = "auto"    the main camera; band extraction can be trusted, no human
            "review"  everything else -- close-ups, celebrations, low or zoomed
                      cameras, interviews, graphics. The band pipeline either
                      cannot see the inventory or reads it wrong, so a person
                      names the brands instead.

Routing is deliberately biased toward "review": over-routing costs a few
seconds of someone's time, under-routing silently loses a sponsor.

Two fields exist for the annotator's benefit:

    cut_severity  how drastic the cut that STARTED this shot was, in units of
                  the detection threshold. Two replays of the same action from
                  different angles score high here even though their colour
                  histograms are nearly identical.
    carry_ok      severity < DRASTIC, i.e. this shot still looks like the
                  previous one, so the previous frame's boxes are a sane
                  starting point. False means annotate from scratch.

A random `qa_frac` of "auto" shots is also flagged `qa: true` -- they stay on
the automatic path but are shown to the reviewer as a spot-check, which is how
drift on a new broadcaster gets caught.
"""

import json
import random
import time
from collections import Counter

import cv2
import numpy as np

from brand_reader.board_regions import rectify_band
from brand_reader.shot_types import (
    DRASTIC, change_severity, classify, cut_points, frame_features,
    frame_signature, shot_change_score, shots_from_cuts,
)
from .store import RunStore


def _video_info(cap, path):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return {
        "path": path,
        "fps": float(fps),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def _review_frames(members, n):
    """`n` frames spread across the shot, for the human to annotate."""
    if n <= 1 or len(members) == 1:
        return [members[len(members) // 2]["frame"]]
    step = (len(members) - 1) / (n - 1)
    idx = sorted({int(round(i * step)) for i in range(n)})
    return [members[i]["frame"] for i in idx]


def _probe_band_conf(cap, members, n_probe, thr):
    """
    Median band confidence over a few full-res frames of the shot.

    This is the expensive part of the pass and the reason it exists: the shot
    LABEL cannot tell a clean main-camera view from a zoomed duel or a high
    overhead angle, because both are "wide_play". Only actually running the band
    extraction says whether there is a readable board there.

    Returns (median_conf, bad_frac, bad_frames, n_probed). conf is 0.0 for a
    frame yielding no band at all, so a shot where extraction simply fails
    scores 0, not None. `bad_frames` are the probed frames that read badly --
    an otherwise automatic shot hands just those to the human, instead of
    condemning the whole shot or losing them.

    `bad_frac` -- the share of probes below the confidence threshold -- is what
    the routing actually uses. The median hides a partly-bad shot: a camera that
    opens on a high overhead angle and settles into a clean wide view reads
    0.00 at the start and 1.00 in the middle, and its median looks healthy while
    the opening seconds are unusable.
    """
    if n_probe <= 0 or not members:
        return None, None, [], 0
    picks = sorted({int(round(i)) for i in
                    np.linspace(0, len(members) - 1, min(n_probe, len(members)))})
    confs, bad_frames = [], []
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, members[i]["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        rect = rectify_band(frame)
        c = 0.0 if rect is None else float(rect["confidence"])
        confs.append(c)
        if c < thr:
            bad_frames.append(members[i]["frame"])
    if not confs:
        return None, None, [], 0
    bad = sum(1 for c in confs if c < thr) / len(confs)
    return float(np.median(confs)), bad, bad_frames, len(confs)


def _route(label, purity, band_conf, bad_frac, cfg):
    """
    "auto" only when everything agrees: it is the main camera, the label is
    stable, and the band reads well across the WHOLE shot. Anything else goes to
    a human -- over-routing costs seconds of their time, under-routing loses a
    sponsor.
    """
    if label != "wide_play":
        return "review", f"{label} shot"
    if purity < cfg.min_purity:
        return "review", f"wide but unstable label (purity {purity:.2f})"
    if band_conf is None:
        return "review", "band not probed"
    if bad_frac is not None and bad_frac > cfg.max_bad_frac:
        return "review", f"{100 * bad_frac:.0f}% of the shot reads badly"
    if band_conf < cfg.band_conf_thr:
        return "review", f"band not trustworthy (conf {band_conf:.2f})"
    return "auto", f"main camera, band conf {band_conf:.2f}"


def run_segment(cfg, progress_every=200):
    """Run pass 0. Returns (store, stats)."""
    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {cfg.video}")

    info = _video_info(cap, cfg.video)
    step = max(1, int(round(info["fps"] / max(cfg.segment_fps, 0.01))))

    store = RunStore(cfg.out_root, cfg.run_name)
    print(f"Run      : {store.dir}")
    print(f"Video    : {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
          f"{info['frames']} frames")
    print(f"Sampling : every {step} frames (~{info['fps'] / step:.1f}/s)")

    t_start = time.perf_counter()

    # --- phase 1: fingerprint + classify every sampled frame ---------------
    samples, sigs = [], []
    frame_idx = 0
    while True:
        if not cap.grab():
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            feats = frame_features(frame)
            label, conf, reason = classify(feats)
            samples.append({"frame": frame_idx,
                            "t": round(frame_idx / info["fps"], 3),
                            "label": label, "conf": conf, "reason": reason})
            sigs.append(frame_signature(frame))
            if progress_every and len(samples) % progress_every == 0:
                print(f"  {len(samples)} frames sampled")
        frame_idx += 1
        if cfg.max_frames and len(samples) >= cfg.max_frames:
            break

    if not samples:
        cap.release()
        raise RuntimeError("No frames sampled")

    # --- phase 2: cuts -> shots -------------------------------------------
    cuts, severities = cut_points(sigs, min_gap=cfg.min_gap)
    spans = shots_from_cuts(len(samples), cuts)
    # severity of the cut that opened each shot; the first shot has no cut
    sev_by_shot = [0.0] + list(severities)

    rng = random.Random(cfg.seed)
    shots, stats_label, stats_route = [], Counter(), Counter()

    with store.open_shots() as out:
        for si, (a, b) in enumerate(spans):
            members = samples[a:b]
            labels = Counter(m["label"] for m in members)
            label, n_dom = labels.most_common(1)[0]
            purity = n_dom / len(members)

            # Only probe where it can change the answer: a non-wide or unstable
            # shot is going to review regardless.
            band_conf, bad_frac, bad_frames, n_probed = (None, None, [], 0)
            if label == "wide_play" and purity >= cfg.min_purity:
                band_conf, bad_frac, bad_frames, n_probed = _probe_band_conf(
                    cap, members, cfg.probe_band, cfg.band_conf_thr)

            route, reason = _route(label, purity, band_conf, bad_frac, cfg)
            sev = float(sev_by_shot[si]) if si < len(sev_by_shot) else 0.0
            qa = route == "auto" and rng.random() < cfg.qa_frac

            rec = {
                "shot": si,
                "f_start": members[0]["frame"],
                "f_end": members[-1]["frame"],
                "t_start": members[0]["t"],
                "t_end": members[-1]["t"],
                "dur_s": round(members[-1]["t"] - members[0]["t"], 2),
                "n_samples": len(members),
                "type": label,
                "purity": round(purity, 2),
                "band_conf": None if band_conf is None else round(band_conf, 3),
                "band_bad_frac": None if bad_frac is None else round(bad_frac, 2),
                "band_probes": n_probed,
                "cut_severity": round(sev, 2),
                # Only meaningful when the previous shot was also reviewed; the
                # annotator uses it to decide whether to offer carry-forward.
                "carry_ok": bool(sev < DRASTIC),
                "route": route,
                "qa": qa,
                "reason": reason,
                # An automatic shot whose camera changes regime part-way (opens
                # on an overhead angle, or drifts off the touchline at the end)
                # is not worth reviewing whole -- only the frames that read
                # badly are queued.
                "partial": bool(route == "auto" and bad_frames),
                "frames": (bad_frames if route == "auto" and bad_frames
                           else _review_frames(members, cfg.per_shot)),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            shots.append(rec)
            stats_label[label] += 1
            stats_route["review" if route == "review" else ("qa" if qa else "auto")] += 1

    cap.release()
    elapsed = time.perf_counter() - t_start

    to_review = [s for s in shots
                 if s["route"] == "review" or s["qa"] or s["partial"]]
    n_annot = sum(len(s["frames"]) for s in to_review)
    review_s = sum(s["dur_s"] for s in shots if s["route"] == "review")

    stats = {
        "frames_sampled": len(samples),
        "shots": len(shots),
        "step": step,
        "by_label": dict(stats_label),
        "by_route": dict(stats_route),
        "review_seconds": round(review_s, 1),
        "review_share": round(review_s / max(info["frames"] / info["fps"], 1e-6), 3),
        "frames_to_annotate": n_annot,
        "drastic_cuts": sum(1 for s in shots if not s["carry_ok"]),
        "seconds": round(elapsed, 1),
    }
    store.write_manifest(cfg, info, stats, stage="segment")

    print(f"\nShots    : {len(shots)}  ({len(samples)} frames sampled)")
    print("  by type : " + "  ".join(f"{k}={v}" for k, v in stats_label.most_common()))
    print("  by route: " + "  ".join(f"{k}={v}" for k, v in stats_route.most_common()))
    n_partial = sum(1 for s in shots if s["partial"])
    print(f"Review   : {len(to_review)} shots ({n_partial} partial), {review_s:.0f}s of footage "
          f"({100 * stats['review_share']:.0f}% of the match)")
    print(f"  -> {n_annot} frames to annotate by hand "
          f"({stats['drastic_cuts']} start on a drastic cut, no carry-forward)")
    print(f"Elapsed  : {elapsed:.1f} s")
    print(f"Artifacts: {store.shots_path}")
    return store, stats
