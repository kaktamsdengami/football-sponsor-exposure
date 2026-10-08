"""
Shot-type + band-confidence preview.

Runs the shot classifier (brand_reader.shot_types) and the new band confidence
(brand_reader.board_regions.band_confidence) over a video, and writes a folder
to eyeball BEFORE any of this is wired into pipeline/:

  output_videos/shot_preview/
    shots.csv            one row per detected shot: frame/time range, dominant
                         label, label purity
    frames.csv           one row per sampled frame: label, confidence, every
                         raw feature, and the band confidence + sub-scores when
                         the frame goes down the wide/medium route
    shots/shot_XXXX.jpg  the middle frame of each shot, annotated: shot label,
                         the band polygon (green = trusted, red = low band
                         confidence) and the deciding numbers
    summary.txt          counts per label; wide_play frames flagged low-conf;
                         the questions to check by eye

Usage:
    python shot_preview.py
    python shot_preview.py input_videos/ucl1.mp4
    python shot_preview.py input_videos/ucl1.mp4 --sample-fps 6 --max-seconds 120
"""

import argparse
import csv
import os
from collections import Counter

import cv2
import numpy as np

from brand_reader.board_regions import rectify_band, band_polygon
from brand_reader.shot_types import (
    classify, cut_points, frame_features, frame_signature, shots_from_cuts,
)

LOW_BAND_CONF = 0.5
_GREEN = (0, 220, 160)
_RED = (60, 60, 235)
_WHITE = (255, 255, 255)


def _put_lines(img, lines, x=16, y=30, color=_WHITE, scale=0.6):
    for ln in lines:
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(28 * scale / 0.6)


def _band_on(frame):
    """(rect, confidence, parts) or (None, None, None)."""
    rect = rectify_band(frame)
    if rect is None:
        return None, None, None
    return rect, rect["confidence"], rect["conf_parts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default="input_videos/ucl1.mp4")
    ap.add_argument("--out-dir", default="output_videos/shot_preview")
    ap.add_argument("--sample-fps", type=float, default=6.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--cut-hist", type=float, default=0.30, help="colour-histogram cut threshold")
    ap.add_argument("--cut-mad", type=float, default=0.11, help="greyscale-thumbnail cut threshold")
    args = ap.parse_args()

    shots_dir = os.path.join(args.out_dir, "shots")
    os.makedirs(shots_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / max(args.sample_fps, 0.01))))
    last_frame = total if args.max_seconds is None else min(total, int(args.max_seconds * fps))

    print(f"{args.video}: {total} frames @ {fps:.1f} fps")
    print(f"Sampling every {step} frames (~{fps / step:.1f}/s), up to frame {last_frame}")

    # --- pass 1: sample, fingerprint, classify, band confidence -------------
    samples = []          # dicts, one per sampled frame
    sigs = []
    frame_idx = 0
    while frame_idx < last_frame:
        if not cap.grab():
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            feats = frame_features(frame)
            label, conf, reason = classify(feats)

            band_conf, band_parts = None, None
            if label in ("wide_play", "medium"):
                _, band_conf, band_parts = _band_on(frame)

            samples.append({
                "frame": frame_idx,
                "t": round(frame_idx / fps, 2),
                "label": label,
                "confidence": round(conf, 3),
                "reason": reason,
                "band_conf": None if band_conf is None else round(band_conf, 3),
                "band_parts": band_parts or {},
                **feats,
            })
            sigs.append(frame_signature(frame))
            if len(samples) % 50 == 0:
                print(f"  {len(samples)} frames sampled")
        frame_idx += 1

    # --- shots -------------------------------------------------------------
    cuts = cut_points(sigs, hist_thr=args.cut_hist, mad_thr=args.cut_mad)
    shots = shots_from_cuts(len(samples), cuts)
    print(f"{len(samples)} frames -> {len(shots)} shots ({len(cuts)} cuts)")

    # --- write frames.csv ------------------------------------------------
    feat_keys = [k for k in samples[0] if k not in
                 ("frame", "t", "label", "confidence", "reason", "band_conf", "band_parts")]
    part_keys = ["span_ratio", "fit_resid", "slope_deg", "bridged_frac",
                 "boundary_y", "break_found", "grass_below"]
    with open(os.path.join(args.out_dir, "frames.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["shot", "frame", "t", "label", "confidence", "band_conf"]
                   + ["bp_" + k for k in part_keys] + feat_keys + ["reason"])
        for si, (a, b) in enumerate(shots):
            for s in samples[a:b]:
                w.writerow([si, s["frame"], s["t"], s["label"], s["confidence"], s["band_conf"]]
                           + [s["band_parts"].get(k, "") for k in part_keys]
                           + [s[k] for k in feat_keys] + [s["reason"]])

    # --- write shots.csv + annotated mid-frames --------------------------
    shot_rows = []
    with open(os.path.join(args.out_dir, "shots.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["shot", "f_start", "f_end", "t_start", "t_end", "dur_s",
                    "n_frames", "dominant_label", "purity",
                    "mean_band_conf", "n_low_band_conf"])
        for si, (a, b) in enumerate(shots):
            members = samples[a:b]
            labels = Counter(m["label"] for m in members)
            dom, dom_n = labels.most_common(1)[0]
            purity = dom_n / len(members)
            bcs = [m["band_conf"] for m in members if m["band_conf"] is not None]
            mean_bc = round(float(np.mean(bcs)), 3) if bcs else ""
            n_low = sum(1 for x in bcs if x < LOW_BAND_CONF)
            t0, t1 = members[0]["t"], members[-1]["t"]
            row = [si, members[0]["frame"], members[-1]["frame"], t0, t1,
                   round(t1 - t0, 2), len(members), dom, round(purity, 2),
                   mean_bc, n_low]
            w.writerow(row)
            shot_rows.append(row)

            # annotate the middle frame of the shot
            mid = members[len(members) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            rect, band_conf, parts = _band_on(frame)
            if rect is not None:
                col = _GREEN if band_conf >= LOW_BAND_CONF else _RED
                cv2.polylines(frame, [band_polygon(rect)], True, col, 3, cv2.LINE_AA)

            lines = [
                f"shot {si:03d}  {dom}  purity {purity:.2f}  ({len(members)} frames, {t1 - t0:.1f}s)",
                f"mid frame {mid['frame']}  t={mid['t']}s  label={mid['label']} conf={mid['confidence']:.2f}  ({mid['reason']})",
                f"green={mid['green_ratio']} green_lo={mid['green_lower']} boundary_span={mid['boundary_span']} "
                f"boundary_y={mid['boundary_y']} centre_subject={mid['centre_subject']} skin={mid['skin_ratio']}",
            ]
            if band_conf is not None:
                lines.append(
                    f"BAND conf={band_conf:.2f}  span_ratio={parts['span_ratio']} "
                    f"fit_resid={parts['fit_resid']} slope_deg={parts['slope_deg']} "
                    f"bridged={parts['bridged_frac']} boundary_y={parts['boundary_y']} "
                    f"break={parts['break_found']} grass_below={parts['grass_below']}"
                    + ("   <-- LOW: send whole frame to LLM" if band_conf < LOW_BAND_CONF else "")
                )
            else:
                lines.append("BAND: none (rectify_band returned None)")
            _put_lines(frame, lines)
            cv2.imwrite(os.path.join(shots_dir, f"shot_{si:04d}.jpg"), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    cap.release()

    # --- summary --------------------------------------------------------
    label_counts = Counter(s["label"] for s in samples)
    wide = [s for s in samples if s["label"] == "wide_play"]
    wide_low = [s for s in wide if s["band_conf"] is not None and s["band_conf"] < LOW_BAND_CONF]
    med_bands = [s for s in samples if s["label"] == "medium" and s["band_conf"] is not None]

    lines = [
        f"video            : {args.video}",
        f"sampled frames   : {len(samples)}  (~{fps / step:.1f}/s)",
        f"shots            : {len(shots)}",
        "",
        "frames per label :",
    ]
    for lab in ("wide_play", "medium", "close", "other"):
        lines.append(f"  {lab:10s} {label_counts.get(lab, 0):5d}")
    lines += [
        "",
        f"wide_play frames with a band          : {sum(1 for s in wide if s['band_conf'] is not None)}",
        f"  of which LOW band confidence (<{LOW_BAND_CONF}) : {len(wide_low)}",
        f"medium frames that still yielded a band : {len(med_bands)}",
        "",
        "CHECK BY EYE (shots/shot_XXXX.jpg):",
        "  1. celebration / low-camera shots -> label should NOT be wide_play,",
        "     or if it is, BAND conf should be low (red polygon).",
        "  2. green polygon should sit on the far perimeter board, whole span,",
        "     no players punched through it, no crowd above it.",
        "  3. red polygon = we would escalate that frame to a full-frame LLM call.",
        "  4. close-ups of players -> label 'close'. Bench / interview -> 'close'",
        "     or 'other' (no dedicated bucket yet).",
        "  5. the upper ring / ribbon board is never captured by any path (known gap).",
    ]
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out_dir}/  (shots/, shots.csv, frames.csv, summary.txt)")


if __name__ == "__main__":
    main()
