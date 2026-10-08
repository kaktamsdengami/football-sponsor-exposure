"""
Read advertiser names anywhere in the frame, with no geometry model at all.

WHY THIS EXISTS. `detect` finds the far touchline, fits it, cuts a band along
it, and reads that. It is precise when it fires and it produces the size and
position numbers a media report wants -- but it can only ever see one thing.
Measured on pilot_match_5min.mp4 it extracts a usable band for 41% of the clip at
its default confidence gate and 52% even wide open, while the by-eye count
finds legible advertising in 82%. The missing third is not noise: it is whole
camera framings -- the goal camera, the corner camera, the dugout-side
hoarding -- where the boards are perfectly readable but are not a horizontal
run along a fitted far touchline. Four of the twelve advertisers on this clip
live mostly in those framings, and three of them scored zero.

So this stage asks a smaller question. Not "where is the board" but "what
advertiser names are on the screen right now". It samples the video on a fixed
clock, blanks the broadcaster's burned-in graphics, runs OCR over the whole
frame, and fuzzy-matches every string against the club's advertiser list. That
is exactly the procedure the hand count was built with, which is the point:
it is the method known to reach the target on this footage.

WHAT IT GIVES UP. No band, so no per-panel geometry: a brand's `area_px` here
is the OCR text box, not the board it sits on, and there is no distance-to-
centre or brightness modelling. It answers "how long was this brand legibly on
screen" and nothing more. `detect` stays for the surface-level detail, and
merge_surfaces can union the two.

    python -m pipeline.cli scan output_videos/RUN --video input.mp4 \
        --sponsors pilot/sponsors.json --scan-fps 1
"""
import csv
import json
import os
import time
from collections import defaultdict

import cv2

from brand_reader.overlay_mask import apply_mask, boxes_from_mask, overlay_mask
from .identify import _index, load_sponsors, match_token

MIN_OCR_CONF = 0.30
MIN_MATCH = 0.62
# A brand must be read on at least this many distinct sampled frames before it
# is reported. One frame is a coin flip on a hard OCR read; the cost of a
# spurious advertiser in a client report is high and the cost of this filter is
# at most one sample of real exposure.
MIN_FRAMES = 2

# EasyOCR runs in two stages: a CRAFT detector proposes text boxes, then a
# recogniser reads each one. Everything measured so far says the DETECTOR is
# what limits this job -- upscaling helped (bigger text is easier to detect),
# CLAHE hurt (more spurious texture for the detector to chase), and the brands
# that score zero are ones a person can plainly read. So these are exposed.
#
#   text_threshold  confidence for a pixel to be text        (lower = more boxes)
#   low_text        how far a box is grown into faint pixels (lower = wider boxes)
#   link_threshold  how eagerly characters join into a word  (lower = more joins)
#   mag_ratio       EasyOCR's own internal upscale before detection
#
# Defaults here are EasyOCR's own. `--sensitive` shifts them toward proposing
# more boxes, which trades recogniser time and junk strings for a chance at
# low-contrast board text -- and junk is cheap, because a string that matches no
# sponsor is discarded anyway.
#
# UNMEASURED. The reasoning is sound and the CLAHE result points the same way,
# but nobody has scored a --sensitive run yet. Do that before believing it, the
# way --clahe looked obviously right and turned out to be a regression.
DETECT_DEFAULTS = {"text_threshold": 0.7, "low_text": 0.4,
                   "link_threshold": 0.4, "mag_ratio": 1.0}
DETECT_SENSITIVE = {"text_threshold": 0.5, "low_text": 0.3,
                    "link_threshold": 0.3, "mag_ratio": 1.5}


def _reader(langs):
    import easyocr                              # heavy; import only when used
    return easyocr.Reader([s.strip() for s in langs.split(",") if s.strip()],
                          gpu=True, verbose=False)


# Rows below the pitch boundary are grass, and a board cannot be in the middle
# of the pitch. Cropping them away before OCR is free accuracy-wise and is the
# single biggest lever on runtime, because the detector's cost scales with image
# area and most of a football frame is grass.
#
# The margin below the boundary is generous on purpose: the pitch-level LED sits
# exactly ON the boundary, and the boundary estimate is a per-column median that
# a running player or a bad frame can pull down. Losing a board to save a second
# would be a terrible trade, so the crop only fires when it can remove a lot.
#
# MEASURED on pilot_match_5min, 120 frames at 1/s, against the no-crop scan:
#
#   margin 0.12   92% of frames cropped, 66% of area saved
#                 1.6 frames/s vs 0.25 -- 6.4x faster
#                 captured 47.0% -> 45.6%, MAE 20.4s -> 21.0s
#                 the only real loss was one brand, 4s -> 1s
#   margin 0.18   91% cropped, 60% saved
#
# 0.18 ships. That brand lives on the pitch-level LED right at the boundary, which is
# exactly what the margin is protecting, and 6 points of area is a cheap
# insurance premium against clipping a board off the bottom of the strip. The
# crop is strictly more conservative than the measured 0.12 case, so its
# accuracy cannot be worse than the numbers above.
GRASS_MARGIN_FRAC = 0.18       # of frame height, kept below the boundary
GRASS_MIN_SAVING = 0.25        # crop only if it removes at least this much
GRASS_ROW_FRAC = 0.70          # a row is "pitch" when this much of it is green


def pitch_crop_row(bgr, row_frac=GRASS_ROW_FRAC, margin_frac=GRASS_MARGIN_FRAC):
    """
    The lowest row worth reading, or None when nothing can safely be cropped.

    Finds the topmost row from which the frame is continuously dominated by
    grass to the bottom. Requiring CONTINUITY is what makes this safe: a single
    green band across a crowd shot (a hi-vis steward row, an LED creative in
    green) does not start the pitch, because the rows below it are not grass
    too.
    """
    from brand_reader.board_regions import green_mask

    h, w = bgr.shape[:2]
    rows = green_mask(bgr).mean(axis=1) / 255.0      # green share per row
    is_pitch = rows >= row_frac
    if not is_pitch.any():
        return None                                   # no pitch: read it all

    # The grass run must actually REACH THE BOTTOM of the frame. Without this
    # test the loop below simply finds the lowest green run wherever it sits, so
    # a grass-coloured advert with crowd beneath it reads as the pitch and
    # everything under it -- including real boards -- gets cropped away. Checked
    # over a band of rows, not the last row alone, because a broadcast often
    # paints a graphic across the very bottom edge.
    foot = max(1, int(0.10 * h))
    if is_pitch[-foot:].mean() < 0.5:
        return None

    # Walk up from the bottom while rows stay green; that run is the pitch.
    top = h
    for y in range(h - 1, -1, -1):
        if is_pitch[y]:
            top = y
        elif top < h:
            break
    cut = min(h, top + int(margin_frac * h))
    if cut >= h * (1.0 - GRASS_MIN_SAVING):
        return None                                   # not worth the risk
    return cut


def _clahe(bgr, clip=2.0, grid=8):
    """
    Normalise local contrast before OCR, on the L channel only.

    Upscaling fixes text that is too SMALL. It does nothing for text that is
    large enough but too LOW-CONTRAST, and on this broadcast that is the single
    largest error in the run: one brand is white-on-light-blue along the top of the
    frame and reads 21s against a hand count of 75s at both 1.0x and 1.5x.

    CLAHE rather than a global stretch because the frame contains both a bright
    pitch and boards in shadow, and one global curve cannot serve both. Colour
    is left alone -- equalising a and b would shift the hues the grass test and
    the LED checks elsewhere depend on.

    MEASURED, AND IT DOES NOT WORK. Over the first 120s of pilot_match_5min,
    with --upscale 1.5 held constant:

        upscale only     captured 51.5%   MAE 18.7s   BRAND_B 21s
        + CLAHE          captured 34.4%   MAE 25.3s   BRAND_B 13s

    It hurt everything, including the brand it was written for. The likely
    reason is that EasyOCR's detector, not its recogniser, is the binding stage
    here: CLAHE amplifies JPEG blocking and crowd texture as enthusiastically as
    it does board text, and the extra false texture costs more than the contrast
    gains. Kept behind a flag and OFF, because the hypothesis is reasonable and
    may hold with a different OCR engine -- but on this one it is a regression.
    Do not turn it on without re-measuring.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def run_scan(run_dir, video=None, sponsors_path=None, scan_fps=1.0,
             langs="en", min_ocr_conf=MIN_OCR_CONF, min_match=MIN_MATCH,
             min_frames=MIN_FRAMES, mask_overlays=True, upscale=1.0,
             clahe=False, pitch_crop=True, sensitive=False, max_frames=None,
             progress_every=25):
    os.makedirs(run_dir, exist_ok=True)
    man_path = os.path.join(run_dir, "manifest.json")
    manifest = {}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)
    video = video or manifest.get("video", {}).get("path")
    if not video:
        raise SystemExit("No --video given and none in the run manifest.")
    sponsors_path = sponsors_path or os.path.join(run_dir, "sponsors.json")
    sponsors = load_sponsors(sponsors_path)
    idx = _index(sponsors)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / max(scan_fps, 0.01))))
    seconds_per_sample = step / fps

    print(f"Run      : {run_dir}")
    print(f"Video    : {video}  {total} frames @ {fps:.2f} fps")
    print(f"Scanning : every {step} frames (~{fps/step:.2f}/s), "
          f"each sample stands for {seconds_per_sample:.2f}s")
    print(f"Sponsors : {len(sponsors)} listed, {len(idx)} names+aliases")

    gmask, ginfo = None, {"ok": False, "reason": "disabled"}
    if mask_overlays:
        gmask, ginfo = overlay_mask(video, n=96)
        if ginfo["ok"]:
            print(f"Graphics : {len(boxes_from_mask(gmask))} overlay(s) masked "
                  f"({100*ginfo['overlay_frac']:.1f}% of frame)")
        else:
            gmask = None
            print(f"Graphics : not masked -- {ginfo['reason']}")

    reader = _reader(langs)
    detect_kw = dict(DETECT_SENSITIVE if sensitive else DETECT_DEFAULTS)
    if sensitive:
        print(f"Detector : sensitive -- {detect_kw}")
    t0 = time.perf_counter()

    rows = []
    brand_frames = defaultdict(set)
    unmatched = defaultdict(int)
    n_seen = 0
    n_cropped = 0
    area_saved = 0.0
    frame_idx = 0
    scan_path = os.path.join(run_dir, "scan.jsonl")
    with open(scan_path, "w", encoding="utf-8") as out:
        while True:
            if not cap.grab():
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if gmask is not None:
                frame = apply_mask(frame, gmask)
            img = frame
            # Drop the pitch before OCR. The crop starts at y=0, so OCR boxes
            # need no offset back into frame coordinates.
            if pitch_crop:
                cut = pitch_crop_row(img)
                if cut is not None:
                    img = img[:cut]
                    n_cropped += 1
                    area_saved += 1.0 - cut / frame.shape[0]
            if clahe:
                img = _clahe(img)
            if upscale and upscale != 1.0:
                # chain off `img`, not `frame` -- otherwise --clahe --upscale
                # silently throws the contrast pass away
                img = cv2.resize(img, None, fx=upscale, fy=upscale,
                                 interpolation=cv2.INTER_CUBIC)

            t = frame_idx / fps
            n_seen += 1
            hits = []
            for box, text, conf in reader.readtext(img, **detect_kw):
                if conf < min_ocr_conf:
                    continue
                sp, score = match_token(text, idx, min_match)
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                area = float((max(xs) - min(xs)) * (max(ys) - min(ys)) / (upscale ** 2))
                if sp is None:
                    if len(text.strip()) >= 4:
                        unmatched[text.strip()] += 1
                    continue
                hits.append({
                    "brand": sp["name"], "paying": sp["paying"],
                    "text": text, "ocr_conf": round(float(conf), 3),
                    "match": round(float(score), 3),
                    "box": [[int(p[0] / upscale), int(p[1] / upscale)] for p in box],
                    "area_px": round(area, 1),
                })
                brand_frames[sp["name"]].add(frame_idx)
            out.write(json.dumps({"frame": frame_idx, "t": round(t, 3),
                                  "hits": hits}, ensure_ascii=False) + "\n")
            for h in hits:
                rows.append({**h, "frame": frame_idx, "t": round(t, 3)})

            if progress_every and n_seen % progress_every == 0:
                named = sum(1 for r in rows if r)
                print(f"  {n_seen} frames scanned, {len(brand_frames)} brands, "
                      f"{named} reads")
            frame_idx += 1
            if max_frames and n_seen >= max_frames:
                break
    cap.release()
    elapsed = time.perf_counter() - t0

    # Drop brands seen on a single frame: one hard OCR read is a coin flip, and
    # a spurious advertiser in a client report costs more than one sample.
    dropped = {b: len(f) for b, f in brand_frames.items() if len(f) < min_frames}
    for b in dropped:
        del brand_frames[b]
    rows = [r for r in rows if r["brand"] in brand_frames]

    paying = {sp["name"]: sp["paying"] for sp in sponsors}
    by_brand = []
    for brand, frames in brand_frames.items():
        rs = [r for r in rows if r["brand"] == brand]
        ts = sorted(r["t"] for r in rs)
        secs = len(frames) * seconds_per_sample
        by_brand.append({
            "brand": brand,
            "paying": paying.get(brand, True),
            "exposure_s": round(secs, 2),
            "time_basis": "per_frame_scan",
            "time_is_upper_bound": False,
            "pct_of_video": round(100 * secs / (total / fps), 2),
            "n_frames": len(frames),
            "mean_panels": round(len(rs) / len(frames), 2),
            "mean_area_px": round(sum(r["area_px"] for r in rs) / len(rs), 0),
            "mean_area_pct": round(
                100 * sum(r["area_px"] for r in rs) / len(rs)
                / (cap_w_h(video)), 4),
            "mean_ocr_conf": round(sum(r["ocr_conf"] for r in rs) / len(rs), 3),
            "mean_match": round(sum(r["match"] for r in rs) / len(rs), 3),
            "first_seen_s": round(ts[0], 2),
            "last_seen_s": round(ts[-1], 2),
            "area_source": "ocr_text_box",
            "identified_by": "scan_ocr",
        })
    by_brand.sort(key=lambda r: (not r["paying"], -r["exposure_s"]))

    _write_csv(os.path.join(run_dir, "exposure_by_brand.csv"), by_brand)
    _write_csv(os.path.join(run_dir, "exposure_detail.csv"), [
        {"brand": r["brand"], "paying": r["paying"], "creative": 0,
         "frame": r["frame"], "t": r["t"], "seconds": round(seconds_per_sample, 4),
         "n_panels": 1, "area_px": r["area_px"],
         "area_pct": 0.0, "dist_center": 0.0,
         "area_source": "ocr_text_box", "time_basis": "per_frame_scan",
         "identified_by": "scan_ocr", "brightness": 0.0, "brightness_rel": 0.0}
        for r in rows])

    top_unmatched = sorted(unmatched.items(), key=lambda kv: -kv[1])[:25]
    with open(os.path.join(run_dir, "scan_unmatched.json"), "w",
              encoding="utf-8") as f:
        json.dump({"_what": "strings OCR read that matched no listed sponsor. "
                            "A real advertiser missing from sponsors.json shows "
                            "up here, repeated, and should be added.",
                   "counts": dict(top_unmatched)}, f, ensure_ascii=False, indent=2)

    stats = {
        "frames_scanned": n_seen,
        "seconds_per_sample": round(seconds_per_sample, 4),
        "step": step,
        "brands": len(by_brand),
        "paying_brands": sum(1 for r in by_brand if r["paying"]),
        "reads": len(rows),
        "pitch_cropped_frames": n_cropped,
        "pitch_mean_area_saved": round(area_saved / max(1, n_seen), 3),
        "dropped_single_frame_brands": dropped,
        "overlay_mask": {**ginfo,
                         "boxes": boxes_from_mask(gmask) if gmask is not None else []},
        "seconds": round(elapsed, 1),
        "fps_processed": round(n_seen / elapsed, 2) if elapsed else None,
    }
    manifest.setdefault("video", {}).update(
        {"path": video, "fps": float(fps), "frames": total})
    manifest.setdefault("stats", {})["scan"] = stats
    manifest.setdefault("config", {})["scan"] = {
        "scan_fps": scan_fps, "langs": langs, "min_ocr_conf": min_ocr_conf,
        "min_match": min_match, "min_frames": min_frames,
        "mask_overlays": mask_overlays, "upscale": upscale, "clahe": clahe,
        "pitch_crop": pitch_crop, "sensitive": sensitive,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n  {n_seen} frames scanned in {elapsed:.0f}s "
          f"({stats['fps_processed']} frames/s)")
    print(f"  {'brand':<32} {'time':>8} {'frames':>7}  match")
    for r in by_brand:
        tag = "" if r["paying"] else "   (not paying)"
        print(f"  {r['brand']:<32} {r['exposure_s']:7.1f}s {r['n_frames']:6d}  "
              f"{r['mean_match']:.2f}{tag}")
    if dropped:
        print(f"\n  dropped (seen on < {min_frames} frames): "
              + ", ".join(f"{b}" for b in dropped))
    if top_unmatched:
        print(f"\n  most common unlisted strings (see scan_unmatched.json):")
        for s, c in top_unmatched[:8]:
            print(f"    {c:4d}x  {s!r}")
    print(f"\n  Wrote {os.path.join(run_dir, 'exposure_by_brand.csv')}")
    return by_brand, stats


_WH_CACHE = {}


def cap_w_h(video):
    """Frame area in px, cached -- used only to express text-box area as a %."""
    if video not in _WH_CACHE:
        c = cv2.VideoCapture(video)
        _WH_CACHE[video] = max(1.0, c.get(cv2.CAP_PROP_FRAME_WIDTH)
                               * c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        c.release()
    return _WH_CACHE[video]


def _write_csv(path, rows):
    if not rows:
        rows = [{"brand": "", "paying": "", "exposure_s": 0}]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
