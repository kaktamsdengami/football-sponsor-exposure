"""
Shot segmentation + a cheap per-frame shot-type label.

The advertising-band pipeline assumes the main tactical camera: far touchline
above grass, near-horizontal, the board a thin uniform strip. Everything else --
goal celebrations on a low camera, player close-ups, bench and interview shots,
replays, full-screen graphics -- either slips through the geometry gates and
produces a distorted partial band, or needs a different extraction entirely
(the whole frame to the vision LLM).

This module produces the routing signal, upstream of any extraction:

  * hsv_signature / shot_change_score / cut_points
        cut detection (HSV histogram), so classification -- and later dedup --
        happen PER SHOT, not per frame.
  * frame_features
        a handful of cheap, scale-insensitive scalars.
  * classify
        a rule-based bucket + a confidence in [0, 1].

Buckets: wide_play | medium | close | other
Numbers here are a FIRST PASS, meant to be tuned on shot_preview.py output.
"""

import cv2
import numpy as np

from .board_regions import green_mask, pitch_boundary

SHOT_LABELS = ("wide_play", "medium", "close", "other")

# Skin, OpenCV HSV (hue 0-180). Two ranges: reddish skin hue wraps around 0.
_SKIN_1_LO = np.array([0, 30, 60], np.uint8)
_SKIN_1_HI = np.array([25, 170, 255], np.uint8)
_SKIN_2_LO = np.array([165, 30, 60], np.uint8)
_SKIN_2_HI = np.array([180, 170, 255], np.uint8)

_WORK_W = 480   # features are scale-insensitive; downscale for speed


def _resize(frame, w=_WORK_W):
    if frame.shape[1] <= w:
        return frame
    h = int(round(frame.shape[0] * w / frame.shape[1]))
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Cut detection
# ---------------------------------------------------------------------------

def frame_signature(frame, h_bins=32, s_bins=32, thumb=32, grid=4):
    """
    Compact shot fingerprint, three complementary views of the same frame:

      hist   global H+S histogram -- catches palette changes (pitch -> crowd ->
             studio). Blind to a cut between two shots of the same scene.
      thumb  32x32 greyscale -- pixel-wise change spikes on any hard cut.
      layout 4x4 grid of mean Lab -- WHERE the colours sit. Two replays of the
             same action from different camera angles have almost the same
             global histogram but a completely different spatial layout, so this
             is the signal that separates them.
    """
    small = _resize(frame)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    hs = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    hist = np.concatenate([hh, hs]).ravel().astype(np.float64)
    n = float(hist.sum())
    hist = hist / n if n else hist

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    tn = cv2.resize(gray, (thumb, thumb), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    layout = cv2.resize(lab, (grid, grid), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    return {"hist": hist, "thumb": tn, "layout": layout}


def shot_change_score(sig_a, sig_b):
    """
    (hist_dist, thumb_mad, layout_dist), each 0 = identical. See
    `frame_signature` for what each one is good at.
    """
    if sig_a is None or sig_b is None:
        return 0.0, 0.0, 0.0
    bc = float(np.sum(np.sqrt(sig_a["hist"] * sig_b["hist"])))
    hist_dist = float(np.clip(1.0 - bc, 0.0, 1.0))
    thumb_mad = float(np.mean(np.abs(sig_a["thumb"] - sig_b["thumb"])))
    layout_dist = float(np.mean(np.abs(sig_a["layout"] - sig_b["layout"])))
    return hist_dist, thumb_mad, layout_dist


# Measured on ucl1.mp4 (348 samples @ 6/s). Within-shot noise vs real cuts:
#   hist    median 0.0008, p95 0.052 | real cuts 0.10 - 0.66
#   mad     median 0.026,  p95 0.116 | real cuts 0.126 - 0.208
#   layout  median 0.004,  p95 0.038 | real cuts 0.042 - 0.157
# Each threshold sits just above that signal's p95 and below its real-cut floor.
# Re-measure on a new broadcaster before trusting these.
CUT_THR = {"hist": 0.25, "mad": 0.12, "layout": 0.042}

# Severity = how many times over threshold a transition sits. Above this, the
# two shots have nothing visually in common and annotations must NOT be carried
# forward (two replays of the same action from different angles land here).
DRASTIC = 2.0


def change_severity(scores, thr=None):
    """
    How far past the cut thresholds a transition sits, as a single number
    (1.0 = exactly at threshold, >= DRASTIC = nothing in common). This is what
    decides whether the previous shot's annotations may be carried forward or
    must be redone from scratch.
    """
    thr = thr or CUT_THR
    hd, md, ld = scores
    return max(hd / thr["hist"], md / thr["mad"], ld / thr["layout"])


def cut_points(sigs, hist_thr=CUT_THR["hist"], mad_thr=CUT_THR["mad"],
               layout_thr=CUT_THR["layout"], min_gap=3):
    """
    (cuts, severities): indices i where sigs[i] starts a new shot -- ANY of the
    three signals exceeding its threshold -- with at least `min_gap` samples
    since the last cut so a flash or a fast double-cut counts once.
    """
    thr = {"hist": hist_thr, "mad": mad_thr, "layout": layout_thr}
    cuts, sev, last = [], [], -min_gap
    for i in range(1, len(sigs)):
        scores = shot_change_score(sigs[i - 1], sigs[i])
        s = change_severity(scores, thr)
        if s > 1.0 and i - last >= min_gap:
            cuts.append(i)
            sev.append(round(s, 3))
            last = i
    return cuts, sev


def shots_from_cuts(n, cuts):
    """[(start, end)] half-open sample-index ranges from a cut list."""
    bounds = [0] + list(cuts) + [n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]


# ---------------------------------------------------------------------------
# Per-frame features + classification
# ---------------------------------------------------------------------------

def _skin_ratio(hsv):
    m = (cv2.inRange(hsv, _SKIN_1_LO, _SKIN_1_HI)
         | cv2.inRange(hsv, _SKIN_2_LO, _SKIN_2_HI))
    return float((m > 0).mean())


def frame_features(frame):
    """Cheap scalars used by `classify`. All scale-insensitive."""
    small = _resize(frame)
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    gm = green_mask(small) > 0
    green_ratio = float(gm.mean())
    green_lower = float(gm[h // 2:].mean())     # pitch tends to sit low in frame

    dark_ratio = float((hsv[:, :, 2] < 40).mean())

    edge = (np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)))
    edge_density = float(np.clip(edge.mean() / 64.0, 0.0, 1.0))

    skin_ratio = _skin_ratio(hsv)

    # A large central subject that is neither grass nor dark crowd -> close-up.
    cy0, cy1 = int(0.25 * h), int(0.80 * h)
    cx0, cx1 = int(0.30 * w), int(0.70 * w)
    c_green = green_mask(small[cy0:cy1, cx0:cx1]) > 0
    c_dark = hsv[cy0:cy1, cx0:cx1, 2] < 40
    centre_subject = float((~c_green & ~c_dark).mean())

    yb = pitch_boundary(small)
    if yb is None:
        boundary_span, boundary_y = 0.0, 1.0
    else:
        ok = (yb > 0.05 * h) & (yb < 0.95 * h)   # clear of both frame edges
        boundary_span = float(ok.mean())
        boundary_y = float(np.median(yb[ok]) / h) if ok.any() else 1.0

    return {
        "green_ratio": round(green_ratio, 3),
        "green_lower": round(green_lower, 3),
        "dark_ratio": round(dark_ratio, 3),
        "edge_density": round(edge_density, 3),
        "skin_ratio": round(skin_ratio, 3),
        "centre_subject": round(centre_subject, 3),
        "boundary_span": round(boundary_span, 3),
        "boundary_y": round(boundary_y, 3),
    }


def classify(f):
    """
    (label, confidence, reason). Rule-based and deliberately transparent: the
    preview exists to move these numbers around against real footage.
    """
    g, gl = f["green_ratio"], f["green_lower"]
    span, cs = f["boundary_span"], f["centre_subject"]
    skin, dark = f["skin_ratio"], f["dark_ratio"]

    # 1. Wide tactical play: grass-dominated low in frame, a long clean
    #    boundary, nothing large filling the middle.
    if span >= 0.45 and gl >= 0.30 and cs < 0.55:
        c = min((span - 0.45) / 0.35, (gl - 0.30) / 0.40, (0.55 - cs) / 0.35)
        return "wide_play", float(np.clip(c, 0.0, 1.0)), "long boundary + grass, clear centre"

    # 2. Close-up: a subject fills the centre, or a lot of skin, little grass.
    if (cs >= 0.70 or skin >= 0.12) and g < 0.30:
        c = max((cs - 0.70) / 0.30, (skin - 0.12) / 0.20)
        return "close", float(np.clip(c, 0.0, 1.0)), "central subject / skin, little grass"

    # 3. Other: no grass and no single subject -> crowd, dark stadium,
    #    full-screen graphic, replay wipe.
    if g < 0.12 and span < 0.20:
        c = np.clip((0.12 - g) / 0.12 + dark, 0.0, 1.0)
        return "other", float(c), "no pitch, no subject"

    # 4. Everything else: some grass but not a clean wide shot -- panned off the
    #    touchline, medium range, or a transitional frame.
    return "medium", 0.3, "between wide and close"
