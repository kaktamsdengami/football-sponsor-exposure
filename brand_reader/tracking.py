"""
Propagate one annotated polygon across the frames around it.

This is what turns a sparse observation into a measurement. A human outlines a
board on one frame; the report needs to know how long that board was on screen
and how big it was at every moment, and nobody is going to outline 300 frames.
The same machinery extends an automatic geometric fit across the frames between
its samples.

Two ideas do the work:

GEOMETRY IS CHEAP, IDENTITY IS EXPENSIVE. Where a surface sits on screen changes
every frame and can be recovered by tracking pixels. WHICH brand is on it changes
rarely -- at a cut, or when an LED board rotates its creative. So the polygon is
tracked densely and the brand label is simply carried along, to be re-verified
only when something suggests it changed.

TRACKING MUST KNOW WHEN IT HAS FAILED. Every step is run forward and then
immediately backward; points that do not return to where they started are
discarded, and the median round-trip error becomes an honest per-frame
confidence. When it degrades past a threshold the track STOPS rather than
emitting a plausible-looking wrong polygon -- the coverage ledger would rather
record an uncovered second than a fabricated one.

An LED board changing creative mid-track shows up as a sudden collapse in
inliers. That is a feature: it is exactly when the brand label needs rechecking.
"""

import cv2
import numpy as np

TRACK_W = 960          # track on a downscale; polygons stay in frame coords

# Confidence lost per FRAME even when tracking is flawless: each warp resamples
# the polygon from the previous one, so error compounds regardless of how clean
# the correspondences look. Per-frame rather than per-step so that tracking
# finely does not look less certain than tracking coarsely.
_DRIFT_PER_FRAME = 0.9994
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _gray(frame, width=TRACK_W):
    h, w = frame.shape[:2]
    s = min(1.0, width / float(w))
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if s < 1.0:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return g, s


def _seed(gray, poly_s, margin=0.35, max_pts=220):
    """
    Corners to track, taken from the polygon AND a margin around it.

    The margin matters: an LED board is a flat panel whose own pixels change
    when the creative rotates, while the touchline, grass edge and crowd around
    it stay put. Including that surround keeps a track alive across a creative
    change instead of losing it exactly when the brand changes.
    """
    h, w = gray.shape[:2]
    pts = np.array(poly_s, np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask,
                  (max(0, int(x1 - mx)), max(0, int(y1 - my))),
                  (min(w - 1, int(x2 + mx)), min(h - 1, int(y2 + my))), 255, -1)
    p = cv2.goodFeaturesToTrack(gray, maxCorners=max_pts, qualityLevel=0.01,
                                minDistance=6, mask=mask, blockSize=7)
    return None if p is None else p.reshape(-1, 2).astype(np.float32)


def _fb_step(g0, g1, p0, fb_thr=1.5):
    """
    One forward-backward LK step. Returns (p0_kept, p1_kept, fb_err_median).

    Points are pushed to the next frame and pulled straight back; whatever does
    not land where it started is a bad correspondence and is dropped before it
    can pollute the transform.
    """
    if p0 is None or len(p0) < 6:
        return None, None, np.inf
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **_LK)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **_LK)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1)
    if ok.sum() < 6:
        return None, None, np.inf
    fb = np.linalg.norm(p0 - p0r, axis=1)
    keep = ok & (fb <= fb_thr)
    if keep.sum() < 6:
        return None, None, float(np.median(fb[ok]))
    return p0[keep], p1[keep], float(np.median(fb[keep]))


def _transform(p0, p1, min_inliers=8):
    """
    (3x3 matrix, inlier_ratio). A perimeter board is planar, so a homography is
    the correct model; with few or noisy correspondences it is unstable, so fall
    back to a similarity transform rather than warping the polygon into nonsense.
    """
    if p0 is None or len(p0) < 4:
        return None, 0.0
    if len(p0) >= min_inliers:
        H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0, maxIters=2000)
        if H is not None and inl is not None:
            r = float(inl.sum()) / len(p0)
            if r >= 0.5:
                return H, r
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0)
    if M is None:
        return None, 0.0
    H = np.vstack([M, [0.0, 0.0, 1.0]])
    r = float(inl.sum()) / len(p0) if inl is not None else 0.0
    return H, r


def _warp(poly, H):
    p = np.array(poly, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def _sane(poly, w, h, ref_area, max_area_ratio=4.0):
    """Reject a warped polygon that has left the frame or exploded in size."""
    xs, ys = poly[:, 0], poly[:, 1]
    if xs.max() < 0 or ys.max() < 0 or xs.min() > w or ys.min() > h:
        return False
    a = cv2.contourArea(poly.astype(np.float32))
    if a <= 1.0 or ref_area <= 1.0:
        return False
    r = a / ref_area
    return 1.0 / max_area_ratio <= r <= max_area_ratio


def track_polygon(cap, anchor_idx, poly, f_start, f_end, step,
                  fb_max=2.5, min_inliers_ratio=0.45, reseed_every=8,
                  min_conf=0.35, track_step=2):
    """
    Propagate `poly` from `anchor_idx` outward to [f_start, f_end], recording a
    sample every `step` frames.

    `track_step` is how far the tracker actually moves between optical-flow
    steps, and it is deliberately much smaller than `step`. Optical flow assumes
    a patch barely moves between frames; a perimeter board drifts slowly and
    survives a coarse step, but a running player's chest travels far enough in a
    quarter of a second that the flow simply loses it. Stepping finely and
    recording coarsely keeps jerseys trackable without inflating the output.

    Returns samples ordered by frame index, each:
        {frame, poly, area_px, fb_err, inlier_ratio, conf, anchor_dist_frames}

    Tracking stops in a direction as soon as it stops being trustworthy, so the
    returned span is usually SHORTER than requested. That is deliberate: an
    honest gap beats a fabricated polygon.
    """
    out = {}

    def read(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        return fr if ok else None

    anchor = read(anchor_idx)
    if anchor is None:
        return []
    H_img, W_img = anchor.shape[:2]
    ref_area = max(cv2.contourArea(np.array(poly, np.float32)), 1.0)
    out[int(anchor_idx)] = {
        "frame": int(anchor_idx), "poly": [[float(x), float(y)] for x, y in poly],
        "area_px": float(ref_area), "fb_err": 0.0, "inlier_ratio": 1.0,
        "conf": 1.0, "anchor_dist_frames": 0,
    }

    for direction in (+1, -1):
        limit = f_end if direction > 0 else f_start
        g_prev, s = _gray(anchor)
        cur = np.array(poly, np.float32) * s
        pts = _seed(g_prev, cur)
        i = anchor_idx
        since_seed = 0
        run_conf = 1.0

        while True:
            i += direction * track_step
            if (direction > 0 and i > limit) or (direction < 0 and i < limit):
                break
            frame = read(i)
            if frame is None:
                break
            g_cur, _ = _gray(frame)

            p0, p1, fb = _fb_step(g_prev, g_cur, pts)
            H, ratio = _transform(p0, p1)
            if H is None or fb > fb_max or ratio < min_inliers_ratio:
                break

            cur = _warp(cur, H)
            full = cur / s
            if not _sane(full, W_img, H_img, ref_area):
                break

            # Confidence is a RUNNING PRODUCT over steps, not a property of the
            # current step. Round-trip error is near zero by construction (bad
            # points were already discarded), so a per-step score saturates at
            # 1.0 and would still claim certainty many seconds from the anchor.
            # Compounding makes it decay the way trust actually does, and the
            # constant drift term means even a flawless chain loses confidence
            # with distance -- each warp resamples the polygon from the last.
            step_q = float(np.clip((1.0 - fb / fb_max) * min(1.0, ratio / 0.9),
                                   0.0, 1.0))
            run_conf *= (_DRIFT_PER_FRAME ** track_step) * step_q
            conf = float(np.clip(run_conf, 0.0, 1.0))
            if abs(i - anchor_idx) % step != 0:
                g_prev, pts = g_cur, p1
                since_seed += 1
                if since_seed >= reseed_every or len(pts) < 20:
                    pts = _seed(g_cur, cur)
                    since_seed = 0
                    if pts is None:
                        break
                continue

            out[int(i)] = {
                "frame": int(i),
                "poly": [[float(x), float(y)] for x, y in full],
                "area_px": float(cv2.contourArea(full.astype(np.float32))),
                "fb_err": round(fb, 3), "inlier_ratio": round(ratio, 3),
                "conf": round(conf, 3),
                "anchor_dist_frames": int(abs(i - anchor_idx)),
            }

            if conf < min_conf:
                break

            g_prev = g_cur
            pts = p1
            since_seed += 1
            if since_seed >= reseed_every or len(pts) < 20:
                pts = _seed(g_cur, cur)
                since_seed = 0
                if pts is None:
                    break

    return [out[k] for k in sorted(out)]
