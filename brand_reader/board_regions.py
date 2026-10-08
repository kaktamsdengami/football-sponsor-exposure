"""
Advertising board localisation by FOLLOWING THE PITCH BOUNDARY.

Scope: the FAR TOUCHLINE only (the board sits above the grass). That is the
dominant inventory on the main camera. The near touchline was removed: the crowd
right behind it is out of focus (shallow depth of field) hence smooth, and the
board frequently runs off the bottom of the frame -- the texture break simply does
not exist there. It comes back with a trained detector.

    1. boundary: for each column, the first row followed by a DEEP COLUMN OF
       GREEN -> the true grass line
    2. ROBUST FIT: a touchline is straight in the world, hence straight in the
       image. Fit a line and reject outlier columns -- typically the pitch
       outline curving around the GOAL, where there is no board at all.
    3. band height: texture break (smooth board / textured crowd), measured once
       per frame over the whole retained span
    4. RECTIFY: resample the band along the boundary into a straight rectangle,
       then REFLOW it into a few stacked rows

Rectify + reflow replaces tiling. A fixed grid cuts a creative into pieces
("bet3" / "65 SPORTW" / "ETTEN"), which breaks identification and makes the
cluster count explode as soon as the camera pans. The reflowed band holds ALL the
advertising in ONE image, readable at a glance: one LLM call per frame, no
creative ever cut.
"""

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# HSV range of the pitch grass (OpenCV: hue on 0-180).
# Deliberately NARROW hue: measured on ucl.mp4, grass sits within H 42-44, while
# the green of LED screens is a cyan-green at H 75-86. A wide range (30-90)
# catches both and the board gets taken for grass, shifting the boundary by a
# whole band height.
# Re-check this on every new source (different turf, broadcaster, snow).
GREEN_LOWER = np.array([32, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([62, 255, 255], dtype=np.uint8)


def green_mask(frame):
    """Raw mask of green pixels (grass, but also hi-vis steward vests)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)


def _rolling_median(a, k):
    if k <= 1:
        return a
    if k % 2 == 0:
        k += 1
    pad = k // 2
    return np.median(sliding_window_view(np.pad(a, pad, mode="edge"), k), axis=-1)


def pitch_boundary(frame, depth=None, frac=0.85, smooth=61,
                   min_cols_ratio=0.10, edge_margin=8):
    """
    Grass line (far touchline) for each column.

    We do NOT take the first green pixel: hi-vis steward vests pass the HSV
    filter and would push the boundary above the board. Instead we require a DEEP
    COLUMN OF GREEN: `depth` pixels below that are at least `frac` green. Grass
    continues for hundreds of rows, a vest does not.

    Returns None when there is no credible boundary (crowd shot, close-up).
    """
    h, w = frame.shape[:2]
    if depth is None:
        # The depth must EXCEED the tallest plausible board: an LED panel showing
        # a green creative is itself a ~40px column of green and would pass a
        # shorter test.
        depth = max(24, h // 20)

    m = (green_mask(frame) > 0).astype(np.float32)

    # prof[y, x] = fraction of green over [y, y+depth)  -> grass below
    prof = cv2.boxFilter(m, -1, (1, depth), anchor=(0, 0),
                         normalize=True, borderType=cv2.BORDER_REPLICATE)
    solid = prof >= frac

    has = solid.any(axis=0)
    if has.sum() < max(10, min_cols_ratio * w):
        return None

    idx = np.argmax(solid, axis=0).astype(np.float32)

    # a boundary hugging the frame edge means the pitch runs out of the image
    usable = has & (idx > edge_margin)
    if usable.sum() < max(10, min_cols_ratio * w):
        return None

    xs = np.arange(w)
    idx = np.interp(xs, xs[usable], idx[usable])
    return _rolling_median(idx, smooth)


def fit_touchline(yb, deg=1, iters=6, k=2.5, min_tol=4.0, max_tol=32.0):
    """
    Fit a ROBUST line to the boundary; returns (fit, inliers, coefficients).

    A touchline is straight in the world, hence straight in the image (perspective
    projection). Where the boundary departs sharply from that, it is no longer the
    touchline: it is the pitch outline curving around the GOAL, the penalty area,
    or a player spilling over. Those columns carry no board and must be dropped,
    otherwise the band samples netting, crowd and grass.

    The tolerance is CAPPED (max_tol): when a large share of columns are outliers
    the MAD blows up and a purely statistical threshold rejects nothing. A real
    touchline never departs from a straight line by more than a few tens of pixels.
    """
    n = yb.size
    xs = np.arange(n, dtype=np.float64)
    keep = np.ones(n, bool)

    for _ in range(iters):
        coef = np.polyfit(xs[keep], yb[keep], deg)
        fit = np.polyval(coef, xs)
        resid = np.abs(yb - fit)
        tol = float(np.clip(k * 1.4826 * float(np.median(resid[keep])), min_tol, max_tol))
        new = resid <= tol
        if new.sum() < 0.2 * n:
            break
        done = bool((new == keep).all())
        keep = new
        if done:
            break

    coef = np.polyfit(xs[keep], yb[keep], deg)
    return np.polyval(coef, xs), keep, coef


def _robust_lsq(A, y, iters=6, k=2.5, min_tol=4.0, max_tol=32.0):
    """
    Robust least squares for an arbitrary design matrix A (n x p): the same
    iteratively-reweighted outlier rejection as fit_touchline, but for any basis.
    Used for the one-knot piecewise fit below.
    """
    n = y.size
    keep = np.ones(n, bool)
    for _ in range(iters):
        beta, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        resid = np.abs(y - A @ beta)
        tol = float(np.clip(k * 1.4826 * float(np.median(resid[keep])),
                            min_tol, max_tol))
        new = resid <= tol
        if new.sum() < 0.2 * n:
            break
        if bool((new == keep).all()):
            keep = new
            break
        keep = new
    beta, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return A @ beta, keep, beta


def fit_touchline_pw(yb, iters=6, k=2.5, min_tol=4.0, max_tol=32.0,
                     max_knots=1, min_seg_frac=0.12, min_slope_change_deg=3.5,
                     tail_trigger=1.6, knot_step=20, scan_sub=4, worse_ratio=1.7):
    """
    Fit the boundary as a CONTINUOUS piecewise-linear curve with up to one knot.

    On a corner shot the perimeter runs along two straight lines -- the far
    touchline, then the goal line -- meeting at the pitch corner. A single
    straight fit (fit_touchline) tolerates the bend (it stays within the robust
    tolerance) and quietly under-covers those goal-line boards.

    So the test is localised. Fit one line. If one END of the boundary sits
    systematically off it -- the tail residual is >= `tail_trigger` x the core
    residual -- a corner is there. Scan a knot into that half and accept it only
    if the two-segment fit cuts the residual IN THAT TAIL by >= `worse_ratio`
    and the slope actually changes by >= `min_slope_change_deg`. A clean straight
    boundary triggers nothing and keeps zero knots -- this reduces to
    fit_touchline.

    Returns (fit, keep, segments, knots):
      fit       (n,) evaluated boundary
      keep      (n,) inlier mask from the final full-resolution fit
      segments  [{slope, intercept, x_lo, x_hi}, ...]  -- 1 or 2 entries, tiling [0,n)
      knots     [[xk, yk]]  -- 0 or 1 entries
    """
    n = int(yb.size)
    xs = np.arange(n, dtype=np.float64)

    A1 = np.column_stack([np.ones(n), xs])
    fit1, keep1, b1 = _robust_lsq(A1, yb, iters, k, min_tol, max_tol)
    seg1 = [{"slope": float(b1[1]), "intercept": float(b1[0]),
             "x_lo": 0, "x_hi": n}]
    if max_knots < 1 or n <= 200:
        return fit1, keep1, seg1, []

    min_seg = max(80, int(min_seg_frac * n))
    res1 = np.abs(yb - fit1)
    core = float(np.median(res1[min_seg:n - min_seg]))
    floor = max(2.0, 1.5 * core)
    tail_r = float(np.median(res1[n - min_seg:]))
    tail_l = float(np.median(res1[:min_seg]))
    if max(tail_r, tail_l) < floor * tail_trigger:
        return fit1, keep1, seg1, []                 # straight enough
    right = tail_r >= tail_l                          # which corner

    ss = max(1, int(scan_sub))
    xs_s, yb_s = xs[::ss], yb[::ss]
    res1_s = res1[::ss]
    tail_s = (xs_s >= n - min_seg) if right else (xs_s < min_seg)
    r1_tail = float(np.median(res1_s[tail_s]))
    eps = 1e-6

    lo, hi = (n // 2, n - min_seg) if right else (min_seg, n // 2)
    best = None                                      # (r2_tail, xk)
    for xk in range(lo, hi, int(knot_step)):
        A2s = np.column_stack([np.ones(xs_s.size), xs_s,
                               np.maximum(0.0, xs_s - xk)])
        f2s, _, b2 = _robust_lsq(A2s, yb_s, iters, k, min_tol, max_tol)
        if float(np.degrees(np.arctan(abs(b2[2])))) < min_slope_change_deg:
            continue
        r2_tail = float(np.median(np.abs(yb_s - f2s)[tail_s]))
        if r1_tail >= r2_tail * worse_ratio and (best is None or r2_tail < best[0]):
            best = (r2_tail, xk)

    if best is None:
        return fit1, keep1, seg1, []

    # refine the knot column around the coarse winner
    xk = best[1]
    for cand in range(max(min_seg, xk - int(knot_step)),
                      min(n - min_seg, xk + int(knot_step) + 1), 4):
        A2s = np.column_stack([np.ones(xs_s.size), xs_s,
                               np.maximum(0.0, xs_s - cand)])
        f2s, _, b2 = _robust_lsq(A2s, yb_s, iters, k, min_tol, max_tol)
        if float(np.degrees(np.arctan(abs(b2[2])))) < min_slope_change_deg:
            continue
        r2_tail = float(np.median(np.abs(yb_s - f2s)[tail_s]))
        if r2_tail < best[0]:
            best = (r2_tail, cand)
    xk = best[1]

    A2 = np.column_stack([np.ones(n), xs, np.maximum(0.0, xs - xk)])
    fit2, keep2, b2 = _robust_lsq(A2, yb, iters, k, min_tol, max_tol)
    slope_a, slope_b = float(b2[1]), float(b2[1] + b2[2])
    yk = float(b2[0] + b2[1] * xk)
    segments = [
        {"slope": slope_a, "intercept": float(b2[0]), "x_lo": 0, "x_hi": int(xk)},
        {"slope": slope_b, "intercept": float(yk - slope_b * xk),
         "x_lo": int(xk), "x_hi": n},
    ]
    return fit2, keep2, segments, [[int(xk), yk]]


def _close_gaps(mask, k):
    """
    Fill small holes in the inlier mask. A player standing on the touchline
    creates an outlier island a few tens of pixels wide that splits the span in
    two: without this closing we would keep only a fragment of the band.
    """
    if k < 3:
        return mask
    ker = np.ones((1, int(k) | 1), np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8).reshape(1, -1),
                              cv2.MORPH_CLOSE, ker)
    return closed.ravel().astype(bool)


def _largest_run(mask):
    """(lo, hi) of the longest contiguous run of True, or None."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    if edges.size == 0:
        return None
    starts, ends = edges[0::2], edges[1::2]
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


# ---------------------------------------------------------------------------
# Band confidence -- is this actually a clean main-camera perimeter read?
# ---------------------------------------------------------------------------
# band_geometry used to be binary: a dict or None. But a frame can clear every
# gate and still be the WRONG kind of shot -- a goal celebration on a low
# camera, a steeply receding board, half the span bridged over a player huddle.
# Those produce a plausible-looking band that captures one creative with
# distorted geometry and silently drops everything else in the frame.
#
# `confidence` (0..1) turns that into a scalar later stages can act on: below a
# threshold, don't trust the band -- send the whole frame to the LLM instead.
# The four sub-scores are also returned so the failure is visible per frame.
# Thresholds are a FIRST PASS, to be tuned on shot_preview.py output.

_CONF_SPAN_FULL = 0.72     # span/width at/above which span is not penalised
_CONF_RESID_OK = 0.25      # boundary-vs-fit residual (in units of H) still fine
_CONF_SLOPE_OK = 8.0       # touchline tilt (deg) still consistent with hard cam
_CONF_BRIDGE_OK = 0.15     # share of span filled by gap-closing still fine
_CONF_GRASS_OK = 0.50      # share of span with genuine deep grass below the line
_CONF_BAND_GRASS_MAX = 0.35  # above this the "board" is just more grass
_CONF_BOUNDARY_Y_OK = 0.24  # grass line higher (smaller y/h) than this = hard cam;
                            # much lower means a zoomed / low shot where an
                            # out-of-focus LED wall fills the top of the frame


def _grass_below_frac(frame, fit, run, H0):
    """
    Fraction of the retained span that has a DEEP RUN OF GREEN just below the
    fitted line. On the hard camera the board sits on grass, so this is ~1. When
    the "boundary" has actually locked onto a line inside a wall of out-of-focus
    LED (a low goalmouth shot), there is little or no grass under it and this
    collapses -- the single strongest tell that the band is not a real read.
    """
    h, w = frame.shape[:2]
    x0, x1 = run
    if x1 - x0 < 2:
        return 0.0
    gm = green_mask(frame) > 0
    depth = int(np.clip(round(2.0 * H0), 16, 80))
    cols = np.arange(x0, x1)
    base = np.clip(np.rint(fit[x0:x1]).astype(np.int64) + 3, 0, h - 1)
    rows = base[None, :] + np.arange(depth)[:, None]
    ok = rows < h
    rows = np.clip(rows, 0, h - 1)
    g = gm[rows, np.broadcast_to(cols, rows.shape)]
    frac_col = (g & ok).sum(0) / np.maximum(ok.sum(0), 1)
    return float((frac_col > 0.5).mean())


def _band_grass_frac(frame, fit, run, H0):
    """
    How much of the band region is grass.

    A band is only a board if it is NOT pitch. On a close-up of a player, or a
    camera pointed at open grass, the boundary fit can still succeed and every
    other check can pass -- the line is straight, there is grass below it, the
    span is wide -- and the result is a confident band containing nothing but
    turf. Measuring the band itself is the check that catches it.

    Safe against green LED creatives: the grass hue range is deliberately narrow
    (H 32-62) while LED green sits at H 75-86, so a green advert does not read
    as grass here.
    """
    h, w = frame.shape[:2]
    x0, x1 = run
    if x1 - x0 < 2 or H0 < 1:
        return 1.0
    gm = green_mask(frame) > 0
    depth = max(4, int(round(H0)))
    cols = np.arange(x0, x1)
    top = np.clip(np.rint(fit[x0:x1]).astype(np.int64) - depth, 0, h - 1)
    rows = top[None, :] + np.arange(depth)[:, None]
    ok = (rows >= 0) & (rows < h)
    rows = np.clip(rows, 0, h - 1)
    g = gm[rows, np.broadcast_to(cols, rows.shape)]
    return float((g & ok).sum() / max(ok.sum(), 1))


def band_confidence(yb_full, fit, keep, run, H0, w, slope_deg=0.0,
                    boundary_y=0.0, break_found=True, grass_below_frac=1.0,
                    band_grass_frac=0.0):
    """
    (score, parts) for a retained band. `parts` carries the raw measurements so
    a low score can be explained: short span, boundary that does not sit on the
    fitted curve (receding board), a steep touchline, a span bridged across
    occluders, a grass line sitting too low (zoomed / low shot, LED wall), or
    little real grass below the line.

    `slope_deg` is the steepest segment tilt of the (piecewise) touchline fit --
    passed in because the fit may now be a two-segment polyline, not one line.

    `break_found` (was there a smooth-board / textured-crowd transition above the
    line) is reported but NOT scored: on the current tuning `estimate_band_height`
    legitimately falls back for a large share of good frames, so it is too noisy
    to penalise directly.
    """
    x0, x1 = run
    span = max(x1 - x0, 1)
    seg = slice(x0, x1)

    span_ratio = span / w
    fit_resid = float(np.median(np.abs(yb_full[seg] - fit[seg]))) / max(H0, 1.0)
    closed = _close_gaps(keep, w // 16)
    bridged = float((closed[seg] & ~keep[seg]).sum()) / span

    score = 1.0
    score -= 2.0 * max(0.0, _CONF_SPAN_FULL - span_ratio)
    score -= 2.0 * max(0.0, fit_resid - _CONF_RESID_OK)
    score -= 0.05 * max(0.0, slope_deg - _CONF_SLOPE_OK)
    score -= 2.0 * max(0.0, bridged - _CONF_BRIDGE_OK)
    score -= 3.0 * max(0.0, boundary_y - _CONF_BOUNDARY_Y_OK)
    score -= 1.5 * max(0.0, _CONF_GRASS_OK - grass_below_frac)
    # The band is mostly turf -> there is no board here at all.
    score -= 2.5 * max(0.0, band_grass_frac - _CONF_BAND_GRASS_MAX)
    score = float(np.clip(score, 0.0, 1.0))

    return score, {"span_ratio": round(span_ratio, 3),
                   "fit_resid": round(fit_resid, 3),
                   "slope_deg": round(slope_deg, 2),
                   "bridged_frac": round(bridged, 3),
                   "boundary_y": round(float(boundary_y), 3),
                   "break_found": bool(break_found),
                   "grass_below": round(float(grass_below_frac), 3),
                   "band_grass": round(float(band_grass_frac), 3)}


def _rectified_profile(gray, yb, h_max):
    """
    hf[d] = horizontal high-frequency energy at (d+1) pixels above the grass,
    measured over the whole retained span. Following the boundary makes the band
    horizontal, so a single 1-D profile suffices -- far more stable than a
    per-tile estimate.
    """
    h, w = gray.shape
    ys = (yb[None, :] - np.arange(1, h_max + 1)[:, None]).astype(np.int32)
    valid = (ys >= 0) & (ys < h)
    xs = np.broadcast_to(np.arange(w)[None, :], ys.shape)

    strip = gray[np.clip(ys, 0, h - 1), xs].astype(np.float32)
    strip[~valid] = np.nan

    with np.errstate(all="ignore"):
        hf = np.nanmedian(np.abs(np.diff(strip, axis=1)), axis=1)
    return np.nan_to_num(hf, nan=np.inf)


def estimate_band_height(gray, yb, h_max=200, skip=3, ref_lo=3, ref_hi=12,
                         mult=1.5, add=1.0, persist=8, confirm=20,
                         margin=1.25, min_band=12, return_info=False):
    """
    Board height above the boundary, from a TEXTURE BREAK.

    An LED board is horizontally smooth (large flat areas); the crowd behind it is
    sharp and heavily textured. We walk upward while high-frequency energy stays
    low and stop at the jump (typically hf ~4 on the board, ~15 above it).

    With return_info=True, also returns {"break_found": bool}: False means no
    smooth-then-textured transition exists above the line -- there is no bounded
    board there (LED wall, crowd-only, graphic), a strong "not a real band" cue.
    """
    def _ret(val, found):
        return (val, {"break_found": found}) if return_info else val

    hf = _rectified_profile(gray, yb, int(h_max))
    if hf.size < ref_hi + persist:
        return _ret(float(min_band), False)

    ref = float(np.median(hf[ref_lo:ref_hi]))
    thr = ref * mult + add

    # A transition must be SUSTAINED: the board's own top edge and its text create
    # transient ~10px bumps that would otherwise stop the walk well before the
    # real crowd and collapse the height onto the lower bound.
    d_break, run = None, 0
    for d in range(skip, hf.size):
        if hf[d] > thr:
            run += 1
            if run >= persist and float(np.median(hf[d:d + confirm])) > thr:
                d_break = d - persist + 1
                break
        else:
            run = 0

    found = d_break is not None
    if not found:
        # No break found: do NOT take h_max (the band would swallow the whole
        # crowd); fall back to a conservative value.
        d_break = 0.5 * hf.size

    return _ret(float(np.clip(d_break * margin, min_band, h_max)), found)


def band_geometry(frame, k_height=0.45, min_band=12, max_band_ratio=0.35,
                  smooth=61, min_span_ratio=0.30, local_lo=0.8, local_hi=1.25):
    """
    Geometry dict, or None -- everything restricted to the span `(x0, x1)` where
    the boundary genuinely follows the touchline.

      yb    : smoothed boundary over the span
      H     : median band height
      local : perspective factor, computed from the FITTED LINE (smooth) rather
              than the raw boundary, so it cannot run away
      span  : retained column interval in the frame
      coef  : touchline fit coefficients

    Guard rail on H: a perimeter board is always a small fraction of the boundary
    depth.
    """
    h, w = frame.shape[:2]
    yb_full = pitch_boundary(frame, smooth=smooth)
    if yb_full is None:
        return None

    # reject whatever is not the touchline (goal outline, etc.); allow one knot
    # so a corner shot keeps BOTH the far touchline and the goal-line boards.
    fit, keep, segments, knots = fit_touchline_pw(yb_full, max_tol=0.06 * h)
    run = _largest_run(_close_gaps(keep, w // 16))
    if run is None:
        return None
    x0, x1 = run
    if (x1 - x0) < max(32, min_span_ratio * w):
        return None

    # A knot only counts if the bend actually shows up WITHIN the retained span:
    # fit_touchline_pw works on the full frame width, so it can place a knot in a
    # region the span later trims off (a noisy frame edge, an overlaid graphic).
    # If the piecewise fit and the best straight line agree to within ~half a
    # band height across the span, drop the knot.
    if knots:
        xs_sp = np.arange(x0, x1)
        lin = np.polyfit(xs_sp, fit[x0:x1], 1)
        if float(np.max(np.abs(fit[x0:x1] - np.polyval(lin, xs_sp)))) < 14.0:
            knots = []
            segments = [{"slope": float(lin[0]), "intercept": float(lin[1]),
                         "x_lo": 0, "x_hi": int(w)}]
            fit = np.polyval(lin, np.arange(w))

    yb = yb_full[x0:x1].astype(np.float32)
    fit_seg = fit[x0:x1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, x0:x1]
    yb_med = float(np.median(yb))

    h_max = float(np.clip(k_height * yb_med, 30, max_band_ratio * h))
    H0, h_info = estimate_band_height(gray, yb, h_max=h_max, min_band=min_band,
                                      return_info=True)
    H0 = float(np.clip(H0, max(min_band, 0.06 * yb_med), 0.25 * yb_med))

    fit_med = float(np.median(fit_seg))
    # A bent boundary (corner shot) has one steep segment that recedes fast; let
    # the perspective factor swing wider there so the band does not thicken.
    clamp_lo, clamp_hi = (0.6, 1.4) if knots else (local_lo, local_hi)
    local = (np.clip(fit_seg / fit_med, clamp_lo, clamp_hi) if fit_med > 1
             else np.ones(x1 - x0))

    slope_deg = max((float(np.degrees(np.arctan(abs(s["slope"]))))
                     for s in segments), default=0.0)
    conf, conf_parts = band_confidence(
        yb_full, fit, keep, (x0, x1), H0, w, slope_deg,
        boundary_y=fit_med / h,
        break_found=h_info["break_found"],
        grass_below_frac=_grass_below_frac(frame, fit, (x0, x1), H0),
        band_grass_frac=_band_grass_frac(frame, fit, (x0, x1), H0))

    # `segments` (+ span + clamp bounds) rebuild the whole geometry from a handful
    # of scalars per frame instead of a 1918-element array -- what keeps pass-1
    # artifacts light and serialisable (JSONL, later an API). `coef` is kept as
    # the first segment for readers that predate the piecewise fit.
    return {"yb": yb, "H": H0, "local": local.astype(np.float32),
            "span": (int(x0), int(x1)), "fit": fit_seg.astype(np.float32),
            "coef": [segments[0]["slope"], segments[0]["intercept"]],
            "segments": segments, "knots": knots,
            "fit_med": fit_med, "local_clamp": (clamp_lo, clamp_hi),
            "confidence": conf, "conf_parts": conf_parts}


# ---------------------------------------------------------------------------
# RECTIFY + REFLOW
# ---------------------------------------------------------------------------

def rectify_band(frame, over_ratio=0.35, band_line=1, line_gap=1.05, **kw):
    """
    Rectify the advertising band into a straight rectangle.

    Output row r corresponds to a constant offset from the boundary; column c
    corresponds to frame column (x0 + c). The perspective slant disappears and the
    apparent height is normalised, which makes the band comparable across frames
    (useful for dedup).

    `band_line` selects which touchline advertising line to cut. `1` (default) is
    the pitch-level LED that sits on the boundary. `2` is the line stacked
    directly above it (at some grounds the paying sponsors are there, with the
    club's own promo on line 1). Line 2 is the same band shifted up by
    `line_gap` band-heights along the fitted boundary -- it inherits line 1's
    height, perspective and confidence, since the same clean boundary read
    implies both lines are on screen and undistorted.

    Returns None, or a dict:
        strip               rectified image (R, W, 3)
        yb, offs, local, x0 to reproject back to the frame
    """
    geo = band_geometry(frame, **kw)
    if geo is None:
        return None
    H0, local, fit = geo["H"], geo["local"], geo["fit"]
    x0, x1 = geo["span"]

    h = frame.shape[0]
    sub = frame[:, x0:x1]
    Hin = int(round(H0))
    over = max(2, int(round(over_ratio * H0)))

    # Line 2 sits `line_gap` band-heights higher up the fitted boundary. Line 1
    # keeps off_shift = 0, so its geometry is byte-for-byte unchanged.
    off_shift = int(round(H0 * line_gap * (band_line - 1)))

    # rows ordered as on screen: above the boundary first
    offs = np.arange(-Hin, over + 1, dtype=np.float32) - off_shift

    # Anchored on the FITTED LINE: the raw boundary keeps a few pixels of noise
    # which shows up as a sawtooth along the board edge in the rectified strip.
    ys = np.rint(fit[None, :] + offs[:, None] * local[None, :]).astype(np.int32)
    valid = (ys >= 0) & (ys < h)
    xs = np.broadcast_to(np.arange(x1 - x0)[None, :], ys.shape)

    strip = sub[np.clip(ys, 0, h - 1), xs]
    strip[~valid] = 0

    return {"strip": strip, "yb": fit, "offs": offs, "local": local,
            "x0": int(x0), "H": H0, "over": int(over),
            "band_line": int(band_line), "off_shift": int(off_shift),
            "coef": geo["coef"], "segments": geo["segments"], "knots": geo["knots"],
            "fit_med": geo["fit_med"],
            "span": geo["span"], "local_clamp": geo["local_clamp"],
            "confidence": geo["confidence"], "conf_parts": geo["conf_parts"]}


def reflow_strip(strip, n_rows=3, upscale=2.0, gap=8, search_ratio=0.15):
    """
    Fold a very elongated band (e.g. 1300x58) into a few stacked rows, so a vision
    model reads it like a paragraph.

    Cuts are placed at the columns of LOWEST vertical edge energy -- that is, in
    the gaps between creatives -- so a logo is never split in two. The search
    window stays narrow (search_ratio) to keep chunk widths close: otherwise the
    canvas is sized on the widest one and the other rows carry a large black area.

    Returns (image, layout); layout maps back to frame coordinates.
    """
    R, W = strip.shape[:2]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

    # vertical edge energy, per column
    col = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    col = cv2.blur(col.reshape(1, -1), (15, 1)).ravel()

    cuts = [0]
    span = W / n_rows
    for i in range(1, n_rows):
        ideal = int(span * i)
        half = int(span * search_ratio / 2)
        lo, hi = max(cuts[-1] + 16, ideal - half), min(W - 16, ideal + half)
        cuts.append(lo + int(np.argmin(col[lo:hi])) if hi > lo else ideal)
    cuts.append(W)

    chunks = [strip[:, cuts[i]:cuts[i + 1]] for i in range(n_rows)]
    rh = max(1, int(round(R * upscale)))
    out_w = max(1, int(round(max(c.shape[1] for c in chunks) * upscale)))

    canvas = np.zeros((n_rows * rh + (n_rows - 1) * gap, out_w, 3), np.uint8)
    rows = []
    y = 0
    for i, c in enumerate(chunks):
        cw = max(1, int(round(c.shape[1] * upscale)))
        canvas[y:y + rh, :cw] = cv2.resize(c, (cw, rh), interpolation=cv2.INTER_CUBIC)
        rows.append({"row": i, "y0": y, "h": rh, "w": cw, "x_off": cuts[i]})
        y += rh + gap

    return canvas, {"rows": rows, "upscale": float(upscale), "strip_shape": (R, W)}


def reflow_box_to_frame(rect, layout, box):
    """
    Convert a box (x1, y1, x2, y2) read on the REFLOWED image into a box in the
    original frame. This is what keeps geometry usable (area, position) even
    though the LLM only ever sees the folded band.
    """
    x1, y1, x2, y2 = box
    s = layout["upscale"]
    R, W = layout["strip_shape"]

    row = min(layout["rows"], key=lambda r: abs((r["y0"] + r["h"] / 2) - (y1 + y2) / 2))

    c1 = int(np.clip(row["x_off"] + x1 / s, 0, W - 1))
    c2 = int(np.clip(row["x_off"] + x2 / s, 0, W - 1))
    if c2 < c1:
        c1, c2 = c2, c1

    r1 = int(np.clip((y1 - row["y0"]) / s, 0, R - 1))
    r2 = int(np.clip((y2 - row["y0"]) / s, 0, R - 1))
    if r2 < r1:
        r1, r2 = r2, r1

    yb, offs, local, x0 = rect["yb"], rect["offs"], rect["local"], rect["x0"]
    cols = np.arange(c1, c2 + 1)
    ys = np.concatenate([yb[cols] + offs[r] * local[cols] for r in (r1, r2)])

    return (x0 + c1, int(max(0, ys.min())), x0 + c2, int(ys.max()))


def band_polygon(rect):
    """
    Outline of the band in ORIGINAL FRAME coordinates.

    The band is not a rectangle: it follows the touchline and its thickness varies
    with perspective. Returns a polygon (N, 2) -- top edge left to right, then
    bottom edge right to left -- useful to visually check the region actually sent
    to the API.
    """
    fit, offs, local, x0 = rect["yb"], rect["offs"], rect["local"], rect["x0"]
    xs = np.arange(fit.size) + x0
    top = fit + offs[0] * local
    bot = fit + offs[-1] * local
    pts = np.concatenate([
        np.stack([xs, top], axis=1),
        np.stack([xs[::-1], bot[::-1]], axis=1),
    ])
    return np.rint(pts).astype(np.int32)


def rect_from_scalars(coef, span, fit_med, local_clamp, H, over,
                      segments=None, knots=None, off_shift=0):
    """
    Rebuild just enough of a `rectify_band` result to draw its polygon, from the
    handful of numbers stored in detections.jsonl.

    This is why the geometry was stored as scalars rather than arrays: a later
    stage can recover the region on screen without decoding the video again.

    `segments` (piecewise touchline, one entry per straight run) rebuilds a bent
    boundary; when absent it falls back to the single-line `coef` (older runs).
    """
    x0, x1 = int(span[0]), int(span[1])
    xs = np.arange(x0, x1)
    if segments:
        fit = np.empty(x1 - x0, np.float64)
        placed = np.zeros(x1 - x0, bool)
        for s in segments:
            m = (xs >= s["x_lo"]) & (xs < s["x_hi"])
            fit[m] = s["slope"] * xs[m] + s["intercept"]
            placed |= m
        if not placed.all():                      # columns past the last x_hi
            s = segments[-1]
            fit[~placed] = s["slope"] * xs[~placed] + s["intercept"]
        fit = fit.astype(np.float32)
    else:
        fit = np.polyval(coef, xs).astype(np.float32)
    lo, hi = local_clamp
    local = (np.clip(fit / fit_med, lo, hi) if fit_med > 1
             else np.ones(x1 - x0, np.float32)).astype(np.float32)
    offs = np.arange(-int(round(H)), int(over) + 1, dtype=np.float32) - int(off_shift)
    return {"yb": fit, "offs": offs, "local": local, "x0": x0}


def band_image(frame, n_rows=3, upscale=2.0, **kw):
    """
    Advertising band ready for an LLM call: ONE image holding all the far-touchline
    advertising, with no creative cut.

    Returns None, or (image, meta) where meta = {"rect":..., "layout":...} to
    reproject the boxes the LLM returns back to the frame.
    """
    rect = rectify_band(frame, **kw)
    if rect is None:
        return None
    img, layout = reflow_strip(rect["strip"], n_rows=n_rows, upscale=upscale)
    return img, {"rect": rect, "layout": layout}


# ---------------------------------------------------------------------------
# Fixed-grid tiling (kept for debugging and as a fallback)
# ---------------------------------------------------------------------------

def board_band_tiles(frame, over_ratio=0.35, tile_aspect=6.0, overlap=0.3,
                     min_band=12, max_band_ratio=0.35, **kw):
    """
    Candidate band tiles on a fixed grid. Cuts creatives that are too wide --
    prefer band_image() for identification; kept for inspection.
    """
    h = frame.shape[0]

    geo = band_geometry(frame, min_band=min_band, max_band_ratio=max_band_ratio, **kw)
    if geo is None:
        return []
    yb, H0, local = geo["yb"], geo["H"], geo["local"]
    x0, x1 = geo["span"]

    n = x1 - x0
    tiles = []
    x = 0
    while x < n:
        H = float(np.clip(H0 * local[min(x, n - 1)], min_band, max_band_ratio * h))
        tw = max(8, int(tile_aspect * H))
        xe = min(n, x + tw)
        seg = yb[x:xe]
        if seg.size == 0:
            break
        y_ref = float(np.median(seg))

        y1 = int(max(0, y_ref - H))
        y2 = int(min(h, y_ref + over_ratio * H))

        if (y2 - y1) >= min_band and (xe - x) >= 8:
            tiles.append({"bbox": (x0 + int(x), y1, x0 + int(xe), y2),
                          "type": "board", "band_h": H, "y_boundary": y_ref})

        x += max(8, int(tw * (1.0 - overlap)))

    return tiles
