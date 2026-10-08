"""
Find the broadcaster's burned-in graphics, so they are not counted as advertising.

THE PROBLEM. On pilot_match_5min.mp4 a streaming-service watermark sits top-right for the
whole clip and a betting-branded score bug sits top-left for most of it. Both are
brands. Neither is inventory the club sells: they belong to the streamer and the
league, and they are on screen ~100% of the time. Read naively they swamp every
real advertiser -- the first merged run credited the watermark brand 59.3s against a hand
count of 15s, purely from the watermark.

THE IDEA, and why it is not tuned to this clip. A burned-in graphic is the only
thing in a broadcast that stays pixel-identical while the camera cuts between
completely different scenes. Stadium content cannot: a board seen from the
centre-line camera and from the goal camera lands on different pixels, and grass
has players crossing it. So: sample frames spread across the WHOLE video, and
mark pixels whose value barely moves. No coordinates, no per-broadcaster
constants, no assumption about which corner the logo is in.

The threshold is in 8-bit levels and is justified by the codec, not by this
video: two frames of genuinely different scene content differ by far more than
compression noise, so a pixel that stays within a couple of levels across
scene cuts is not showing scene content. `_STATIC_LEVELS` is that noise floor.

Sampling across cuts is what makes it safe. On a video that never cuts, a truly
static camera could mark real scenery as overlay, so `overlay_mask` refuses to
return a mask when the sampled frames are too similar to each other to tell the
two apart, and says so.

KNOWN LIMIT. The test is "did this pixel move", so scenery that is genuinely
motionless for the whole sample can read as a graphic -- a locked-off camera on
an empty stand, say. Two things keep that from doing damage: broadcast cameras
pan and breathe, so real scenery moves by more than the noise floor within a
few frames; and a detection covering more than `_MAX_OVERLAY_FRAC` of the frame
is rejected outright as a failure rather than trusted. If a future broadcast
does trip it, the symptom is visible -- `detect` prints every masked box, and
they will not look like a logo.
"""
import numpy as np

try:                                            # cv2 is required at runtime
    import cv2
except ImportError:                             # pragma: no cover
    cv2 = None

_STATIC_LEVELS = 3.0      # 8-bit MAD below this is codec noise, not scene change
# Guard on the BUSIEST tenth of the frame, not the median. Most of a football
# frame is grass: uniform green that keeps almost the same value however the
# camera moves, so the median pixel barely budges (7 levels on pilot_match_5min)
# even in a clip full of hard cuts. What tells us the sample has real scene
# variety is that its most-changing regions really do change.
_SCENE_PCTL = 90
_MIN_SCENE_MAD = 12.0
_GLYPH_MERGE = 25         # merge anti-aliased letters into one region first...
_MIN_BLOB_PX = 600        # ...then drop anything too small to be a graphic
_MAX_OVERLAY_FRAC = 0.25  # a "mask" covering more than this is a failed detection
_DILATE = 5               # grow slightly: glyph edges do vary a little


def sample_frames(video_path, n=96, max_frames=None):
    """`n` frames spread evenly over the whole video, as greyscale float32."""
    if cv2 is None:
        raise RuntimeError("opencv is required")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, int(max_frames))
    if total <= 0:
        cap.release()
        return []
    idx = np.linspace(0, total - 1, num=min(n, total), dtype=int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return out


def overlay_mask(video_path, n=96, max_frames=None, frames=None):
    """
    uint8 mask, 255 where a burned-in broadcast graphic sits.

    Returns (mask, info). `info["ok"]` is False when the video does not cut
    enough for the test to mean anything; the mask is then all zeros and the
    caller should mask nothing rather than mask the wrong thing.
    """
    frames = frames if frames is not None else sample_frames(video_path, n, max_frames)
    info = {"n_frames": len(frames), "ok": False, "reason": None,
            "scene_mad": 0.0, "overlay_frac": 0.0}
    if len(frames) < 8:
        info["reason"] = "too few frames sampled"
        h, w = (frames[0].shape if frames else (1, 1))
        return np.zeros((h, w), np.uint8), info

    stack = np.stack(frames, axis=0)
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med), axis=0)         # per-pixel, over time

    # Does this sample contain genuinely different scenes? If the whole frame
    # barely moves there is nothing to separate overlay from scenery.
    scene_mad = float(np.percentile(mad, _SCENE_PCTL))
    info["scene_mad"] = round(scene_mad, 2)
    if scene_mad < _MIN_SCENE_MAD:
        info["reason"] = (f"sampled frames are too alike (p{_SCENE_PCTL} MAD "
                          f"{scene_mad:.1f} < {_MIN_SCENE_MAD}) -- cannot tell a "
                          f"burned-in graphic from a static camera")
        return np.zeros(mad.shape, np.uint8), info

    raw = (mad < _STATIC_LEVELS).astype(np.uint8) * 255

    # Merge first, THEN filter by size. A watermark is drawn as anti-aliased
    # glyphs on a rounded box: the letter interiors are pixel-stable but the
    # gaps between them are not, so labelling the raw mask yields a scatter of
    # 30-100px specks that any sane size filter throws away. Closing them into
    # one region first is what makes "big enough to be a graphic" testable.
    merged = cv2.morphologyEx(
        raw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_GLYPH_MERGE, _GLYPH_MERGE)))
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    keep = np.zeros_like(merged)
    blobs = 0
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= _MIN_BLOB_PX:
            keep[lbl == i] = 255
            blobs += 1
    mask = cv2.dilate(keep, np.ones((_DILATE, _DILATE), np.uint8), iterations=1)

    frac = float((mask > 0).mean())
    info.update(overlay_frac=round(frac, 4), blobs=blobs)
    if frac > _MAX_OVERLAY_FRAC:
        info["reason"] = (f"{100*frac:.0f}% of the frame reads as static -- that "
                          f"is a failed detection, not a watermark")
        return np.zeros(mask.shape, np.uint8), info

    info["ok"] = True
    return mask, info


def boxes_from_mask(mask, pad=2):
    """[(x, y, w, h)] for each masked region, for logging and debug overlays."""
    if cv2 is None or mask is None or not mask.any():
        return []
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8),
                                                          connectivity=8)
    out = []
    for i in range(1, n_lbl):
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        out.append((max(0, x - pad), max(0, y - pad), w + 2 * pad, h + 2 * pad))
    return sorted(out, key=lambda b: -b[2] * b[3])


def apply_mask(frame, mask, fill=0):
    """Blank the overlay regions of `frame` so nothing downstream reads them."""
    if mask is None or not mask.any():
        return frame
    out = frame.copy()
    out[mask > 0] = fill
    return out
