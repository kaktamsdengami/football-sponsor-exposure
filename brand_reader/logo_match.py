"""
Find known logos by matching them, not by reading them.

WHY, AND WHY IT IS NOT A STEP BACKWARDS. Relo Metrics and Nielsen Sports
measured sponsor exposure to a sellable standard years before any of this was
called AI, and they did it by never attempting the hard problem. They did not
read arbitrary text in arbitrary frames. They took the client's logo kit --
a closed set, typically a dozen brands -- and asked a much smaller question of
every frame: is THIS mark here, and where.

That reframing is the whole point, and it is exactly the shape of this job. The
club hands over its advertiser list. Twelve brands, known in advance. That is
classification against a gallery, not open-ended reading.

WHAT IT BUYS US. OCR credits exposure in proportion to how readable a wordmark
is, which is why one brand lands at 33% of its hand count and another at zero --
EasyOCR did not produce a single string resembling it across the entire clip,
in 25 unmatched candidates. Correlation does not care that a mark is stylised.
A logo that is hopeless to read is often *easier* to match, because it is a
distinctive fixed shape. OCR's weakness is this method's strength.

WHAT IT COSTS. A gallery. Either the club supplies the logo kit -- the normal
industry path, and worth asking for -- or templates are cut from the first
match and corrected by hand once. And it is closed-set by construction: a brand
absent from the gallery is invisible to it, so this complements the discovery
paths (`scan`'s unlisted strings, the vision model's unlisted brands) rather
than replacing them.

    from brand_reader.logo_match import load_gallery, find_logos
    gallery = load_gallery("pilot/logos")
    hits = find_logos(frame, gallery)
"""
import os

import cv2
import numpy as np

# Scales searched. A perimeter board's apparent size changes with camera
# distance far more than its shape does, so scale is the axis that matters;
# rotation is small on a touchline and is not searched.
SCALES = (0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 1.9)
# Normalised-correlation score at which a match is believed. TM_CCOEFF_NORMED
# is in [-1, 1]; broadcast compression, motion blur and LED refresh banding pull
# a true match well below 1, while unrelated content rarely clears 0.5.
MIN_SCORE = 0.62
# Two hits closer than this (in template widths) are the same board seen twice
# by neighbouring scales, not two panels.
NMS_FRAC = 0.5


def load_gallery(path):
    """
    {brand -> [template BGR]} from a directory of images.

    The file name is the brand, so a gallery is inspectable and correctable
    with a file manager. `BRAND.png` and `BRAND__2.png` are two views of one
    brand -- a second view is how you handle a logo that appears both boxed and
    bare, or in two colourways.
    """
    gallery = {}
    if not os.path.isdir(path):
        return gallery
    for fn in sorted(os.listdir(path)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        img = cv2.imread(os.path.join(path, fn))
        if img is None or img.size == 0:
            continue
        brand = stem.split("__")[0].strip()
        gallery.setdefault(brand, []).append(img)
    return gallery


def _prep(img):
    """Greyscale + light blur. Colour is deliberately discarded: an LED board
    shifts hue hard with exposure and refresh banding, while the mark's shape
    survives. Matching on shape is what makes one template cover a creative
    shown at several brightnesses."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0)


def match_one(frame_gray, tpl_gray, scales=SCALES, min_score=MIN_SCORE):
    """[(score, x, y, w, h)] for one template over one frame, best scale wins."""
    fh, fw = frame_gray.shape[:2]
    out = []
    for s in scales:
        w = max(8, int(tpl_gray.shape[1] * s))
        h = max(6, int(tpl_gray.shape[0] * s))
        if w >= fw or h >= fh:
            continue
        t = cv2.resize(tpl_gray, (w, h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(frame_gray, t, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= min_score)
        for x, y in zip(xs, ys):
            out.append((float(res[y, x]), int(x), int(y), w, h))
    return _nms(out)


def _nms(hits):
    """Keep the best of every cluster of overlapping hits."""
    hits = sorted(hits, key=lambda h: -h[0])
    kept = []
    for hit in hits:
        s, x, y, w, h = hit
        if any(abs(x - kx) < NMS_FRAC * max(w, kw)
               and abs(y - ky) < NMS_FRAC * max(h, kh)
               for _, kx, ky, kw, kh in kept):
            continue
        kept.append(hit)
    return kept


def find_logos(frame, gallery, scales=SCALES, min_score=MIN_SCORE, roi=None):
    """
    {brand -> [(score, x, y, w, h)]} in FRAME coordinates.

    `roi` is an optional (y0, y1) band to search, which is how the pitch crop
    gets applied: correlation cost scales with area and grass holds no boards.
    """
    y0 = 0 if roi is None else max(0, roi[0])
    sub = frame if roi is None else frame[y0:roi[1]]
    if sub.size == 0:
        return {}
    fg = _prep(sub)
    found = {}
    for brand, tpls in gallery.items():
        hits = []
        for t in tpls:
            hits += match_one(fg, _prep(t), scales, min_score)
        hits = _nms(hits)
        if hits:
            found[brand] = [(s, x, y + y0, w, h) for s, x, y, w, h in hits]
    return found
