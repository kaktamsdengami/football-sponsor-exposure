"""
Lock the burned-in-graphic detector on synthetic footage. No video needed.

The detector's whole claim is that it separates "a brand painted into the
broadcast feed" from "a brand in the stadium" using nothing but how the pixels
move over time. These cases check that claim, including the two ways it is
allowed to fail: a video with no scene variety, and a mask so large it must be
wrong.

    python overlay_mask_selftest.py
"""
import sys

import numpy as np

from brand_reader.overlay_mask import boxes_from_mask, overlay_mask

H, W = 270, 480
RNG = np.random.default_rng(7)


def _scene(n=40, static_boxes=(), noise=2.0):
    """`n` frames of wholly different scenes, with `static_boxes` burned in."""
    frames = []
    for _ in range(n):
        # Each frame is a different scene: blocky content at a different level.
        f = np.zeros((H, W), np.float32)
        for by in range(0, H, 30):
            for bx in range(0, W, 30):
                f[by:by + 30, bx:bx + 30] = RNG.integers(0, 256)
        f += RNG.normal(0, noise, f.shape)
        for (x, y, w, h, val) in static_boxes:
            f[y:y + h, x:x + w] = val          # identical in every frame
        frames.append(np.clip(f, 0, 255).astype(np.float32))
    return frames


def _covers(boxes, x, y, w, h):
    """Is (x,y,w,h) inside one of `boxes`, allowing for the dilation margin?"""
    for (bx, by, bw, bh) in boxes:
        if bx <= x + 2 and by <= y + 2 and bx + bw >= x + w - 2 and by + bh >= y + h - 2:
            return True
    return False


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def main():
    ok = True
    print("overlay_mask")

    # 1. Two burned-in graphics in a cutting scene are both found, and nothing
    #    else is.
    boxes_in = [(20, 15, 90, 34, 240.0), (380, 200, 70, 40, 12.0)]
    m, info = overlay_mask(None, frames=_scene(static_boxes=boxes_in))
    found = boxes_from_mask(m)
    ok &= check("detects a static graphic in a cutting scene", info["ok"],
                info.get("reason") or "")
    ok &= check("finds both graphics", len(found) == 2, f"found {len(found)}")
    for (x, y, w, h, _) in boxes_in:
        ok &= check(f"covers the graphic at {x},{y}", _covers(found, x, y, w, h))
    ok &= check("masks only a small share of the frame",
                info["overlay_frac"] < 0.10, f"{100*info['overlay_frac']:.1f}%")

    # 2. Content that changes is never masked, however few states it has. Two
    #    scenes that differ at EVERY pixel: nothing here is static, so nothing
    #    may be reported. (Built as a complement rather than a second random
    #    draw -- two random scenes collide on a block now and then by pure
    #    chance, which is an artifact of a 2-state toy, not of the detector.)
    a = _scene(n=1, noise=0.0)[0]
    b = (a + 100.0) % 256.0        # every pixel moves by 100 or 156 levels
    m, info = overlay_mask(None, frames=[a, b] * 20)
    ok &= check("content that changes is never masked", not m.any(),
                f"{100*info['overlay_frac']:.1f}% masked")

    # 3. A video that never changes cannot be judged, and must refuse rather
    #    than mask the whole stadium.
    still = _scene(n=1, noise=0.0)[0]
    m, info = overlay_mask(None, frames=[still.copy() for _ in range(30)])
    ok &= check("refuses when there is no scene variety", not info["ok"])
    ok &= check("...and masks nothing", not m.any())
    ok &= check("...and says why", bool(info["reason"]))

    # 4. Too few frames to say anything.
    m, info = overlay_mask(None, frames=_scene(n=4))
    ok &= check("refuses on too few frames", not info["ok"] and not m.any())

    # 5. A graphic smaller than the noise floor of the size filter is dropped,
    #    not reported -- specks must not become "advertising".
    m, info = overlay_mask(None, frames=_scene(static_boxes=[(10, 10, 6, 6, 200.0)]))
    ok &= check("ignores a speck too small to be a graphic",
                not _covers(boxes_from_mask(m), 10, 10, 6, 6))

    print("\n" + ("all checks passed" if ok else "FAILURES -- see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
