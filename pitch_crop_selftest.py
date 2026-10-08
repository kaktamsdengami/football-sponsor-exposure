"""
Lock the pitch crop. No video needed.

This one is worth locking hard because its failure is invisible and expensive:
crop too high and a board silently vanishes from the report, and nothing
downstream can tell the difference between "the board was not there" and "we
cut it off". Every case below is a way that could happen on real footage.

    python pitch_crop_selftest.py
"""
import sys

import cv2
import numpy as np

from pipeline.scan import GRASS_MARGIN_FRAC, pitch_crop_row

H, W = 1080, 1920
# HSV grass green, inside board_regions' deliberately narrow hue window.
GRASS = cv2.cvtColor(np.uint8([[[45, 180, 120]]]), cv2.COLOR_HSV2BGR)[0, 0]
# LED cyan-green, OUTSIDE it -- the whole reason that window is narrow.
LED_GREEN = cv2.cvtColor(np.uint8([[[80, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
CROWD = (90, 70, 60)
BOARD = (160, 90, 40)


def frame(bands):
    """bands = [(y0, y1, colour)] painted onto a crowd-coloured frame."""
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :] = CROWD
    for y0, y1, c in bands:
        f[y0:y1, :] = c
    return f


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def main():
    ok = True
    print("pitch_crop_row")
    margin = int(GRASS_MARGIN_FRAC * H)

    # 1. The ordinary case: crowd, a board strip, then pitch to the bottom.
    #    The cut must sit BELOW the board, never through it.
    f = frame([(0, 300, CROWD), (300, 360, BOARD), (360, H, GRASS)])
    cut = pitch_crop_row(f)
    ok &= check("crops a normal frame", cut is not None)
    if cut is not None:
        ok &= check("keeps the whole board", cut >= 360, f"cut={cut}, board ends 360")
        ok &= check("still saves real area", cut < H * 0.75, f"cut={cut}")
        ok &= check("cut is boundary + margin", abs(cut - (360 + margin)) <= 2,
                    f"cut={cut}, expected ~{360 + margin}")

    # 2. All grass (a pitch close-up): croppable to almost nothing, and it must
    #    not return something past the frame.
    cut = pitch_crop_row(frame([(0, H, GRASS)]))
    ok &= check("a pure-grass frame crops hard", cut is not None and cut <= margin + 2,
                f"cut={cut}")

    # 3. No grass at all (crowd, roof, an interview): nothing to crop, and the
    #    answer must be None rather than 0 -- 0 would read as "crop everything".
    ok &= check("no pitch -> no crop", pitch_crop_row(frame([(0, H, CROWD)])) is None)

    # 4. THE ONE THAT MATTERS. A green LED creative across the middle of a crowd
    #    shot must not be mistaken for the pitch. Continuity is the defence: the
    #    rows below the band are crowd, not grass. Note the band is LED green,
    #    outside the grass hue window, so this is belt and braces.
    f = frame([(0, 400, CROWD), (400, 470, LED_GREEN), (470, H, CROWD)])
    ok &= check("a green LED band mid-frame is not the pitch",
                pitch_crop_row(f) is None)

    # 5. Same, but with the band in ACTUAL grass hue -- a green advert that
    #    happens to match the turf. Continuity must still save us.
    f = frame([(0, 400, CROWD), (400, 470, GRASS), (470, H, CROWD)])
    ok &= check("even a grass-coloured band mid-frame is not the pitch",
                pitch_crop_row(f) is None)

    # 6. Grass at the bottom but only a thin sliver: the saving is under the
    #    floor, so it must decline rather than shave a few rows for nothing.
    f = frame([(0, H - 40, CROWD), (H - 40, H, GRASS)])
    ok &= check("declines when there is little to gain",
                pitch_crop_row(f) is None)

    # 7. The cut never exceeds the frame, whatever the margin.
    cut = pitch_crop_row(frame([(0, H, GRASS)]), margin_frac=5.0)
    ok &= check("cut is clamped to the frame", cut is None or cut <= H, f"cut={cut}")

    print("\n" + ("all checks passed" if ok else "FAILURES -- see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
