"""
Synthetic checks for the piecewise touchline fit (fit_touchline_pw) and the
scalar round-trip (rect_from_scalars). No video, no ground truth needed -- just
the geometry maths.

    python board_regions_selftest.py
"""
import numpy as np

from brand_reader.board_regions import fit_touchline_pw, rect_from_scalars

RS = np.random.RandomState(0)
N = 1900
XS = np.arange(N)


def _noise(scale=1.6):
    return RS.randn(N) * scale


def _slope_deg(s):
    return abs(np.degrees(np.arctan(s)))


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("straight boundary -> no knot")
def _():
    yb = 0.02 * XS + 300 + _noise()
    _, _, seg, kn = fit_touchline_pw(yb)
    assert len(kn) == 0 and len(seg) == 1, (kn, seg)


@case("straight + a player blob on the line -> no knot")
def _():
    yb = 0.02 * XS + 300 + _noise()
    yb[900:965] -= 22
    _, _, _, kn = fit_touchline_pw(yb)
    assert len(kn) == 0, kn


@case("straight + noisy right edge -> no knot")
def _():
    yb = -0.015 * XS + 500 + _noise()
    yb[1780:1870] -= 16
    _, _, _, kn = fit_touchline_pw(yb)
    assert len(kn) == 0, kn


@case("gentle smooth curve (not a corner) -> no knot")
def _():
    yb = 300 + 3e-5 * (XS - 950) ** 2 + _noise()
    _, _, _, kn = fit_touchline_pw(yb)
    assert len(kn) == 0, kn


@case("right corner: knot within 60px, steeper 2nd segment")
def _():
    xk_true = 1550
    yb = np.where(XS < xk_true, 0.01 * XS + 300,
                  0.01 * xk_true + 300 - 0.16 * (XS - xk_true)) + _noise()
    _, _, seg, kn = fit_touchline_pw(yb)
    assert len(kn) == 1, "no knot found"
    assert abs(kn[0][0] - xk_true) <= 60, kn
    assert _slope_deg(seg[1]["slope"]) > _slope_deg(seg[0]["slope"]) + 3.0, seg


@case("left corner: knot within 60px")
def _():
    xk_true = 380
    yb = np.where(XS < xk_true, 300 - 0.15 * (xk_true - XS),
                  0.012 * (XS - xk_true) + 300) + _noise()
    _, _, seg, kn = fit_touchline_pw(yb)
    assert len(kn) == 1 and abs(kn[0][0] - xk_true) <= 60, kn


@case("mild edge corner (~10px/100px) -> knot found")
def _():
    xk_true = 1670
    yb = np.where(XS < xk_true, 0.008 * XS + 300,
                  0.008 * xk_true + 300 - 0.10 * (XS - xk_true)) + _noise()
    _, _, _, kn = fit_touchline_pw(yb)
    assert len(kn) == 1, "mild corner missed"


@case("piecewise fit is continuous at the knot")
def _():
    yb = np.where(XS < 1400, 0.012 * XS + 250,
                  0.012 * 1400 + 250 - 0.13 * (XS - 1400)) + _noise()
    fit, _, seg, kn = fit_touchline_pw(yb)
    xk = kn[0][0]
    ya = seg[0]["slope"] * xk + seg[0]["intercept"]
    yb2 = seg[1]["slope"] * xk + seg[1]["intercept"]
    assert abs(ya - yb2) < 1e-6, (ya, yb2)


@case("rect_from_scalars rebuilds the piecewise boundary")
def _():
    yb = np.where(XS < 1500, 0.01 * XS + 300,
                  0.01 * 1500 + 300 - 0.15 * (XS - 1500)) + _noise()
    fit, keep, seg, kn = fit_touchline_pw(yb)
    span = (200, 1850)
    fit_med = float(np.median(fit[span[0]:span[1]]))
    r = rect_from_scalars([seg[0]["slope"], seg[0]["intercept"]], span, fit_med,
                          (0.6, 1.4), 40, 14, segments=seg, knots=kn)
    want = fit[span[0]:span[1]]
    assert np.max(np.abs(r["yb"] - want)) < 1e-3, np.max(np.abs(r["yb"] - want))


@case("rect_from_scalars falls back to coef when segments absent")
def _():
    coef = [-0.02, 480.0]
    span = (0, 1900)
    r = rect_from_scalars(coef, span, 400.0, (0.8, 1.25), 40, 14)
    want = np.polyval(coef, np.arange(*span))
    assert np.max(np.abs(r["yb"] - want)) < 1e-3


def main():
    ok = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  PASS  {name}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL  {name}\n        {e}")
    print(f"\n{ok}/{len(CASES)} passed")
    raise SystemExit(0 if ok == len(CASES) else 1)


if __name__ == "__main__":
    main()
