"""
Lock the surface-merge arithmetic. No video, no runs.

The one thing this stage must never do is double-count. A brand on the
pitch-level LED and on the board line above it in the same second is one second
of exposure; summing the per-surface tables would inflate exactly the brands
that appear most often, which is the worst place to be wrong. And because
`detect` samples at 4/s while `scan` samples at 1/s, "the same frame" is not a
question that can be asked across surfaces -- only overlapping time is.

    python merge_surfaces_selftest.py
"""
import sys

from pipeline.merge_surfaces import _union_seconds as union

CASES = [
    ([], 0.0, "no observations"),
    ([(0, 1)], 1.0, "one sample"),
    ([(0, 1), (1, 2)], 2.0, "abutting samples add"),
    ([(0, 1), (0.5, 1.5)], 1.5, "partial overlap counts once"),
    ([(0, 1), (0, 1), (0, 1)], 1.0, "the same second on three surfaces is one second"),
    ([(0, 1), (2, 3)], 2.0, "a gap is not filled in"),
    ([(0, 4), (1, 2)], 4.0, "a contained span adds nothing"),
    ([(2, 3), (0, 1), (0.5, 2.5)], 3.0, "unsorted spans that chain together"),
    # The mixed-cadence case the interval refactor exists for.
    ([(10, 11)] + [(10 + i * 0.25, 10.25 + i * 0.25) for i in range(4)], 1.0,
     "one 1s scan sample + four 0.25s detect samples inside it"),
    ([(0, 0.25), (0.5, 0.75)], 0.5, "two detect samples with a gap between"),
]


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    print("merge_surfaces -- union of exposure spans")
    ok = True
    for spans, want, why in CASES:
        got = union(spans)
        good = abs(got - want) < 1e-9
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'}  {why:<52} "
              f"{got:.3f}s (want {want}s)")
    print("\n" + ("all checks passed" if ok else "FAILURES -- see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
