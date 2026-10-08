# Sponsor-exposure analytics for football broadcasts

Given a match video, report **how much on-screen exposure each sponsor got**:
seconds on screen, share of the broadcast, and (where measurable) size. The
aim is an independent, auditable measurement that European clubs can hand to
their commercial partners.

Developed in collaboration with a professional second-tier football club,
whose broadcast and advertiser list were used to build and validate it.

It started as a school prototype that produced an annotated demo video. It is
now a measurement pipeline whose output is a per-advertiser table, and whose
accuracy is scored against a hand count rather than eyeballed.

## Why this is hard

Perimeter LED boards rotate creatives every few seconds, the camera cuts
between very different framings, boards are seen at steep angles, and
broadcasters overlay their own graphics that look exactly like sponsors but are
not inventory the club sells. Plain OCR over every frame gets you a number, but
the wrong one.

## Pipeline

Each stage is a pure function over a run directory, so it is callable from the
CLI today and from an API later.

![From a broadcast frame to a sponsor reading: pitch edge and board band found, band straightened into a strip, folded into rows, matched to the advertiser list](docs/pipeline.png)

*One sampled frame through the board path. Frames are from a match broadcast clip and are shown to illustrate the method.*

| Stage | What it does |
|---|---|
| `segment` | Cuts the match into shots (histogram, thumbnail and Lab colour-layout signals) and routes each to automatic or human review |
| `detect` | Finds the far-touchline advertising band per frame and rectifies it into a straight strip |
| `dedup` | Groups strips that show the same view, so the reader runs once per view, not per frame |
| `identify` | OCR plus fuzzy matching against the club's advertiser list, with a vision-LLM fallback |
| `scan` | Geometry-free: reads brands anywhere in the frame, for framings the touchline model cannot fit |
| `merge_surfaces` | Unions several surfaces by time, so a brand on two boards in one frame counts once |
| `coverage` | A gap-free partition of the match into measured / not measured, naming the stage that owes each gap |
| `aggregate` / `report` | Per-brand time, surface and brightness, rendered as a client report |

## Ideas worth a look

![The detected band follows the advertising board across three moments of a 17-second clip, including a corner view](docs/tracking.png)

- **Board finder (`brand_reader/board_regions.py`).** Zero-shot YOLO-World
  failed on perimeter boards, so the finder follows the pitch boundary: first
  row with a deep run of grass below it, a robust piecewise-linear fit (one
  knot, accepted only when a tail of the boundary sits systematically off the
  straight fit), a texture break to find the board height, then a perspective
  un-warp. The fit maths is locked by `board_regions_selftest.py`.
- **Burned-in graphics are measured, not hardcoded (`overlay_mask.py`).** A
  watermark is the only thing that stays pixel-identical while the camera cuts
  between scenes. Found from the video's own statistics, and it refuses to run
  rather than guess when the evidence is thin.
- **The coverage ledger.** The original failure was not a wrong number, it was
  that nothing knew those seconds existed. Spans must tile the whole match with
  no gaps or overlaps.
- **Honest scoring (`score.py`).** Separates "a sponsor credited zero seconds"
  (loses the contract) from "a sponsor with the wrong number" (accuracy).
  Each metric can move against the other, so per-brand rows are read, not just
  headlines.

## Measured results

On a hand-counted 5-minute window of a real match (12 advertisers):

| method | captured | mean abs. error |
|---|---|---|
| touchline `detect`, line 1 only | 10.9% | 45.3 s |
| + second line, graphics masked | 18.4% | 41.5 s |
| `scan` (geometry-free OCR) | 51.3% | 24.8 s |
| all OCR surfaces unioned | 56.1% | 22.3 s |
| vision-model reading | 95.9% | 4.3 s |

The last row is **partly circular**: the same session built the hand count and
did the reading, so it shows the surrounding machinery is sound, not that a
vision model beats OCR in general. An independent count from the club was
collected afterwards to close that loop.

What the OCR rows reveal is a **legibility bias**: time is credited in
proportion to how readable a wordmark is, not whether the brand is present.
Plain horizontal wordmarks land; stylised marks fail, and no amount of
aliasing fixes that without overfitting to one clip. That is the argument for
the vision path.

## What is tuned to one clip

Cut-detection thresholds, the `band_confidence` constants and the grass hue
range were tuned on the footage I had. They would need re-measuring on another
broadcaster. The overlay mask, the matcher's containment rule and the union
arithmetic are not tuned to a clip.

## Layout

```
brand_reader/   board finder, overlay mask, shot typing, tracking, OCR matching
pipeline/       the stages above, one module each
run_clip.py     video in, scored per-advertiser table out
score.py        compare a run against a hand count
*_selftest.py   synthetic-data checks (no video needed)
```

## Running the self-tests

The selftests use synthetic data and need no footage:

```
python board_regions_selftest.py
python overlay_mask_selftest.py
python aggregate_selftest.py
python pitch_crop_selftest.py
python merge_surfaces_selftest.py
```

Broadcast footage, the club's advertiser list and the hand count are not
included in this repository.
