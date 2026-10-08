# Manual review — how to annotate

    python -m pipeline.cli segment --video input_videos/ucl1.mp4 --run-name seg_ucl1
    python annotate.py output_videos/seg_ucl1

## What you are being asked for

Only two things: **which brand**, and **where in the frame**.

Duration comes from the shot boundary. Size and position come from the shape you
draw. Never type either.

You only see shots the machine could not read. Clean main-camera play is handled
automatically and never reaches you.

## The loop, per frame

1. **Outline the logo**
   - **drag** → rectangle. Right for jerseys.
   - **`w`** → 4-corner mode. Click each corner in order. Use this for perimeter
     boards seen at an angle: a rectangle around a slanted board can overstate
     its area by 80%+, which would inflate that sponsor in the report.
     `u` removes the last corner, `w` or `ESC` goes back to rectangle.
2. **Name the brand** — just start typing. The list filters as you go
   (`ju` → JUST EAT, `qat` → QATAR AIRWAYS). `TAB` cycles, `ENTER` picks.
   The last entry is always `+ create "…"`, so a new brand is the same gesture.
   Never invent a second spelling of an existing sponsor — pick the match.
3. **Pick the surface** — one letter:

   | | | | |
   |---|---|---|---|
   | `l` perimeter_led | `r` ribbon | `j` jersey_front | `k` sleeve |
   | `h` shorts | `m` manufacturer | `p` backdrop | `e` bench_board |
   | `g` graphic | `t` bottle | `o` other | `ENTER` = default for this shot type |

4. **`SPACE`** → save and move on.

## Other keys

| key | |
|---|---|
| `x` | **nothing visible here.** Use it. An explicit empty is data; a skip is a hole. |
| `c` | copy the previous frame's boxes. Blocked after a drastic cut. |
| `u` / `d` | undo last box / clear this frame |
| `b` | back one frame |
| `s` | skip the rest of this shot |
| `q` | save and quit — safe at any moment, restarting resumes |

## What the header tells you

    [4/14] shot 6  close  19.8-20.2s (0.3s)  REVIEW
    frame 1/2   cut severity 2.03   DRASTIC CUT - annotate from scratch

- **REVIEW** — the machine could not read this shot at all.
- **PARTIAL — bad frames only** — the shot is mostly fine automatically; you are
  seeing just the frames that failed.
- **QA spot-check** — the machine *did* read this one. You are checking whether
  it was right. Annotate what you actually see.
- **cut severity** — how different this shot is from the previous one.
  `DRASTIC` (≥ 2.0) means it has nothing in common with the last shot — two
  replays from different angles land here — so `c` is disabled and you start
  clean. Below that, `c` is a safe shortcut.

## Two rules that matter downstream

- **One sponsor, one name.** `QATAR` and `QATAR AIRWAYS` become two rows in the
  report and split that sponsor's exposure in half. Always take the suggestion.
- **`x` is not a skip.** Marking a frame empty tells us the close-up track found
  nothing there, which is how we learn whether reviewing these shots is worth
  it. Skipping tells us nothing.

## Output

`output_videos/<run>/annotations.jsonl`, one record per frame, appended live.
Each box carries `brand`, `surface`, the 4-point `poly`, its `bbox`, and the
true `area_px`. This is both the exposure data and the ground truth for tuning
the automatic path.
