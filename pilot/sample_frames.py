"""
Dump evenly-spaced full frames from a match video for the by-eye manual tally
of the surfaces the pipeline does not measure (jersey, second advertising line,
corner banners, 3D carpets, near-touchline LED).

    python pilot/sample_frames.py input_videos/MATCH.mp4 --every 4 --out output_videos/<run>/manual_frames

Each frame is written as `t<seconds>.jpg` (zero-padded, one decimal) so the
reviewer can read a timestamp straight off the filename. To turn a count into
seconds for `manual_tally`: seconds = (frames where the logo is present and
readable) x --every.

Deliberately dumb: no detection, no cropping. The reviewer looks at each frame
and tallies per surface. `--scale` shrinks the JPEGs so a few hundred fit in a
folder view without filling the disk.
"""

import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=4.0,
                    help="seconds between sampled frames (default 4)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <video dir>/manual_frames)")
    ap.add_argument("--scale", type=float, default=0.6,
                    help="resize factor for the written JPEGs (default 0.6)")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / fps if total else 0.0
    out = args.out or os.path.join(os.path.dirname(args.video) or ".", "manual_frames")
    os.makedirs(out, exist_ok=True)

    end = args.end if args.end is not None else dur
    step = max(1, int(round(args.every * fps)))
    print(f"{args.video}: {fps:.1f} fps, {dur:.0f}s -> every {args.every}s "
          f"({step} frames), {out}")

    i, written = 0, 0
    want_next = int(args.start * fps)
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = i / fps
        if i >= want_next and t <= end:
            ok, frame = cap.retrieve()
            if ok:
                if args.scale != 1.0:
                    frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale,
                                       interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(out, f"t{t:07.1f}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                written += 1
                if written % 100 == 0:
                    print(f"  {written} frames, t={t:.0f}s")
            want_next += step
        i += 1
    cap.release()
    print(f"done: {written} frames in {out}  (each frame = {args.every}s of match)")


if __name__ == "__main__":
    main()
