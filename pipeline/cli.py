"""
Command-line entry point.

    python -m pipeline.cli detect --video input_videos/ucl.mp4
    python -m pipeline.cli detect --video match.mp4 --sample-fps 1 --debug-overlay

The CLI only builds a RunConfig and calls the stage. An API will build the same
RunConfig from JSON and call the same function.
"""

import argparse
import sys

from .config import RunConfig
from .aggregate import run_aggregate
from .board_states import run_states
from .coverage import run_coverage
from .dedup import run_dedup
from .detect import run_detect
from .expose import run_expose
from .identify import run_identify
from .report import run_report
from .segment import run_segment


def _add_segment(sub):
    p = sub.add_parser("segment", help="pass 0: video -> shots.jsonl + routing")
    p.add_argument("--video", required=True)
    p.add_argument("--out-root", default="output_videos")
    p.add_argument("--run-name", default=None)
    p.add_argument("--segment-fps", type=float, default=6.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--per-shot", type=int, default=2,
                   help="frames per shot offered to the human annotator")
    p.add_argument("--qa-frac", type=float, default=0.05,
                   help="share of automatic shots sampled for spot-checking")
    p.add_argument("--min-purity", type=float, default=0.60)
    p.add_argument("--probe-band", type=int, default=5,
                   help="full-res frames per candidate shot used to test the band")
    p.add_argument("--band-conf", type=float, default=0.55,
                   help="band confidence below which a wide shot goes to review")
    p.add_argument("--max-bad-frac", type=float, default=0.25,
                   help="share of probed frames allowed to read badly before review")
    p.add_argument("--seed", type=int, default=0)
    return p


def _add_identify(sub):
    p = sub.add_parser("identify", help="read the grouped strips -> brand names")
    p.add_argument("run_dir")
    p.add_argument("--sponsors", default=None,
                   help="sponsor list JSON (default: <run>/sponsors.json)")
    p.add_argument("--min-ocr-conf", type=float, default=0.30)
    p.add_argument("--ocr-every", type=int, default=1,
                   help="OCR only every Nth frame per group (default 1 = every "
                        "frame). Use 3-4 for a full match; keep 1 for score.py runs")
    p.add_argument("--min-match", type=float, default=0.62,
                   help="fuzzy score at which OCR text is accepted as a sponsor")
    p.add_argument("--langs", default="en")
    p.add_argument("--min-unknown-seconds", type=float, default=1.0,
                   help="board time a readable-but-unlisted string must hold "
                        "before it is reported as an advertiser nobody listed "
                        "(0 disables the discovery pass)")
    p.add_argument("--carry-states", action="store_true",
                   help="carry a brand across the whole board state it was read on "
                        "(needs `states`; EXPERIMENTAL -- overcredits when states "
                        "are under-segmented, see CLAUDE.md ACCURACY)")
    p.add_argument("--readings", default=None,
                   help="JSON of brand readings made outside the API (a person, or "
                        "a Claude session looking at the strips) -- the pilot path")
    p.add_argument("--llm", action="store_true",
                   help="send what OCR could not read to the vision model (costs money)")
    p.add_argument("--llm-model", default="claude-opus-5")
    p.add_argument("--llm-max", type=int, default=25, help="cap the number of calls")
    p.add_argument("--llm-dry-run", action="store_true",
                   help="show how many calls and what they would cost, then stop")
    p.add_argument("--llm-min-seconds", type=float, default=0.4,
                   help="do not spend a call on a fragment shorter than this")
    return p


def _add_expose(sub):
    p = sub.add_parser("expose", help="track annotations outward -> exposure.jsonl")
    p.add_argument("run_dir")
    p.add_argument("--video", default=None, help="override the video from the manifest")
    p.add_argument("--sample-fps", type=float, default=4.0,
                   help="exposure resolution; 4/s is 0.25s, plenty for a report")
    p.add_argument("--min-conf", type=float, default=0.35,
                   help="stop a track once confidence falls below this")
    p.add_argument("--min-auto-conf", type=float, default=0.55,
                   help="lowest band confidence to accept from detect")
    return p


def _add_aggregate(sub):
    p = sub.add_parser("aggregate", help="identified strips -> per-brand CSV")
    p.add_argument("run_dir")
    p.add_argument("--video", default=None, help="override the video from the manifest")
    p.add_argument("--min-seconds", type=float, default=0.0,
                   help="drop brands seen for less than this")
    p.add_argument("--no-brightness", action="store_true",
                   help="skip the video pass that measures brightness")
    return p


def _add_dedup(sub):
    p = sub.add_parser("dedup", help="group band strips that show the same view")
    p.add_argument("run_dir")
    p.add_argument("--min-conf", type=float, default=0.55,
                   help="lowest band confidence worth grouping")
    p.add_argument("--thr", type=float, default=0.30,
                   help="distance below which two strips are the same view")
    return p


def _add_states(sub):
    p = sub.add_parser("states",
                       help="cut each view's timeline into held LED board states")
    p.add_argument("run_dir")
    p.add_argument("--change-thr", type=float, default=None,
                   help="strip-to-strip distance that means the creative changed "
                        "(default: self-calibrated from the run)")
    return p


def _add_coverage(sub):
    p = sub.add_parser("coverage", help="which seconds of the match were measured, and why not")
    p.add_argument("run_dir")
    p.add_argument("--obs-window", type=float, default=0.5,
                   help="seconds a single observation is assumed to prove")
    p.add_argument("--auto-conf", type=float, default=0.55)
    p.add_argument("--top", type=int, default=12, help="largest gaps to print")
    return p


def _add_report(sub):
    p = sub.add_parser("report", help="per-partner client report (HTML + JSON)")
    p.add_argument("run_dir")
    p.add_argument("--config", default=None,
                   help="report_config.json (default: <run>/report_config.json, "
                        "then pilot/report_config.json)")
    p.add_argument("--out", default=None, help="output HTML path")
    p.add_argument("--surface-run", action="append", default=[], metavar="SURF=RUN_DIR",
                   help="feed one surface from a different run's aggregate output, "
                        "e.g. --surface-run second_line=output_videos/pilot_run5_L2 "
                        "(repeatable)")
    return p


def _add_scan(sub):
    p = sub.add_parser("scan", help="geometry-free: read advertiser names "
                                    "anywhere in the frame")
    p.add_argument("run_dir")
    p.add_argument("--video", default=None, help="override the video from the manifest")
    p.add_argument("--sponsors", default=None,
                   help="sponsor list JSON (default: <run>/sponsors.json)")
    p.add_argument("--scan-fps", type=float, default=1.0,
                   help="frames per second to read; each sample stands for 1/fps")
    p.add_argument("--langs", default="en")
    p.add_argument("--min-ocr-conf", type=float, default=0.30)
    p.add_argument("--min-match", type=float, default=0.62,
                   help="fuzzy score at which OCR text is accepted as a sponsor")
    p.add_argument("--min-frames", type=int, default=2,
                   help="distinct frames a brand must be read on to be reported")
    p.add_argument("--upscale", type=float, default=1.0,
                   help="magnify before OCR; helps small far-side board text")
    p.add_argument("--no-pitch-crop", action="store_true",
                   help="do NOT crop the pitch away before OCR. The crop is the "
                        "biggest lever on runtime (grass cannot hold a board) "
                        "and defaults on")
    p.add_argument("--sensitive", action="store_true",
                   help="make EasyOCR's text DETECTOR propose more boxes. "
                        "UNMEASURED -- score it before trusting it")
    p.add_argument("--clahe", action="store_true",
                   help="normalise local contrast before OCR. Targets text that "
                        "is large enough but low-contrast (white on light blue), "
                        "which upscaling does not help")
    p.add_argument("--no-mask-overlays", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def _add_scan_llm(sub):
    p = sub.add_parser("scan-llm", help="geometry-free: a vision model reads "
                                        "whole frames (costs money)")
    p.add_argument("run_dir")
    p.add_argument("--video", default=None)
    p.add_argument("--sponsors", default=None)
    p.add_argument("--scan-fps", type=float, default=1.0)
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="reading a wordmark is recognition, not reasoning; "
                        "Haiku is 3x cheaper and this task repeats thousands "
                        "of times per match")
    p.add_argument("--max-calls", type=int, default=None, help="cap the spend")
    p.add_argument("--group-dist", type=float, default=0.06,
                   help="appearance distance below which consecutive samples "
                        "share one call")
    p.add_argument("--dry-run", action="store_true",
                   help="show the call count and estimated cost, then stop. "
                        "Needs no SDK and no credentials")
    p.add_argument("--export-frames", default=None, metavar="DIR",
                   help="write the grouped frames out as montages instead of "
                        "sending them, for a person or a Claude session to read. "
                        "The free pilot path; sends nothing")
    p.add_argument("--readings", default=None,
                   help="JSON of {\"<group id>\": [\"BRAND\", ...]} made outside "
                        "the API. Produces the same records as the API path")
    p.add_argument("--no-mask-overlays", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def _add_detect(sub):
    p = sub.add_parser("detect", help="pass 1: video -> bands + detections.jsonl")
    p.add_argument("--video", required=True)
    p.add_argument("--out-root", default="output_videos")
    p.add_argument("--run-name", default=None)
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--n-rows", type=int, default=3)
    p.add_argument("--upscale", type=float, default=2.0)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--debug-overlay", action="store_true",
                   help="also write the original frame with the region outlined")
    p.add_argument("--shot-types", default="wide_play,medium",
                   help="shot kinds to read; 'all' reads every shot")
    p.add_argument("--band-line", type=int, default=1,
                   help="which touchline advertising line to read: 1 = pitch-level "
                        "LED (default), 2 = the line stacked above it")
    p.add_argument("--no-mask-overlays", action="store_true",
                   help="do NOT blank the broadcaster's burned-in graphics. They "
                        "carry brands (watermark, score bug) that are on screen "
                        "the whole match and are not the club's inventory, so "
                        "leaving them in inflates those brands")
    p.add_argument("--overlay-probe-frames", type=int, default=96,
                   help="frames sampled across the video to locate the overlays")
    return p


def main(argv=None):
    # Brand names can be non-Latin, and a
    # Windows console defaults to cp1252, which raises rather than printing
    # them. Losing a run at the final print because a sponsor is spelled in a
    # non-Latin alphabet is not acceptable.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(prog="pipeline")
    sub = ap.add_subparsers(dest="stage", required=True)
    _add_segment(sub)
    _add_detect(sub)
    _add_scan(sub)
    _add_scan_llm(sub)
    _add_dedup(sub)
    _add_states(sub)
    _add_identify(sub)
    _add_aggregate(sub)
    _add_expose(sub)
    _add_coverage(sub)
    _add_report(sub)
    args = ap.parse_args(argv)

    if args.stage == "segment":
        cfg = RunConfig(
            video=args.video,
            out_root=args.out_root,
            run_name=args.run_name,
            segment_fps=args.segment_fps,
            max_frames=args.max_frames,
            per_shot=args.per_shot,
            qa_frac=args.qa_frac,
            min_purity=args.min_purity,
            probe_band=args.probe_band,
            band_conf_thr=args.band_conf,
            max_bad_frac=args.max_bad_frac,
            seed=args.seed,
        )
        run_segment(cfg)

    elif args.stage == "dedup":
        run_dedup(args.run_dir, min_conf=args.min_conf, thr=args.thr)

    elif args.stage == "states":
        run_states(args.run_dir, change_thr=args.change_thr)

    elif args.stage == "identify":
        run_identify(args.run_dir, sponsors_path=args.sponsors,
                     min_ocr_conf=args.min_ocr_conf, min_match=args.min_match,
                     langs=tuple(args.langs.split(",")),
                     readings_path=args.readings, use_llm=args.llm,
                     llm_model=args.llm_model, llm_max=args.llm_max,
                     llm_min_seconds=args.llm_min_seconds,
                     llm_dry_run=args.llm_dry_run,
                     min_unknown_seconds=args.min_unknown_seconds,
                     carry_states=args.carry_states, ocr_every=args.ocr_every)

    elif args.stage == "aggregate":
        run_aggregate(args.run_dir, video=args.video,
                      min_seconds=args.min_seconds,
                      brightness=not args.no_brightness)

    elif args.stage == "expose":
        run_expose(args.run_dir, video=args.video,
                   sample_fps=args.sample_fps, min_conf=args.min_conf,
                   min_auto_conf=args.min_auto_conf)

    elif args.stage == "coverage":
        run_coverage(args.run_dir, obs_window=args.obs_window,
                     auto_conf=args.auto_conf, top=args.top)

    elif args.stage == "report":
        surface_runs = {}
        for spec in args.surface_run:
            if "=" not in spec:
                ap.error(f"--surface-run expects SURF=RUN_DIR, got {spec!r}")
            k, v = spec.split("=", 1)
            surface_runs[k.strip()] = v.strip()
        run_report(args.run_dir, config_path=args.config, out_path=args.out,
                   surface_runs=surface_runs)

    elif args.stage == "scan":
        from .scan import run_scan
        run_scan(args.run_dir, video=args.video, sponsors_path=args.sponsors,
                 scan_fps=args.scan_fps, langs=args.langs,
                 min_ocr_conf=args.min_ocr_conf, min_match=args.min_match,
                 min_frames=args.min_frames, upscale=args.upscale,
                 clahe=args.clahe, pitch_crop=not args.no_pitch_crop,
                 sensitive=args.sensitive,
                 mask_overlays=not args.no_mask_overlays,
                 max_frames=args.max_frames)

    elif args.stage == "scan-llm":
        from .scan_llm import run_scan_llm
        run_scan_llm(args.run_dir, video=args.video, sponsors_path=args.sponsors,
                     scan_fps=args.scan_fps, model=args.model,
                     max_calls=args.max_calls, dry_run=args.dry_run,
                     group_dist=args.group_dist,
                     mask_overlays=not args.no_mask_overlays,
                     max_frames=args.max_frames,
                     export_frames_dir=args.export_frames,
                     readings=args.readings)

    elif args.stage == "detect":
        cfg = RunConfig(
            video=args.video,
            out_root=args.out_root,
            run_name=args.run_name,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            n_rows=args.n_rows,
            upscale=args.upscale,
            jpeg_quality=args.jpeg_quality,
            debug_overlay=args.debug_overlay,
            detect_shot_types=() if args.shot_types == "all" else tuple(args.shot_types.split(",")),
            band_kwargs={"band_line": args.band_line} if args.band_line != 1 else {},
            mask_overlays=not args.no_mask_overlays,
            overlay_probe_frames=args.overlay_probe_frames,
        )
        run_detect(cfg)


if __name__ == "__main__":
    main()
