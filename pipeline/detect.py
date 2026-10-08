"""
PASS 1 -- detection: video -> rectified advertising bands + per-frame records.

No network calls here. We walk the video, sample frames, and for each sampled
frame extract the far-touchline advertising band (rectified then reflowed), write
it as a JPEG, and append one JSON line describing its geometry.

Later stages (dedup, LLM identification, aggregation) read only these artifacts;
they never decode the video again.

Geometry is stored as SCALARS (touchline fit coefficients, span, band height)
rather than per-column arrays -- a few hundred bytes per frame instead of several
kilobytes, so records can travel through an API as-is.
"""

import json
import time

import cv2

from brand_reader.board_regions import band_image, band_polygon
from brand_reader.overlay_mask import apply_mask, boxes_from_mask, overlay_mask
from .store import RunStore


def _shot_index(store):
    """frame -> shot record, so every detection knows where it belongs."""
    try:
        return store.read_shots()
    except Exception:
        return []


def _shot_at(shots, frame_idx):
    for s in shots:
        if s["f_start"] <= frame_idx <= s["f_end"]:
            return s
    return None


def _video_info(cap, path):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return {
        "path": path,
        "fps": float(fps),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def _record(frame_idx, t, rect, layout, relpath, img_shape):
    return {
        "frame": frame_idx,
        "t": round(t, 3),
        "band": relpath,
        "band_size": [int(img_shape[1]), int(img_shape[0])],
        "H": round(float(rect["H"]), 2),
        "over": int(rect["over"]),
        "band_line": int(rect.get("band_line", 1)),
        "off_shift": int(rect.get("off_shift", 0)),
        "span": list(rect["span"]),
        "coef": [round(c, 6) for c in rect["coef"]],
        # piecewise touchline: 1 entry for a straight boundary, 2 across a corner
        "segments": [{"slope": round(float(s["slope"]), 8),
                      "intercept": round(float(s["intercept"]), 3),
                      "x_lo": int(s["x_lo"]), "x_hi": int(s["x_hi"])}
                     for s in rect.get("segments", [])],
        "knots": [[int(x), round(float(y), 2)] for x, y in rect.get("knots", [])],
        "fit_med": round(float(rect["fit_med"]), 3),
        "local_clamp": list(rect["local_clamp"]),
        # How much to trust this band. Low -> the frame is not a clean hard-cam
        # perimeter read (celebration, receding board, occluded span); a later
        # stage should send the whole frame to the LLM instead.
        "conf": round(float(rect["confidence"]), 3),
        "conf_parts": rect["conf_parts"],
        "strip": [int(layout["strip_shape"][0]), int(layout["strip_shape"][1])],
        "upscale": layout["upscale"],
        "rows": [{"y0": r["y0"], "h": r["h"], "w": r["w"], "x_off": r["x_off"]}
                 for r in layout["rows"]],
    }


def _write_overlay(store, frame, frame_idx, t, rect, quality):
    """Original frame with the extracted region outlined -- visual check only."""
    vis = frame.copy()
    poly = band_polygon(rect)
    conf = float(rect["confidence"])
    col = (0, 220, 160) if conf >= 0.5 else (0, 170, 255)
    cv2.polylines(vis, [poly], True, col, 2, cv2.LINE_AA)
    cv2.putText(vis, f"t={t:6.1f}s  frame {frame_idx}  conf={conf:.2f}  "
                     f"H={rect['H']:.0f}px  span={rect['span'][0]}-{rect['span'][1]}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    cv2.imwrite(store.overlay_path(frame_idx), vis,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def _write_noband_overlay(store, frame, frame_idx, t, why, quality):
    """A sampled frame where the band finder returned nothing -- still write it,
    so `--debug-overlay` shows every sampled second, not only the hits."""
    vis = frame.copy()
    cv2.putText(vis, f"t={t:6.1f}s  frame {frame_idx}  NO BAND  ({why})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2, cv2.LINE_AA)
    cv2.imwrite(store.overlay_path(frame_idx), vis,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def run_detect(cfg, progress_every=50):
    """Run pass 1. Returns (store, stats)."""
    cap = cv2.VideoCapture(cfg.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {cfg.video}")

    info = _video_info(cap, cfg.video)
    step = max(1, int(round(info["fps"] / max(cfg.sample_fps, 0.01))))

    store = RunStore(cfg.out_root, cfg.run_name)
    shots = _shot_index(store)
    # Run on every shot that could plausibly show a far-touchline board, whatever
    # segment routed the SHOT to. Routing per-shot threw away the readable frames
    # of any shot that was clean for part of its length -- ~36s on the clip_b clip.
    # Each frame now stands on its own band confidence.
    types = set(cfg.detect_shot_types or ())
    if shots:
        keep = sum(1 for x in shots if not types or x["type"] in types)
        print(f"Shots    : {len(shots)} from segment, "
              f"reading {keep} of type {sorted(types) or 'any'}")
    print(f"Run      : {store.dir}")
    print(f"Video    : {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
          f"{info['frames']} frames")
    print(f"Sampling : every {step} frames (~{info['fps'] / step:.1f} img/s)")
    if cfg.debug_overlay:
        print("Overlay  : on")

    # Find the broadcaster's burned-in graphics once, up front, and blank them
    # in every frame before the band finder or OCR sees it. Without this the
    # watermark's brand is credited exposure for the whole match.
    gmask, gmask_info = None, {"ok": False, "reason": "disabled"}
    if getattr(cfg, "mask_overlays", True):
        gmask, gmask_info = overlay_mask(cfg.video, n=cfg.overlay_probe_frames,
                                         max_frames=None)
        if gmask_info["ok"]:
            boxes = boxes_from_mask(gmask)
            print(f"Graphics : {len(boxes)} burned-in overlay(s) masked "
                  f"({100 * gmask_info['overlay_frac']:.1f}% of frame): "
                  + ", ".join(f"{w}x{h}@{x},{y}" for x, y, w, h in boxes[:4]))
        else:
            gmask = None
            print(f"Graphics : not masked -- {gmask_info['reason']}")

    t_start = time.perf_counter()
    n_seen = n_band = n_band_low = 0
    frame_idx = 0
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality]

    with store.open_detections() as out:
        while True:
            if not cap.grab():
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok:
                break
            if gmask is not None:
                frame = apply_mask(frame, gmask)

            shot = _shot_at(shots, frame_idx) if shots else None
            # Skip only shot kinds that cannot have a touchline board at all
            # (close-ups, crowd, graphics). A wide or medium shot is worth a
            # look even if segment sent it to a human.
            if shot is not None and types and shot["type"] not in types:
                frame_idx += 1
                continue

            n_seen += 1
            t = frame_idx / info["fps"]
            res = band_image(frame, n_rows=cfg.n_rows, upscale=cfg.upscale,
                             **cfg.band_kwargs)

            if res is None:
                # No usable touchline: close-up, crowd shot, replay, graphics...
                # Recorded rather than skipped so coverage stays measurable.
                rec = {"frame": frame_idx, "t": round(t, 3), "band": None}
                if shot is not None:
                    rec["shot"], rec["route"] = shot["shot"], shot["route"]
                if cfg.debug_overlay:
                    why = shot["type"] if shot is not None else "no shot"
                    _write_noband_overlay(store, frame, frame_idx, t, why,
                                          cfg.jpeg_quality)
            else:
                img, meta = res
                cv2.imwrite(store.band_path(frame_idx), img, enc)
                rec = _record(frame_idx, t, meta["rect"], meta["layout"],
                              store.band_relpath(frame_idx), img.shape)
                if shot is not None:
                    rec["shot"], rec["route"] = shot["shot"], shot["route"]
                if cfg.debug_overlay:
                    _write_overlay(store, frame, frame_idx, t, meta["rect"],
                                   cfg.jpeg_quality)
                n_band += 1
                if meta["rect"]["confidence"] < 0.5:
                    n_band_low += 1

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if progress_every and n_seen % progress_every == 0:
                print(f"  {n_seen} frames sampled, {n_band} bands")

            frame_idx += 1
            if cfg.max_frames and n_seen >= cfg.max_frames:
                break

    cap.release()
    elapsed = time.perf_counter() - t_start

    n_ok = n_band - n_band_low
    stats = {
        "frames_sampled": n_seen,
        "bands_found": n_band,
        "bands_low_conf": n_band_low,   # conf < 0.5: band found but not trusted
        "bands_good": n_ok,             # conf >= 0.5: measured automatically
        "coverage": round(n_band / n_seen, 4) if n_seen else 0.0,
        "step": step,
        "seconds": round(elapsed, 1),
        "fps_processed": round(n_seen / elapsed, 1) if elapsed else None,
        "overlay_mask": {
            **gmask_info,
            "boxes": boxes_from_mask(gmask) if gmask is not None else [],
        },
    }
    store.write_manifest(cfg, info, stats, stage="detect")

    good_s = n_ok * step / info["fps"]
    print(f"\nBands    : {n_band}/{n_seen} frames read; {n_ok} good (~{good_s:.0f}s "
          f"auto), {n_band_low} low-confidence (still need a human)")
    print(f"Elapsed  : {elapsed:.1f} s ({stats['fps_processed']} frames/s)")
    print(f"Artifacts: {store.detections_path}")
    return store, stats
