"""
Read advertiser names off whole frames with a vision model instead of OCR.

WHY. `scan` (EasyOCR over the whole frame) roughly tripled what the touchline
band model captured on pilot_match_5min, and what it still misses is the failure
mode CLAUDE.md has documented since the UCL clip and that no amount of aliasing
fixes: OCR credits time in proportion to how READABLE a wordmark is, not to
whether the brand is present. Plain horizontal text lands; stylised marks,
glyph logos and small blue-on-blue board text read short or not at all. On the
pilot clip that is BRAND_B (33% of its hand count) and BRAND_C (48%), against
BRAND_A at 78%.

A vision model does not have that bias. It is also, precisely, the method the
hand count was built with -- a reader looking at sampled frames and naming what
is legible -- so it is the method already known to reach the target on this
footage.

COST is not the constraint the design works around; repetition is. Consecutive
samples of a football broadcast are mostly the same picture, so frames are
grouped by appearance first and one call answers for the whole group. A 5-minute
clip at 1 sample/s collapses to well under 300 calls, and `--dry-run` prints the
count and the estimated spend without an SDK, credentials, or a single request.

    python -m pipeline.cli scan-llm output_videos/RUN --video in.mp4 --dry-run
    python -m pipeline.cli scan-llm output_videos/RUN --video in.mp4   # needs a key

Needs `pip install anthropic` and a key. Broadcast frames leave the machine on
this path -- that trade is recorded as accepted in CLAUDE.md, but it is a real
one and the dry run exists so it can be weighed before committing.

THE FREE PATH, FOR THE PILOT. `--export-frames` writes the same grouped
representatives out as montages instead of sending them, and `--readings` folds
the answers back in. Anything that can look at a picture can then do the
reading: a person, or a Claude session with the images in front of it. The
records it writes are identical to the API path's, so `score.py` cannot tell
them apart -- which makes this the honest way to find out whether the vision
approach is worth paying for BEFORE paying for it.

It does not scale: a 90-minute match is thousands of groups and somebody has to
look at all of them. It is a pilot tool and a measurement instrument, not the
product. The product is the API path above, and this is how you justify it.

    python -m pipeline.cli scan-llm output_videos/RUN --video in.mp4 \
        --export-frames output_videos/RUN/frames      # then go and read them
    python -m pipeline.cli scan-llm output_videos/RUN --video in.mp4 \
        --readings output_videos/RUN/readings.json    # fold the answers back in
"""
import base64
import csv
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from brand_reader.overlay_mask import apply_mask, boxes_from_mask, overlay_mask
from .identify import _index, _norm, load_sponsors, match_token
from .scan import pitch_crop_row

# Haiku by default. Reading a wordmark off a board is recognition, not
# reasoning, and it is the task the whole pipeline repeats thousands of times --
# exactly where model choice decides whether a match costs $3 or $25. Switch to
# claude-sonnet-5 with --model if a scored run shows Haiku dropping brands.
MODEL = "claude-haiku-4-5-20251001"
PRICES = {                             # $/1M tokens (input, output)
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
}
PRICE_IN, PRICE_OUT = PRICES[MODEL]
OUT_TOKENS = 220                       # the JSON reply is short
JPEG_Q = 80
# Frames are sent at most this wide. Board text on a 1080p broadcast survives
# 1280px comfortably and the token bill scales with area.
MAX_W = 1280
# Consecutive samples closer than this (Lab histogram + row profile, as in
# dedup) are the same picture and share one call.
#
# Chosen against measured group LENGTH, not against the call count, because the
# failure mode here is the one `--carry-states` already fell into: carry a read
# across a stretch where the LED actually changed and you credit one advertiser
# for another's seconds. Measured on pilot_match_5min at 1 sample/s:
#
#   dist    groups   median    p90    max
#   0.12       19      8.0s   32.0s  82.0s   <- one read covering 82s: unusable
#   0.06      151      1.0s    4.0s  11.0s   <- shorter than a creative hold
#   0.03      258      1.0s    1.0s   5.0s
#
# 0.06 keeps the median group at 1s, but its worst is 11s, and the claim above
# ("shorter than a creative hold") is simply false for that tail: creatives at
# this ground hold ~8-20s, so an 11s group CAN straddle a change.
#
# The cap below is reasoning, not a measured regression -- worth saying plainly,
# because it was nearly justified with a bad number. BRAND_D looked like proof
# (41s read against 23s counted) until a second continuous count found 12s more
# and made it 35s, at which point the read was within tolerance and proved
# nothing. The cap stays on its own merits: appearance distance cannot detect an
# LED swap that covers a small part of a wide frame, so the signal is weakest
# exactly when groups are longest, and a hard limit in SECONDS is the only thing
# that bounds the damage.
GROUP_DIST = 0.06
# Half a short creative hold. Creatives at this ground hold ~8-20s, so a group
# capped at 5s cannot straddle a change without at least one group boundary
# landing inside the new creative. Costs more calls; calls are cents and a
# misattributed sponsor is the report.
MAX_GROUP_S = 5.0

PROMPT = """This is one frame from a football broadcast.

List every advertiser or brand name that is LEGIBLY visible on an advertising \
surface: perimeter LED boards, the fixed board rows behind them, hoardings \
beside the pitch or dugout, interview backdrops, corner banners.

Rules:
- Report a name only if you can actually read it in this frame. Do not infer it \
from a colour, a shape, or what is usually there. An empty list is a correct \
and useful answer.
- Report the name as printed, in its own script (Cyrillic stays Cyrillic).
- The same advertiser is usually repeated along the boards. Report it ONCE.
- Do NOT report the broadcaster's own overlaid graphics -- the score clock, the \
channel watermark, lower-third name captions. Those are painted on top of the \
picture, not sold by the club. If a brand appears BOTH as an overlay and on a \
board, report it (it is on a board) and say so in `notes`.
- Do not report shirt sponsors, kit makers, stadium names on architecture, \
crowd banners, players, or pitch markings.

These advertisers are known at this ground. If what you read is one of them, \
put the exact list name in `matches_known` so spellings merge:
{known}

A brand NOT in that list is still worth reporting -- leave `matches_known` \
empty. The list is known to be incomplete and a missing advertiser is exactly \
what this pass is for."""


KEY_FILE = ".anthropic_key"


def _api_key():
    """
    The billing key for this pipeline, preferring a name of its own.

    SPONSOR_LLM_API_KEY comes first deliberately. The Claude Code CLI a
    developer is likely running in the next window authenticates by OAuth, and
    setting ANTHROPIC_API_KEY machine-wide can make it switch to that key and
    bill it -- a surprising way to lose an afternoon. A separate name means this
    pipeline's credentials cannot disturb anything else.

    The file fallback exists because a Windows env var set in one terminal is
    not visible to a process started from another. Keep it out of git.
    """
    for var in ("SPONSOR_LLM_API_KEY", "ANTHROPIC_API_KEY"):
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip()
    for d in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        p = os.path.join(d, KEY_FILE)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                v = f.read().strip()
            if v:
                return v
    return None


def _schema():
    return {
        "type": "object",
        "properties": {
            "brands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string",
                                 "description": "the brand as printed"},
                        "matches_known": {"type": "string",
                                          "description": "exact name from the known list, or empty"},
                        "surface": {"type": "string",
                                    "description": "led board / fixed board / hoarding / backdrop / other"},
                        "legible": {"type": "boolean",
                                    "description": "false if only partly readable"},
                        "confidence": {"type": "number", "description": "0-1"},
                    },
                    "required": ["name", "matches_known", "surface", "legible",
                                 "confidence"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string",
                      "description": "one short line, or empty"},
        },
        "required": ["brands", "notes"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------
# frame sampling and grouping
# --------------------------------------------------------------------------

def _signature(img):
    """Cheap appearance signature: coarse Lab histogram + row-mean profile."""
    small = cv2.resize(img, (128, 72), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    rows = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean(axis=1)
    rows = (rows - rows.mean()) / (rows.std() + 1e-9)
    return hist, rows


def _distance(a, b):
    hist_d = 0.5 * float(np.abs(a[0] - b[0]).sum())
    rows_d = float(np.abs(a[1] - b[1]).mean()) / 4.0
    return 0.6 * hist_d + 0.4 * min(1.0, rows_d)


def sample_and_group(video, scan_fps, mask_overlays=True, max_frames=None,
                     group_dist=GROUP_DIST, max_group_s=MAX_GROUP_S,
                     pitch_crop=True):
    """
    [(rep_frame_idx, rep_t, [member frame idx], rep_image)] plus info.

    Groups only CONSECUTIVE samples: two visits to the same camera angle minutes
    apart can show different LED creatives, and merging them would carry one
    read across a change it never saw.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / max(scan_fps, 0.01))))

    gmask, ginfo = None, {"ok": False, "reason": "disabled"}
    if mask_overlays:
        gmask, ginfo = overlay_mask(video, n=96)
        if not ginfo["ok"]:
            gmask = None

    groups, sig_prev = [], None
    idx, n = 0, 0
    while True:
        if not cap.grab():
            break
        if idx % step != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        if gmask is not None:
            frame = apply_mask(frame, gmask)
        if frame.shape[1] > MAX_W:
            sc = MAX_W / frame.shape[1]
            frame = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        # Grass cannot hold a board, and every row of it is billed. Cropping the
        # pitch away before the image is sent cuts the image to ~41% of its
        # height on this broadcast, and the token bill with it.
        if pitch_crop:
            cut = pitch_crop_row(frame)
            if cut is not None:
                frame = frame[:cut]
        sig = _signature(frame)
        spf_now = step / fps
        too_long = (groups and
                    len(groups[-1]["members"]) * spf_now >= max_group_s)
        if (sig_prev is not None and _distance(sig, sig_prev) < group_dist
                and not too_long):
            groups[-1]["members"].append(idx)
        else:
            groups.append({"gid": len(groups), "rep_frame": idx,
                           "rep_t": idx / fps, "members": [idx],
                           "image": frame})
        sig_prev = sig
        idx += 1
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    return groups, {"fps": fps, "frames": total, "step": step,
                    "sampled": n, "groups": len(groups),
                    "seconds_per_sample": step / fps, "overlay_mask": ginfo,
                    "overlay_boxes": boxes_from_mask(gmask) if gmask is not None else []}


# The perimeter occupies roughly the top quarter of the frame in every camera
# framing this broadcast uses -- the same window the ground-truth sampler uses,
# and established the same way (by checking every framing, not the first one).
# Cropping to it keeps board text at native width in a montage, which is what
# makes it readable; a whole frame scaled to fit three-up is not.
CROP_Y = (0, 250)
PER_MONTAGE = 3


def export_frames(groups, out_dir, spf, crop_y=CROP_Y, per_montage=PER_MONTAGE):
    """
    Write grouped representatives as montages for a human or a Claude session.

    Each montage carries `per_montage` groups, labelled with their id and time
    span so a reading can be attributed back without counting rows.
    """
    os.makedirs(out_dir, exist_ok=True)
    index, batch, n = {}, [], 0
    for g in groups:
        img = g["image"]
        y0, y1 = crop_y
        scale = img.shape[0] / 1080.0            # frames were downscaled to MAX_W
        crop = img[int(y0 * scale):int(min(y1 * scale, img.shape[0]))]
        if crop.size == 0:
            crop = img
        dur = len(g["members"]) * spf
        label = f"#{g['gid']}  t={g['rep_t']:.0f}-{g['rep_t'] + dur:.0f}s"
        crop = crop.copy()
        cv2.rectangle(crop, (0, 0), (330, 26), (0, 0, 0), -1)
        cv2.putText(crop, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        batch.append((g, crop))
        if len(batch) == per_montage:
            n = _flush_montage(batch, out_dir, n, index)
            batch = []
    if batch:
        n = _flush_montage(batch, out_dir, n, index)

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({
            "_what": "One montage per few groups. Read each panel and record the "
                     "advertisers legible on any advertising surface. Write "
                     "readings.json as {\"<group id>\": [\"BRAND\", ...]}, using "
                     "the names from sponsors.json where you recognise them. An "
                     "empty list is a valid and useful answer.",
            "montages": index,
        }, f, ensure_ascii=False, indent=2)
    return n, index


def _flush_montage(batch, out_dir, n, index):
    w = max(c.shape[1] for _, c in batch)
    rows = []
    for _, c in batch:
        if c.shape[1] < w:
            c = cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1],
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        rows.append(c)
        rows.append(c[:3] * 0 + 255)             # a white rule between panels
    name = f"m_{n:04d}.jpg"
    cv2.imwrite(os.path.join(out_dir, name), cv2.vconcat(rows[:-1]),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    index[name] = [{"gid": g["gid"], "rep_t": round(g["rep_t"], 2),
                    "members": len(g["members"])} for g, _ in batch]
    return n + 1


def _encode(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    if not ok:
        return None, 0
    return base64.b64encode(buf.tobytes()).decode("ascii"), buf.size


def _est_tokens(img, prompt_chars):
    """Anthropic bills an image at about (w x h)/750 tokens."""
    return int(img.shape[0] * img.shape[1] / 750) + prompt_chars // 4


def _prices(model):
    """(in, out) $/1M. Unknown models fall back to the dearest known rate, so a
    cost estimate is never optimistic about a model nobody has priced."""
    return PRICES.get(model, max(PRICES.values()))


# --------------------------------------------------------------------------

def run_scan_llm(run_dir, video=None, sponsors_path=None, scan_fps=1.0,
                 model=MODEL, max_calls=None, dry_run=False,
                 mask_overlays=True, max_frames=None, group_dist=GROUP_DIST,
                 export_frames_dir=None, readings=None, pitch_crop=True):
    os.makedirs(run_dir, exist_ok=True)
    man_path = os.path.join(run_dir, "manifest.json")
    manifest = {}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)
    video = video or manifest.get("video", {}).get("path")
    if not video:
        raise SystemExit("No --video given and none in the run manifest.")
    sponsors_path = sponsors_path or os.path.join(run_dir, "sponsors.json")
    sponsors = load_sponsors(sponsors_path)
    idx = _index(sponsors)
    known = "\n".join(f"- {s['name']}" for s in sponsors)
    prompt = PROMPT.format(known=known)

    print(f"Run      : {run_dir}")
    print(f"Video    : {video}")
    t0 = time.perf_counter()
    groups, info = sample_and_group(video, scan_fps, mask_overlays, max_frames,
                                    group_dist, MAX_GROUP_S, pitch_crop)
    info["path"] = video
    spf = info["seconds_per_sample"]
    print(f"Sampled  : {info['sampled']} frames every {info['step']} "
          f"(~{scan_fps}/s), each standing for {spf:.2f}s")
    print(f"Grouped  : {len(groups)} distinct pictures "
          f"({info['sampled'] / max(1, len(groups)):.1f}x fewer calls)")
    if info["overlay_mask"]["ok"]:
        print(f"Graphics : {len(info['overlay_boxes'])} overlay(s) masked")
    else:
        print(f"Graphics : not masked -- {info['overlay_mask']['reason']}")

    todo = groups if not max_calls else groups[:max_calls]
    tok_in = sum(_est_tokens(g["image"], len(prompt)) for g in todo)
    tok_out = OUT_TOKENS * len(todo)
    p_in, p_out = _prices(model)
    cost = tok_in / 1e6 * p_in + tok_out / 1e6 * p_out
    covered = sum(len(g["members"]) for g in todo) * spf
    print(f"\n  PLAN -- {len(todo)} call(s) to {model}, covering {covered:.0f}s")
    print(f"         ~{tok_in} in / ~{tok_out} out  =  ~${cost:.2f}")
    print("         (estimate only -- billed tokens come back on each response)")
    if dry_run:
        print("\n  --dry-run: nothing was sent.")
        return None, {"dry_run": True, "calls": len(todo), "est_cost_usd": round(cost, 4),
                      **info}

    # --- the free path: write the frames out for someone to read --------------
    if export_frames_dir:
        n_m, index = export_frames(todo, export_frames_dir, spf)
        print(f"\n  Exported {len(todo)} group(s) as {n_m} montage(s) to "
              f"{export_frames_dir}")
        print(f"  Read them, then write readings.json as "
              f"{{\"<group id>\": [\"BRAND\", ...]}} and re-run with --readings.")
        print(f"  Nothing was sent to any API. Estimated saving: ${cost:.2f}")
        return None, {"exported": len(todo), "montages": n_m,
                      "export_dir": export_frames_dir, **info}

    rows, per_brand = [], defaultdict(list)
    paying = {s["name"]: s["paying"] for s in sponsors}
    unlisted = defaultdict(int)
    n_calls, in_tok, out_tok = 0, 0, 0
    results = []

    # --- the free path, part two: fold in readings made outside the API -------
    if readings:
        # Accept both shapes. .jsonl is the append-only store a long match is
        # read into (resumable, parallel-safe); .json is the flat map a single
        # short pass produces. Same downstream path either way.
        if readings.endswith(".jsonl"):
            from .readings_store import load as _load_readings
            read_map = {str(k): v for k, v in _load_readings(readings).items()}
        else:
            with open(readings, encoding="utf-8") as f:
                read_map = json.load(f)
        by_gid = {g["gid"]: g for g in todo}
        unknown_gids, unknown_names = [], defaultdict(int)
        for gid_s, names in read_map.items():
            if gid_s.startswith("_"):            # allow _comment style keys
                continue
            try:
                g = by_gid[int(gid_s)]
            except (KeyError, ValueError):
                unknown_gids.append(gid_s)
                continue
            got = []
            for raw in names:
                sp = next((s for s in sponsors if s["name"] == raw), None)
                if sp is None:
                    sp, _sc = match_token(raw, idx)
                if sp is None:
                    if len(_norm(raw)) >= 4:
                        unlisted[raw] += len(g["members"])
                        unknown_names[raw] += 1
                    continue
                got.append(sp["name"])
                for m in g["members"]:
                    t0_m = m / info["fps"]
                    per_brand[sp["name"]].append((t0_m, t0_m + spf))
                    rows.append({"brand": sp["name"], "paying": sp["paying"],
                                 "creative": g["gid"], "frame": m,
                                 "t": round(t0_m, 3), "seconds": round(spf, 4),
                                 "n_panels": 1, "area_px": 0.0, "area_pct": 0.0,
                                 "dist_center": 0.0, "area_source": "not_measured",
                                 "time_basis": "per_frame_llm",
                                 "identified_by": "vision_readings",
                                 "brightness": 0.0, "brightness_rel": 0.0})
            results.append({"group": g["gid"], "rep_frame": g["rep_frame"],
                            "rep_t": round(g["rep_t"], 2),
                            "members": len(g["members"]), "brands": got,
                            "notes": "read outside the API"})
        covered_gids = {r["group"] for r in results}
        missing = [g for g in todo if g["gid"] not in covered_gids]
        print(f"\n  Folded in readings for {len(covered_gids)}/{len(todo)} groups")
        if missing:
            # Silence here would look identical to "nobody was on those boards".
            miss_s = sum(len(g['members']) for g in missing) * spf
            print(f"  {len(missing)} group(s) ({miss_s:.0f}s) have NO reading and "
                  f"are credited to nobody -- gids "
                  f"{', '.join(str(g['gid']) for g in missing[:12])}"
                  f"{' ...' if len(missing) > 12 else ''}")
        if unknown_gids:
            print(f"  ignored {len(unknown_gids)} unknown group id(s): "
                  f"{', '.join(unknown_gids[:8])}")
        if unknown_names:
            print("  names that matched no sponsor (kept in the unlisted file): "
                  + ", ".join(sorted(unknown_names)[:8]))
        return _finalise(run_dir, manifest, man_path, rows, per_brand, paying,
                         unlisted, results, info, spf, sponsors,
                         {"source": "readings", "readings_file": readings,
                          "groups": len(groups), "groups_read": len(covered_gids),
                          "groups_unread": len(missing), "cost_usd": 0.0,
                          "est_cost_usd_if_api": round(cost, 4),
                          "sampled": info["sampled"], "step": info["step"],
                          "seconds_per_sample": round(spf, 4)})

    try:
        import anthropic
    except ImportError:
        raise SystemExit("  the anthropic SDK is not installed -- `pip install anthropic`")
    key = _api_key()
    if not key:
        raise SystemExit(
            "  No API key found. Set one of, in order of preference:\n"
            "    SPONSOR_LLM_API_KEY   env var (preferred -- cannot disturb the\n"
            "                          Claude Code CLI's own OAuth login)\n"
            "    .anthropic_key        a file in the repo root holding just the key\n"
            "    ANTHROPIC_API_KEY     env var\n"
            "  Get a key at https://console.anthropic.com/settings/keys")
    try:
        client = anthropic.Anthropic(api_key=key)
    except Exception as e:
        raise SystemExit(f"  could not create the Anthropic client: {e}")

    rows, per_brand = [], defaultdict(list)
    paying = {s["name"]: s["paying"] for s in sponsors}
    unlisted = defaultdict(int)
    n_calls, in_tok, out_tok = 0, 0, 0
    results = []

    for i, g in enumerate(todo, 1):
        b64, _ = _encode(g["image"])
        if b64 is None:
            continue
        try:
            resp = client.messages.create(
                model=model, max_tokens=1024,
                tools=[{"name": "report_brands",
                        "description": "Report the advertisers legible in this frame.",
                        "input_schema": _schema()}],
                tool_choice={"type": "tool", "name": "report_brands"},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg",
                                                 "data": b64}},
                    {"type": "text", "text": prompt}]}])
        except Exception as e:
            print(f"  call {i}/{len(todo)} failed: {e}")
            continue
        n_calls += 1
        in_tok += getattr(resp.usage, "input_tokens", 0)
        out_tok += getattr(resp.usage, "output_tokens", 0)
        payload = next((c.input for c in resp.content
                        if getattr(c, "type", "") == "tool_use"), {"brands": []})
        got = []
        for b in payload.get("brands", []):
            if not b.get("legible", True):
                continue
            name = (b.get("matches_known") or "").strip()
            sp = next((s for s in sponsors if s["name"] == name), None)
            if sp is None:
                sp, _score = match_token(b.get("name", ""), idx)
            if sp is None:
                raw = (b.get("name") or "").strip()
                if len(_norm(raw)) >= 4:
                    unlisted[raw] += len(g["members"])
                continue
            got.append(sp["name"])
            for m in g["members"]:
                per_brand[sp["name"]].append((m / info["fps"],
                                              m / info["fps"] + spf))
                rows.append({"brand": sp["name"], "paying": sp["paying"],
                             "creative": i, "frame": m, "t": round(m / info["fps"], 3),
                             "seconds": round(spf, 4), "n_panels": 1,
                             "area_px": 0.0, "area_pct": 0.0, "dist_center": 0.0,
                             "area_source": "not_measured",
                             "time_basis": "per_frame_llm",
                             "identified_by": "vision_llm",
                             "brightness": 0.0, "brightness_rel": 0.0})
        results.append({"group": i, "rep_frame": g["rep_frame"],
                        "rep_t": round(g["rep_t"], 2),
                        "members": len(g["members"]), "brands": got,
                        "notes": payload.get("notes", "")})
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} calls, {len(per_brand)} brands so far")

    p_in, p_out = _prices(model)
    cost_real = in_tok / 1e6 * p_in + out_tok / 1e6 * p_out
    return _finalise(run_dir, manifest, man_path, rows, per_brand, paying,
                     unlisted, results, info, spf, sponsors,
                     {"source": "api", "calls": n_calls, "groups": len(groups),
                      "sampled": info["sampled"], "step": info["step"],
                      "seconds_per_sample": round(spf, 4),
                      "input_tokens": in_tok, "output_tokens": out_tok,
                      "cost_usd": round(cost_real, 4),
                      "est_cost_usd": round(cost, 4), "model": model})


def _finalise(run_dir, manifest, man_path, rows, per_brand, paying, unlisted,
              results, info, spf, sponsors, extra_stats):
    """
    Turn per-brand spans into the run's outputs.

    Shared by the API path and the readings path on purpose: the two must
    produce byte-identical artifacts, or the free pilot pass would not actually
    be measuring the thing the paid pass will do.
    """
    from .merge_surfaces import _union_seconds

    by_brand = []
    duration = info["frames"] / info["fps"]
    ident = "vision_readings" if extra_stats.get("source") == "readings" else "vision_llm"
    for brand, spans in per_brand.items():
        secs = _union_seconds(spans)
        ts = sorted(s for s, _ in spans)
        by_brand.append({
            "brand": brand, "paying": paying.get(brand, True),
            "exposure_s": round(secs, 2), "time_basis": "per_frame_llm",
            "time_is_upper_bound": False,
            "pct_of_video": round(100 * secs / duration, 2) if duration else 0.0,
            "n_frames": len(spans), "mean_panels": 1.0,
            "mean_area_px": 0.0, "mean_area_pct": 0.0, "mean_dist_center": 0.0,
            "mean_brightness": 0.0, "mean_brightness_rel": 0.0,
            "first_seen_s": round(ts[0], 2), "last_seen_s": round(ts[-1], 2),
            "area_source": "not_measured", "identified_by": ident,
        })
    by_brand.sort(key=lambda r: (not r["paying"], -r["exposure_s"]))

    _write_csv(os.path.join(run_dir, "exposure_by_brand.csv"), by_brand)
    _write_csv(os.path.join(run_dir, "exposure_detail.csv"), rows)
    with open(os.path.join(run_dir, "scan_llm.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "scan_llm_unlisted.json"), "w",
              encoding="utf-8") as f:
        json.dump({"_what": "brands read that are not in sponsors.json, with the "
                            "seconds they would be worth. Review before adding -- "
                            "this is the discovery path.",
                   "seconds": {k: round(v * spf, 1) for k, v in
                               sorted(unlisted.items(), key=lambda kv: -kv[1])}},
                  f, ensure_ascii=False, indent=2)

    stats = {**extra_stats,
             "brands": len(by_brand),
             "paying_brands": sum(1 for r in by_brand if r["paying"]),
             "unlisted": dict(sorted(unlisted.items(), key=lambda kv: -kv[1])[:20]),
             "overlay_mask": info["overlay_mask"]}
    manifest.setdefault("video", {}).update(
        {"path": info.get("path", manifest.get("video", {}).get("path")),
         "fps": info["fps"], "frames": info["frames"]})
    manifest.setdefault("stats", {})["scan_llm"] = stats
    manifest["stats"].setdefault("scan", {}).setdefault("step", info["step"])
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n  {'brand':<32} {'time':>8}")
    for r in by_brand:
        tag = "" if r["paying"] else "   (not paying)"
        print(f"  {r['brand']:<32} {r['exposure_s']:7.1f}s{tag}")
    if unlisted:
        print("\n  read but not on the sponsor list (see scan_llm_unlisted.json):")
        for k, v in sorted(unlisted.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {v * spf:6.1f}s  {k}")
    print(f"\n  Wrote {os.path.join(run_dir, 'exposure_by_brand.csv')}")
    return by_brand, stats


def _write_csv(path, rows):
    if not rows:
        rows = [{"brand": "", "paying": "", "exposure_s": 0}]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
