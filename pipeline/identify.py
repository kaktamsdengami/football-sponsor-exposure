"""
Put brand names on the grouped band strips.

`dedup` leaves one representative strip per distinct view. This stage reads the
text on each one and decides which sponsor it belongs to.

TWO PASSES, CHEAP ONE FIRST. Perimeter board text is large, horizontal and high
contrast -- the easy case for OCR. It is free, runs offline (so no footage
leaves the machine, which the club has not yet agreed to) and takes about a
quarter of a second per group. Whatever OCR cannot resolve is left marked for a
vision model, which is slower and costs money, so it should only see the
leftovers.

OCR ALONE IS NOT ENOUGH. On real footage one sponsor came back as a dozen
different strings, none identical. Raw OCR text is unusable as
an identity. What makes it usable is matching against a KNOWN SPONSOR LIST: the
mangled spellings all collapse onto the right name, and anything that matches
nothing is honestly reported as unidentified rather than invented.

The list also carries `paying`. A perimeter carries the club's own promos and
the stadium name alongside the advertisers who actually bought space; the report
is about the latter, but the former still have to be recognised so they are not
mistaken for a sponsor.

Where OCR found the text is mapped back into frame coordinates, so a brand gets
its own position and area rather than inheriting the whole strip's.
"""

import base64
import json
import os
import time
from collections import defaultdict
from difflib import SequenceMatcher

import cv2

from brand_reader.board_regions import rect_from_scalars, reflow_box_to_frame
from .board_states import carry_within_states
from .store import RunStore

MIN_OCR_CONF = 0.30      # below this the text is noise, not a weak read
MIN_MATCH = 0.62         # fuzzy score at which a token is accepted as a sponsor

# Text that is readable but matches no sponsor on the list. A single stray
# string is OCR noise; the SAME string recurring across seconds of board time
# is an advertiser nobody wrote down. These separate the two.
MIN_UNKNOWN_LEN = 4          # normalised characters; shorter is not a brand
MIN_UNKNOWN_MERGE = 0.75     # fuzzy score at which two spellings are one brand
MIN_UNKNOWN_SECONDS = 1.0    # board time a string must hold to be reported
MIN_UNKNOWN_FRAMES = 4       # ...on at least this many distinct frames
MIN_UNKNOWN_CONF = 0.60      # OCR's own confidence; noise reads low
MAX_UNKNOWN_SIM = 0.45       # above this it is a misread of a LISTED sponsor,
                             # not a new one -- do not invent a brand for it

# The vision pass. Opus is worth it here: these are the strips OCR already
# failed on, they are a small minority, and a wrong brand name goes straight
# into a client's invoice.
LLM_MODEL = "claude-opus-5"
LLM_PRICE_IN, LLM_PRICE_OUT = 5.00, 25.00      # $/1M tokens, for the cost line


# ---------------------------------------------------------------------------
# sponsor list
# ---------------------------------------------------------------------------

# Cyrillic letters that are drawn identically to a Latin one. EasyOCR is run
# with both alphabets loaded when a board mixes scripts, and it picks
# between two glyphs it cannot tell apart essentially at random: OPPO came back
# as both "oppo" (Latin o, U+006F) and "[Cyrillic o]ppo" (U+043E) on the same
# board, seconds apart. Left alone those are two different brands to every
# string comparison in this file -- they never merge with each other and
# neither matches a Latin sponsor list.
#
# Folding is applied to the sponsor list and to the OCR text through the same
# function, so it stays symmetric: a genuinely Cyrillic-only sponsor
# folds the same way on both sides and still matches itself.
_HOMOGLYPHS = str.maketrans({
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041A": "K", "\u041C": "M", "\u041D": "H", "\u041E": "O",
    "\u0420": "P", "\u0421": "C", "\u0422": "T", "\u0423": "Y", "\u0425": "X", "\u0406": "I", "\u0405": "S",
    "\u0408": "J", "\u0401": "E", "\u04AE": "Y", "\u050C": "G", "\u0524": "P",
})


def _norm(s):
    """Uppercase, drop punctuation, and fold Cyrillic/Latin lookalikes."""
    return "".join(c for c in s.upper().translate(_HOMOGLYPHS) if c.isalnum())


def load_sponsors(path):
    """
    [{name, aliases, paying}] from JSON. Accepts either the full form or a bare
    list of names, so a club can hand over the simplest possible file.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("sponsors", data) if isinstance(data, dict) else data
    out = []
    for it in items:
        if isinstance(it, str):
            it = {"name": it}
        out.append({
            "name": it["name"],
            "aliases": list(it.get("aliases", [])),
            "paying": bool(it.get("paying", True)),
            "note": it.get("note", ""),
        })
    return out


def _index(sponsors):
    """(normalised string -> sponsor) for every name and alias."""
    idx = []
    for sp in sponsors:
        for s in [sp["name"]] + sp["aliases"]:
            n = _norm(s)
            if n:
                idx.append((n, sp))
    return idx


# A substring only counts as "the same brand" when it is most of the name.
# Without this, any short word inside a longer alias scores 0.86 and wins:
# a club-name token landed on a longer sponsor name 12 times in 30 frames of one
# clip, and a short word inside a long slogan alias captured the sponsor that
# owned the slogan. OCR does
# drop characters off the ends, which is what the rule is for -- but it drops
# one or two, not two thirds of the word.
_CONTAIN_MIN_LEN = 6
_CONTAIN_MIN_FRAC = 0.60
# Two different sponsors this close together is not a read, it is a coin flip.
_AMBIGUOUS_MARGIN = 0.03


def match_token(token, idx, min_match=MIN_MATCH):
    """
    (sponsor, score) or (None, best_score).

    Containment counts as a strong match -- OCR routinely drops or adds a
    character at either end -- but only when the token is substantial next to
    the name it sits inside. A token that fits two different sponsors equally
    well is rejected rather than assigned to whichever sorted first.
    """
    t = _norm(token)
    if len(t) < 3:
        return None, 0.0
    scores = {}                                  # sponsor name -> best score
    best, best_s = None, 0.0
    for n, sp in idx:
        r = SequenceMatcher(None, t, n).ratio()
        if (t in n or n in t) and len(t) >= _CONTAIN_MIN_LEN \
                and len(t) >= _CONTAIN_MIN_FRAC * len(n):
            r = max(r, 0.86)
        if r > scores.get(sp["name"], 0.0):
            scores[sp["name"]] = r
        if r > best_s:
            best, best_s = sp, r
    if best_s < min_match:
        return None, best_s
    rivals = sorted((v for k, v in scores.items() if k != best["name"]),
                    reverse=True)
    if rivals and best_s - rivals[0] < _AMBIGUOUS_MARGIN:
        return None, best_s                      # ambiguous between sponsors
    return best, best_s


# ---------------------------------------------------------------------------
# mapping OCR boxes back onto the frame
# ---------------------------------------------------------------------------

def _frame_poly(det, ocr_box):
    """
    Where this text sat in the original frame.

    The strip is a rectified band folded into rows, so a box on it means nothing
    on its own; `reflow_box_to_frame` unwinds the fold and the perspective. It
    is what turns "this strip mentions BRAND_A" into "BRAND_A occupied this
    region of the picture", which is what the report is actually measuring.
    """
    try:
        rect = rect_from_scalars(det["coef"], det["span"], det["fit_med"],
                                 det["local_clamp"], det["H"], det["over"],
                                 det.get("segments"), det.get("knots"),
                                 det.get("off_shift", 0))
        layout = {"rows": det["rows"], "upscale": det["upscale"],
                  "strip_shape": det["strip"]}
        xs = [p[0] for p in ocr_box]
        ys = [p[1] for p in ocr_box]
        x1, y1, x2, y2 = reflow_box_to_frame(
            rect, layout, (min(xs), min(ys), max(xs), max(ys)))
        return [[int(x1), int(y1)], [int(x2), int(y1)],
                [int(x2), int(y2)], [int(x1), int(y2)]]
    except Exception:
        return None


# ---------------------------------------------------------------------------

_LLM_PROMPT = """This is a strip cut from a football broadcast, showing the \
perimeter advertising boards along one touchline. The strip has been \
straightened and then folded into {n_rows} stacked rows -- read it like lines of \
text: each row continues the one above, left to right.

List every distinct advertiser or brand name you can read on the boards.

Rules:
- Report a name only if you can actually read it. Do not guess from a partial \
word or a colour. An empty list is a correct answer.
- Report the name as printed, in its own script (Cyrillic stays Cyrillic).
- The same advertiser is often repeated along the board. Report it ONCE.
- Ignore the broadcaster's overlaid score clock and channel logo -- those are \
graphics on top of the picture, not advertising on the boards.
- Ignore stadium furniture, crowd, players and pitch markings.

These sponsors are known to appear at this ground. If what you read is one of \
them, use `matches_known` to say which -- OCR mangles spellings and this is how \
they get merged:
{known}

If you read a brand that is NOT in that list, still report it, and leave \
`matches_known` empty. The list is known to be incomplete, and a missing \
advertiser is exactly what this pass is for."""


def _llm_schema():
    return {
        "type": "object",
        "properties": {
            "brands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string",
                                 "description": "the brand as printed on the board"},
                        "matches_known": {
                            "type": "string",
                            "description": "exact name from the known list, or empty"},
                        "legible": {"type": "boolean",
                                    "description": "false if only partly readable"},
                        "confidence": {"type": "number",
                                       "description": "0-1, how sure you are"},
                    },
                    "required": ["name", "matches_known", "legible", "confidence"],
                    "additionalProperties": False,
                },
            },
            "notes": {"type": "string",
                      "description": "one short line: why anything was unreadable"},
        },
        "required": ["brands", "notes"],
        "additionalProperties": False,
    }


def _est_tokens(path, prompt_chars):
    """
    Rough input tokens for one call, without contacting the API.

    Anthropic bills an image at about (width x height) / 750 tokens. This is an
    estimate so a spend can be seen BEFORE any money is committed -- it is not
    the billed figure, which comes back on the response.
    """
    try:
        im = cv2.imread(path)
        img_tok = int(im.shape[0] * im.shape[1] / 750) if im is not None else 800
    except Exception:
        img_tok = 800
    return img_tok + prompt_chars // 4


def _llm_plan(store, creatives, sponsors, model, max_calls, min_seconds):
    """What a run would cost, listed per call. Makes no API calls."""
    todo = sorted((c for c in creatives
                   if c.get("needs_llm") and c["seconds"] >= min_seconds),
                  key=lambda c: -c["seconds"])[:max_calls]
    known = "\n".join(f"- {s['name']}" for s in sponsors)
    prompt_chars = len(_LLM_PROMPT.format(n_rows=3, known=known))
    rows, tok_in = [], 0
    for c in todo:
        t = _est_tokens(os.path.join(store.dir, c["rep_band"]), prompt_chars)
        tok_in += t
        rows.append((c, t))
    tok_out = 150 * len(todo)          # the JSON reply is short
    cost = tok_in / 1e6 * LLM_PRICE_IN + tok_out / 1e6 * LLM_PRICE_OUT
    return rows, tok_in, tok_out, cost


def _print_plan(rows, tok_in, tok_out, cost, model, seconds_total):
    print(f"\n  [llm] PLAN -- {len(rows)} call(s) to {model}")
    print(f"        covering {seconds_total:.1f}s of unread board time")
    for c, t in rows[:12]:
        print(f"          call -> #{c['creative']:<3} {c['seconds']:5.1f}s  "
              f"~{t:5d} tok  {c['rep_band']}")
    if len(rows) > 12:
        print(f"          ... and {len(rows) - 12} more")
    print(f"        estimated ~{tok_in} in / ~{tok_out} out  =  ~${cost:.3f}")
    print("        (estimate only -- billed tokens come back on each response)")


def _llm_pass(store, creatives, sponsors, idx, model, max_calls, min_seconds,
              min_match, progress=True, dry_run=False):
    """
    Send the strips OCR could not read to a vision model.

    Only the leftovers, largest first: this is the expensive path, and a 0.2s
    fragment is not worth a call. The model is allowed to return brands that are
    NOT in the sponsor list -- on real footage the biggest miss was an advertiser
    nobody had written down, which list-matching alone can never recover.
    """
    # The plan is built and shown BEFORE anything is imported or called, so the
    # spend can be inspected on a machine with no SDK and no credentials.
    rows, e_in, e_out, e_cost = _llm_plan(store, creatives, sponsors, model,
                                          max_calls, min_seconds)
    todo = [c for c, _ in rows]
    if not todo:
        print("\n  [llm] nothing left worth sending")
        return {}, 0.0, 0
    _print_plan(rows, e_in, e_out, e_cost, model,
                sum(c["seconds"] for c in todo))
    if dry_run:
        print("        DRY RUN -- no calls made")
        return {}, 0.0, 0

    try:
        import anthropic
    except ImportError:
        print("  [llm] the anthropic SDK is not installed -- `pip install anthropic`")
        return {}, 0.0, 0

    try:
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"  [llm] no Anthropic credentials ({e}).\n"
              "        Set ANTHROPIC_API_KEY, or run `ant auth login`.")
        return {}, 0.0, 0

    known = "\n".join(f"- {s['name']}" for s in sponsors)
    secs, tok_in, tok_out, n_calls = defaultdict(float), 0, 0, 0
    for c in todo:
        path = os.path.join(store.dir, c["rep_band"])
        try:
            with open(path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
        except OSError:
            continue
        n_rows = len(c.get("rows") or []) or 3
        n_calls += 1
        if progress:
            print(f"    [call {n_calls}/{len(todo)}] #{c['creative']} "
                  f"({c['seconds']:.1f}s)", end=" ")
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg",
                                                 "data": b64}},
                    {"type": "text",
                     "text": _LLM_PROMPT.format(n_rows=n_rows, known=known)},
                ]}],
                output_config={"format": {"type": "json_schema",
                                          "schema": _llm_schema()}},
            )
        except Exception as e:
            print(f"call failed: {type(e).__name__}: {e}")
            continue

        tok_in += resp.usage.input_tokens
        tok_out += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            print("declined by the model")
            continue
        try:
            data = json.loads(next(b.text for b in resp.content if b.type == "text"))
        except (StopIteration, json.JSONDecodeError):
            continue

        found = []
        for b in data.get("brands", []):
            raw = (b.get("name") or "").strip()
            if not raw or not b.get("legible", True):
                continue
            # Prefer the model's own mapping onto a known sponsor; fall back to
            # fuzzy matching its raw reading; otherwise keep it as a new name.
            sp, score = None, float(b.get("confidence", 0.5))
            hinted = (b.get("matches_known") or "").strip()
            if hinted:
                sp, _ = match_token(hinted, idx, min_match)
            if sp is None:
                sp, _ = match_token(raw, idx, min_match)
            found.append({
                "name": sp["name"] if sp else raw,
                "paying": sp["paying"] if sp else True,
                "in_sponsor_list": sp is not None,
                "score": round(score, 3),
                "ocr_text": raw,
                # No boxes and one frame read: the model saw the representative
                # strip only, so an LLM-named brand inherits the whole view's
                # geometry AND its whole duration. `aggregate` marks both as an
                # over-estimate; point `--llm` at more frames per group to
                # tighten the time.
                "polys": [],
                "frames": list(c.get("frames") or []),
                "instances": 1,
                "time_basis": "group",
                "source": "llm",
            })
        c["brands"] = found
        c["needs_llm"] = not found
        c["llm_notes"] = data.get("notes", "")
        for b in found:
            secs[b["name"]] += c["seconds"]
        if progress:
            new = [b["name"] for b in found if not b["in_sponsor_list"]]
            tag = "  NEW: " + ", ".join(new) if new else ""
            print(f"-> {', '.join(b['name'] for b in found) or '(nothing)'}{tag}")

    cost = tok_in / 1e6 * LLM_PRICE_IN + tok_out / 1e6 * LLM_PRICE_OUT
    print(f"\n  [llm] {n_calls} call(s) made | {tok_in} in / {tok_out} out "
          f"tokens | ${cost:.4f}  (estimated ~${e_cost:.4f})")
    return secs, cost, n_calls


def _best_sim(token, idx):
    """Similarity of `token` to the closest sponsor already on the list."""
    t = _norm(token)
    best = 0.0
    for n, _ in idx:
        r = SequenceMatcher(None, t, n).ratio()
        if t and (t in n or n in t):
            r = max(r, 0.88)
        best = max(best, r)
    return best


def _cluster_unknown(unknown, per_frame_s, idx=(), min_seconds=MIN_UNKNOWN_SECONDS,
                     min_frames=MIN_UNKNOWN_FRAMES, merge=MIN_UNKNOWN_MERGE,
                     min_conf=MIN_UNKNOWN_CONF, max_sim=MAX_UNKNOWN_SIM):
    """
    Group readable-but-unlisted OCR text into candidate advertisers.

    The sponsor list is written by hand from whatever the club remembered, and
    on real footage the biggest single miss was an advertiser nobody had
    written down -- OPPO ran for 20s of a 5-minute clip and scored zero,
    because `match_token` found no entry and the read was discarded. Silence is
    the worst possible answer there: the club sends the report to the
    advertiser and the advertiser is not on it.

    Raw OCR text is still not an identity -- one board came back as BETSITY,
    BEtSITI, 3Etsiti and more. So a string is only promoted to a brand when it
    RECURS: the same reading (fuzzily) has to hold the boards for at least
    `min_seconds` across `min_frames` distinct frames. That is what separates a
    real LED creative, which sits there for seconds, from a one-off misread of
    a shirt or a banner in the crowd.

    Returns [{name, norm, seconds, frames, hits, spellings}], largest first.
    """
    clusters = []                       # [{norm, hits: [...]}]
    for hit in sorted(unknown, key=lambda h: -h["conf"]):
        best, best_r = None, 0.0
        for cl in clusters:
            r = SequenceMatcher(None, hit["norm"], cl["norm"]).ratio()
            if hit["norm"] in cl["norm"] or cl["norm"] in hit["norm"]:
                r = max(r, 0.88)
            if r > best_r:
                best, best_r = cl, r
        if best is not None and best_r >= merge:
            best["hits"].append(hit)
        else:
            # Seeded by the highest-confidence reading, which is the spelling
            # most likely to be the real one.
            clusters.append({"norm": hit["norm"], "hits": [hit]})

    out = []
    for cl in clusters:
        seen = {(h["creative"], h["frame"]) for h in cl["hits"]}
        seconds = sum(per_frame_s.get(cr, 0.0) for cr, _ in seen)
        # Label it with the spelling that was read most often, breaking ties on
        # confidence -- the modal reading beats a single lucky one.
        tally = defaultdict(lambda: [0, 0.0])
        for h in cl["hits"]:
            t = tally[h["text"]]
            t[0] += 1
            t[1] = max(t[1], h["conf"])
        name = max(tally.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
        conf = sum(h["conf"] for h in cl["hits"]) / len(cl["hits"])
        sim = _best_sim(name, idx) if idx else 0.0

        # Four things separate an advertiser nobody listed from OCR noise, and
        # all four are needed. On the UCL clip they cut 38 candidates to 1 while
        # keeping the real find (OPPO, 20s):
        #   held      a real LED creative sits on the boards for seconds
        #   long      a 3-character string is not a brand; Lay's sunburst logo
        #             read as AGS / A9S / ARS / GGS / AIS, all noise
        #   legible   OCR's own confidence; mangles score low
        #   unlike    if it is nearly a sponsor already on the list it is a
        #             MISREAD of that sponsor (HEIND for Heineken, OEOSI for
        #             Pepsi), and inventing a second brand for it is worse than
        #             dropping it
        reasons = []
        if seconds < min_seconds or len(seen) < min_frames:
            reasons.append("too brief")
        if len(cl["norm"]) < MIN_UNKNOWN_LEN:
            reasons.append("too short")
        if conf < min_conf:
            reasons.append("low confidence")
        if sim >= max_sim:
            reasons.append("looks like a misread of a listed sponsor")
        # A run that is mostly digits is a clock, a countdown or a price ticker
        # read off the broadcast graphic sitting in the band, never an
        # advertiser name -- they recur every second and would otherwise promote
        # (04.26, 01:59, ...). Measured on the normalised form so a Cyrillic-O
        # misread of a zero ("O125") is caught too; a short alphANUMERIC brand
        # (3M, BET365) stays under the 60%-digit bar.
        n = cl["norm"]
        if n and sum(c.isdigit() for c in n) / len(n) >= 0.6:
            reasons.append("mostly digits (clock / ticker, not a brand)")

        out.append({
            "name": name,
            "norm": cl["norm"],
            "seconds": round(seconds, 2),
            "frames": len(seen),
            "mean_conf": round(conf, 3),
            "sim_to_listed": round(sim, 3),
            "hits": cl["hits"],
            "spellings": sorted(tally, key=lambda s: -tally[s][0])[:6],
            # Every candidate is returned, promoted or not, with the reason it
            # was rejected. A real advertiser sitting just under a threshold is
            # exactly the failure this pass exists to prevent, and it must not
            # be invisible a second time.
            "promoted": not reasons,
            "rejected_for": reasons,
        })
    out.sort(key=lambda c: -c["seconds"])
    return out


def _attach_unknown(creatives, clusters):
    """
    Write discovered brands onto the creatives that actually showed them.

    `paying` is True so the brand is visible in the report rather than quietly
    filtered out, but `in_sponsor_list` is False and the caller prints the list
    loudly: whether it is an advertiser, the competition's own branding or the
    stadium name is the operator's call, not OCR's.
    """
    by_id = {c["creative"]: c for c in creatives}
    n = 0
    for cl in clusters:
        per_creative = defaultdict(list)
        for h in cl["hits"]:
            per_creative[h["creative"]].append(h)
        for cid, hits in per_creative.items():
            c = by_id.get(cid)
            if c is None:
                continue
            boxes = defaultdict(list)
            for h in hits:
                if h["poly"]:
                    boxes[str(h["frame"])].append(h["poly"])
            c.setdefault("brands", []).append({
                "name": cl["name"],
                "paying": True,
                "in_sponsor_list": False,
                "score": round(max(h["conf"] for h in hits), 3),
                "ocr_text": cl["name"],
                "polys": [p for ps in boxes.values() for p in ps],
                "boxes_by_frame": dict(boxes),
                "frames": sorted({h["frame"] for h in hits}),
                "instances": len(hits),
                "time_basis": "per_frame",
                "source": "ocr_unlisted",
            })
            c["needs_llm"] = False
            n += 1
    return n


def _apply_readings(creatives, path, sponsors, idx, min_match):
    """
    Load brand readings produced OUTSIDE the API -- by a person, or by a Claude
    session looking at the strips directly.

    For a pilot this is the cheap path: the leftovers are a handful of images,
    and someone with eyes can name them for nothing. It is not the product path
    (it does not scale to a full match, and it needs a human or a session in the
    loop), so the API pass exists alongside it. Both write the same records.

    Format -- {"<creative id>": ["BRAND", ...]} or
              {"<creative id>": [{"name": ..., "paying": true}, ...]}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    by_id = {c["creative"]: c for c in creatives}
    added, n = defaultdict(float), 0
    for key, brands in data.items():
        if key.startswith("_"):          # comment keys
            continue
        c = by_id.get(int(key))
        if c is None or not brands:
            continue
        out = []
        for b in brands:
            raw = b if isinstance(b, str) else b.get("name", "")
            raw = raw.strip()
            if not raw:
                continue
            sp, score = match_token(raw, idx, min_match)
            out.append({
                "name": sp["name"] if sp else raw,
                "paying": sp["paying"] if sp else
                          (b.get("paying", True) if isinstance(b, dict) else True),
                "in_sponsor_list": sp is not None,
                "score": round(score if sp else 1.0, 3),
                "ocr_text": raw,
                "polys": [],
                # Group-level reading: no per-frame data and no box, so this
                # brand can only be credited the WHOLE view's frames. That
                # over-states its time when other advertisers share the view --
                # `aggregate` flags it as an upper bound.
                "frames": list(c.get("frames") or []),
                "instances": 1,
                "time_basis": "group",
                "source": "reading",
            })
        if out:
            c["brands"] = out
            c["needs_llm"] = False
            n += 1
            for b in out:
                added[b["name"]] += c["seconds"]
    return n, added


def run_identify(run_dir, sponsors_path=None, min_ocr_conf=MIN_OCR_CONF,
                 min_match=MIN_MATCH, langs=("en",), progress=True,
                 use_llm=False, llm_model=LLM_MODEL, llm_max=25,
                 llm_min_seconds=0.4, readings_path=None, llm_dry_run=False,
                 min_unknown_seconds=MIN_UNKNOWN_SECONDS, carry_states=False,
                 ocr_every=1):
    """Fill in the `brands` field of creatives.jsonl. Returns (store, stats)."""
    out_root, name = os.path.split(os.path.normpath(run_dir))
    store = RunStore(out_root, name, create=False)

    cre_path = os.path.join(store.dir, "creatives.jsonl")
    if not os.path.exists(cre_path):
        raise RuntimeError(f"No creatives.jsonl in {run_dir} -- run `dedup` first.")
    creatives = [json.loads(l) for l in open(cre_path, encoding="utf-8") if l.strip()]

    # Board states from the `states` stage. When present, a brand read on any
    # frame of a held LED state is credited to the whole of that state (within
    # the same camera view only), and a state nobody could read is flagged.
    states_path = os.path.join(store.dir, "states.jsonl")
    states = []
    if carry_states and os.path.exists(states_path):
        states = [json.loads(l) for l in open(states_path, encoding="utf-8")
                  if l.strip()]
    states_by_cre = defaultdict(list)
    for s in states:
        states_by_cre[s["creative"]].append(s)

    sponsors_path = sponsors_path or os.path.join(store.dir, "sponsors.json")
    if not os.path.exists(sponsors_path):
        raise RuntimeError(
            f"No sponsor list at {sponsors_path}.\n"
            "identify needs to know who it is looking for -- raw OCR text is not "
            "an identity. Write a JSON list of names (or names with aliases), or "
            "pass --sponsors.")
    sponsors = load_sponsors(sponsors_path)
    idx = _index(sponsors)

    dets = {d["frame"]: d for d in store.read_detections() if d.get("band")}

    import easyocr
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    print(f"Identify : {store.dir}")
    print(f"Sponsors : {len(sponsors)} from {sponsors_path} "
          f"({sum(1 for s in sponsors if s['paying'])} paying)")
    print(f"Groups   : {len(creatives)} to read (OCR, gpu={gpu})")

    t0 = time.perf_counter()
    secs = defaultdict(float)
    n_named = 0
    n_frames_ocr = 0
    unknown = []            # readable text matching no sponsor, judged later
    per_frame_s = {}        # creative -> seconds each of its frames stands for
    for c in creatives:
        frames = c.get("frames") or (
            [c["rep_frame"]] if c.get("rep_frame") is not None else [])
        # `ocr_every` > 1 reads only every Nth frame of the group (in time
        # order) and lets each read stand for N sampling steps via `per_frame_s`
        # below. OCR is ~0.8s/frame on a weak GPU, so a full match is hours at
        # stride 1; a report tolerates the coarser time grid. A brand that is up
        # only between two sampled frames can be missed -- keep stride 1 when
        # accuracy matters (the score.py runs).
        if ocr_every > 1 and len(frames) > 1:
            frames = sorted(frames)[::ocr_every]
        # OCR EVERY band image in the group, not just the representative one.
        # A perimeter LED unit cycles through several advertisers while one
        # camera view stays up, so a single frame names only whoever was on the
        # boards at that instant. Reading every sampled frame is what lets a
        # brand be credited the seconds it was ACTUALLY on screen (`frames`
        # below) instead of inheriting the whole view's duration.
        #
        # A perimeter also repeats the same advertiser along its length -- four
        # `mastercard` panels in one frame are one brand, but four panels' worth
        # of surface. So the brand is stored ONCE per group with EVERY box it
        # was found at, across every frame it appeared in.
        found = {}
        n_ocr = 0
        ocr_frames = []                 # frames of this group we actually read
        for fr in frames:
            img = cv2.imread(os.path.join(store.dir, f"bands/{fr:07d}.jpg"))
            if img is None:
                continue
            n_ocr += 1
            ocr_frames.append(fr)
            det = dets.get(fr)
            for box, text, conf in reader.readtext(img, detail=1, paragraph=False):
                if conf < min_ocr_conf:
                    continue
                sp, score = match_token(text, idx, min_match)
                if sp is None:
                    # Readable, but not on the sponsor list. Do NOT drop it: the
                    # list is written by hand and is routinely incomplete, and a
                    # dropped read is an advertiser who silently gets zero
                    # seconds in the client's report. Park it and let
                    # `_cluster_unknown` decide later whether the same string
                    # recurs often enough to be a real board.
                    t = _norm(text)
                    if len(t) >= MIN_UNKNOWN_LEN:
                        unknown.append({
                            "norm": t,
                            "text": text.strip(),
                            "conf": float(conf),
                            "creative": c["creative"],
                            "frame": fr,
                            "poly": _frame_poly(det, box) if det else None,
                        })
                    continue
                rec = found.get(sp["name"])
                if rec is None:
                    rec = found[sp["name"]] = {
                        "name": sp["name"],
                        "paying": sp["paying"],
                        "score": round(score, 3),
                        "ocr_text": text,
                        "ocr_conf": round(float(conf), 3),
                        "polys": [],
                        "boxes_by_frame": {},
                        "frames": set(),
                        "instances": 0,
                        "time_basis": "per_frame",
                        "source": "ocr",
                    }
                rec["instances"] += 1
                rec["frames"].add(fr)
                poly = _frame_poly(det, box) if det else None
                if poly:
                    # Boxes are kept PER FRAME: the brand moves and the camera
                    # pans, so a box read on one frame is not where the brand
                    # sits on the next. `aggregate` measures surface frame by
                    # frame from these, never by stamping one frame's boxes onto
                    # the whole group.
                    rec["boxes_by_frame"].setdefault(str(fr), []).append(poly)
                    rec["polys"].append(poly)
                if score > rec["score"]:  # keep the clearest reading as the label
                    rec["score"] = round(score, 3)
                    rec["ocr_text"] = text
                    rec["ocr_conf"] = round(float(conf), 3)
        n_frames_ocr += n_ocr
        for rec in found.values():
            rec["frames_read"] = sorted(rec["frames"])
            rec["frames"] = sorted(rec["frames"])

        # Carry each brand across the held LED state(s) it was read on. States
        # are cut WITHIN this one camera view (`states` stage), so this only
        # fills the frames where the same creative was up but its logo was too
        # blurred / banded to OCR -- it never reaches past a creative change or a
        # cut. `frames_read` keeps the literal reads for auditing.
        cre_states = states_by_cre.get(c["creative"])
        n_carried = 0
        if cre_states and found:
            ok = set(ocr_frames)
            state_frames = {st["state"]: [f for f in st["frames"] if f in ok]
                            for st in cre_states}
            frame_to_state = {f: sid for sid, fs in state_frames.items() for f in fs}
            carried = carry_within_states(
                {n: rec["frames_read"] for n, rec in found.items()},
                frame_to_state, state_frames)
            for n, rec in found.items():
                rec["frames"] = carried.get(n, rec["frames_read"])
                n_carried += len(rec["frames"]) - len(rec["frames_read"])
        if cre_states:
            read_any = set()
            for rec in found.values():
                read_any.update(rec["frames_read"])
            for st in cre_states:
                st["status"] = ("read" if any(f in read_any for f in st["frames"])
                                else "unread")

        c["brands"] = sorted(found.values(), key=lambda b: -b["score"])
        c["ocr_frames_read"] = n_ocr
        # Nothing recognised anywhere in the group: point the vision model here.
        c["needs_llm"] = not c["brands"]
        # Per-frame credit: each frame stands for c["seconds"] / n_ocr of time.
        per_frame_s[c["creative"]] = c["seconds"] / max(n_ocr, 1)
        if c["brands"]:
            n_named += 1
            for b in c["brands"]:
                secs[b["name"]] += len(b["frames"]) * per_frame_s[c["creative"]]
        if progress:
            names = ", ".join(
                (f"{b['name']} {len(b['frames_read'])}->{len(b['frames'])}/{n_ocr}"
                 if len(b["frames"]) != len(b["frames_read"])
                 else f"{b['name']} {len(b['frames'])}/{n_ocr}")
                for b in c["brands"]) or "-"
            tag = f"  (+{n_carried} carried)" if n_carried else ""
            print(f"  #{c['creative']:<3} {c['seconds']:5.1f}s  {names}{tag}")

    # Readable text that matched nothing on the list. Promoted to brands only
    # where the same string held the boards long enough to be a real creative.
    candidates = _cluster_unknown(unknown, per_frame_s, idx=idx,
                                  min_seconds=min_unknown_seconds)
    unlisted = [c for c in candidates if c["promoted"]]
    if candidates:
        # Everything the pass saw, promoted or not, goes to disk. Tuning the
        # threshold is then an inspection, not another five-minute OCR run.
        with open(os.path.join(store.dir, "unlisted.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "_comment": "OCR text from the boards matching no sponsor on "
                            "the list. `promoted` entries were counted as "
                            "brands; the rest were judged too brief or too rare "
                            "to be a real creative. Move real advertisers into "
                            "sponsors.json.",
                "min_unknown_seconds": min_unknown_seconds,
                "candidates": [{k: v for k, v in c.items() if k != "hits"}
                               for c in candidates],
            }, f, ensure_ascii=False, indent=2)
    if unlisted:
        n_named += _attach_unknown(creatives, unlisted)
        for cl in unlisted:
            secs[cl["name"]] += cl["seconds"]
        print(f"\n  [unlisted] {len(unlisted)} brand(s) read on the boards that "
              f"are NOT in {os.path.basename(sponsors_path)}:")
        for cl in unlisted[:12]:
            alt = [s for s in cl["spellings"] if s != cl["name"]][:3]
            tail = f"   (also read as {', '.join(alt)})" if alt else ""
            print(f"    {cl['name']:<24} {cl['seconds']:6.1f}s  "
                  f"{cl['frames']} frames{tail}")
        if len(unlisted) > 12:
            print(f"    ... and {len(unlisted) - 12} more")
        print(f"    {len(candidates) - len(unlisted)} weaker candidate(s) not "
              f"counted -- all of them in unlisted.json")
        print("    -> add the real advertisers to the sponsor list, and mark "
              "competition/stadium branding `paying: false`.")

    llm_cost, llm_calls = 0.0, 0
    if readings_path:
        n, added = _apply_readings(creatives, readings_path, sponsors, idx, min_match)
        for k, v in added.items():
            secs[k] += v
        n_named += n
        print(f"\n  [readings] {n} groups named from {readings_path}")
    if use_llm:
        added, llm_cost, llm_calls = _llm_pass(
            store, creatives, sponsors, idx, llm_model, llm_max,
            llm_min_seconds, min_match, progress, dry_run=llm_dry_run)
        for k, v in added.items():
            secs[k] += v
        n_named += sum(1 for c in creatives
                       if c["brands"] and any(b["source"] == "llm" for b in c["brands"]))

    with open(cre_path, "w", encoding="utf-8") as f:
        for c in creatives:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # A state OCR could not read but a reading / the vision model / the unlisted
    # pass has since named is no longer an unread gap. Those fallbacks carry no
    # per-frame evidence, so promote every state of a now-named group.
    if states:
        named_cre = {c["creative"] for c in creatives if c["brands"]}
        for s in states:
            if s.get("status") == "unread" and s["creative"] in named_cre:
                s["status"] = "read"
        with open(states_path, "w", encoding="utf-8") as f:
            for s in states:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    n_states_read = sum(1 for s in states if s.get("status") == "read")
    n_states_unread = sum(1 for s in states if s.get("status") == "unread")
    unread_board_s = round(
        sum(s["seconds"] for s in states if s.get("status") == "unread"), 1)

    total_s = sum(c["seconds"] for c in creatives)
    named_s = sum(c["seconds"] for c in creatives if c["brands"])
    todo = [c for c in creatives if c["needs_llm"]]
    paying = {s["name"] for s in sponsors if s["paying"]}
    elapsed = time.perf_counter() - t0

    stats = {
        "creatives": len(creatives),
        "frames_ocr": n_frames_ocr,
        "unlisted_brands": [{"name": c["name"], "seconds": c["seconds"],
                             "frames": c["frames"]} for c in unlisted],
        "named": n_named,
        "needs_llm": len(todo),
        "board_seconds": round(total_s, 1),
        "named_seconds": round(named_s, 1),
        "named_share": round(named_s / total_s, 3) if total_s else 0.0,
        "seconds_per_sponsor": {k: round(v, 1) for k, v in secs.items()},
        "states_read": n_states_read,
        "states_unread": n_states_unread,
        "unread_board_seconds": unread_board_s,
        "carry_states": bool(states),
        "seconds": round(elapsed, 1),
        "llm_calls": llm_calls,
        "llm_cost_usd": round(llm_cost, 4),
    }
    # Cumulative, so the manifest answers "what has this run cost me so far"
    # rather than only what the last invocation cost.
    prev = (store.read_manifest().get("stats", {}).get("identify", {})
            if os.path.exists(store.manifest_path) else {})
    stats["llm_calls_total"] = prev.get("llm_calls_total", 0) + llm_calls
    stats["llm_cost_usd_total"] = round(
        prev.get("llm_cost_usd_total", 0.0) + llm_cost, 4)
    store.write_manifest({"sponsors": sponsors_path, "min_match": min_match},
                         None, stats, stage="identify")

    print(f"\n  named {n_named}/{len(creatives)} groups "
          f"= {named_s:.1f}s of {total_s:.1f}s board time "
          f"({100 * stats['named_share']:.0f} %) in {elapsed:.0f}s")
    if states:
        print(f"  board states: {n_states_read} read, {n_states_unread} unread "
              f"({unread_board_s:.1f}s of board on screen with no advertiser "
              f"identified -- see `coverage`)")
    elif carry_states:
        print("  (no states.jsonl -- run `states` before `identify` to carry a "
              "read across the LED state it was on)")
    print(f"\n  BOARD SECONDS PER SPONSOR (raw, before size/position weighting)")
    for k, v in sorted(secs.items(), key=lambda kv: -kv[1]):
        tag = "" if k in paying else "   (not a paying sponsor)"
        print(f"    {k:<34} {v:6.1f}s{tag}")
    if llm_calls or stats["llm_calls_total"]:
        print(f"\n  API CALLS: {llm_calls} this run (${llm_cost:.4f}) | "
              f"{stats['llm_calls_total']} total for this run dir "
              f"(${stats['llm_cost_usd_total']:.4f})")
    if todo:
        big = sorted(todo, key=lambda c: -c["seconds"])[:5]
        print(f"\n  {len(todo)} groups unread, worth "
              f"{sum(c['seconds'] for c in todo):.1f}s -- for the vision model:")
        for c in big:
            print(f"    #{c['creative']:<3} {c['seconds']:5.1f}s  {c['rep_band']}")
    return store, stats
