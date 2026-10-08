"""
The client deliverable: a per-partner sponsor-exposure report for one match.

`aggregate` produces raw per-brand measurements (time, surface, position,
brightness) for whatever the automatic path could read -- in this pilot, the
far-touchline LED perimeter. This stage folds those together with:

  - the club's partner list and, per partner, the surfaces they are entitled to
    (`report_config.json` -> `partners`);
  - a by-eye tally for the surfaces the pipeline does not measure yet -- jersey,
    second advertising line, corner banners, 3D carpets (`manual_tally`);
  - the coverage ledger, so the report states plainly how much of the match was
    actually measured and where it was blind.

Output: `report.html` (self-contained, for the client) and
`report.json` (the same numbers, machine-readable). It applies NO media-value
weighting -- that is a later product decision. This is an exposure measurement.

    python -m pipeline.cli report output_videos/<run>
    python -m pipeline.cli report output_videos/<run> --config pilot/report_config.json
"""

import csv
import html
import json
import os
from collections import defaultdict
from datetime import datetime

from .store import RunStore

# Cyrillic/Latin lookalikes, folded so a brand keyed one way in the CSV matches
# the same brand keyed the other way in the config. Same idea as identify._norm.
_HOMO = str.maketrans({
    "A": "\u0410", "B": "\u0412", "C": "\u0421", "E": "\u0415", "H": "\u041D", "K": "\u041A", "M": "\u041C",
    "O": "\u041E", "P": "\u0420", "T": "\u0422", "X": "\u0425", "Y": "\u0423",
})


def _norm(s):
    return (s or "").upper().translate(_HOMO).replace(" ", "").replace(".", "").strip()


def _mmss(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _dur(seconds):
    """`M:SS (X.X s)` -- mm:ss is readable, the raw seconds keep it a measurement."""
    seconds = max(0.0, float(seconds or 0.0))
    return f"{_mmss(seconds)} ({seconds:.1f} s)"


def _plural(n, one, few, many):
    """English count agreement: 1 surface, 2 surfaces."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _read_by_brand(run_dir):
    """exposure_by_brand.csv -> {norm(brand): row dict with floats}."""
    path = os.path.join(run_dir, "exposure_by_brand.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"No exposure_by_brand.csv in {run_dir} -- run "
                           "`aggregate` first.")
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for k in ("exposure_s", "pct_of_video", "n_frames", "mean_area_pct",
                      "mean_dist_center", "mean_brightness_rel", "mean_panels",
                      "first_seen_s", "last_seen_s", "area_from_whole_band_pct"):
                try:
                    row[k] = float(row[k]) if row.get(k, "") != "" else None
                except (TypeError, ValueError):
                    row[k] = None
            row["_upper_bound"] = str(row.get("time_is_upper_bound", "")).lower() \
                in ("true", "1", "yes")
            out[_norm(row["brand"])] = row
    return out


def _appearances(run_dir, fps, step):
    """
    Count distinct on-screen spells per brand from exposure_detail.csv: runs of
    sampled frames with no gap longer than a few sampling steps.
    """
    path = os.path.join(run_dir, "exposure_detail.csv")
    if not os.path.exists(path) or not step:
        return {}
    frames = defaultdict(set)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                frames[_norm(row["brand"])].add(int(row["frame"]))
            except (TypeError, ValueError, KeyError):
                pass
    gap = step * 3
    out = {}
    for b, fs in frames.items():
        fs = sorted(fs)
        runs = 1
        for a, c in zip(fs, fs[1:]):
            if c - a > gap:
                runs += 1
        out[b] = runs
    return out


def _load_config(run_dir, config_path):
    for cand in (config_path,
                 os.path.join(run_dir, "report_config.json"),
                 os.path.join("pilot", "report_config.json")):
        if cand and os.path.exists(cand):
            with open(cand, encoding="utf-8") as f:
                return json.load(f), cand
    raise RuntimeError("No report_config.json (looked in the run dir and "
                       "pilot/). Pass --config.")


def _centrality(dist_center):
    """0..1 distance-from-centre -> a plain-language position label (English)."""
    if dist_center is None:
        return "-"
    if dist_center < 0.35:
        return "near frame centre"
    if dist_center < 0.6:
        return "middle zone"
    return "frame edge"


def run_report(run_dir, config_path=None, out_path=None, surface_labelns=None):
    """
    `surface_labelns` maps a surface id to a DIFFERENT run directory whose
    `aggregate` output feeds that surface. This is how the second advertising
    line is folded in: it is a separate `detect --band-line 2` run, so
    `report second_line=output_videos/<run>_L2` pulls the `second_line` surface
    from that run's CSV while every other auto surface stays on the base run.
    """
    run_dir = os.path.normpath(run_dir)
    out_root, name = os.path.split(run_dir)
    store = RunStore(out_root, name, create=False)
    manifest = store.read_manifest()
    info = manifest.get("video", {})
    stats = manifest.get("stats", {})
    fps = float(info.get("fps") or 25.0)
    duration = float(info.get("frames") or 0) / fps
    step = int(stats.get("detect", {}).get("step") or 0)

    cfg, cfg_used = _load_config(run_dir, config_path)

    # One (by_brand, appearances) pair per distinct source run.
    surface_labelns = {k: os.path.normpath(v) for k, v in (surface_labelns or {}).items()}
    src_dirs = {run_dir} | set(surface_labelns.values())
    by_brand_src = {d: _read_by_brand(d) for d in src_dirs}
    appears_src = {d: _appearances(d, fps, step) for d in src_dirs}
    by_brand = by_brand_src[run_dir]
    appears = appears_src[run_dir]

    def _auto_src(surf):
        d = surface_labelns.get(surf, run_dir)
        return by_brand_src[d], appears_src[d]

    cov = stats.get("coverage", {})
    cov_share = cov.get("measured_share")
    cov_measured_s = cov.get("measured_s")
    cov_by_status = cov.get("by_status", {})

    surfaces = cfg.get("surfaces", {})
    manual = defaultdict(dict)          # (partner_norm, surface) -> tally row
    for t in cfg.get("manual_tally", []):
        if t.get("_example"):
            continue
        manual[(_norm(t["partner"]), t["surface"])] = t

    # ---- assemble per-partner rows --------------------------------------
    partners_out = []
    for p in cfg.get("partners", []):
        pn = _norm(p["name"])
        rows = []
        total_s = 0.0
        any_measured = False
        # Two entitled surfaces can resolve to the SAME source run -- e.g. a
        # partner on both "LED boards" and "Second advertising line" when no
        # --surface-run splits them, or any surface reading from a merged run
        # that already unioned the lines. The row in that CSV is one
        # measurement of that brand's time, so adding it once per surface
        # double-counts. It put Brand A at 4:51 / 96.96% of the broadcast when
        # its measured time was 2:25. The second surface still gets a row --
        # the club is owed a line for every surface it sold -- but it is
        # labelled as already counted and contributes nothing to the total.
        counted_srcs = set()
        for surf in p.get("surfaces", []):
            sdef = surfaces.get(surf, {"label": surf, "measured_by": "manual"})
            entry = {
                "surface": surf,
                "surface_label": sdef.get("label", surf),
                "measured_by": sdef.get("measured_by", "manual"),
                "note": sdef.get("note", ""),
                "exposure_s": None, "pct": None, "appearances": None,
                "area_pct": None, "position": None, "brightness_rel": None,
                "status": "not measured", "source": "",
                "upper_bound": False, "area_upper_bound": False,
            }
            s_by_brand, s_appears = _auto_src(surf)
            src_key = (surface_labelns.get(surf, run_dir), pn)
            if (sdef.get("measured_by") == "auto" and pn in s_by_brand
                    and src_key in counted_srcs):
                # Same measurement, already counted on an earlier surface.
                r = s_by_brand[pn]
                entry.update(
                    exposure_s=None, pct=None,
                    status="counted in the measurement above",
                    note=(sdef.get("note", "") or "") +
                            (" This surface is measured by the same pass as the previous "
                             "one; its time is not added again, so the same seconds "
                             "are not counted twice."),
                )
                any_measured = True
            elif sdef.get("measured_by") == "auto" and pn in s_by_brand:
                counted_srcs.add(src_key)
                r = s_by_brand[pn]
                entry.update(
                    exposure_s=r["exposure_s"], pct=r["pct_of_video"],
                    appearances=s_appears.get(pn),
                    area_pct=r["mean_area_pct"],
                    position=_centrality(r["mean_dist_center"]),
                    brightness_rel=r["mean_brightness_rel"],
                    status="measured automatically",
                    source=sdef.get("source")
                    or "far-touchline LED, geometry + per-frame reading",
                    upper_bound=r["_upper_bound"],
                    area_upper_bound=(r["area_from_whole_band_pct"] or 0) > 50,
                )
                total_s += r["exposure_s"] or 0.0
                any_measured = True
            elif (pn, surf) in manual:
                t = manual[(pn, surf)]
                entry.update(
                    exposure_s=float(t.get("exposure_s") or 0.0),
                    pct=(100.0 * float(t.get("exposure_s") or 0.0) / duration
                         if duration else None),
                    appearances=t.get("appearances"),
                    status="manual tally",
                    source=t.get("method", "manual review of sampled frames"),
                    note=t.get("notes", "") or sdef.get("note", ""),
                )
                total_s += float(t.get("exposure_s") or 0.0)
                any_measured = True
            rows.append(entry)

        # Auto path read this partner on the LED perimeter but the club did not
        # list `led` among its surfaces -- surface it as a finding, don't drop it.
        auto_used = any(r["measured_by"] == "auto"
                        and r["status"] == "measured automatically"
                        for r in rows)
        if pn in by_brand and not auto_used:
            r = by_brand[pn]
            rows.append({
                "surface": "led_found", "surface_label": "LED boards (detected)",
                "measured_by": "auto",
                "note": "Brand recognised on the LED boards although the club did not "
                           "list them as a surface for this partner.",
                "exposure_s": r["exposure_s"], "pct": r["pct_of_video"],
                "appearances": appears.get(pn), "area_pct": r["mean_area_pct"],
                "position": _centrality(r["mean_dist_center"]),
                "brightness_rel": r["mean_brightness_rel"],
                "status": "measured automatically",
                "source": "far-touchline LED, geometry + per-frame reading",
                "upper_bound": r["_upper_bound"],
                "area_upper_bound": (r["area_from_whole_band_pct"] or 0) > 50,
            })
            total_s += r["exposure_s"] or 0.0
            any_measured = True

        partners_out.append({
            "name": p["name"], "display": p.get("display", p["name"]),
            "rows": rows, "total_s": total_s,
            "total_pct": (100.0 * total_s / duration) if duration else None,
            "any_measured": any_measured,
        })

    # ---- every advertiser detected, partner or not ----------------------
    # The client asked for a full census, not just the five pilot partners.
    # One row per brand, exposure split by the line it was read on. Club /
    # stadium / broadcast-graphic items (paying:false) are excluded here.
    partner_norms = {_norm(p["name"]) for p in cfg.get("partners", [])}
    line_label = {run_dir: "led_far"}
    line_label.update({v: k for k, v in surface_labelns.items()})
    all_ads = {}                      # norm -> {"name","by_line":{},"partner":bool}
    for src, bb in by_brand_src.items():
        lbl = line_label.get(src, os.path.basename(src))
        for nb, r in bb.items():
            if str(r.get("paying", "True")).lower() not in ("true", "1", "yes"):
                continue
            a = all_ads.setdefault(nb, {"name": r["brand"], "by_line": {},
                                        "partner": nb in partner_norms})
            a["by_line"][lbl] = round((a["by_line"].get(lbl, 0.0)
                                       or 0.0) + (r["exposure_s"] or 0.0), 2)
    advertisers = sorted(
        ({"name": a["name"], "pilot_partner": a["partner"],
          "by_line": a["by_line"],
          "total_s": round(sum(a["by_line"].values()), 2),
          "pct_of_video": (round(100.0 * sum(a["by_line"].values()) / duration, 2)
                           if duration else None)}
         for a in all_ads.values()),
        key=lambda a: -a["total_s"])

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run": name, "git": manifest.get("git"),
        "config_used": cfg_used,
        "match": cfg.get("match", {}),
        "video": {"duration_s": round(duration, 1), "fps": fps,
                  "w": info.get("width"), "h": info.get("height")},
        "coverage": {"measured_share": cov_share, "measured_s": cov_measured_s,
                     "by_status": cov_by_status},
        "scope": cfg.get("scope", {}),
        "partners": partners_out,
        "advertisers": advertisers,
    }
    out_path = out_path or os.path.join(run_dir, "report.html")
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_render_html(report))

    # ---- console summary ----------------------------------------------
    print(f"Report   : {out_path}")
    m = report["match"]
    print(f"  match      {m.get('home','?')} - {m.get('away','?')}  "
          f"{m.get('competition','')} {m.get('date','')}".rstrip())
    print(f"  broadcast  {_mmss(duration)}  ({duration:.0f}s)")
    if cov_share is not None:
        print(f"  measured   {cov_share * 100:.0f}% automatically "
              f"({_mmss(cov_measured_s or 0)})")
    print(f"\n  {'partner':<14} {'measured':>9} {'% bcast':>8}  surfaces")
    print("  " + "-" * 52)
    for p in partners_out:
        got = sum(1 for r in p["rows"] if r["status"] != "not measured")
        n = len(p["rows"])
        print(f"  {p['display']:<14} {_mmss(p['total_s']):>9} "
              f"{(p['total_pct'] or 0):7.2f}%  {got}/{n} "
              f"{_plural(n, 'surface', 'surfaces', 'surfaces')}")
    missing = [p["display"] for p in partners_out if not p["any_measured"]]
    if missing:
        print(f"\n  NO DATA YET for: {', '.join(missing)} "
              "-- run aggregate and/or fill manual_tally")

    if advertisers:
        print(f"\n  ALL ADVERTISERS DETECTED ({len(advertisers)}, partner or not)")
        for a in advertisers:
            tag = " *partner" if a["pilot_partner"] else ""
            by = " ".join(f"{k}={v:.0f}s" for k, v in a["by_line"].items())
            print(f"    {a['name']:<32} {_mmss(a['total_s']):>7}  {by}{tag}")

    print(f"\n  Wrote {out_path}")
    print(f"        {os.path.join(run_dir, 'report.json')}")
    return store, {"partners": len(partners_out),
                   "partners_with_data": sum(1 for p in partners_out
                                             if p["any_measured"]),
                   "advertisers": len(advertisers)}


# ---------------------------------------------------------------------------
# HTML rendering. Client-facing text is English; it is data, not code.
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; }
body { font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif;
       color: #1a1a1a; background: #fff; margin: 0; padding: 40px 32px 80px; }
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 40px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid #111; }
h3 { font-size: 15px; margin: 24px 0 8px; }
.sub { color: #666; margin: 0 0 24px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 14px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #e3e3e3;
         vertical-align: top; }
th { background: #f5f5f5; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.tag { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 10px;
       background: #eee; color: #444; white-space: nowrap; }
.tag.auto { background: #e2f0e2; color: #1c5c2e; }
.tag.manual { background: #fdefda; color: #8a5a10; }
.tag.none { background: #f2e2e2; color: #8a2020; }
.note { color: #777; font-size: 12.5px; }
.card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 18px; margin: 14px 0; }
.card h3 { margin-top: 0; }
.total { font-weight: 600; }
.callout { background: #f7f7f7; border-left: 3px solid #999; padding: 12px 16px;
           margin: 16px 0; font-size: 13.5px; color: #444; }
footer { margin-top: 48px; color: #888; font-size: 12px; }
"""


def _fmt(v, nd=2, suffix=""):
    if v is None or v == "":
        return "&mdash;"
    return f"{v:.{nd}f}{suffix}"


def _render_html(rep):
    e = html.escape
    m = rep["match"]
    dur = rep["video"]["duration_s"] or 0
    cov = rep["coverage"]
    parts = []
    parts.append(f"<!-- generated {rep['generated']} run={rep['run']} -->")
    parts.append(f"<style>{_CSS}</style><div class='wrap'>")

    title = f"{m.get('home','')} — {m.get('away','') or '?'}".strip(" —")
    parts.append(f"<h1>Sponsor advertising exposure report</h1>")
    line2 = " · ".join(x for x in [title, m.get("competition", ""),
                                   m.get("round", ""), m.get("date", "")] if x)
    parts.append(f"<p class='sub'>{e(line2)}</p>")

    parts.append("<table>")
    meta = [
        ("Match", title or "&mdash;"),
        ("Competition", m.get("competition", "")),
        ("Date", m.get("date", "")),
        ("Venue", m.get("venue", "")),
        ("Broadcast source", m.get("broadcast_source", "")),
        ("Analysed duration", f"{_mmss(dur)} ({dur:.0f} s)"),
        ("Report prepared", rep["generated"][:10]),
    ]
    for k, v in meta:
        if v:
            parts.append(f"<tr><th style='width:210px'>{e(k)}</th><td>{e(str(v))}</td></tr>")
    parts.append("</table>")

    if m.get("notes"):
        parts.append(f"<p class='note'>{e(m['notes'])}</p>")

    # ---- coverage / honesty -----------------------------------------
    parts.append("<h2>Measurement coverage</h2>")
    if cov.get("measured_share") is not None:
        pct = cov["measured_share"] * 100
        parts.append(
            f"<p>Measured automatically: <b>{pct:.0f}%</b> of the broadcast "
            f"({_mmss(cov.get('measured_s') or 0)} of {_mmss(dur)}). "
            "The rest is shots where no advertising board was in frame, out of focus "
            "or unreadable; those intervals are marked as not measured, not as zero.</p>")
        bs = cov.get("by_status", {})
        if bs:
            parts.append("<table><tr><th>Status</th><th class='num'>Time</th></tr>")
            labels = {"human": "from manual annotation", "auto": "automatic",
                      "scan": "whole-frame read (advertiser recognised)",
                      "tracked": "tracked", "uncovered": "not measured",
                      # Must not read as measured time: the frame WAS read and
                      # named nobody, which is either "nothing was on offer"
                      # (close-up, beauty shot) or "a board was there and could
                      # not be read". The pipeline cannot tell those apart.
                      "scan_blank": "frame read, no advertiser recognised"}
            for k, s in sorted(bs.items(), key=lambda kv: -kv[1]):
                parts.append(f"<tr><td>{labels.get(k, k)}</td>"
                             f"<td class='num'>{_mmss(s)}</td></tr>")
            parts.append("</table>")
            if bs.get("scan_blank"):
                parts.append(
                    "<p class='note'>'Frame read, no advertiser recognised' "
                    "is <b>not</b> measured advertising time. The whole frame was "
                    "looked at but no brand was read: either there was no "
                    "advertising in shot (close-up, panorama) or a board was there "
                    "but stayed unreadable. The two cannot be told apart "
                    "automatically, so this time is counted as neither "
                    "measured nor missed.</p>")
    else:
        parts.append("<p class='note'>Coverage ledger unavailable: the "
                     "<code>coverage</code> stage was not run for this run.</p>")

    scope = rep.get("scope", {})
    if scope.get("automated") or scope.get("manual"):
        parts.append("<div class='callout'>")
        if scope.get("automated"):
            parts.append("<b>Measured automatically:</b> "
                         + e(", ".join(scope["automated"])) + "<br>")
        if scope.get("manual"):
            parts.append("<b>Manual tally / not measured:</b> "
                         + e(", ".join(scope["manual"])))
        parts.append("</div>")

    # ---- summary table --------------------------------------------
    parts.append("<h2>Partner summary</h2>")
    parts.append("<table><tr><th>Partner</th><th class='num'>Exposure</th>"
                 "<th class='num'>% of broadcast</th><th>Surfaces with data</th></tr>")
    for p in rep["partners"]:
        got = sum(1 for r in p["rows"] if r["status"] != "not measured")
        parts.append(
            f"<tr><td>{e(p['display'])}</td>"
            f"<td class='num total'>{_dur(p['total_s'])}</td>"
            f"<td class='num'>{_fmt(p['total_pct'], 2, '%')}</td>"
            f"<td>{got} of {len(p['rows'])}</td></tr>")
    parts.append("</table>")
    parts.append("<p class='note'>'Exposure' is the total on-screen time across all of a "
                 "partner's surfaces that have data. Surfaces without data are not "
                 "included, so the total is a lower bound.</p>")

    # ---- every advertiser detected --------------------------------
    ads = rep.get("advertisers", [])
    if ads:
        line_names = {"led_far": "Far-touchline LED", "second_line": "Second line"}
        lines = []
        for a in ads:
            lines.extend(a["by_line"].keys())
        lines = sorted(set(lines), key=lambda x: (x != "led_far", x))
        parts.append("<h2>All advertisers detected</h2>")
        parts.append("<p class='note'>Every brand recognised on the advertising boards "
                     "in the analysed window, not only the listed partners. Club, "
                     "stadium and broadcast furniture (club name, stadium, score bug "
                     "and similar) is excluded. 'Partner' marks brands from the "
                     "club list.</p>")
        parts.append("<table><tr><th>Brand</th><th>Listed partner</th>"
                     + "".join(f"<th class='num'>{e(line_names.get(l, l))}</th>"
                              for l in lines)
                     + "<th class='num'>Total</th><th class='num'>% bcast</th></tr>")
        for a in ads:
            parts.append(
                "<tr><td>" + e(a["name"]) + "</td>"
                "<td>" + ("yes" if a["pilot_partner"] else "—") + "</td>"
                + "".join(f"<td class='num'>{_dur(a['by_line'].get(l))}</td>"
                          if a["by_line"].get(l) else "<td class='num'>&mdash;</td>"
                          for l in lines)
                + f"<td class='num total'>{_dur(a['total_s'])}</td>"
                + f"<td class='num'>{_fmt(a['pct_of_video'], 2, '%')}</td></tr>")
        parts.append("</table>")

    # ---- per partner --------------------------------------------
    parts.append("<h2>Detail by partner</h2>")
    tagcls = {"auto": "auto", "manual": "manual"}
    for p in rep["partners"]:
        parts.append("<div class='card'>")
        parts.append(f"<h3>{e(p['display'])} "
                     f"<span class='note'>— total {_dur(p['total_s'])} "
                     f"({_fmt(p['total_pct'], 2, '%')} of the broadcast)</span></h3>")
        parts.append("<table><tr><th>Surface</th><th>How measured</th>"
                     "<th class='num'>Time</th><th class='num'>% bcast</th>"
                     "<th class='num'>Appearances</th><th class='num'>Mean size</th>"
                     "<th>Position</th></tr>")
        for r in p["rows"]:
            if r["status"] == "not measured":
                tcls = "none"
            else:
                tcls = tagcls.get(r["measured_by"], "manual")
            area = _fmt(r["area_pct"], 2, "% of frame") if r["area_pct"] is not None else "&mdash;"
            if r["area_upper_bound"]:
                area += " <span class='note'>(whole board)</span>"
            tm = _dur(r["exposure_s"]) if r["exposure_s"] is not None else "&mdash;"
            if r["upper_bound"] and r["exposure_s"]:
                tm += " <span class='note'>(upper bound)</span>"
            parts.append(
                f"<tr><td>{e(r['surface_label'])}</td>"
                f"<td><span class='tag {tcls}'>{e(r['status'])}</span></td>"
                f"<td class='num'>{tm}</td>"
                f"<td class='num'>{_fmt(r['pct'], 2, '%')}</td>"
                f"<td class='num'>{r['appearances'] if r['appearances'] is not None else '&mdash;'}</td>"
                f"<td class='num'>{area}</td>"
                f"<td>{e(r['position'] or '&mdash;') if r['position'] else '&mdash;'}</td></tr>")
            if r.get("source") or r.get("note"):
                extra = " · ".join(x for x in [r.get("source"), r.get("note")] if x)
                parts.append(f"<tr><td colspan='7' class='note'>{e(extra)}</td></tr>")
        parts.append("</table></div>")

    # ---- method --------------------------------------------
    parts.append("<h2>Method and caveats</h2><ul class='note'>")
    parts.append("<li>Far-touchline LED boards and the upper board row (second "
                 "advertising line) are measured automatically: the pitch boundary "
                 "is followed frame by frame, the advertising strip is straightened "
                 "and read, and a brand is credited on the frames where it was "
                 "actually recognised. The second line comes from a separate run "
                 "with the strip shifted up by one board height.</li>")
    parts.append("<li>Shirts, corner banners, 3D carpets and the near board line are "
                 "tallied by hand from sampled frames or marked as not "
                 "measured.</li>")
    if rep.get("scope", {}).get("disclaimer"):
        parts.append(f"<li>{e(rep['scope']['disclaimer'])}</li>")
    parts.append("<li>'Upper bound' on time means the brand was named from one "
                 "reading of a camera view without per-frame confirmation; 'whole "
                 "board' on size means the area is taken from the whole advertising "
                 "strip, not the logo itself.</li>")
    parts.append("</ul>")

    parts.append(f"<footer>Run <code>{e(rep['run'])}</code> · "
                 f"code {e(str(rep.get('git') or '?'))} · "
                 f"generated {e(rep['generated'])}</footer>")
    parts.append("</div>")
    return "\n".join(parts)
