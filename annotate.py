"""
Manual review tool: name the brands in the shots the machine cannot read.

Reads a run produced by `python -m pipeline.cli segment`, builds the review
queue (every shot routed "review", plus the random QA sample of automatic
shots), and walks you through a couple of frames per shot.

You outline the logo, search for the brand, pick the surface. The duration comes
from the shot boundary, the size and position come from the outline, so those
are never typed.

    python annotate.py output_videos/seg_ucl1
    python annotate.py output_videos/seg_ucl1 --min-dur 1.0 --redo

Shapes
    A perimeter board seen at an angle is a PARALLELOGRAM, not a rectangle. Its
    axis-aligned bounding box can be twice the true area, which would inflate
    the size weighting in the report. So every outline is stored as a 4-point
    polygon: drag gives you a rectangle (fast, right for jerseys), `w` switches
    to 4-corner mode for slanted boards.

Keys
    drag            rectangle outline
    w               4-corner mode: click each corner (u removes the last)
    then            type to search the brand -> ENTER, then a letter for surface
    c               copy the previous frame's boxes (blocked after a drastic cut)
    u / d           undo last box / clear this frame
    x               nothing visible here  (an explicit empty, not a skip)
    SPACE or n      save this frame, next
    b               back one frame
    s               skip the rest of this shot
    q or ESC        save and quit

Progress is appended to annotations.jsonl as you go, so closing the window at
any point is safe -- restarting resumes where you stopped.
"""

import argparse
import os
import sys
from difflib import SequenceMatcher

import cv2
import numpy as np

from pipeline.store import RunStore
from pipeline.timeline import poly_area, poly_bbox, rect_poly

# key -> surface. These are the placement types aggregation weights differently.
SURFACES = {
    ord("l"): "perimeter_led",
    ord("r"): "ribbon",
    ord("j"): "jersey_front",
    ord("k"): "sleeve",
    ord("h"): "shorts",
    ord("m"): "manufacturer",
    ord("p"): "backdrop",
    ord("e"): "bench_board",
    ord("g"): "graphic",
    ord("t"): "bottle",
    ord("o"): "other",
}
DEFAULT_SURFACE = {"close": "jersey_front", "medium": "perimeter_led",
                   "wide_play": "perimeter_led", "other": "other"}

MAX_MATCHES = 7
# Constant, so the window does not resize when the panel switches mode. Tall
# enough for the brand search, which is the tallest layout.
PANEL_H = 156
_BOX = (0, 220, 160)
_PENDING = (60, 200, 255)
_WARN = (60, 60, 235)
_DIM = (170, 170, 170)
_SEL = (120, 255, 200)

ENTER, BKSP, TAB, ESC = 13, 8, 9, 27


def screen_size(default=(1920, 1080)):
    """Usable screen size in the units OpenCV will lay the window out in."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        wh = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return wh if wh[0] > 200 and wh[1] > 200 else default
    except Exception:
        return default


def _text(img, s, xy, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def rank_brands(query, brands):
    """Brands matching `query`, best first: prefix, then substring, then fuzzy."""
    q = query.strip().upper()
    if not q:
        return list(brands)
    scored = []
    for b in brands:
        if b.startswith(q):
            s = 3.0 + len(q) / max(len(b), 1)
        elif q in b:
            s = 2.0 + len(q) / max(len(b), 1)
        else:
            r = SequenceMatcher(None, q, b).ratio()
            if r < 0.45:
                continue
            s = r
        scored.append((s, b))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [b for _, b in scored]


class Annotator:
    def __init__(self, store, video, shots, args):
        self.store = store
        self.args = args
        self.shots = shots
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        self.brands = store.read_brands()
        self.done = self._done_frames()
        self.prev_boxes = []

        self.boxes = []
        self.mode = "idle"            # idle | quad | brand | surface
        self.pending = None           # polygon awaiting brand + surface
        self.pending_brand = None
        self.corners = []             # 4-corner mode, in frame coords
        self.drag_from = self.drag_to = self.cursor = None
        self.query = ""
        self.sel = 0
        self.scale = 1.0
        self.frame_wh = (1, 1)
        self.win = "annotate"

    # --- persistence ----------------------------------------------------

    def _done_frames(self):
        if self.args.redo:
            return set()
        return {(r["shot"], r["frame"]) for r in self.store.read_annotations()}

    def _save(self, shot, frame_idx, t):
        self.store.append_annotation({
            "shot": shot["shot"], "frame": int(frame_idx), "t": round(float(t), 3),
            "shot_type": shot["type"], "route": shot["route"], "qa": shot["qa"],
            "empty": not self.boxes,
            "boxes": [dict(b) for b in self.boxes],
        })
        self.done.add((shot["shot"], frame_idx))

    def _commit_box(self, surface):
        poly = self.pending
        self.boxes.append({
            "brand": self.pending_brand,
            "surface": surface,
            "poly": poly,
            "bbox": poly_bbox(poly),          # convenience for readers
            "area_px": round(poly_area(poly), 1),
        })
        self.pending = self.pending_brand = None
        self.mode = "idle"

    # --- coordinates ----------------------------------------------------

    def _to_frame(self, x, y):
        """Canvas point -> frame coords, clamped to the image (the info panel
        sits below it, so an overshooting drag must not leave the frame)."""
        fw, fh = self.frame_wh
        return [max(0, min(int(x / self.scale), fw - 1)),
                max(0, min(int(y / self.scale), fh - 1))]

    def _to_disp(self, pt):
        return (int(pt[0] * self.scale), int(pt[1] * self.scale))

    def _on_mouse(self, event, x, y, flags, _param):
        self.cursor = (x, y)
        if self.mode == "quad":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.corners.append(self._to_frame(x, y))
                if len(self.corners) == 4:
                    self.pending = self.corners
                    self.corners = []
                    self._open_brand_search()
            return
        if self.mode != "idle":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_from = self.drag_to = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_from:
            self.drag_to = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_from:
            x0, y0 = self.drag_from
            self.drag_from = self.drag_to = None
            if abs(x - x0) < 6 or abs(y - y0) < 6:
                return
            a = self._to_frame(min(x0, x), min(y0, y))
            b = self._to_frame(max(x0, x), max(y0, y))
            self.pending = rect_poly(a[0], a[1], b[0], b[1])
            self._open_brand_search()

    # --- brand search ---------------------------------------------------

    def _open_brand_search(self):
        self.query = ""
        self.sel = 0
        self.mode = "brand"

    def _options(self):
        """Ranked matches plus a 'create' entry, so one list covers both."""
        opts = rank_brands(self.query, self.brands)[:MAX_MATCHES]
        q = self.query.strip().upper()
        if q and q not in opts:
            opts.append(("NEW", q))
        return opts

    def _pick(self, opt):
        name = opt[1] if isinstance(opt, tuple) else opt
        if name not in self.brands:
            self.brands.append(name)
            self.store.write_brands(self.brands)
        self.pending_brand = name
        self.mode = "surface"

    # --- drawing --------------------------------------------------------

    def _make_window(self):
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.win, 20, 20)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _draw_polys(self, disp):
        for b in self.boxes:
            poly = b.get("poly") or rect_poly(*b["bbox"])
            pts = np.array([self._to_disp(p) for p in poly], np.int32)
            cv2.polylines(disp, [pts], True, _BOX, 2, cv2.LINE_AA)
            x, y = pts[:, 0].min(), pts[:, 1].min()
            _text(disp, f"{b['brand']} / {b['surface']}", (x + 4, max(14, y - 6)), _BOX, 0.5)

        if self.pending:
            pts = np.array([self._to_disp(p) for p in self.pending], np.int32)
            cv2.polylines(disp, [pts], True, _PENDING, 2, cv2.LINE_AA)
        elif self.mode == "quad" and self.corners:
            pts = [self._to_disp(p) for p in self.corners]
            for p in pts:
                cv2.circle(disp, p, 4, _PENDING, -1)
            if len(pts) > 1:
                cv2.polylines(disp, [np.array(pts, np.int32)], False, _PENDING, 2, cv2.LINE_AA)
            if self.cursor:
                cv2.line(disp, pts[-1], self.cursor, _PENDING, 1, cv2.LINE_AA)
        elif self.drag_from and self.drag_to:
            cv2.rectangle(disp, self.drag_from, self.drag_to, _PENDING, 2)

    def _panel(self, dw, shot, k, n_frames, qi, n_queue):
        panel = np.full((PANEL_H, dw, 3), 24, np.uint8)
        tag = ("QA spot-check" if shot["qa"]
               else "PARTIAL - bad frames only" if shot.get("partial")
               else shot["route"].upper())
        _text(panel, f"[{qi}/{n_queue}] shot {shot['shot']}  {shot['type']}  "
                     f"{shot['t_start']:.1f}-{shot['t_end']:.1f}s ({shot['dur_s']:.1f}s)  {tag}"
                     f"   frame {k + 1}/{n_frames}   {len(self.boxes)} box(es)",
              (12, 24), (255, 255, 255), 0.55)

        if self.mode == "brand":
            _text(panel, f"BRAND: {self.query}_", (12, 50), _PENDING, 0.6)
            opts = self._options()
            for i, o in enumerate(opts[:MAX_MATCHES + 1]):
                is_new = isinstance(o, tuple)
                label = f"+ create \"{o[1]}\"" if is_new else o
                col = _SEL if i == self.sel else (_WARN if is_new else _DIM)
                mark = ">" if i == self.sel else " "
                _text(panel, f"{mark} {label}", (24 + (i % 3) * (dw // 3), 74 + (i // 3) * 22),
                      col, 0.5)
            _text(panel, "type to search  |  TAB next  |  ENTER pick  |  ESC cancel",
                  (12, 74 + ((len(opts) - 1) // 3 + 1) * 22 + 4), _DIM, 0.45)
            return panel

        if self.mode == "surface":
            d = DEFAULT_SURFACE.get(shot["type"], "other")
            _text(panel, f"BRAND = {self.pending_brand}   now pick the SURFACE:",
                  (12, 50), _PENDING, 0.55)
            _text(panel, "l led   r ribbon   j jersey   k sleeve   h shorts   m manufacturer",
                  (12, 74), _DIM, 0.5)
            _text(panel, f"p backdrop   e bench   g graphic   t bottle   o other"
                         f"   |   ENTER = {d}", (12, 96), _DIM, 0.5)
            return panel

        if self.mode == "quad":
            _text(panel, f"4-CORNER MODE: click corner {len(self.corners) + 1} of 4"
                         f"   (u = remove last, w or ESC = back to rectangle)",
                  (12, 50), _PENDING, 0.55)
            return panel

        carry = shot["carry_ok"] and self.prev_boxes
        _text(panel, f"cut severity {shot['cut_severity']:.2f}"
                     + ("" if shot["carry_ok"] else "   DRASTIC CUT - annotate from scratch"),
              (12, 50), _DIM if shot["carry_ok"] else _WARN, 0.5)
        _text(panel, "drag = rectangle    w = 4-corner (slanted boards)    "
                     + ("c copy prev    " if carry else "") + "u undo    d clear",
              (12, 74), _DIM, 0.5)
        _text(panel, "x nothing here    SPACE next    b back    s skip shot    q quit",
              (12, 96), _DIM, 0.5)
        return panel

    def _render(self, frame, shot, k, n_frames, qi, n_queue):
        h, w = frame.shape[:2]
        self.frame_wh = (w, h)
        # Fit BOTH dimensions: the panel sits under the image, so sizing on
        # width alone pushes the instructions off the bottom of the screen.
        self.scale = min(1.0, self.args.max_width / w,
                         max(self.args.max_height - PANEL_H, 200) / h)
        disp = (cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_AREA) if self.scale < 1.0 else frame.copy())
        self._draw_polys(disp)
        panel = self._panel(disp.shape[1], shot, k, n_frames, qi, n_queue)
        return np.vstack([disp, panel])

    # --- main loop ------------------------------------------------------

    def run(self):
        self._make_window()
        qi = 0
        while qi < len(self.shots):
            shot = self.shots[qi]
            frames = [f for f in shot["frames"] if (shot["shot"], f) not in self.done]
            if frames and self._do_shot(shot, frames, qi + 1, len(self.shots)) == "quit":
                break
            qi += 1

        self.cap.release()
        cv2.destroyAllWindows()
        n = len(self.store.read_annotations())
        print(f"\nSaved {n} annotated frames -> {self.store.annotations_path}")
        print(f"Brands ({len(self.brands)}): {', '.join(self.brands)}")

    def _do_shot(self, shot, frames, qi, n_queue):
        k = 0
        while 0 <= k < len(frames):
            fi = frames[k]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = self.cap.read()
            if not ok:
                return "next"
            self.boxes = []
            self.mode, self.pending, self.pending_brand = "idle", None, None
            self.corners = []
            res = self._edit_frame(frame, shot, k, len(frames), qi, n_queue, fi)
            if res == "quit":
                return "quit"
            if res == "skip":
                return "next"
            k = max(0, k - 1) if res == "back" else k + 1
        return "next"

    def _edit_frame(self, frame, shot, k, n_frames, qi, n_queue, fi):
        t = fi / max(self.cap.get(cv2.CAP_PROP_FPS) or 25.0, 1e-6)
        while True:
            cv2.imshow(self.win, self._render(frame, shot, k, n_frames, qi, n_queue))
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
            if key == 255:
                continue

            if self.mode == "brand":
                opts = self._options()
                if key == ESC:
                    self.mode, self.pending = "idle", None
                elif key == TAB and opts:
                    self.sel = (self.sel + 1) % len(opts)
                elif key == ENTER and opts:
                    self._pick(opts[min(self.sel, len(opts) - 1)])
                elif key == BKSP:
                    self.query, self.sel = self.query[:-1], 0
                elif 32 <= key < 127 and len(self.query) < 28:
                    self.query, self.sel = self.query + chr(key).upper(), 0
                continue

            if self.mode == "surface":
                if key == ESC:
                    self.mode, self.pending, self.pending_brand = "idle", None, None
                elif key == ENTER:
                    self._commit_box(DEFAULT_SURFACE.get(shot["type"], "other"))
                elif key in SURFACES:
                    self._commit_box(SURFACES[key])
                continue

            if self.mode == "quad":
                if key in (ESC, ord("w")):
                    self.mode, self.corners = "idle", []
                elif key == ord("u") and self.corners:
                    self.corners.pop()
                continue

            # idle
            if key in (ord("q"), ESC):
                self._save(shot, fi, t)
                return "quit"
            if key in (ord(" "), ord("n")):
                self._save(shot, fi, t)
                self.prev_boxes = [dict(b) for b in self.boxes]
                return "next"
            if key == ord("x"):
                self.boxes = []
                self._save(shot, fi, t)
                self.prev_boxes = []
                return "next"
            if key == ord("b"):
                return "back"
            if key == ord("s"):
                return "skip"
            if key == ord("w"):
                self.mode, self.corners = "quad", []
            elif key == ord("u") and self.boxes:
                self.boxes.pop()
            elif key == ord("d"):
                self.boxes = []
            elif key == ord("c") and shot["carry_ok"] and self.prev_boxes:
                self.boxes = [dict(b) for b in self.prev_boxes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a run directory produced by pipeline.cli segment")
    ap.add_argument("--video", default=None, help="override the video from the manifest")
    ap.add_argument("--min-dur", type=float, default=0.0,
                    help="skip review shots shorter than this (little exposure in them)")
    ap.add_argument("--order", choices=["time", "duration"], default="time",
                    help="'duration' does the highest-value shots first, but breaks "
                         "carry-forward (which assumes chronological order)")
    ap.add_argument("--fit", type=float, default=0.82,
                    help="share of the screen the window may occupy")
    ap.add_argument("--max-width", type=int, default=0, help="0 = derive from --fit")
    ap.add_argument("--max-height", type=int, default=0, help="0 = derive from --fit")
    ap.add_argument("--redo", action="store_true", help="re-annotate frames already done")
    ap.add_argument("--auto-conf", type=float, default=0.55,
                    help="band confidence at which detect covers a review frame for you")
    ap.add_argument("--auto-window", type=float, default=0.6,
                    help="seconds around a good detection it is taken to cover")
    args = ap.parse_args()

    sw, sh = screen_size()
    if not args.max_width:
        args.max_width = int(sw * args.fit)
    if not args.max_height:
        args.max_height = int(sh * args.fit)

    out_root, name = os.path.split(os.path.normpath(args.run_dir))
    store = RunStore(out_root, name, create=False)
    if not os.path.exists(store.shots_path):
        sys.exit(f"No shots.jsonl in {args.run_dir} -- run `pipeline.cli segment` first.")

    manifest = store.read_manifest()
    video = args.video or manifest.get("video", {}).get("path") \
        or manifest.get("config", {}).get("segment", {}).get("video")
    fps = float(manifest.get("video", {}).get("fps") or 25.0)

    shots = [s for s in store.read_shots()
             if (s["route"] == "review" or s["qa"] or s.get("partial"))
             and s["dur_s"] >= args.min_dur]

    # If detect has already run, drop review frames it read cleanly: the human
    # only needs the frames the automatic path could not measure. This is what
    # keeps a wide shot that is clean for 12s of its 20s from costing full
    # review time.
    if not args.redo:
        good = [d["t"] for d in store.read_detections()
                if d.get("band") is not None and float(d.get("conf", 0)) >= args.auto_conf]
        if good:
            tol = args.auto_window / 2.0
            trimmed, kept = [], 0
            for s in shots:
                if s["type"] not in ("wide_play", "medium"):
                    trimmed.append(s)
                    kept += len(s["frames"])
                    continue
                need = [f for f in s["frames"]
                        if not any(abs(f / fps - g) <= tol for g in good)]
                if need:
                    s = dict(s, frames=need)
                    trimmed.append(s)
                    kept += len(need)
            dropped = sum(len(s["frames"]) for s in shots) - kept
            shots = trimmed
            print(f"detect covered {dropped} review frame(s); {kept} still need you")

    if args.order == "duration":
        shots.sort(key=lambda s: -s["dur_s"])

    done = {(r["shot"], r["frame"]) for r in store.read_annotations()}
    todo = sum(1 for s in shots for f in s["frames"] if (s["shot"], f) not in done)

    print(f"Run    : {store.dir}")
    print(f"Video  : {video}")
    print(f"Screen : {sw}x{sh}  ->  window fits {args.max_width}x{args.max_height}")
    print(f"Queue  : {len(shots)} shots, {todo} frames left"
          + (f" ({len(done)} already done)" if done and not args.redo else ""))
    if not todo and not args.redo:
        print("Nothing left. Use --redo to go through them again.")
        return
    print("Keys   : drag=rect  w=4-corner  type=search brand  ENTER  letter=surface  SPACE=next\n")

    Annotator(store, video, shots, args).run()


if __name__ == "__main__":
    main()
