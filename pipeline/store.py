"""
Run directory: pipeline stages exchange their artifacts on disk.

    run_<timestamp>/
        manifest.json      config + code revision + video info + stats
        detections.jsonl   one record per sampled frame
        bands/             the reflowed advertising band for each frame
        overlay/           optional: original frame with the region outlined

Each stage reads the previous stage's files instead of holding everything in
memory. Intended consequences:
  - identification or aggregation can be re-run without re-decoding the video;
  - every step stays inspectable after the fact when a number looks wrong;
  - an API can expose a stage without code changes (it reads and writes the same
    directory, possibly on object storage).
"""

import json
import os
import subprocess
from datetime import datetime


def _git_revision():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


class RunStore:
    def __init__(self, out_root, run_name=None, create=True):
        name = run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self.dir = os.path.join(out_root, name)
        self.name = name
        self.bands_dir = os.path.join(self.dir, "bands")
        self.overlay_dir = os.path.join(self.dir, "overlay")
        self.manifest_path = os.path.join(self.dir, "manifest.json")
        self.detections_path = os.path.join(self.dir, "detections.jsonl")
        self.shots_path = os.path.join(self.dir, "shots.jsonl")
        self.annotations_path = os.path.join(self.dir, "annotations.jsonl")
        self.brands_path = os.path.join(self.dir, "brands.json")
        if create:
            os.makedirs(self.bands_dir, exist_ok=True)

    # --- bands ---

    def band_path(self, frame_idx):
        return os.path.join(self.bands_dir, f"{frame_idx:07d}.jpg")

    def band_relpath(self, frame_idx):
        # Always "/" separated: this path goes into the JSONL and must stay
        # readable from another OS (an API served on Linux, for instance).
        return f"bands/{frame_idx:07d}.jpg"

    def overlay_path(self, frame_idx):
        os.makedirs(self.overlay_dir, exist_ok=True)
        return os.path.join(self.overlay_dir, f"{frame_idx:07d}.jpg")

    # --- records ---

    def open_detections(self):
        return open(self.detections_path, "w", encoding="utf-8")

    def read_detections(self):
        if not os.path.exists(self.detections_path):
            return
        with open(self.detections_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # --- shots (segmentation) ---

    def open_shots(self):
        return open(self.shots_path, "w", encoding="utf-8")

    def read_shots(self):
        return list(self._read_jsonl(self.shots_path))

    # --- annotations (human review) ---
    #
    # Appended one record per reviewed frame, never rewritten: the annotator is
    # a long interactive session and must survive being closed at any point.

    def append_annotation(self, rec):
        with open(self.annotations_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def read_annotations(self, history=False):
        """
        Latest annotation per (shot, frame), in first-seen order.

        The file is append-only so an interrupted session loses nothing, which
        means going BACK in the annotator and re-saving a frame leaves an older
        record behind it. Last-wins turns that into an edit; without it a
        corrected frame would be counted twice. `history=True` returns every
        record, including superseded ones.
        """
        rows = list(self._read_jsonl(self.annotations_path))
        if history:
            return rows
        latest = {}
        for r in rows:
            latest[(r["shot"], r["frame"])] = r
        return list(latest.values())

    def read_brands(self):
        if not os.path.exists(self.brands_path):
            return []
        with open(self.brands_path, encoding="utf-8") as f:
            return json.load(f)

    def write_brands(self, brands):
        with open(self.brands_path, "w", encoding="utf-8") as f:
            json.dump(list(brands), f, indent=2, ensure_ascii=False)

    # --- shared ---

    @staticmethod
    def _read_jsonl(path):
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # --- manifest ---

    def write_manifest(self, config, video_info=None, stats=None, stage=None):
        """
        Merge this stage's config and stats into the manifest.

        Stages share a run directory, so a later stage must not erase what an
        earlier one recorded -- `detect` writing over `segment`'s numbers would
        lose the routing that produced its own input.
        """
        cfg = config.to_dict() if hasattr(config, "to_dict") else config
        payload = {}
        if stage and os.path.exists(self.manifest_path):
            try:
                payload = self.read_manifest()
            except Exception:
                payload = {}
        payload.setdefault("run", self.name)
        payload["updated"] = datetime.now().isoformat(timespec="seconds")
        payload.setdefault("created", payload["updated"])
        payload["git"] = _git_revision()
        payload["video"] = video_info or payload.get("video", {})
        if stage:
            payload.setdefault("config", {})[stage] = cfg
            payload.setdefault("stats", {})[stage] = stats or {}
        else:
            payload["config"] = cfg
            payload["stats"] = stats or {}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return payload

    def read_manifest(self):
        with open(self.manifest_path, encoding="utf-8") as f:
            return json.load(f)
