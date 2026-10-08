"""
Detection of advertising REGIONS in a frame (boards, perimeter LED, jerseys).

This module does NOT read the text or the brand. It only returns bounding boxes
plus a region "type" ("board" or "jersey") to route into the right analysis
track. Brand identification happens further down (embedding + LLM).

Based on YOLO-World (open-vocabulary): classes are described in plain text, so no
training is required to get started. It can later be swapped for a trained
one-class detector without touching the rest of the pipeline.

Note: measured on broadcast wide shots, zero-shot confidences stay at 0.02-0.15,
which is too weak to threshold. board_regions.py supersedes this for the far
touchline; this is kept as a pluggable fallback and for the jersey track.
"""

from ultralytics import YOLOWorld

# Open-vocabulary prompts -> region type.
# Several phrasings per type: YOLO-World is sensitive to wording.
PROMPT_GROUPS = {
    "board": [
        "advertising billboard",
        "led perimeter advertising board",
        "stadium advertising banner",
        "sponsor sign board",
    ],
    "jersey": [
        "sponsor logo on soccer jersey",
        "logo on football shirt",
    ],
}

_ALL_PROMPTS = [p for group in PROMPT_GROUPS.values() for p in group]
_PROMPT_TO_TYPE = {p: t for t, group in PROMPT_GROUPS.items() for p in group}


class AdRegionDetector:
    def __init__(self, model_path="yolov8s-worldv2.pt", conf=0.015, iou=0.5,
                 imgsz=960, device=None, prompts=None):
        self.model = YOLOWorld(model_path)
        self.prompts = list(prompts) if prompts is not None else list(_ALL_PROMPTS)
        self.model.set_classes(self.prompts)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame):
        """
        frame: BGR image (numpy, OpenCV layout).
        Returns a list of dicts:
            {"bbox": (x1, y1, x2, y2), "score": float, "prompt": str,
             "type": "board"|"jersey"}
        """
        res = self.model.predict(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]

        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            prompt = self.prompts[int(box.cls[0])]
            out.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "score": float(box.conf[0]),
                "prompt": prompt,
                "type": _PROMPT_TO_TYPE.get(prompt, "board"),
            })
        out.sort(key=lambda r: r["score"], reverse=True)
        return out
