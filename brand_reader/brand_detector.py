# cv2 for video, easyocr to read text in the image, numpy for the maths.
import math
from difflib import SequenceMatcher
import cv2
import numpy as np
import easyocr

# EasyOCR init
reader = easyocr.Reader(['en'], gpu=True)


# 1. AD DETECTION ON A FRAME


# List of brands we want to detect in the images.
KNOWN_BRANDS = [
    "ADIDAS", "NIKE", "PUMA",
    "QATAR", "EMIRATES", "VISIT",
    "ETIHAD", "RAKUTEN", "UNICEF",
    "COCA", "PEPSI", "HEINEKEN",
    "LAYS", "OPPO", "TURKISH",
    "COLA", "FIFA", "WANDA",
    "HYUNDAI", "VISA",
    "HISENSE", "BUD", "FEDEX",
    "GATORADE", "CRYPTO.COM",
    "BET365", "PS5", "UCL",
    "POWERADE", "WORLD CUP", "DELIVER",
    "VIVO", "BUUUUUUD", "CRYPTO",
    "WE DELIVER", "HONDA", "SAP",
    "ENERGIZER"
]

# Without special characters: CRYPTO.COM -> CRYPTO COM.
NORMALIZED_BRANDS = [
    (" ".join("".join(ch if ch.isalnum() else " " for ch in brand.upper()).split()), brand)
    for brand in KNOWN_BRANDS
]


# Strip punctuation and uppercase everything before any comparison.
def normalize_text(text):
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.upper())
    return " ".join(cleaned.split())


# Similarity score between two strings, 0 (nothing in common) to 1 (identical).
def similarity_score(text1, text2):
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


# Find brands in the list whose text is close enough to what the OCR read.
def best_brand_matches(text):
    text_norm = normalize_text(text)
    tokens = text_norm.split()  # ex. "CRYPTO COM LEAGUE" -> ["CRYPTO", "COM", "LEAGUE"]
    matches = []

    for brand_norm, brand in NORMALIZED_BRANDS:
        if not brand_norm:
            continue

        if brand_norm in text_norm:
            # exact match, top score straight away
            score = 1.0
        else:
            # global comparison, whole text vs brand
            score = similarity_score(brand_norm, text_norm)
            # also test word by word, useful when the OCR returns a single word
            for token in tokens:
                score = max(score, similarity_score(brand_norm, token))

            # for multi-word brands (e.g. WORLD CUP), compare consecutive segments of equal length
            brand_words = brand_norm.split()
            if len(brand_words) > 1 and len(tokens) >= len(brand_words):
                for i in range(len(tokens) - len(brand_words) + 1):
                    segment = " ".join(tokens[i:i + len(brand_words)])
                    score = max(score, similarity_score(brand_norm, segment))

        # stricter threshold for short brands (<=4 letters), they resemble many words
        min_similarity = 0.85 if len(brand_norm.replace(" ", "")) <= 4 else 0.65
        if score >= min_similarity:
            matches.append((brand, score))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


# OCR on the whole frame, returns the detected ads
def detect_ads_in_frame(frame):
    """
    Returns a list of (brand_name, bbox)
    bbox = (x1, y1, x2, y2)
    """
    results = reader.readtext(frame)

    detections = []

    for bbox, text, conf in results:
        matches = best_brand_matches(text)
        if not matches:
            continue

        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, x2 = int(min(x_coords)), int(max(x_coords))
        y1, y2 = int(min(y_coords)), int(max(y_coords))

        seen_brands = set()
        for brand, score in matches:
            if brand in seen_brands:
                continue
            seen_brands.add(brand)
            detections.append((brand, (x1, y1, x2, y2)))

    return detections



# 2. SPATIAL METRICS



def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy


# Compute area and distance to the frame centre.
def compute_metrics(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox

    area = max(0, (x2 - x1)) * max(0, (y2 - y1))

    cx, cy = get_bbox_center(bbox)

    dist_center = ((cx - frame_w / 2) ** 2 + (cy - frame_h / 2) ** 2) ** 0.5

    return area, dist_center


# Brightness of the box relative to the whole frame
def compute_lum_relative(bbox, lum_map, mean_lum_frame, frame_w, frame_h):
    """Mean ROI brightness divided by mean frame brightness."""
    x1, y1, x2, y2 = bbox
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(frame_w, x2), min(frame_h, y2)
    roi = lum_map[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 1.0
    return float(roi.mean()) / (mean_lum_frame + 1e-6)


# Overall visibility score
def compute_visibility_score(s_n, dc_n, lum_relative, lambda_c=2.0):
    """
    f(s, dc, lum) = s_n * lum_relative * exp(-lambda_c * dc_n**2)
    """
    gauss_c = math.exp(-lambda_c * dc_n ** 2)
    return s_n * lum_relative * gauss_c



# 3. STATS UPDATE


# Accumulate the metrics of each ad.
def update_brand_stats(brands_stats, detections, frame, fps, frame_step, lambda_c=2.0):
    # real duration covered by this frame
    delta_t = frame_step / fps
    frame_h, frame_w = frame.shape[:2]

    diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
    frame_area = frame_w * frame_h

    # frame luminance map
    frame_f = frame.astype(np.float32)
    lum_map = 0.299 * frame_f[:, :, 2] + 0.587 * frame_f[:, :, 1] + 0.114 * frame_f[:, :, 0]
    mean_lum_frame = float(lum_map.mean())

    # de-duplicated list of brands
    brands_seen = set(brand for brand, _ in detections)

    for brand in brands_seen:
        if brand not in brands_stats:
            # initialise the entry on first sight
            brands_stats[brand] = {
                "time_seconds": 0.0,
                "n_frames_seen": 0,
                "n_detections": 0,
                "total_area": 0.0,
                "total_center_dist": 0.0,
                "total_lum": 0.0,
                "total_visibility": 0.0,
            }

        brands_stats[brand]["time_seconds"] += delta_t
        brands_stats[brand]["n_frames_seen"] += 1

    # accumulate metrics for each detected bbox
    for brand, bbox in detections:
        area, dist_center = compute_metrics(bbox, frame_w, frame_h)
        lum_rel = compute_lum_relative(bbox, lum_map, mean_lum_frame, frame_w, frame_h)

        brands_stats[brand]["n_detections"] += 1
        brands_stats[brand]["total_area"] += area
        brands_stats[brand]["total_center_dist"] += dist_center
        brands_stats[brand]["total_lum"] += lum_rel

        # normalise the metrics
        s_n = area / frame_area
        dc_n = dist_center / diag

        f = compute_visibility_score(s_n, dc_n, lum_rel, lambda_c)
        brands_stats[brand]["total_visibility"] += f * delta_t



# 4. VISUAL ANNOTATION


# Draw a rectangle and the brand name on the frame to visualise detections.
def draw_brand_detections(frame, detections):
    """Draw brand detections on the frame (bbox + label)."""
    COLOR   = (0, 220, 160)   # teal-green
    BG      = (15, 15, 15)
    FONT    = cv2.FONT_HERSHEY_SIMPLEX
    SCALE   = 0.48
    THICK   = 1

    for brand, bbox in detections:
        x1, y1, x2, y2 = bbox

        # Border around the detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)

        (tw, th), _ = cv2.getTextSize(brand, FONT, SCALE, THICK)
        pad = 4

        
        lx = x1
        if y1 - th - pad * 2 >= 0:
            ly_bg_top = y1 - th - pad * 2
            ly_bg_bot = y1
            ly_text   = y1 - pad
        else:
            ly_bg_top = y2
            ly_bg_bot = y2 + th + pad * 2
            ly_text   = y2 + th + pad

        cv2.rectangle(frame, (lx, ly_bg_top), (lx + tw + pad * 2, ly_bg_bot), BG, -1)
        cv2.putText(frame, brand, (lx + pad, ly_text), FONT, SCALE, COLOR, THICK, cv2.LINE_AA)

    return frame



# 5. FINAL STATS


# Compute the final statistics
def finalize_brand_stats(brands_stats):
    final_stats = {}

    for brand, s in brands_stats.items():
        n_det = s["n_detections"]

        final_stats[brand] = {
            "total_time_s": round(s["time_seconds"], 2),
            "mean_area": s["total_area"] / (n_det if n_det else 1),
            "mean_dist_center": s["total_center_dist"] / (n_det if n_det else 1),
            "mean_rel_luminance": s["total_lum"] / (n_det if n_det else 1),
            "V_total": round(s["total_visibility"], 4),
        }

    return final_stats
