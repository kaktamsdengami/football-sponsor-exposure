import cv2
import os
import csv
import time
from datetime import datetime
from brand_reader.brand_detector import (
    detect_ads_in_frame,
    update_brand_stats,
    finalize_brand_stats,
    draw_brand_detections,
)

# --- CONFIG ---
INPUT_PATH = "input_videos/ucl1.mp4"
FRAME_STEP = 15                 # analyse 1 frame out of FRAME_STEP
SAVE_ANNOTATED_VIDEO = False    # annotated video (debug/demo only): expensive, off by default


def main():

    # OUTPUT DIRECTORY
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"output_videos/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    csv_out_path = f"{run_dir}/brand_stats.csv"
    video_out_path = f"{run_dir}/output_video.mp4"

    # OPEN THE VIDEO
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps} | frames: {total_frames or '?'}")
    print("Streaming video processing...")
    if SAVE_ANNOTATED_VIDEO:
        print("Annotated video: ON (slower)")

    brands_stats = {}
    last_detections = []
    out = None

    t_start = time.perf_counter()
    t_ocr = 0.0
    t_video = 0.0
    n_analyzed = 0
    frame_idx = 0

    # READ THE VIDEO
    while True:
        # grab() advances the stream without decoding the image (fast)
        if not cap.grab():
            break

        analyze = (frame_idx % FRAME_STEP == 0)

        # only decode the image when we actually need it
        if not analyze and not SAVE_ANNOTATED_VIDEO:
            frame_idx += 1
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

        # ANALYSIS (OCR + stats) on 1 frame out of FRAME_STEP
        if analyze:
            t0 = time.perf_counter()
            detections = detect_ads_in_frame(frame)
            t_ocr += time.perf_counter() - t0

            last_detections = detections
            update_brand_stats(brands_stats, detections, frame, fps, FRAME_STEP)
            n_analyzed += 1

        # ANNOTATED VIDEO (optional)
        if SAVE_ANNOTATED_VIDEO:
            t0 = time.perf_counter()
            annotated = draw_brand_detections(frame.copy(), last_detections)
            if out is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
                print("Writing video ->", video_out_path)
            out.write(annotated)
            t_video += time.perf_counter() - t0

        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()

    # FINAL STATISTICS + CSV
    final_stats = finalize_brand_stats(brands_stats)

    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Brand",
            "Total_Time_s",
            "Average_surface",
            "Average_Distance_to_center",
            "Average_Lum_relative",
            "V_total",
        ])
        for brand, s in final_stats.items():
            writer.writerow([
                brand,
                f"{s['total_time_s']:.2f}",
                f"{s['mean_area']:.1f}",
                f"{s['mean_dist_center']:.1f}",
                f"{s['mean_rel_luminance']:.3f}",
                f"{s['V_total']:.4f}",
            ])

    # TIMING
    total = time.perf_counter() - t_start
    print(f"\nResults saved to: {run_dir}")
    print("CSV of statistics")
    if SAVE_ANNOTATED_VIDEO:
        print("Annotated video")

    print("\n--- Timing ---")
    print(f"Frames analysed : {n_analyzed} / {frame_idx} read")
    print(f"Total time      : {total:.1f} s")
    if total > 0:
        print(f"  of which OCR  : {t_ocr:.1f} s ({100 * t_ocr / total:.0f} %)")
        if SAVE_ANNOTATED_VIDEO:
            print(f"  of which video: {t_video:.1f} s ({100 * t_video / total:.0f} %)")
        print(f"Speed           : {n_analyzed / total:.1f} analysed frames/s")
    print("Done.")


if __name__ == "__main__":
    main()
