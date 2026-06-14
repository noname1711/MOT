import csv
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# COCO class IDs used by this project for vehicle detection.
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Mapping from class IDs to readable labels used in outputs.
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    -1: "predicted"
}

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


# Create a directory if needed and return its string path.
def ensure_dir(path: str | Path) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


# Build timestamp strings for unique output file names.
def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Validate uploaded video file extensions.
def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


# Convert numeric class IDs to readable names.
def get_vehicle_class_name(class_id: int) -> str:
    return VEHICLE_CLASS_NAMES.get(int(class_id), f"class_{class_id}")


# Save dictionaries as UTF-8 formatted JSON files.
def save_json(data: Dict[str, Any], output_path: str) -> str:
    ensure_dir(os.path.dirname(output_path))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return output_path


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
    run_id: str | None = None,
    suffix: str = "metrics"
) -> str:
    ensure_dir(output_dir)

    if run_id is None:
        run_id = timestamp_now()

    output_path = os.path.join(output_dir, f"{run_id}_{suffix}.json")
    return save_json(metrics, output_path)


def safe_copy(src: str, dst: str) -> str:
    ensure_dir(os.path.dirname(dst))
    shutil.copyfile(src, dst)
    return dst


# Convert OpenCV output videos into browser-friendly H.264 MP4.
def convert_video_to_h264(input_path: str, output_path: str) -> None:
    """
    Convert a video to H.264 for stable browser playback.

    If FFmpeg is not available, fall back to copying the temporary file.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video to convert not found: {input_path}")

    ensure_dir(os.path.dirname(output_path))

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        safe_copy(input_path, output_path)


# Read basic metadata from a video file using OpenCV.
def get_video_info(video_path: str) -> Dict[str, Any]:
    import cv2

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_sec": duration_sec
    }


# Read exported tracking TXT rows back into dictionaries.
def read_tracking_txt(tracking_txt_path: str) -> List[Dict[str, Any]]:
    """
    Read tracking records from a CSV-style TXT file.

    Expected format:
        frame,id,x,y,w,h,conf,class,visibility
    """

    if not os.path.exists(tracking_txt_path):
        return []

    records: List[Dict[str, Any]] = []

    with open(tracking_txt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])

                records.append({
                    "frame": int(float(row["frame"])),
                    "track_id": str(row["id"]),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "bbox_xywh": [x, y, w, h],
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "confidence": float(row["conf"]),
                    "class_id": int(float(row["class"])),
                    "visibility": int(float(row["visibility"]))
                })
            except Exception:
                continue

    return records


# Write the standard tracking TXT header.
def write_tracking_header(csv_writer) -> None:
    csv_writer.writerow([
        "frame",
        "id",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "class",
        "visibility"
    ])


# Write all tracks from one frame to the tracking TXT file.
def write_tracking_rows(csv_writer, frame_id: int, tracks: List[Dict[str, Any]]) -> None:
    for track in tracks:
        x, y, w, h = track["bbox_xywh"]

        csv_writer.writerow([
            frame_id,
            track["track_id"],
            int(x),
            int(y),
            int(w),
            int(h),
            round(float(track.get("confidence", 0.0)), 4),
            int(track.get("class_id", -1)),
            int(track.get("visibility", 0))
        ])
