import csv
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, List


def save_metrics(metrics: Dict[str, Any], output_dir: str = "results/metrics") -> str:
    """
    Lưu thông tin metrics sau khi xử lý video ra file JSON.
    """

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(output_dir, f"{timestamp}_metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    return metrics_path


def save_tracking_statistics(
    statistics: Dict[str, Any],
    output_dir: str = "results/metrics",
    timestamp: str | None = None
) -> str:
    """
    Lưu thống kê tracking ra file JSON.
    """

    os.makedirs(output_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    statistics_path = os.path.join(output_dir, f"{timestamp}_tracking_statistics.json")

    with open(statistics_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=4)

    return statistics_path


def convert_video_to_h264(input_path: str, output_path: str) -> None:
    """
    Convert video sang H.264 để trình duyệt phát được ổn định.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy video cần convert: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def analyze_tracking_txt(
    tracking_txt_path: str,
    fps: float = 25.0
) -> Dict[str, Any]:
    """
    Phân tích file tracking txt để thống kê các đối tượng đã tracking được.

    File input có dạng:
        frame,id,x,y,w,h,conf,class,visibility

    Trả về:
        {
            "total_tracks": ...,
            "total_tracking_rows": ...,
            "tracks": [...]
        }
    """

    if not os.path.exists(tracking_txt_path):
        raise FileNotFoundError(f"Không tìm thấy tracking txt: {tracking_txt_path}")

    if fps <= 0:
        fps = 25.0

    tracks: Dict[str, List[Dict[str, Any]]] = {}

    with open(tracking_txt_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {"frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"}
        current_columns = set(reader.fieldnames or [])

        if not required_columns.issubset(current_columns):
            raise ValueError(
                "File tracking txt không đúng format. "
                "Cần header: frame,id,x,y,w,h,conf,class,visibility"
            )

        for row in reader:
            try:
                track_id = str(row["id"])
                frame = int(float(row["frame"]))
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])
                conf = float(row["conf"])
                class_id = int(float(row["class"]))
                visibility = float(row["visibility"])
            except (ValueError, KeyError):
                continue

            if track_id not in tracks:
                tracks[track_id] = []

            tracks[track_id].append({
                "frame": frame,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf,
                "class_id": class_id,
                "visibility": visibility
            })

    track_statistics = []
    total_tracking_rows = 0

    for track_id, records in tracks.items():
        if not records:
            continue

        records = sorted(records, key=lambda item: item["frame"])

        frames = [item["frame"] for item in records]
        xs = [item["x"] for item in records]
        ys = [item["y"] for item in records]
        ws = [item["w"] for item in records]
        hs = [item["h"] for item in records]
        confs = [item["conf"] for item in records]
        visibilities = [item["visibility"] for item in records]

        first_frame = min(frames)
        last_frame = max(frames)
        frames_tracked = len(records)

        # Số giây thực sự được tracker ghi nhận.
        tracked_duration_seconds = frames_tracked / fps

        # Khoảng thời gian từ frame đầu đến frame cuối.
        lifespan_seconds = (last_frame - first_frame + 1) / fps

        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        avg_w = sum(ws) / len(ws)
        avg_h = sum(hs) / len(hs)
        avg_conf = sum(confs) / len(confs)
        avg_visibility = sum(visibilities) / len(visibilities)

        class_id = records[0]["class_id"]
        class_name = "person" if class_id == 0 else str(class_id)

        total_tracking_rows += frames_tracked

        track_statistics.append({
            "track_id": track_id,
            "class_id": class_id,
            "class_name": class_name,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frames_tracked": frames_tracked,
            "tracked_duration_seconds": round(tracked_duration_seconds, 2),
            "lifespan_seconds": round(lifespan_seconds, 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_visibility": round(avg_visibility, 4),
            "avg_bbox": {
                "x": round(avg_x, 2),
                "y": round(avg_y, 2),
                "w": round(avg_w, 2),
                "h": round(avg_h, 2)
            }
        })

    # Sắp xếp ID theo số frame xuất hiện nhiều nhất
    track_statistics = sorted(
        track_statistics,
        key=lambda item: item["frames_tracked"],
        reverse=True
    )

    return {
        "tracking_txt_path": tracking_txt_path,
        "fps": fps,
        "total_tracks": len(track_statistics),
        "total_tracking_rows": total_tracking_rows,
        "tracks": track_statistics
    }