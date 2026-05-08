import csv
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, List


VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    -1: "predicted"
}


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


def get_vehicle_class_name(class_id: int) -> str:
    """
    Lấy tên class vehicle từ class_id.
    """

    return VEHICLE_CLASS_NAMES.get(class_id, f"class_{class_id}")


def get_main_class_id(records: List[Dict[str, Any]]) -> int:
    """
    Lấy class_id chính của một track.

    Vì DeepSORT có thể duy trì track ở các frame không match được YOLO,
    những frame đó có class_id = -1. Khi thống kê, ta bỏ qua -1 và chọn
    class_id thật xuất hiện nhiều nhất trong track.
    """

    class_counts: Dict[int, int] = {}

    for item in records:
        class_id = int(item["class_id"])

        if class_id == -1:
            continue

        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    if not class_counts:
        return -1

    return max(class_counts, key=class_counts.get)


def analyze_tracking_txt(
    tracking_txt_path: str,
    fps: float = 25.0
) -> Dict[str, Any]:
    """
    Phân tích file tracking txt để thống kê các phương tiện đã tracking được.

    File input có dạng:
        frame,id,x,y,w,h,conf,class,visibility

    Với vehicle tracking:
        class = 2  -> car
        class = 3  -> motorcycle
        class = 5  -> bus
        class = 7  -> truck
        class = -1 -> predicted, track được DeepSORT dự đoán nhưng không match YOLO

    Trả về:
        {
            "target": "vehicle",
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

        detection_records = [
            item for item in records
            if int(item["visibility"]) == 1 and int(item["class_id"]) != -1
        ]

        predicted_records = [
            item for item in records
            if int(item["visibility"]) == 0 or int(item["class_id"]) == -1
        ]

        first_frame = min(frames)
        last_frame = max(frames)
        frames_tracked = len(records)
        detection_frames = len(detection_records)
        predicted_frames = len(predicted_records)

        tracked_duration_seconds = frames_tracked / fps
        lifespan_seconds = (last_frame - first_frame + 1) / fps

        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        avg_w = sum(ws) / len(ws)
        avg_h = sum(hs) / len(hs)
        avg_conf = sum(confs) / len(confs)
        avg_visibility = sum(visibilities) / len(visibilities)

        if detection_records:
            detection_confs = [item["conf"] for item in detection_records]
            avg_detection_confidence = sum(detection_confs) / len(detection_confs)
        else:
            avg_detection_confidence = 0.0

        main_class_id = get_main_class_id(records)
        class_name = get_vehicle_class_name(main_class_id)

        total_tracking_rows += frames_tracked

        track_statistics.append({
            "track_id": track_id,
            "class_id": main_class_id,
            "class_name": class_name,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frames_tracked": frames_tracked,
            "detection_frames": detection_frames,
            "predicted_frames": predicted_frames,
            "tracked_duration_seconds": round(tracked_duration_seconds, 2),
            "lifespan_seconds": round(lifespan_seconds, 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_detection_confidence": round(avg_detection_confidence, 4),
            "avg_visibility": round(avg_visibility, 4),
            "avg_bbox": {
                "x": round(avg_x, 2),
                "y": round(avg_y, 2),
                "w": round(avg_w, 2),
                "h": round(avg_h, 2)
            }
        })

    track_statistics = sorted(
        track_statistics,
        key=lambda item: item["frames_tracked"],
        reverse=True
    )

    return {
        "target": "vehicle",
        "tracking_txt_path": tracking_txt_path,
        "fps": fps,
        "total_tracks": len(track_statistics),
        "total_tracking_rows": total_tracking_rows,
        "vehicle_class_names": VEHICLE_CLASS_NAMES,
        "tracks": track_statistics
    }