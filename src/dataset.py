import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from src.utils import get_video_info


DATASET_ROOT = "data/vehicle_tracking"


def list_datasets(dataset_root: str = DATASET_ROOT) -> List[Dict[str, Any]]:
    root = Path(dataset_root)

    if not root.exists():
        return []

    datasets = []

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        info = get_dataset_info(dataset_dir.name, dataset_root=dataset_root)
        datasets.append(info)

    return datasets


def get_dataset_info(dataset_name: str, dataset_root: str = DATASET_ROOT) -> Dict[str, Any]:
    dataset_dir = Path(dataset_root) / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_name}")

    original_videos = list(dataset_dir.glob("*_Original-video.mp4"))
    gt_videos = list(dataset_dir.glob("*_GroundTruth-video.mp4"))
    gt_json_files = list(dataset_dir.glob("*_GroundTruth.json"))
    gt_txt_files = list(dataset_dir.glob("*_GroundTruth.txt"))

    original_video_path = str(original_videos[0]) if original_videos else None
    gt_video_path = str(gt_videos[0]) if gt_videos else None
    gt_json_path = str(gt_json_files[0]) if gt_json_files else None
    gt_txt_path = str(gt_txt_files[0]) if gt_txt_files else None

    video_meta = None
    if original_video_path and os.path.exists(original_video_path):
        video_meta = get_video_info(original_video_path)

    return {
        "name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "original_video_path": original_video_path,
        "groundtruth_video_path": gt_video_path,
        "groundtruth_json_path": gt_json_path,
        "groundtruth_txt_path": gt_txt_path,
        "video_meta": video_meta
    }


def load_ground_truth_records(dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    txt_path = dataset_info.get("groundtruth_txt_path")
    json_path = dataset_info.get("groundtruth_json_path")

    if txt_path and os.path.exists(txt_path):
        records = parse_ground_truth_txt(txt_path)
        if records:
            return records

    if json_path and os.path.exists(json_path):
        return parse_ground_truth_json(json_path)

    return []


def parse_ground_truth_txt(txt_path: str) -> List[Dict[str, Any]]:
    """
    Parser linh hoạt cho GroundTruth.txt.

    Hỗ trợ:
        - CSV có header: frame,id,x,y,w,h,...
        - MOT-like không header: frame,id,x,y,w,h,...
        - Space separated: frame id x y w h ...
    """

    records: List[Dict[str, Any]] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return records

    first_line = lines[0]
    has_header = bool(re.search(r"[a-zA-Z]", first_line))

    if has_header:
        with open(txt_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                normalized = _normalize_gt_row(row)
                if normalized:
                    records.append(normalized)

        return records

    for line in lines:
        parts = re.split(r"[,\s]+", line.strip())
        nums = []

        for part in parts:
            try:
                nums.append(float(part))
            except ValueError:
                pass

        if len(nums) < 6:
            continue

        frame = int(nums[0])
        track_id = str(int(nums[1]))
        x = float(nums[2])
        y = float(nums[3])
        w = float(nums[4])
        h = float(nums[5])

        if w <= 0 or h <= 0:
            continue

        records.append(_build_gt_record(frame, track_id, x, y, w, h))

    return records


def parse_ground_truth_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items: List[Dict[str, Any]] = []

    def walk(obj):
        if isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, dict):
            keys = set(obj.keys())

            has_frame = bool(keys & {"frame", "frame_id", "frameId"})
            has_track = bool(keys & {"id", "track_id", "trackId"})
            has_bbox = bool(keys & {"bbox", "box", "bounding_box", "bbox_xywh"})

            if has_frame and has_bbox:
                raw_items.append(obj)

            for value in obj.values():
                if isinstance(value, (list, dict)):
                    walk(value)

    walk(data)

    records: List[Dict[str, Any]] = []

    for item in raw_items:
        frame = item.get("frame_id", item.get("frame", item.get("frameId")))
        track_id = item.get("track_id", item.get("trackId", item.get("id", -1)))
        bbox = item.get("bbox", item.get("box", item.get("bounding_box", item.get("bbox_xywh"))))

        if frame is None or bbox is None:
            continue

        try:
            frame = int(frame)
            track_id = str(int(track_id)) if str(track_id).replace(".", "", 1).isdigit() else str(track_id)

            if isinstance(bbox, dict):
                x = float(bbox.get("x", bbox.get("left", bbox.get("xmin", 0))))
                y = float(bbox.get("y", bbox.get("top", bbox.get("ymin", 0))))
                w = bbox.get("w", bbox.get("width"))
                h = bbox.get("h", bbox.get("height"))

                if w is None or h is None:
                    xmax = float(bbox.get("xmax", bbox.get("right", 0)))
                    ymax = float(bbox.get("ymax", bbox.get("bottom", 0)))
                    w = xmax - x
                    h = ymax - y

                w = float(w)
                h = float(h)

            elif isinstance(bbox, list) and len(bbox) >= 4:
                x = float(bbox[0])
                y = float(bbox[1])
                w = float(bbox[2])
                h = float(bbox[3])
            else:
                continue

            if w <= 0 or h <= 0:
                continue

            records.append(_build_gt_record(frame, track_id, x, y, w, h))

        except Exception:
            continue

    return records


def _normalize_gt_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    frame_key = _find_key(row, ["frame", "frame_id", "frameId"])
    id_key = _find_key(row, ["id", "track_id", "trackId"])
    x_key = _find_key(row, ["x", "left", "xmin"])
    y_key = _find_key(row, ["y", "top", "ymin"])
    w_key = _find_key(row, ["w", "width"])
    h_key = _find_key(row, ["h", "height"])

    if not all([frame_key, id_key, x_key, y_key, w_key, h_key]):
        return None

    try:
        frame = int(float(row[frame_key]))
        track_id = str(row[id_key])
        x = float(row[x_key])
        y = float(row[y_key])
        w = float(row[w_key])
        h = float(row[h_key])

        if w <= 0 or h <= 0:
            return None

        return _build_gt_record(frame, track_id, x, y, w, h)

    except Exception:
        return None


def _find_key(row: Dict[str, Any], candidates: List[str]) -> str | None:
    lowered = {key.lower(): key for key in row.keys()}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def _build_gt_record(frame: int, track_id: str, x: float, y: float, w: float, h: float) -> Dict[str, Any]:
    return {
        "frame": int(frame),
        "track_id": str(track_id),
        "x": float(x),
        "y": float(y),
        "w": float(w),
        "h": float(h),
        "bbox_xywh": [float(x), float(y), float(w), float(h)],
        "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)]
    }


def summarize_ground_truth(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "total_groundtruth_boxes": 0,
            "total_groundtruth_tracks": 0,
            "annotated_frames": 0,
            "avg_objects_per_frame": 0,
            "max_objects_per_frame": 0,
            "avg_track_length_frames": 0
        }

    frames = {}
    tracks = {}

    for item in records:
        frame = int(item["frame"])
        track_id = str(item["track_id"])

        frames.setdefault(frame, 0)
        frames[frame] += 1

        tracks.setdefault(track_id, set())
        tracks[track_id].add(frame)

    track_lengths = [len(v) for v in tracks.values()]

    return {
        "total_groundtruth_boxes": len(records),
        "total_groundtruth_tracks": len(tracks),
        "annotated_frames": len(frames),
        "avg_objects_per_frame": round(sum(frames.values()) / len(frames), 4) if frames else 0,
        "max_objects_per_frame": max(frames.values()) if frames else 0,
        "avg_track_length_frames": round(sum(track_lengths) / len(track_lengths), 4) if track_lengths else 0
    }
