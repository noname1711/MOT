from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.utils import get_vehicle_class_name, read_tracking_txt


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def analyze_tracking_txt(tracking_txt_path: str, fps: float = 25.0) -> Dict[str, Any]:
    records = read_tracking_txt(tracking_txt_path)

    if fps <= 0:
        fps = 25.0

    tracks = defaultdict(list)

    for item in records:
        tracks[str(item["track_id"])].append(item)

    track_summaries = []
    total_visible_rows = 0
    total_predicted_rows = 0

    for track_id, items in tracks.items():
        items = sorted(items, key=lambda x: x["frame"])

        frames = [int(x["frame"]) for x in items]
        confs = [float(x["confidence"]) for x in items]
        visible_items = [x for x in items if int(x["visibility"]) == 1 and int(x["class_id"]) != -1]
        predicted_items = [x for x in items if int(x["visibility"]) == 0 or int(x["class_id"]) == -1]

        total_visible_rows += len(visible_items)
        total_predicted_rows += len(predicted_items)

        class_id = _get_main_class_id(items)

        first_frame = min(frames)
        last_frame = max(frames)
        frames_tracked = len(items)

        duration_seconds = frames_tracked / fps
        lifespan_seconds = (last_frame - first_frame + 1) / fps

        avg_conf = sum(confs) / len(confs) if confs else 0

        track_summaries.append({
            "track_id": track_id,
            "class_id": class_id,
            "class_name": get_vehicle_class_name(class_id),
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frames_tracked": frames_tracked,
            "detection_frames": len(visible_items),
            "predicted_frames": len(predicted_items),
            "tracked_duration_seconds": round(duration_seconds, 3),
            "lifespan_seconds": round(lifespan_seconds, 3),
            "avg_confidence": round(avg_conf, 4)
        })

    track_summaries = sorted(
        track_summaries,
        key=lambda x: x["frames_tracked"],
        reverse=True
    )

    total_frames_with_tracks = len(set(item["frame"] for item in records))
    objects_per_frame = defaultdict(int)

    for item in records:
        objects_per_frame[int(item["frame"])] += 1

    track_lengths = [item["frames_tracked"] for item in track_summaries]
    short_tracks = [length for length in track_lengths if length <= max(3, int(fps * 0.5))]

    return {
        "tracking_txt_path": tracking_txt_path,
        "fps": fps,
        "total_tracking_rows": len(records),
        "total_visible_rows": total_visible_rows,
        "total_predicted_rows": total_predicted_rows,
        "total_tracks": len(track_summaries),
        "frames_with_tracks": total_frames_with_tracks,
        "avg_objects_per_frame": round(sum(objects_per_frame.values()) / len(objects_per_frame), 4) if objects_per_frame else 0,
        "max_objects_per_frame": max(objects_per_frame.values()) if objects_per_frame else 0,
        "avg_track_length_frames": round(sum(track_lengths) / len(track_lengths), 4) if track_lengths else 0,
        "short_track_count": len(short_tracks),
        "short_track_ratio": round(len(short_tracks) / len(track_lengths), 4) if track_lengths else 0,
        "tracks": track_summaries
    }


def build_upload_metrics(
    tracking_txt_path: str,
    process_info: Dict[str, Any]
) -> Dict[str, Any]:
    stats = analyze_tracking_txt(
        tracking_txt_path=tracking_txt_path,
        fps=float(process_info.get("original_fps", 25.0))
    )

    return {
        "mode": "upload",
        "video_statistics": {
            "total_frames": process_info.get("total_frames", 0),
            "width": process_info.get("width", 0),
            "height": process_info.get("height", 0),
            "original_fps": round(float(process_info.get("original_fps", 0)), 4),
            "duration_seconds": round(
                process_info.get("total_frames", 0) / process_info.get("original_fps", 25.0),
                4
            ) if process_info.get("original_fps", 0) else 0
        },
        "vehicle_statistics": {
            "total_tracking_rows": stats["total_tracking_rows"],
            "total_visible_rows": stats["total_visible_rows"],
            "unique_tracked_vehicles": stats["total_tracks"],
            "avg_vehicles_per_frame": stats["avg_objects_per_frame"],
            "max_vehicles_per_frame": stats["max_objects_per_frame"]
        },
        "tracking_behavior": {
            "avg_track_length_frames": stats["avg_track_length_frames"],
            "short_track_count": stats["short_track_count"],
            "short_track_ratio": stats["short_track_ratio"]
        },
        "performance": {
            "elapsed_time_sec": round(float(process_info.get("elapsed_time", 0)), 4),
            "processing_fps": round(float(process_info.get("process_fps", 0)), 4),
            "avg_detector_time_ms": round(float(process_info.get("avg_detector_time", 0)) * 1000, 4),
            "avg_tracker_time_ms": round(float(process_info.get("avg_tracker_time", 0)) * 1000, 4)
        },
        "track_details": stats["tracks"]
    }


def evaluate_with_ground_truth(
    ground_truth_records: List[Dict[str, Any]],
    prediction_records: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluation đơn giản, đủ dùng cho đồ án:
        - Greedy IoU matching theo từng frame
        - Precision / Recall / F1
        - FP / FN
        - MOTP proxy
        - ID switch proxy
        - MOTA proxy
    """

    gt_by_frame = defaultdict(list)
    pred_by_frame = defaultdict(list)

    for item in ground_truth_records:
        gt_by_frame[int(item["frame"])].append(item)

    for item in prediction_records:
        pred_by_frame[int(item["frame"])].append(item)

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    matches: List[Tuple[int, str, str, float]] = []
    total_gt = 0
    total_pred = 0
    false_positive = 0
    false_negative = 0

    for frame in all_frames:
        gt_items = gt_by_frame.get(frame, [])
        pred_items = pred_by_frame.get(frame, [])

        total_gt += len(gt_items)
        total_pred += len(pred_items)

        candidate_pairs = []

        for gi, gt in enumerate(gt_items):
            for pi, pred in enumerate(pred_items):
                iou = compute_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])

                if iou >= iou_threshold:
                    candidate_pairs.append((iou, gi, pi))

        candidate_pairs.sort(reverse=True, key=lambda x: x[0])

        used_gt = set()
        used_pred = set()

        for iou, gi, pi in candidate_pairs:
            if gi in used_gt or pi in used_pred:
                continue

            used_gt.add(gi)
            used_pred.add(pi)

            matches.append((
                frame,
                str(gt_items[gi]["track_id"]),
                str(pred_items[pi]["track_id"]),
                iou
            ))

        false_negative += len(gt_items) - len(used_gt)
        false_positive += len(pred_items) - len(used_pred)

    true_positive = len(matches)

    precision = true_positive / total_pred if total_pred else 0
    recall = true_positive / total_gt if total_gt else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0
    motp_proxy = sum(item[3] for item in matches) / len(matches) if matches else 0

    id_switches = _count_id_switches(matches)

    mota_proxy = 1 - ((false_negative + false_positive + id_switches) / total_gt) if total_gt else 0

    return {
        "iou_threshold": iou_threshold,
        "total_groundtruth_boxes": total_gt,
        "total_prediction_boxes": total_pred,
        "true_positive_matches": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "average_matched_iou": round(motp_proxy, 4),
        "motp_proxy": round(motp_proxy, 4),
        "id_switches_proxy": id_switches,
        "mota_proxy": round(mota_proxy, 4)
    }


def build_dataset_metrics(
    dataset_name: str,
    ground_truth_records: List[Dict[str, Any]],
    prediction_txt_path: str,
    process_info: Dict[str, Any],
    groundtruth_summary: Dict[str, Any]
) -> Dict[str, Any]:
    prediction_records = read_tracking_txt(prediction_txt_path)

    tracking_stats = analyze_tracking_txt(
        tracking_txt_path=prediction_txt_path,
        fps=float(process_info.get("original_fps", 25.0))
    )

    evaluation = evaluate_with_ground_truth(
        ground_truth_records=ground_truth_records,
        prediction_records=prediction_records,
        iou_threshold=0.5
    )

    return {
        "mode": "dataset",
        "dataset": dataset_name,
        "video_statistics": {
            "total_frames": process_info.get("total_frames", 0),
            "width": process_info.get("width", 0),
            "height": process_info.get("height", 0),
            "original_fps": round(float(process_info.get("original_fps", 0)), 4)
        },
        "groundtruth_statistics": groundtruth_summary,
        "prediction_statistics": {
            "total_prediction_rows": tracking_stats["total_tracking_rows"],
            "total_visible_prediction_rows": tracking_stats["total_visible_rows"],
            "unique_predicted_tracks": tracking_stats["total_tracks"],
            "avg_predicted_objects_per_frame": tracking_stats["avg_objects_per_frame"],
            "max_predicted_objects_per_frame": tracking_stats["max_objects_per_frame"]
        },
        "evaluation": evaluation,
        "performance": {
            "elapsed_time_sec": round(float(process_info.get("elapsed_time", 0)), 4),
            "processing_fps": round(float(process_info.get("process_fps", 0)), 4),
            "avg_detector_time_ms": round(float(process_info.get("avg_detector_time", 0)) * 1000, 4),
            "avg_tracker_time_ms": round(float(process_info.get("avg_tracker_time", 0)) * 1000, 4)
        },
        "track_details": tracking_stats["tracks"]
    }


def _get_main_class_id(records: List[Dict[str, Any]]) -> int:
    counts = defaultdict(int)

    for item in records:
        class_id = int(item["class_id"])

        if class_id == -1:
            continue

        counts[class_id] += 1

    if not counts:
        return -1

    return max(counts, key=counts.get)


def _count_id_switches(matches: List[Tuple[int, str, str, float]]) -> int:
    by_gt_id = defaultdict(list)

    for frame, gt_id, pred_id, iou in matches:
        by_gt_id[gt_id].append((frame, pred_id))

    id_switches = 0

    for gt_id, pairs in by_gt_id.items():
        pairs = sorted(pairs, key=lambda x: x[0])
        last_pred_id = None

        for _, pred_id in pairs:
            if last_pred_id is not None and pred_id != last_pred_id:
                id_switches += 1

            last_pred_id = pred_id

    return id_switches
