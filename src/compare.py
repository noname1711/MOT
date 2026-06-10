import os
import shutil
from typing import Any, Dict, List, Optional

from src.dataset import get_dataset_info, load_ground_truth_records, summarize_ground_truth
from src.metrics import build_dataset_metrics, build_upload_metrics
from src.utils import ensure_dir, save_json, timestamp_now
from src.video_processor import VideoProcessor


TRACKERS_TO_COMPARE = [
    ("deepsort", "DeepSORT (original)"),
    ("custom", "Custom DeepSORT"),
]


def compare_dataset_trackers(
    dataset_name: str,
    yolo_model_path: str = "models/yolov5n.pt",
    conf_threshold: float = 0.35,
    image_size: int = 416,
    device: str = "cpu",
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_info = get_dataset_info(dataset_name)
    input_video_path = dataset_info.get("original_video_path")

    if not input_video_path:
        raise FileNotFoundError(
            f"Dataset {dataset_name} does not contain Original-video.mp4"
        )

    ground_truth_records = load_ground_truth_records(dataset_info)

    if max_frames is not None:
        ground_truth_records = [
            item for item in ground_truth_records
            if int(item["frame"]) <= int(max_frames)
        ]

    groundtruth_summary = summarize_ground_truth(ground_truth_records)

    return compare_trackers_for_video(
        input_video_path=input_video_path,
        run_name=dataset_name,
        mode="dataset",
        yolo_model_path=yolo_model_path,
        conf_threshold=conf_threshold,
        image_size=image_size,
        device=device,
        max_frames=max_frames,
        ground_truth_records=ground_truth_records,
        groundtruth_summary=groundtruth_summary,
        dataset_name=dataset_name,
    )


def compare_upload_trackers(
    input_video_path: str,
    run_name: str,
    yolo_model_path: str = "models/yolov5n.pt",
    conf_threshold: float = 0.35,
    image_size: int = 416,
    device: str = "cpu",
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    return compare_trackers_for_video(
        input_video_path=input_video_path,
        run_name=run_name,
        mode="upload",
        yolo_model_path=yolo_model_path,
        conf_threshold=conf_threshold,
        image_size=image_size,
        device=device,
        max_frames=max_frames,
    )


def compare_trackers_for_video(
    input_video_path: str,
    run_name: str,
    mode: str,
    yolo_model_path: str,
    conf_threshold: float,
    image_size: int,
    device: str,
    max_frames: Optional[int] = None,
    ground_truth_records: Optional[List[Dict[str, Any]]] = None,
    groundtruth_summary: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
) -> Dict[str, Any]:
    safe_name = _safe_name(run_name)
    run_id = f"{timestamp_now()}_{safe_name}"

    result_root = "results/compare"
    static_root = "static/outputs/compare"

    video_result_dir = ensure_dir(os.path.join(result_root, "videos"))
    txt_result_dir = ensure_dir(os.path.join(result_root, "txt"))
    metrics_result_dir = ensure_dir(os.path.join(result_root, "metrics"))
    static_video_dir = ensure_dir(static_root)

    tracker_results = {}

    for tracker_key, tracker_label in TRACKERS_TO_COMPARE:
        output_video_name = f"{run_id}_{tracker_key}_tracking.mp4"
        output_txt_name = f"{run_id}_{tracker_key}_tracking.txt"
        output_metrics_name = f"{run_id}_{tracker_key}_metrics.json"

        static_output_video_path = os.path.join(static_video_dir, output_video_name)
        result_video_path = os.path.join(video_result_dir, output_video_name)
        result_txt_path = os.path.join(txt_result_dir, output_txt_name)
        result_metrics_path = os.path.join(metrics_result_dir, output_metrics_name)

        processor = VideoProcessor(
            yolo_model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            image_size=image_size,
            device=device,
            tracker_type=tracker_key,
        )

        process_info = processor.process(
            input_video_path=input_video_path,
            output_video_path=static_output_video_path,
            output_txt_path=result_txt_path,
            max_frames=max_frames,
        )

        shutil.copyfile(static_output_video_path, result_video_path)

        if ground_truth_records is not None:
            metrics_data = build_dataset_metrics(
                dataset_name=dataset_name or run_name,
                ground_truth_records=ground_truth_records,
                prediction_txt_path=result_txt_path,
                process_info=process_info,
                groundtruth_summary=groundtruth_summary or {},
            )
        else:
            metrics_data = build_upload_metrics(
                tracking_txt_path=result_txt_path,
                process_info=process_info,
            )

        metrics_data.update({
            "run_id": run_id,
            "compare_mode": True,
            "mode": mode,
            "tracker": tracker_key,
            "tracker_label": tracker_label,
            "model": f"YOLOv5n + {tracker_label}",
            "yolo_model_path": yolo_model_path,
            "device": device,
            "confidence_threshold": conf_threshold,
            "image_size": image_size,
            "input_video_path": input_video_path,
            "output_video_path": result_video_path,
            "web_output_video_path": static_output_video_path,
            "tracking_txt_path": result_txt_path,
        })

        save_json(metrics_data, result_metrics_path)

        tracker_results[tracker_key] = {
            "tracker": tracker_key,
            "tracker_label": tracker_label,
            "video_path": result_video_path,
            "web_video_path": static_output_video_path,
            "txt_path": result_txt_path,
            "metrics_path": result_metrics_path,
            "process_info": process_info,
            "metrics": metrics_data,
        }

    summary = _build_comparison_summary(
        run_id=run_id,
        mode=mode,
        run_name=run_name,
        input_video_path=input_video_path,
        tracker_results=tracker_results,
        max_frames=max_frames,
    )

    summary_path = os.path.join(
        metrics_result_dir,
        f"{run_id}_comparison_summary.json",
    )

    save_json(summary, summary_path)
    summary["summary_path"] = summary_path

    return summary


def _build_comparison_summary(
    run_id: str,
    mode: str,
    run_name: str,
    input_video_path: str,
    tracker_results: Dict[str, Any],
    max_frames: Optional[int],
) -> Dict[str, Any]:
    rows = []

    for tracker_key, item in tracker_results.items():
        metrics = item["metrics"]
        process_info = item["process_info"]

        evaluation = metrics.get("evaluation", {})
        prediction_stats = metrics.get("prediction_statistics", {})
        vehicle_stats = metrics.get("vehicle_statistics", {})

        rows.append({
            "tracker": tracker_key,
            "tracker_label": item["tracker_label"],

            "processing_fps": round(
                float(process_info.get("process_fps", 0)),
                4,
            ),
            "avg_tracker_time_ms": round(
                float(process_info.get("avg_tracker_time", 0)) * 1000,
                4,
            ),
            "total_frames": int(process_info.get("total_frames", 0)),

            "unique_tracks": (
                prediction_stats.get("unique_predicted_tracks")
                if prediction_stats
                else vehicle_stats.get("unique_tracked_vehicles")
            ),
            "avg_objects_per_frame": (
                prediction_stats.get("avg_predicted_objects_per_frame")
                if prediction_stats
                else vehicle_stats.get("avg_vehicles_per_frame")
            ),

            "precision": evaluation.get("precision"),
            "recall": evaluation.get("recall"),
            "f1_score": evaluation.get("f1_score"),

            # Dataset mode only:
            # MOTP proxy is average matched IoU.
            # Upload mode has no ground truth, so this value will be None.
            "motp_proxy": evaluation.get("motp_proxy"),
            "average_matched_iou": evaluation.get("average_matched_iou"),
            "mota_proxy": evaluation.get("mota_proxy"),
            "id_switches_proxy": evaluation.get("id_switches_proxy"),

            "video_path": item["video_path"],
            "txt_path": item["txt_path"],
            "metrics_path": item["metrics_path"],
        })

    return {
        "run_id": run_id,
        "mode": mode,
        "run_name": run_name,
        "input_video_path": input_video_path,
        "max_frames": max_frames,
        "comparison": rows,
        "outputs": tracker_results,
    }


def _safe_name(name: str) -> str:
    cleaned = "".join(
        c if c.isalnum() or c in {"-", "_"} else "_"
        for c in str(name)
    )

    return cleaned.strip("_") or "video"