import os
import shutil
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from src.dataset import (
    get_dataset_info,
    list_datasets,
    load_ground_truth_records,
    summarize_ground_truth,
)
from src.metrics import (
    analyze_tracking_txt,
    build_dataset_metrics,
    build_upload_metrics,
)
from src.utils import (
    allowed_video_file,
    ensure_dir,
    save_metrics,
    timestamp_now,
)
from src.video_processor import VideoProcessor
from src.webcam_processor import WebcamProcessor
from src.compare import compare_dataset_trackers, compare_upload_trackers


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_OUTPUT_ROOT = "static/outputs"
RESULT_ROOT = "results"

MODEL_PATH = "models/yolov5n.pt"
CONF_THRESHOLD = 0.35
IMAGE_SIZE = 416
DEVICE = "cpu"

VEHICLE_MODEL_NAME = "YOLOv5n + DeepSORT/Custom DeepSORT Vehicle Tracking"

TRACKER_OPTIONS = [
    {
        "value": "deepsort",
        "label": "Original DeepSORT",
        "description": "Baseline tracker using the deep-sort-realtime library and a MobileNet embedder."
    },
    {
        "value": "custom",
        "label": "Custom DeepSORT",
        "description": "Custom tracker implemented with Kalman Filter, IoU Matching, Appearance Matching, Hungarian Assignment, and Track Lifecycle."
    },
    {
        "value": "compare",
        "label": "Compare both",
        "description": "Run the original DeepSORT and Custom DeepSORT on the same video for comparison."
    },
]

DEFAULT_WEBCAM_TRACKER = "deepsort"
WEBCAM_TRACKER_OPTIONS = [
    item for item in TRACKER_OPTIONS
    if item["value"] in {"deepsort", "custom"}
]

MODES = ["dataset", "upload", "webcam", "compare"]

webcam_processor = None

latest_webcam_recording = {
    "run_id": None,
    "filename": None,
    "txt_filename": None,
    "static_path": None,
    "result_path": None,
    "txt_path": None,
    "tracker_type": DEFAULT_WEBCAM_TRACKER,
    "tracker_label": "Original DeepSORT",
}


def init_directories():
    ensure_dir(UPLOAD_FOLDER)

    for mode in MODES:
        ensure_dir(os.path.join(STATIC_OUTPUT_ROOT, mode))
        ensure_dir(os.path.join(RESULT_ROOT, mode, "videos"))
        ensure_dir(os.path.join(RESULT_ROOT, mode, "txt"))
        ensure_dir(os.path.join(RESULT_ROOT, mode, "metrics"))


def normalize_webcam_tracker(tracker_type: str | None = None) -> str:
    normalized = str(tracker_type or DEFAULT_WEBCAM_TRACKER).strip().lower()

    if normalized in {"custom", "custom_deepsort", "custom-deepsort"}:
        return "custom"

    return "deepsort"


def get_current_webcam_tracker_type() -> str:
    if webcam_processor is None:
        return DEFAULT_WEBCAM_TRACKER

    return normalize_webcam_tracker(
        getattr(webcam_processor, "tracker_type", DEFAULT_WEBCAM_TRACKER)
    )


def get_webcam_processor(tracker_type: str | None = None) -> WebcamProcessor:
    global webcam_processor

    selected_tracker = normalize_webcam_tracker(
        tracker_type or get_current_webcam_tracker_type()
    )

    if webcam_processor is not None:
        current_tracker = normalize_webcam_tracker(
            getattr(webcam_processor, "tracker_type", DEFAULT_WEBCAM_TRACKER)
        )

        if current_tracker != selected_tracker:
            live_info = webcam_processor.get_live_metrics()

            if not live_info.get("is_recording"):
                webcam_processor.stop()
                webcam_processor = None
            else:
                selected_tracker = current_tracker

    if webcam_processor is None:
        webcam_processor = WebcamProcessor(
            camera_index=0,
            yolo_model_path=MODEL_PATH,
            conf_threshold=CONF_THRESHOLD,
            image_size=IMAGE_SIZE,
            device=DEVICE,
            tracker_type=selected_tracker,
        )

    return webcam_processor


def create_video_processor(tracker_type: str = "deepsort") -> VideoProcessor:
    return VideoProcessor(
        yolo_model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        image_size=IMAGE_SIZE,
        device=DEVICE,
        tracker_type=tracker_type,
    )


def get_result_dir(mode: str, kind: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode}")

    if kind not in {"videos", "txt", "metrics"}:
        raise ValueError(f"Invalid result kind: {kind}")

    return os.path.join(RESULT_ROOT, mode, kind)


def get_static_output_dir(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode}")

    return os.path.join(STATIC_OUTPUT_ROOT, mode)


def get_tracker_label(tracker_type: str) -> str:
    tracker_type = str(tracker_type).strip().lower()

    for item in TRACKER_OPTIONS:
        if item["value"] == tracker_type:
            return item["label"]

    return tracker_type


def get_selected_tracker(default: str = "deepsort") -> str:
    tracker_type = request.form.get("tracker_type", default).strip().lower()

    valid_values = {item["value"] for item in TRACKER_OPTIONS}

    if tracker_type not in valid_values:
        return default

    return tracker_type


def get_optional_max_frames():
    raw_value = request.form.get("max_frames", "").strip()

    if not raw_value:
        return None

    try:
        value = int(raw_value)
    except ValueError:
        return None

    if value <= 0:
        return None

    return value


def build_compare_template_result(summary: dict) -> dict:
    rows = []

    for row in summary.get("comparison", []):
        item = dict(row)

        video_name = os.path.basename(str(row.get("video_path", "")))
        txt_name = os.path.basename(str(row.get("txt_path", "")))
        metrics_name = os.path.basename(str(row.get("metrics_path", "")))

        item.update({
            "video_name": video_name,
            "txt_name": txt_name,
            "metrics_name": metrics_name,
            "video_url": url_for(
                "static",
                filename=f"outputs/compare/{video_name}",
            ) if video_name else None,
            "video_download_url": url_for(
                "download_file",
                mode="compare",
                kind="videos",
                filename=video_name,
            ) if video_name else None,
            "txt_download_url": url_for(
                "download_file",
                mode="compare",
                kind="txt",
                filename=txt_name,
            ) if txt_name else None,
            "metrics_download_url": url_for(
                "download_file",
                mode="compare",
                kind="metrics",
                filename=metrics_name,
            ) if metrics_name else None,
        })

        rows.append(item)

    summary_path = summary.get("summary_path")
    summary_name = os.path.basename(summary_path) if summary_path else None

    return {
        "compare": True,
        "run_id": summary.get("run_id"),
        "run_name": summary.get("run_name"),
        "input_video_path": summary.get("input_video_path"),
        "max_frames": summary.get("max_frames"),
        "comparison": rows,
        "summary": summary,
        "summary_download_url": url_for(
            "download_file",
            mode="compare",
            kind="metrics",
            filename=summary_name,
        ) if summary_name else None,
    }


@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# Dataset mode
# ============================================================

@app.route("/dataset", methods=["GET", "POST"])
def dataset_page():
    datasets = list_datasets()
    result = None
    error = None
    selected_tracker = "deepsort"

    if request.method == "POST":
        dataset_name = request.form.get("dataset_name", "").strip()
        selected_tracker = get_selected_tracker()
        max_frames = get_optional_max_frames()

        if not dataset_name:
            error = "No dataset was selected."
            return render_template(
                "dataset.html",
                datasets=datasets,
                result=result,
                error=error,
                tracker_options=TRACKER_OPTIONS,
                selected_tracker=selected_tracker,
            )

        try:
            if selected_tracker == "compare":
                summary = compare_dataset_trackers(
                    dataset_name=dataset_name,
                    yolo_model_path=MODEL_PATH,
                    conf_threshold=CONF_THRESHOLD,
                    image_size=IMAGE_SIZE,
                    device=DEVICE,
                    max_frames=max_frames,
                )

                result = build_compare_template_result(summary)
                result.update({
                    "dataset_name": dataset_name,
                    "mode": "dataset",
                })

            else:
                dataset_info = get_dataset_info(dataset_name)
                input_video_path = dataset_info.get("original_video_path")

                if not input_video_path:
                    raise FileNotFoundError(
                        f"Dataset {dataset_name} does not contain Original-video.mp4"
                    )

                run_id = f"{timestamp_now()}_{dataset_name}_{selected_tracker}"
                output_video_name = f"{run_id}_tracking.mp4"
                output_txt_name = f"{run_id}_tracking.txt"

                static_output_video_path = os.path.join(
                    get_static_output_dir("dataset"),
                    output_video_name,
                )

                result_video_path = os.path.join(
                    get_result_dir("dataset", "videos"),
                    output_video_name,
                )

                result_txt_path = os.path.join(
                    get_result_dir("dataset", "txt"),
                    output_txt_name,
                )

                processor = create_video_processor(tracker_type=selected_tracker)

                process_info = processor.process(
                    input_video_path=input_video_path,
                    output_video_path=static_output_video_path,
                    output_txt_path=result_txt_path,
                    max_frames=max_frames,
                )

                shutil.copyfile(static_output_video_path, result_video_path)

                ground_truth_records = load_ground_truth_records(dataset_info)

                if max_frames is not None:
                    ground_truth_records = [
                        item for item in ground_truth_records
                        if int(item["frame"]) <= int(max_frames)
                    ]

                groundtruth_summary = summarize_ground_truth(ground_truth_records)

                metrics_data = build_dataset_metrics(
                    dataset_name=dataset_name,
                    ground_truth_records=ground_truth_records,
                    prediction_txt_path=result_txt_path,
                    process_info=process_info,
                    groundtruth_summary=groundtruth_summary,
                )

                metrics_data.update({
                    "run_id": run_id,
                    "model": f"YOLOv5n + {get_tracker_label(selected_tracker)} Vehicle Tracking",
                    "yolo_model_path": MODEL_PATH,
                    "tracker": selected_tracker,
                    "tracker_label": get_tracker_label(selected_tracker),
                    "device": DEVICE,
                    "confidence_threshold": CONF_THRESHOLD,
                    "image_size": IMAGE_SIZE,
                    "max_frames": max_frames,
                    "input_video_path": input_video_path,
                    "output_video_path": result_video_path,
                    "web_output_video_path": static_output_video_path,
                    "tracking_txt_path": result_txt_path,
                })

                metrics_path = save_metrics(
                    metrics=metrics_data,
                    output_dir=get_result_dir("dataset", "metrics"),
                    run_id=run_id,
                )

                result = {
                    "compare": False,
                    "run_id": run_id,
                    "dataset_name": dataset_name,
                    "tracker": selected_tracker,
                    "tracker_label": get_tracker_label(selected_tracker),
                    "video_url": url_for(
                        "static",
                        filename=f"outputs/dataset/{output_video_name}",
                    ),
                    "video_download_url": url_for(
                        "download_file",
                        mode="dataset",
                        kind="videos",
                        filename=output_video_name,
                    ),
                    "txt_download_url": url_for(
                        "download_file",
                        mode="dataset",
                        kind="txt",
                        filename=output_txt_name,
                    ),
                    "metrics_download_url": url_for(
                        "download_file",
                        mode="dataset",
                        kind="metrics",
                        filename=os.path.basename(metrics_path),
                    ),
                    "metrics": metrics_data,
                }

        except Exception as e:
            error = f"Error while processing dataset: {str(e)}"

    return render_template(
        "dataset.html",
        datasets=datasets,
        result=result,
        error=error,
        tracker_options=TRACKER_OPTIONS,
        selected_tracker=selected_tracker,
    )


# ============================================================
# Upload mode
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    result = None
    error = None
    selected_tracker = "deepsort"

    if request.method == "POST":
        selected_tracker = get_selected_tracker()
        max_frames = get_optional_max_frames()

        if "video" not in request.files:
            error = "No video file was found in the request."
            return render_template(
                "upload.html",
                result=result,
                error=error,
                tracker_options=TRACKER_OPTIONS,
                selected_tracker=selected_tracker,
            )

        file = request.files["video"]

        if file.filename == "":
            error = "No video was selected."
            return render_template(
                "upload.html",
                result=result,
                error=error,
                tracker_options=TRACKER_OPTIONS,
                selected_tracker=selected_tracker,
            )

        if not allowed_video_file(file.filename):
            error = "Invalid video format. Supported formats: mp4, avi, mov, mkv."
            return render_template(
                "upload.html",
                result=result,
                error=error,
                tracker_options=TRACKER_OPTIONS,
                selected_tracker=selected_tracker,
            )

        try:
            original_filename = secure_filename(file.filename)
            run_id = f"{timestamp_now()}_{Path(original_filename).stem}"

            input_filename = f"{run_id}{Path(original_filename).suffix}"
            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            file.save(input_path)

            if selected_tracker == "compare":
                summary = compare_upload_trackers(
                    input_video_path=input_path,
                    run_name=Path(original_filename).stem,
                    yolo_model_path=MODEL_PATH,
                    conf_threshold=CONF_THRESHOLD,
                    image_size=IMAGE_SIZE,
                    device=DEVICE,
                    max_frames=max_frames,
                )

                result = build_compare_template_result(summary)
                result.update({
                    "mode": "upload",
                    "original_filename": original_filename,
                })

            else:
                run_id = f"{run_id}_{selected_tracker}"
                output_video_name = f"{run_id}_tracking.mp4"
                output_txt_name = f"{run_id}_tracking.txt"

                static_output_video_path = os.path.join(
                    get_static_output_dir("upload"),
                    output_video_name,
                )

                result_video_path = os.path.join(
                    get_result_dir("upload", "videos"),
                    output_video_name,
                )

                result_txt_path = os.path.join(
                    get_result_dir("upload", "txt"),
                    output_txt_name,
                )

                processor = create_video_processor(tracker_type=selected_tracker)

                process_info = processor.process(
                    input_video_path=input_path,
                    output_video_path=static_output_video_path,
                    output_txt_path=result_txt_path,
                    max_frames=max_frames,
                )

                shutil.copyfile(static_output_video_path, result_video_path)

                metrics_data = build_upload_metrics(
                    tracking_txt_path=result_txt_path,
                    process_info=process_info,
                )

                metrics_data.update({
                    "run_id": run_id,
                    "model": f"YOLOv5n + {get_tracker_label(selected_tracker)} Vehicle Tracking",
                    "yolo_model_path": MODEL_PATH,
                    "tracker": selected_tracker,
                    "tracker_label": get_tracker_label(selected_tracker),
                    "device": DEVICE,
                    "confidence_threshold": CONF_THRESHOLD,
                    "image_size": IMAGE_SIZE,
                    "max_frames": max_frames,
                    "input_video_path": input_path,
                    "output_video_path": result_video_path,
                    "web_output_video_path": static_output_video_path,
                    "tracking_txt_path": result_txt_path,
                })

                metrics_path = save_metrics(
                    metrics=metrics_data,
                    output_dir=get_result_dir("upload", "metrics"),
                    run_id=run_id,
                )

                result = {
                    "compare": False,
                    "run_id": run_id,
                    "tracker": selected_tracker,
                    "tracker_label": get_tracker_label(selected_tracker),
                    "video_url": url_for(
                        "static",
                        filename=f"outputs/upload/{output_video_name}",
                    ),
                    "video_download_url": url_for(
                        "download_file",
                        mode="upload",
                        kind="videos",
                        filename=output_video_name,
                    ),
                    "txt_download_url": url_for(
                        "download_file",
                        mode="upload",
                        kind="txt",
                        filename=output_txt_name,
                    ),
                    "metrics_download_url": url_for(
                        "download_file",
                        mode="upload",
                        kind="metrics",
                        filename=os.path.basename(metrics_path),
                    ),
                    "metrics": metrics_data,
                }

        except Exception as e:
            error = f"Error while processing video: {str(e)}"

    return render_template(
        "upload.html",
        result=result,
        error=error,
        tracker_options=TRACKER_OPTIONS,
        selected_tracker=selected_tracker,
    )


# ============================================================
# Webcam mode
# ============================================================

@app.route("/webcam")
def webcam():
    selected_tracker = normalize_webcam_tracker(
        request.args.get("tracker_type", get_current_webcam_tracker_type())
    )

    processor = get_webcam_processor(tracker_type=selected_tracker)
    processor.start()

    return render_template(
        "webcam.html",
        tracker_options=WEBCAM_TRACKER_OPTIONS,
        selected_tracker=selected_tracker,
        selected_tracker_label=get_tracker_label(selected_tracker),
    )


@app.route("/webcam_feed")
def webcam_feed():
    processor = get_webcam_processor()

    return Response(
        processor.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/webcam/metrics")
def webcam_metrics():
    processor = get_webcam_processor()
    return jsonify(processor.get_live_metrics())


@app.route("/webcam/start_worker", methods=["POST"])
def webcam_start_worker():
    """
    Internal route used by the frontend to restart the webcam worker.

    This endpoint is not exposed as a visible UI button.
    """

    payload = request.get_json(silent=True) or {}
    selected_tracker = normalize_webcam_tracker(
        payload.get("tracker_type")
        or request.form.get("tracker_type")
        or request.args.get("tracker_type")
        or get_current_webcam_tracker_type()
    )

    processor = get_webcam_processor(tracker_type=selected_tracker)
    info = processor.start()
    info["tracker_type"] = getattr(processor, "tracker_type", selected_tracker)
    info["tracker_label"] = get_tracker_label(info["tracker_type"])

    return jsonify(info)


@app.route("/webcam/stop_worker", methods=["POST"])
def webcam_stop_worker():
    """
    Internal route used by the frontend to stop the webcam worker.

    If recording is active, the system tries to finalize the recording before
    releasing the camera.
    """

    processor = get_webcam_processor()
    live_info = processor.get_live_metrics()
    recording_result = None

    if live_info.get("is_recording"):
        stop_record_info = processor.stop_recording()

        if stop_record_info.get("success"):
            recording_result = _save_webcam_recording_metrics(stop_record_info)
        else:
            recording_result = stop_record_info

    stop_info = processor.stop()

    return jsonify({
        "success": True,
        "message": "Webcam worker stopped.",
        "worker": stop_info,
        "recording": recording_result,
    })


@app.route("/webcam/start_record", methods=["POST"])
def webcam_start_record():
    global latest_webcam_recording

    processor = get_webcam_processor()
    tracker_type = normalize_webcam_tracker(
        getattr(processor, "tracker_type", DEFAULT_WEBCAM_TRACKER)
    )
    tracker_label = get_tracker_label(tracker_type)

    run_id = f"{timestamp_now()}_webcam_{tracker_type}"
    output_video_name = f"{run_id}_tracking.mp4"
    output_txt_name = f"{run_id}_tracking.txt"

    static_output_path = os.path.join(
        get_static_output_dir("webcam"),
        output_video_name,
    )

    result_output_path = os.path.join(
        get_result_dir("webcam", "videos"),
        output_video_name,
    )

    txt_output_path = os.path.join(
        get_result_dir("webcam", "txt"),
        output_txt_name,
    )

    info = processor.start_recording(
        static_output_path=static_output_path,
        result_output_path=result_output_path,
        txt_output_path=txt_output_path,
        fps=20.0,
    )

    if info.get("success"):
        latest_webcam_recording = {
            "run_id": run_id,
            "filename": output_video_name,
            "txt_filename": output_txt_name,
            "static_path": static_output_path,
            "result_path": result_output_path,
            "txt_path": txt_output_path,
            "tracker_type": tracker_type,
            "tracker_label": tracker_label,
        }

    return jsonify({
        "success": info.get("success", False),
        "message": info.get("message", ""),
        "run_id": run_id if info.get("success") else latest_webcam_recording.get("run_id"),
        "filename": output_video_name if info.get("success") else latest_webcam_recording.get("filename"),
        "tracker_type": tracker_type,
        "tracker_label": tracker_label,
    })


@app.route("/webcam/stop_record", methods=["POST"])
def webcam_stop_record():
    processor = get_webcam_processor()
    info = processor.stop_recording()

    if not info.get("success"):
        return jsonify(info)

    return jsonify(_save_webcam_recording_metrics(info))


def _save_webcam_recording_metrics(info):
    """
    Save metrics for a completed webcam recording session.

    This helper is used by:
        - /webcam/stop_record
        - /webcam/stop_worker when the user leaves while recording
    """

    run_id = latest_webcam_recording.get("run_id") or timestamp_now()

    filename = latest_webcam_recording.get("filename")
    txt_filename = latest_webcam_recording.get("txt_filename")

    if not filename and info.get("result_output_path"):
        filename = os.path.basename(info["result_output_path"])

    if not txt_filename and info.get("txt_output_path"):
        txt_filename = os.path.basename(info["txt_output_path"])

    tracking_txt_path = info["txt_output_path"]

    tracking_stats = analyze_tracking_txt(
        tracking_txt_path=tracking_txt_path,
        fps=float(info.get("fps", 20.0)),
    )

    tracker_type = normalize_webcam_tracker(
        latest_webcam_recording.get("tracker_type")
        or info.get("tracker_type")
        or get_current_webcam_tracker_type()
    )
    tracker_label = get_tracker_label(tracker_type)

    metrics_data = {
        "run_id": run_id,
        "mode": "webcam",
        "model": f"YOLOv5n + {tracker_label} Webcam Tracking",
        "yolo_model_path": MODEL_PATH,
        "tracker": tracker_type,
        "tracker_label": tracker_label,
        "device": DEVICE,
        "confidence_threshold": CONF_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "recording": info,
        "tracking_statistics": tracking_stats,
    }

    metrics_path = save_metrics(
        metrics=metrics_data,
        output_dir=get_result_dir("webcam", "metrics"),
        run_id=run_id,
    )

    return {
        "success": True,
        "message": info.get("message", "Webcam recording stopped."),
        "run_id": run_id,
        "tracker_type": tracker_type,
        "tracker_label": tracker_label,
        "frame_count": info.get("frame_count", 0),
        "duration_sec": info.get("duration_sec", 0),
        "fps": info.get("fps", 0),
        "unique_tracks": info.get("unique_tracks", 0),
        "max_active_tracks": info.get("max_active_tracks", 0),
        "video_url": url_for(
            "static",
            filename=f"outputs/webcam/{filename}",
        ) if filename else None,
        "video_download_url": url_for(
            "download_file",
            mode="webcam",
            kind="videos",
            filename=filename,
        ) if filename else None,
        "txt_download_url": url_for(
            "download_file",
            mode="webcam",
            kind="txt",
            filename=txt_filename,
        ) if txt_filename else None,
        "metrics_download_url": url_for(
            "download_file",
            mode="webcam",
            kind="metrics",
            filename=os.path.basename(metrics_path),
        ),
    }


# ============================================================
# Results
# ============================================================

@app.route("/results")
def results_page():
    grouped_results = {}

    for mode in MODES:
        grouped_results[mode] = _group_result_runs(mode)

    return render_template("results.html", results=grouped_results)


def _group_result_runs(mode: str):
    """
    Group result artifacts by run ID for dataset, upload, webcam, and compare modes.

    A run is shown only when it has either:
        - an output video, or
        - a metrics JSON file.

    TXT-only runs are hidden because they usually come from incomplete webcam
    recording sessions that did not produce a valid video.
    """

    grouped = {}

    kind_to_suffix = {
        "videos": "_tracking.mp4",
        "txt": "_tracking.txt",
        "metrics": "_metrics.json",
    }

    for kind, suffix in kind_to_suffix.items():
        directory = get_result_dir(mode, kind)
        files = _list_files(directory)

        for filename in files:
            run_id = _extract_run_id(filename, suffix)

            if not run_id:
                continue

            if run_id not in grouped:
                grouped[run_id] = {
                    "run_id": run_id,
                    "mode": mode,
                    "display_time": _format_run_time(run_id),
                    "display_name": _format_run_name(run_id),
                    "video": None,
                    "txt": None,
                    "metrics": None,
                }

            item = {
                "filename": filename,
                "download_url": url_for(
                    "download_file",
                    mode=mode,
                    kind=kind,
                    filename=filename,
                ),
            }

            if kind == "videos":
                static_path = os.path.join(STATIC_OUTPUT_ROOT, mode, filename)

                if os.path.exists(static_path):
                    item["open_url"] = url_for(
                        "static",
                        filename=f"outputs/{mode}/{filename}",
                    )
                else:
                    item["open_url"] = None

                grouped[run_id]["video"] = item

            elif kind == "txt":
                grouped[run_id]["txt"] = item

            elif kind == "metrics":
                grouped[run_id]["metrics"] = item

    valid_runs = []

    for run in grouped.values():
        has_video = run.get("video") is not None
        has_metrics = run.get("metrics") is not None

        if not has_video and not has_metrics:
            continue

        valid_runs.append(run)

    return sorted(
        valid_runs,
        key=lambda item: item["run_id"],
        reverse=True,
    )


def _extract_run_id(filename: str, suffix: str):
    if not filename.endswith(suffix):
        return None

    return filename[:-len(suffix)]


def _format_run_time(run_id: str) -> str:
    if len(run_id) < 15:
        return run_id

    date_part = run_id[:8]
    time_part = run_id[9:15]

    if not (date_part.isdigit() and time_part.isdigit()):
        return run_id

    yyyy = date_part[:4]
    mm = date_part[4:6]
    dd = date_part[6:8]

    hh = time_part[:2]
    mi = time_part[2:4]
    ss = time_part[4:6]

    return f"{yyyy}-{mm}-{dd} {hh}:{mi}:{ss}"


def _format_run_name(run_id: str) -> str:
    parts = run_id.split("_", 2)

    if len(parts) >= 3:
        return parts[2]

    return run_id


# ============================================================
# Download
# ============================================================

@app.route("/download/<mode>/<kind>/<filename>")
def download_file(mode, kind, filename):
    directory = get_result_dir(mode, kind)

    return send_from_directory(
        directory,
        filename,
        as_attachment=True,
    )


def _list_files(directory: str):
    if not os.path.exists(directory):
        return []

    files = []

    for item in sorted(os.listdir(directory), reverse=True):
        path = os.path.join(directory, item)

        if os.path.isfile(path):
            files.append(item)

    return files


if __name__ == "__main__":
    init_directories()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False,
    )
