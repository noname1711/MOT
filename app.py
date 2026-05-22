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


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_OUTPUT_ROOT = "static/outputs"
RESULT_ROOT = "results"

MODEL_PATH = "models/yolov5n.pt"
CONF_THRESHOLD = 0.35
IMAGE_SIZE = 416
DEVICE = "cpu"

VEHICLE_MODEL_NAME = "YOLOv5n + DeepSORT Vehicle Tracking"

MODES = ["dataset", "upload", "webcam"]

webcam_processor = None

latest_webcam_recording = {
    "run_id": None,
    "filename": None,
    "txt_filename": None,
    "static_path": None,
    "result_path": None,
    "txt_path": None,
}


def init_directories():
    ensure_dir(UPLOAD_FOLDER)

    for mode in MODES:
        ensure_dir(os.path.join(STATIC_OUTPUT_ROOT, mode))
        ensure_dir(os.path.join(RESULT_ROOT, mode, "videos"))
        ensure_dir(os.path.join(RESULT_ROOT, mode, "txt"))
        ensure_dir(os.path.join(RESULT_ROOT, mode, "metrics"))


def get_webcam_processor() -> WebcamProcessor:
    global webcam_processor

    if webcam_processor is None:
        webcam_processor = WebcamProcessor(
            camera_index=0,
            yolo_model_path=MODEL_PATH,
            conf_threshold=CONF_THRESHOLD,
            image_size=IMAGE_SIZE,
            device=DEVICE,
        )

    return webcam_processor


def create_video_processor() -> VideoProcessor:
    return VideoProcessor(
        yolo_model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        image_size=IMAGE_SIZE,
        device=DEVICE,
    )


def get_result_dir(mode: str, kind: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Mode không hợp lệ: {mode}")

    if kind not in {"videos", "txt", "metrics"}:
        raise ValueError(f"Kind không hợp lệ: {kind}")

    return os.path.join(RESULT_ROOT, mode, kind)


def get_static_output_dir(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Mode không hợp lệ: {mode}")

    return os.path.join(STATIC_OUTPUT_ROOT, mode)


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

    if request.method == "POST":
        dataset_name = request.form.get("dataset_name", "").strip()

        if not dataset_name:
            error = "Bạn chưa chọn dataset."
            return render_template(
                "dataset.html",
                datasets=datasets,
                result=result,
                error=error,
            )

        try:
            dataset_info = get_dataset_info(dataset_name)
            input_video_path = dataset_info.get("original_video_path")

            if not input_video_path:
                raise FileNotFoundError(
                    f"Dataset {dataset_name} không có Original-video.mp4"
                )

            run_id = f"{timestamp_now()}_{dataset_name}"
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

            processor = create_video_processor()

            process_info = processor.process(
                input_video_path=input_video_path,
                output_video_path=static_output_video_path,
                output_txt_path=result_txt_path,
            )

            shutil.copyfile(static_output_video_path, result_video_path)

            ground_truth_records = load_ground_truth_records(dataset_info)
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
                "model": VEHICLE_MODEL_NAME,
                "yolo_model_path": MODEL_PATH,
                "tracker": "DeepSORT",
                "device": DEVICE,
                "confidence_threshold": CONF_THRESHOLD,
                "image_size": IMAGE_SIZE,
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
                "run_id": run_id,
                "dataset_name": dataset_name,
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
            error = f"Lỗi khi xử lý dataset: {str(e)}"

    return render_template(
        "dataset.html",
        datasets=datasets,
        result=result,
        error=error,
    )


# ============================================================
# Upload mode
# ============================================================

@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    result = None
    error = None

    if request.method == "POST":
        if "video" not in request.files:
            error = "Không tìm thấy file video trong request."
            return render_template("upload.html", result=result, error=error)

        file = request.files["video"]

        if file.filename == "":
            error = "Bạn chưa chọn video."
            return render_template("upload.html", result=result, error=error)

        if not allowed_video_file(file.filename):
            error = "Định dạng video không hợp lệ. Chỉ hỗ trợ: mp4, avi, mov, mkv."
            return render_template("upload.html", result=result, error=error)

        try:
            original_filename = secure_filename(file.filename)
            run_id = f"{timestamp_now()}_{Path(original_filename).stem}"

            input_filename = f"{run_id}{Path(original_filename).suffix}"
            input_path = os.path.join(UPLOAD_FOLDER, input_filename)
            file.save(input_path)

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

            processor = create_video_processor()

            process_info = processor.process(
                input_video_path=input_path,
                output_video_path=static_output_video_path,
                output_txt_path=result_txt_path,
            )

            shutil.copyfile(static_output_video_path, result_video_path)

            metrics_data = build_upload_metrics(
                tracking_txt_path=result_txt_path,
                process_info=process_info,
            )

            metrics_data.update({
                "run_id": run_id,
                "model": VEHICLE_MODEL_NAME,
                "yolo_model_path": MODEL_PATH,
                "tracker": "DeepSORT",
                "device": DEVICE,
                "confidence_threshold": CONF_THRESHOLD,
                "image_size": IMAGE_SIZE,
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
                "run_id": run_id,
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
            error = f"Lỗi khi xử lý video: {str(e)}"

    return render_template("upload.html", result=result, error=error)


# ============================================================
# Webcam mode
# ============================================================

@app.route("/webcam")
def webcam():
    processor = get_webcam_processor()
    processor.start()
    return render_template("webcam.html")


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
    Route ẩn để frontend tự bật webcam worker khi quay lại trang webcam.
    Không hiển thị nút này trên giao diện.
    """

    processor = get_webcam_processor()
    return jsonify(processor.start())


@app.route("/webcam/stop_worker", methods=["POST"])
def webcam_stop_worker():
    """
    Route ẩn để frontend tự tắt webcam worker khi rời trang webcam.
    Nếu đang record, hệ thống sẽ cố finalize record trước khi tắt camera.
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
        "message": "Đã dừng webcam worker.",
        "worker": stop_info,
        "recording": recording_result,
    })


@app.route("/webcam/start_record", methods=["POST"])
def webcam_start_record():
    global latest_webcam_recording

    run_id = f"{timestamp_now()}_webcam"
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

    processor = get_webcam_processor()

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
        }

    return jsonify({
        "success": info.get("success", False),
        "message": info.get("message", ""),
        "run_id": run_id if info.get("success") else latest_webcam_recording.get("run_id"),
        "filename": output_video_name if info.get("success") else latest_webcam_recording.get("filename"),
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
    Lưu metrics cho phiên webcam record đã hoàn tất.

    Hàm này được dùng bởi:
        - /webcam/stop_record
        - /webcam/stop_worker nếu người dùng rời trang khi đang record
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

    metrics_data = {
        "run_id": run_id,
        "mode": "webcam",
        "model": VEHICLE_MODEL_NAME,
        "yolo_model_path": MODEL_PATH,
        "tracker": "DeepSORT",
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
        "message": info.get("message", "Đã dừng ghi webcam."),
        "run_id": run_id,
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
    Gom kết quả theo từng lần chạy cho cả dataset/upload/webcam.

    Chỉ hiển thị run hợp lệ:
        - Có output video, hoặc
        - Có metrics JSON

    Những run chỉ có TXT sẽ bị ẩn khỏi Results.
    Trường hợp này hay xảy ra khi webcam bấm Start Record nhưng chưa ghi được frame/video.
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
