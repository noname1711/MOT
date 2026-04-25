import os
import shutil
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    url_for,
    Response,
    send_from_directory,
    jsonify
)
from werkzeug.utils import secure_filename

from src.utils import save_metrics, analyze_tracking_txt, save_tracking_statistics
from src.video_processor import VideoProcessor
from src.webcam_processor import WebcamProcessor


app = Flask(__name__)

# ===== Cấu hình thư mục =====
UPLOAD_FOLDER = "uploads"
STATIC_OUTPUT_FOLDER = "static/outputs"
RESULT_VIDEO_FOLDER = "results/videos"
RESULT_TXT_FOLDER = "results/txt"
RESULT_METRICS_FOLDER = "results/metrics"

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_OUTPUT_FOLDER"] = STATIC_OUTPUT_FOLDER
app.config["RESULT_VIDEO_FOLDER"] = RESULT_VIDEO_FOLDER
app.config["RESULT_TXT_FOLDER"] = RESULT_TXT_FOLDER
app.config["RESULT_METRICS_FOLDER"] = RESULT_METRICS_FOLDER

# Webcam processor dùng chung cho stream và record
webcam_processor = None
latest_webcam_recording = {
    "filename": None,
    "static_path": None,
    "result_path": None
}


def get_webcam_processor() -> WebcamProcessor:
    """
    Tạo hoặc lấy WebcamProcessor dùng chung.
    """

    global webcam_processor

    if webcam_processor is None:
        webcam_processor = WebcamProcessor(
            camera_index=0,
            yolo_model_path="models/yolov5n.pt",
            conf_threshold=0.35,
            image_size=416,
            device="cpu"
        )

    return webcam_processor


def allowed_file(filename: str) -> bool:
    """
    Kiểm tra file upload có đúng định dạng video cho phép không.
    """

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """
    Trang chủ.
    """

    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    """
    Trang upload video và chạy multi-object tracking.
    """

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

        if not allowed_file(file.filename):
            error = "Định dạng video không hợp lệ. Chỉ hỗ trợ: mp4, avi, mov, mkv."
            return render_template("upload.html", result=result, error=error)

        original_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        input_filename = f"{timestamp}_{original_filename}"
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)

        output_video_name = f"{timestamp}_tracking.mp4"
        output_txt_name = f"{timestamp}_tracking.txt"

        static_output_video_path = os.path.join(
            app.config["STATIC_OUTPUT_FOLDER"],
            output_video_name
        )

        result_video_path = os.path.join(
            app.config["RESULT_VIDEO_FOLDER"],
            output_video_name
        )

        result_txt_path = os.path.join(
            app.config["RESULT_TXT_FOLDER"],
            output_txt_name
        )

        try:
            file.save(input_path)

            processor = VideoProcessor(
                yolo_model_path="models/yolov5n.pt",
                conf_threshold=0.35,
                image_size=416,
                device="cpu"
            )

            info = processor.process(
                input_video_path=input_path,
                output_video_path=static_output_video_path,
                output_txt_path=result_txt_path
            )

            if static_output_video_path != result_video_path:
                shutil.copyfile(static_output_video_path, result_video_path)

            tracking_statistics = analyze_tracking_txt(
                tracking_txt_path=result_txt_path,
                fps=info["original_fps"]
            )

            statistics_path = save_tracking_statistics(
                statistics=tracking_statistics,
                output_dir=app.config["RESULT_METRICS_FOLDER"],
                timestamp=timestamp
            )

            metrics_data = {
                "input_video_path": input_path,
                "output_video_path": result_video_path,
                "web_output_video_path": static_output_video_path,
                "tracking_txt_path": result_txt_path,
                "statistics_path": statistics_path,
                "total_frames": info["total_frames"],
                "elapsed_time": round(info["elapsed_time"], 2),
                "process_fps": round(info["process_fps"], 2),
                "original_fps": round(info["original_fps"], 2),
                "width": info["width"],
                "height": info["height"],
                "model": "YOLOv5n + DeepSORT",
                "yolo_model_path": "models/yolov5n.pt",
                "device": "CPU",
                "confidence_threshold": 0.35,
                "image_size": 416,
                "created_at": timestamp,
                "total_tracks": tracking_statistics["total_tracks"],
                "total_tracking_rows": tracking_statistics["total_tracking_rows"]
            }

            metrics_path = save_metrics(
                metrics=metrics_data,
                output_dir=app.config["RESULT_METRICS_FOLDER"]
            )

            result = {
                "total_frames": info["total_frames"],
                "elapsed_time": round(info["elapsed_time"], 2),
                "process_fps": round(info["process_fps"], 2),
                "original_fps": round(info["original_fps"], 2),
                "width": info["width"],
                "height": info["height"],
                "video_url": url_for("static", filename=f"outputs/{output_video_name}"),
                "txt_path": result_txt_path,
                "video_path": result_video_path,
                "metrics_path": metrics_path,
                "statistics_path": statistics_path,
                "tracking_statistics": tracking_statistics,
                "txt_download_url": url_for("download_txt", filename=output_txt_name),
                "video_download_url": url_for("download_video", filename=output_video_name)
            }

        except Exception as e:
            error = f"Lỗi khi xử lý video: {str(e)}"

    return render_template("upload.html", result=result, error=error)


@app.route("/webcam")
def webcam():
    """
    Trang hiển thị webcam realtime.
    """

    return render_template("webcam.html")


@app.route("/webcam_feed")
def webcam_feed():
    """
    Route stream webcam frame cho trình duyệt.
    """

    processor = get_webcam_processor()

    return Response(
        processor.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/webcam/start_record", methods=["POST"])
def webcam_start_record():
    """
    Bắt đầu ghi webcam overlay.
    """

    global latest_webcam_recording

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{timestamp}_webcam_recording.mp4"

    static_output_path = os.path.join(STATIC_OUTPUT_FOLDER, output_name)
    result_output_path = os.path.join(RESULT_VIDEO_FOLDER, output_name)

    processor = get_webcam_processor()

    info = processor.start_recording(
        static_output_path=static_output_path,
        result_output_path=result_output_path,
        fps=20.0
    )

    if info.get("success"):
        latest_webcam_recording = {
            "filename": output_name,
            "static_path": static_output_path,
            "result_path": result_output_path
        }

    return jsonify({
        "success": info.get("success", False),
        "message": info.get("message", ""),
        "filename": output_name if info.get("success") else latest_webcam_recording.get("filename")
    })


@app.route("/webcam/stop_record", methods=["POST"])
def webcam_stop_record():
    """
    Dừng ghi webcam overlay.
    """

    processor = get_webcam_processor()
    info = processor.stop_recording()

    download_url = None
    video_url = None

    if info.get("success") and latest_webcam_recording.get("filename"):
        download_url = url_for(
            "download_video",
            filename=latest_webcam_recording["filename"]
        )

        video_url = url_for(
            "static",
            filename=f"outputs/{latest_webcam_recording['filename']}"
        )

    return jsonify({
        "success": info.get("success", False),
        "message": info.get("message", ""),
        "frame_count": info.get("frame_count", 0),
        "fps": info.get("fps", 0),
        "download_url": download_url,
        "video_url": video_url,
        "filename": latest_webcam_recording.get("filename")
    })


@app.route("/download/txt/<filename>")
def download_txt(filename):
    """
    Tải file tracking txt.
    """

    return send_from_directory(
        RESULT_TXT_FOLDER,
        filename,
        as_attachment=True
    )


@app.route("/download/video/<filename>")
def download_video(filename):
    """
    Tải video tracking kết quả.
    """

    return send_from_directory(
        RESULT_VIDEO_FOLDER,
        filename,
        as_attachment=True
    )


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(RESULT_VIDEO_FOLDER, exist_ok=True)
    os.makedirs(RESULT_TXT_FOLDER, exist_ok=True)
    os.makedirs(RESULT_METRICS_FOLDER, exist_ok=True)

    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)