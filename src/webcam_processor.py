import os
import shutil
from typing import Dict, Any, Optional

import cv2

from src.detector import YOLODetector
from src.tracker import DeepSORTTracker
from src.utils import convert_video_to_h264


class WebcamProcessor:
    """
    WebcamProcessor xử lý luồng webcam realtime.

    Chức năng:
        - Stream webcam lên web
        - YOLODetector phát hiện người
        - DeepSORTTracker gán ID
        - Có thể ghi lại video đã overlay bounding box + ID
    """

    def __init__(
        self,
        camera_index: int = 0,
        yolo_model_path: str = "models/yolov5n.pt",
        conf_threshold: float = 0.35,
        image_size: int = 416,
        device: str = "cpu"
    ):
        self.camera_index = camera_index

        self.detector = YOLODetector(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            image_size=image_size,
            device=device
        )

        self.tracker = DeepSORTTracker(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.4
        )

        self.cap = None

        # Recording state
        self.is_recording = False
        self.record_writer = None
        self.record_temp_path = None
        self.record_static_path = None
        self.record_result_path = None
        self.record_fps = 20.0
        self.record_frame_count = 0
        self.record_size = None

    def open_camera(self):
        """
        Mở webcam.
        """

        if self.cap is not None and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được webcam index {self.camera_index}")

        # Giảm độ phân giải để chạy CPU mượt hơn
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def release_camera(self):
        """
        Giải phóng webcam.
        """

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def start_recording(
        self,
        static_output_path: str,
        result_output_path: str,
        fps: float = 20.0
    ) -> Dict[str, Any]:
        """
        Bắt đầu ghi video webcam đã overlay.

        Video được ghi tạm bằng mp4v, sau đó khi stop sẽ convert sang H.264.
        """

        if self.is_recording:
            return {
                "success": False,
                "message": "Webcam đang ghi hình.",
                "static_output_path": self.record_static_path,
                "result_output_path": self.record_result_path
            }

        os.makedirs(os.path.dirname(static_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(result_output_path), exist_ok=True)

        base_name, ext = os.path.splitext(static_output_path)
        temp_path = f"{base_name}_temp{ext}"

        self.is_recording = True
        self.record_writer = None
        self.record_temp_path = temp_path
        self.record_static_path = static_output_path
        self.record_result_path = result_output_path
        self.record_fps = fps if fps > 0 else 20.0
        self.record_frame_count = 0
        self.record_size = None

        return {
            "success": True,
            "message": "Đã bắt đầu ghi hình webcam.",
            "static_output_path": static_output_path,
            "result_output_path": result_output_path
        }

    def stop_recording(self) -> Dict[str, Any]:
        """
        Dừng ghi video webcam và convert sang H.264.
        """

        if not self.is_recording:
            return {
                "success": False,
                "message": "Webcam hiện không ghi hình."
            }

        self.is_recording = False

        if self.record_writer is not None:
            self.record_writer.release()
            self.record_writer = None

        if self.record_temp_path is None or not os.path.exists(self.record_temp_path):
            return {
                "success": False,
                "message": "Chưa có frame nào được ghi."
            }

        if self.record_frame_count <= 0:
            if os.path.exists(self.record_temp_path):
                os.remove(self.record_temp_path)

            return {
                "success": False,
                "message": "Video ghi hình không có frame nào."
            }

        # Convert file tạm sang H.264 để trình duyệt phát được
        convert_video_to_h264(
            input_path=self.record_temp_path,
            output_path=self.record_static_path
        )

        # Copy sang results/videos để lưu kết quả
        if self.record_static_path != self.record_result_path:
            shutil.copyfile(self.record_static_path, self.record_result_path)

        if os.path.exists(self.record_temp_path):
            os.remove(self.record_temp_path)

        result = {
            "success": True,
            "message": "Đã dừng ghi hình webcam.",
            "frame_count": self.record_frame_count,
            "fps": self.record_fps,
            "static_output_path": self.record_static_path,
            "result_output_path": self.record_result_path
        }

        self.record_temp_path = None
        self.record_static_path = None
        self.record_result_path = None
        self.record_frame_count = 0
        self.record_size = None

        return result

    def _write_recording_frame(self, frame):
        """
        Ghi một frame đã overlay vào video recording.
        """

        if not self.is_recording:
            return

        height, width = frame.shape[:2]

        if self.record_writer is None:
            self.record_size = (width, height)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.record_writer = cv2.VideoWriter(
                self.record_temp_path,
                fourcc,
                self.record_fps,
                self.record_size
            )

            if not self.record_writer.isOpened():
                self.is_recording = False
                raise RuntimeError(f"Không tạo được file ghi webcam: {self.record_temp_path}")

        self.record_writer.write(frame)
        self.record_frame_count += 1

    def generate_frames(self):
        """
        Generator trả về frame JPEG cho Flask streaming.

        Flask dùng multipart/x-mixed-replace để stream ảnh liên tục.
        """

        self.open_camera()

        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    break

                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)
                output_frame = self.tracker.draw_tracks(frame, tracks)

                # Ghi lại đúng frame đã overlay
                self._write_recording_frame(output_frame)

                success, buffer = cv2.imencode(".jpg", output_frame)

                if not success:
                    continue

                frame_bytes = buffer.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame_bytes +
                    b"\r\n"
                )

        finally:
            if self.is_recording:
                self.stop_recording()

            self.release_camera()