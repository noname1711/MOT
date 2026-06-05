import csv
import os
import shutil
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.pipeline import VehicleTrackingPipeline
from src.utils import (
    convert_video_to_h264,
    ensure_dir,
    write_tracking_header,
    write_tracking_rows,
)
from src.visualizer import draw_hud


class WebcamProcessor:
    """
    Background-thread webcam processor.

    The worker thread keeps camera capture, YOLO inference, tracking, metrics,
    and optional recording independent from the browser stream consumer.
    """

    def __init__(
        self,
        camera_index: int = 0,
        yolo_model_path: str = "models/yolov5n.pt",
        conf_threshold: float = 0.35,
        image_size: int = 416,
        device: str = "cpu",
        tracker_type: str = "deepsort",
    ):
        self.camera_index = camera_index
        self.tracker_type = tracker_type

        self.pipeline = VehicleTrackingPipeline(
            yolo_model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            image_size=image_size,
            device=device,
            tracker_type=tracker_type,
        )

        self.cap = None

        # Thread state
        self.lock = threading.RLock()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.latest_frame_jpeg: Optional[bytes] = None
        self.latest_error: Optional[str] = None

        # Session metrics
        self.session_started_at = time.time()
        self.session_frame_count = 0
        self.session_track_ids = set()
        self.session_max_active_tracks = 0
        self.current_active_tracks = 0
        self.last_fps = 0.0
        self.average_fps = 0.0
        self.last_frame_time = time.time()

        # Recording state
        self.is_recording = False
        self.record_writer = None
        self.record_temp_path = None
        self.record_static_path = None
        self.record_result_path = None
        self.record_txt_path = None
        self.record_txt_file = None
        self.record_csv_writer = None
        self.record_fps = 20.0
        self.record_frame_count = 0
        self.record_started_at = None
        self.record_track_ids = set()
        self.record_max_active_tracks = 0

    # ============================================================
    # Worker lifecycle
    # ============================================================

    def start(self) -> Dict[str, Any]:
        """
        Start the background worker if it is not already running.
        """

        with self.lock:
            if self.running:
                return {
                    "success": True,
                    "message": "Webcam worker is already running.",
                }

            self.running = True
            self.latest_error = None
            self.latest_frame_jpeg = None

            self.session_started_at = time.time()
            self.session_frame_count = 0
            self.session_track_ids = set()
            self.session_max_active_tracks = 0
            self.current_active_tracks = 0
            self.last_fps = 0.0
            self.average_fps = 0.0
            self.last_frame_time = time.time()

            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
            )
            self.worker_thread.start()

        return {
            "success": True,
            "message": "Webcam worker started.",
        }

    def stop(self) -> Dict[str, Any]:
        """
        Stop the background worker and release the webcam.
        """

        with self.lock:
            self.running = False

        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None

        with self.lock:
            if self.is_recording:
                self._stop_recording_locked()

        self._release_camera()

        return {
            "success": True,
            "message": "Webcam worker stopped.",
        }

    def _open_camera(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _release_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ============================================================
    # Recording
    # ============================================================

    def start_recording(
        self,
        static_output_path: str,
        result_output_path: str,
        txt_output_path: str,
        fps: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Start webcam recording.

        The worker is started automatically if it is not already running.
        """

        self.start()

        with self.lock:
            if self.is_recording:
                return {
                    "success": False,
                    "message": "Webcam is already recording.",
                }

            ensure_dir(os.path.dirname(static_output_path))
            ensure_dir(os.path.dirname(result_output_path))
            ensure_dir(os.path.dirname(txt_output_path))

            base_name, ext = os.path.splitext(static_output_path)
            temp_path = f"{base_name}_temp{ext}"

            self.is_recording = True
            self.record_writer = None
            self.record_temp_path = temp_path
            self.record_static_path = static_output_path
            self.record_result_path = result_output_path
            self.record_txt_path = txt_output_path
            self.record_fps = fps if fps > 0 else 20.0
            self.record_frame_count = 0
            self.record_started_at = time.time()
            self.record_track_ids = set()
            self.record_max_active_tracks = 0

            self.record_txt_file = open(txt_output_path, "w", newline="", encoding="utf-8")
            self.record_csv_writer = csv.writer(self.record_txt_file)
            write_tracking_header(self.record_csv_writer)
            self.record_txt_file.flush()

        return {
            "success": True,
            "message": "Webcam recording started.",
            "static_output_path": static_output_path,
            "result_output_path": result_output_path,
            "txt_output_path": txt_output_path,
        }

    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop webcam recording, convert the video, and return recording metrics.
        """

        with self.lock:
            return self._stop_recording_locked()

    def _stop_recording_locked(self) -> Dict[str, Any]:
        if not self.is_recording:
            return {
                "success": False,
                "message": "Webcam is not recording.",
            }

        self.is_recording = False

        if self.record_writer is not None:
            self.record_writer.release()
            self.record_writer = None

        if self.record_txt_file is not None:
            self.record_txt_file.flush()
            self.record_txt_file.close()
            self.record_txt_file = None
            self.record_csv_writer = None

        if not self.record_temp_path or not os.path.exists(self.record_temp_path):
            self._clear_recording_state()
            return {
                "success": False,
                "message": "No frames were recorded. Wait a few seconds after the webcam appears, then press Stop.",
            }

        if self.record_frame_count <= 0:
            if os.path.exists(self.record_temp_path):
                os.remove(self.record_temp_path)

            self._clear_recording_state()
            return {
                "success": False,
                "message": "The recorded video does not contain any frames.",
            }

        convert_video_to_h264(
            input_path=self.record_temp_path,
            output_path=self.record_static_path,
        )

        if self.record_static_path != self.record_result_path:
            shutil.copyfile(self.record_static_path, self.record_result_path)

        if os.path.exists(self.record_temp_path):
            os.remove(self.record_temp_path)

        duration = time.time() - self.record_started_at if self.record_started_at else 0
        avg_fps = self.record_frame_count / duration if duration > 0 else 0

        result = {
            "success": True,
            "message": "Webcam recording stopped.",
            "tracker_type": self.tracker_type,
            "frame_count": self.record_frame_count,
            "duration_sec": round(duration, 4),
            "fps": round(avg_fps, 4),
            "unique_tracks": len(self.record_track_ids),
            "max_active_tracks": self.record_max_active_tracks,
            "static_output_path": self.record_static_path,
            "result_output_path": self.record_result_path,
            "txt_output_path": self.record_txt_path,
        }

        self._clear_recording_state()

        return result

    def _clear_recording_state(self) -> None:
        self.record_temp_path = None
        self.record_static_path = None
        self.record_result_path = None
        self.record_txt_path = None
        self.record_frame_count = 0
        self.record_started_at = None
        self.record_track_ids = set()
        self.record_max_active_tracks = 0

    def _write_recording_frame_locked(self, frame, tracks) -> None:
        if not self.is_recording:
            return

        height, width = frame.shape[:2]

        if self.record_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.record_writer = cv2.VideoWriter(
                self.record_temp_path,
                fourcc,
                self.record_fps,
                (width, height),
            )

            if not self.record_writer.isOpened():
                self.is_recording = False
                self.latest_error = f"Cannot create webcam recording file: {self.record_temp_path}"
                raise RuntimeError(self.latest_error)

        self.record_frame_count += 1

        for track in tracks:
            self.record_track_ids.add(str(track["track_id"]))

        self.record_max_active_tracks = max(
            self.record_max_active_tracks,
            len(tracks),
        )

        self.record_writer.write(frame)

        if self.record_csv_writer is not None:
            write_tracking_rows(self.record_csv_writer, self.record_frame_count, tracks)

            if self.record_frame_count % 10 == 0 and self.record_txt_file is not None:
                self.record_txt_file.flush()

    # ============================================================
    # Metrics and stream
    # ============================================================

    def get_live_metrics(self) -> Dict[str, Any]:
        with self.lock:
            duration = time.time() - self.session_started_at
            avg_fps = self.session_frame_count / duration if duration > 0 else 0

            return {
                "worker_running": self.running,
                "tracker_type": self.tracker_type,
                "camera_opened": self.cap is not None and self.cap.isOpened(),
                "session_duration_sec": round(duration, 2),
                "session_frames": self.session_frame_count,
                "live_fps": round(self.last_fps, 2),
                "average_fps": round(avg_fps, 2),
                "total_unique_tracks": len(self.session_track_ids),
                "current_active_tracks": self.current_active_tracks,
                "max_active_tracks": self.session_max_active_tracks,
                "is_recording": self.is_recording,
                "record_frame_count": self.record_frame_count,
                "latest_error": self.latest_error,
            }

    def generate_frames(self):
        """
        Yield MJPEG frames for the /webcam_feed route.

        YOLO and tracking are handled by the worker thread; this generator only
        streams the latest encoded JPEG frame.
        """

        self.start()

        while True:
            with self.lock:
                frame_bytes = self.latest_frame_jpeg
                error = self.latest_error

            if frame_bytes is None:
                placeholder = self._make_placeholder_frame(error or "Starting webcam...")
                success, buffer = cv2.imencode(".jpg", placeholder)

                if not success:
                    time.sleep(0.1)
                    continue

                frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

            time.sleep(0.03)

    def _worker_loop(self) -> None:
        try:
            self._open_camera()

            while True:
                with self.lock:
                    if not self.running:
                        break

                ret, frame = self.cap.read()

                if not ret:
                    with self.lock:
                        self.latest_error = "Cannot read frames from the webcam."
                    time.sleep(0.05)
                    continue

                now = time.time()
                delta = now - self.last_frame_time
                live_fps = 1.0 / delta if delta > 0 else 0.0
                self.last_frame_time = now

                result = self.pipeline.process_frame(frame, draw=True)
                output_frame = result["frame"]
                tracks = result["tracks"]

                with self.lock:
                    self.session_frame_count += 1
                    self.last_fps = live_fps

                    for track in tracks:
                        self.session_track_ids.add(str(track["track_id"]))

                    self.current_active_tracks = len(tracks)
                    self.session_max_active_tracks = max(
                        self.session_max_active_tracks,
                        len(tracks),
                    )

                    duration = now - self.session_started_at
                    self.average_fps = (
                        self.session_frame_count / duration if duration > 0 else 0
                    )

                    hud = {
                        "Tracker": self.tracker_type,
                        "FPS": f"{self.last_fps:.1f}",
                        "Current Vehicles": self.current_active_tracks,
                        "Total IDs": len(self.session_track_ids),
                        "Recording": "ON" if self.is_recording else "OFF",
                        "Record Frames": self.record_frame_count,
                    }

                output_frame = draw_hud(output_frame, hud)

                with self.lock:
                    self._write_recording_frame_locked(output_frame, tracks)

                success, buffer = cv2.imencode(".jpg", output_frame)

                if success:
                    with self.lock:
                        self.latest_frame_jpeg = buffer.tobytes()
                        self.latest_error = None

        except Exception as exc:
            with self.lock:
                self.latest_error = str(exc)
                self.running = False

        finally:
            self._release_camera()

    @staticmethod
    def _make_placeholder_frame(message: str):
        frame = np.full((360, 640, 3), 255, dtype=np.uint8)

        cv2.putText(
            frame,
            "Webcam",
            (32, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (2, 132, 199),
            3,
        )

        cv2.putText(
            frame,
            message[:70],
            (32, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (15, 23, 42),
            2,
        )

        return frame
