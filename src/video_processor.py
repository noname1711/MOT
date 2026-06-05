import csv
import os
import time
from typing import Any, Dict

import cv2

from src.pipeline import VehicleTrackingPipeline
from src.utils import convert_video_to_h264, ensure_dir, write_tracking_header, write_tracking_rows


class VideoProcessor:
    """
    Process video files for upload, dataset, and comparison modes.
    """

    def __init__(
        self,
        yolo_model_path: str = "models/yolov5n.pt",
        conf_threshold: float = 0.35,
        image_size: int = 416,
        device: str = "cpu",
        tracker_type: str = "deepsort",
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
    ):
        self.tracker_type = tracker_type

        self.pipeline = VehicleTrackingPipeline(
            yolo_model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            image_size=image_size,
            device=device,
            tracker_type=tracker_type,
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
        )

    def process(
        self,
        input_video_path: str,
        output_video_path: str,
        output_txt_path: str,
        max_frames: int | None = None
    ) -> Dict[str, Any]:
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ensure_dir(os.path.dirname(output_video_path))
        ensure_dir(os.path.dirname(output_txt_path))

        base_name, ext = os.path.splitext(output_video_path)
        temp_output_video_path = f"{base_name}_temp{ext}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            temp_output_video_path,
            fourcc,
            original_fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {temp_output_video_path}")

        frame_id = 0
        detector_total_time = 0.0
        tracker_total_time = 0.0
        start_time = time.time()

        try:
            with open(output_txt_path, "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.writer(f)
                write_tracking_header(csv_writer)

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_id += 1

                    result = self.pipeline.process_frame(frame, draw=True)

                    detector_total_time += result["detector_time"]
                    tracker_total_time += result["tracker_time"]

                    output_frame = result["frame"]
                    tracks = result["tracks"]

                    write_tracking_rows(csv_writer, frame_id, tracks)

                    writer.write(output_frame)

                    if frame_id % 20 == 0:
                        print(
                            f"[{self.tracker_type}] Processed "
                            f"{frame_id}/{total_input_frames} frames"
                        )

                    if max_frames is not None and frame_id >= max_frames:
                        break

        finally:
            cap.release()
            writer.release()

        elapsed_time = time.time() - start_time
        process_fps = frame_id / elapsed_time if elapsed_time > 0 else 0

        convert_video_to_h264(
            input_path=temp_output_video_path,
            output_path=output_video_path
        )

        if os.path.exists(temp_output_video_path):
            os.remove(temp_output_video_path)

        return {
            "input_video_path": input_video_path,
            "output_video_path": output_video_path,
            "output_txt_path": output_txt_path,
            "tracker_type": self.tracker_type,
            "total_frames": frame_id,
            "total_input_frames": total_input_frames,
            "elapsed_time": elapsed_time,
            "process_fps": process_fps,
            "original_fps": original_fps,
            "width": width,
            "height": height,
            "avg_detector_time": detector_total_time / frame_id if frame_id else 0,
            "avg_tracker_time": tracker_total_time / frame_id if frame_id else 0
        }
