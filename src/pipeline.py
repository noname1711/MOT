import time
from typing import Any, Dict

from src.detector import YOLODetector
from src.tracker import create_tracker
from src.visualizer import draw_tracks


class VehicleTrackingPipeline:
    """
    Shared frame-processing pipeline.

    The same pipeline is used by dataset, upload, and webcam modes.

    Flow:
        frame -> YOLODetector -> Tracker -> Visualizer
    """

    def __init__(
        self,
        yolo_model_path: str = "models/yolov5n.pt",
        conf_threshold: float = 0.35,
        image_size: int = 416,
        device: str = "cpu",
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        tracker_type: str = "deepsort",
    ):
        self.tracker_type = tracker_type

        self.detector = YOLODetector(
            model_path=yolo_model_path,
            conf_threshold=conf_threshold,
            image_size=image_size,
            device=device
        )

        self.tracker = create_tracker(
            tracker_type=tracker_type,
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance
        )

    def process_frame(self, frame, draw: bool = True) -> Dict[str, Any]:
        detector_start = time.time()
        detections = self.detector.detect(frame)
        detector_time = time.time() - detector_start

        tracker_start = time.time()
        tracks = self.tracker.update(detections, frame)
        tracker_time = time.time() - tracker_start

        if draw:
            output_frame = draw_tracks(frame, tracks)
        else:
            output_frame = frame

        return {
            "frame": output_frame,
            "detections": detections,
            "tracks": tracks,
            "detector_time": detector_time,
            "tracker_time": tracker_time,
            "total_time": detector_time + tracker_time,
            "tracker_type": self.tracker_type,
        }
