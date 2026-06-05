from typing import Any, Dict, List

import cv2
from ultralytics import YOLO

from src.utils import VEHICLE_CLASS_IDS, VEHICLE_CLASS_NAMES


class YOLODetector:
    """
    YOLOv5/Ultralytics wrapper for vehicle detection.

    Target COCO vehicle classes:
        2 -> car
        3 -> motorcycle
        5 -> bus
        7 -> truck
    """

    def __init__(
        self,
        model_path: str = "models/yolov5n.pt",
        conf_threshold: float = 0.35,
        image_size: int = 416,
        device: str = "cpu",
        target_classes: List[int] | None = None
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        self.device = device
        self.target_classes = target_classes or VEHICLE_CLASS_IDS

        self.model = YOLO(self.model_path)

    def detect(self, frame) -> List[Dict[str, Any]]:
        if frame is None:
            return []

        frame_height, frame_width = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        boxes = results[0].boxes

        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            if class_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = xyxy

            x1 = int(max(0, min(frame_width - 1, x1)))
            y1 = int(max(0, min(frame_height - 1, y1)))
            x2 = int(max(0, min(frame_width - 1, x2)))
            y2 = int(max(0, min(frame_height - 1, y2)))

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            detections.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, w, h],
                "confidence": conf,
                "class_id": class_id,
                "class_name": VEHICLE_CLASS_NAMES.get(class_id, f"class_{class_id}")
            })

        return detections

    def draw_detections(self, frame, detections: List[Dict[str, Any]]):
        output_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            label = f'{det["class_name"]} {det["confidence"]:.2f}'

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output_frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return output_frame
