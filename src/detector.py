from typing import List, Dict, Any

import cv2
from ultralytics import YOLO


class YOLODetector:
    """
    YOLODetector dùng để phát hiện người trong từng frame.

    Model sử dụng:
        models/yolov5n.pt

    Output của detect() là danh sách detection dạng:

        [
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x, y, w, h],
                "confidence": 0.91,
                "class_id": 0,
                "class_name": "person"
            },
            ...
        ]

    Trong COCO dataset:
        class_id = 0 là person.
    """

    def __init__(
        self,
        model_path: str = "models/yolov5n.pt",
        conf_threshold: float = 0.35,
        image_size: int = 416,
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        self.device = device

        # Load YOLO model
        self.model = YOLO(self.model_path)

    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Phát hiện người trong một frame.

        Args:
            frame: ảnh dạng numpy array, đọc từ OpenCV, định dạng BGR.

        Returns:
            List[Dict]: danh sách detection.
        """

        if frame is None:
            return []

        height, width = frame.shape[:2]

        # Chỉ detect class person = 0
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            classes=[0],
            device=self.device,
            verbose=False
        )

        detections = []

        if len(results) == 0:
            return detections

        boxes = results[0].boxes

        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            x1, y1, x2, y2 = xyxy

            # Giới hạn tọa độ nằm trong frame
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(width - 1, x2))
            y2 = int(min(height - 1, y2))

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            detections.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, w, h],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": "person"
            })

        return detections

    def draw_detections(self, frame, detections):
        """
        Vẽ kết quả detection lên frame.
        Hàm này dùng để test riêng detector, chưa có tracking ID.
        """

        output_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            confidence = det["confidence"]

            label = f"person {confidence:.2f}"

            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output_frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return output_frame