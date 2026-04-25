from typing import List, Dict, Any, Optional

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSORTTracker:
    """
    DeepSORTTracker dùng để gán ID cho các detection qua nhiều frame.

    Input của update() là output từ YOLODetector.detect():

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

    Output của update() là list track:

        [
            {
                "track_id": "1",
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x, y, w, h],
                "confidence": 0.91,
                "class_id": 0,
                "class_name": "person",
                "visibility": 1,
                "matched_iou": 0.83
            },
            ...
        ]
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        nn_budget=None
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_gpu=False
        )

    def update(self, detections: List[Dict[str, Any]], frame) -> List[Dict[str, Any]]:
        """
        Cập nhật tracker bằng detection của frame hiện tại.

        Với mỗi track, hàm này tìm detection hiện tại có IoU cao nhất
        để lấy confidence/class thật từ YOLO.
        """

        if frame is None:
            return []

        height, width = frame.shape[:2]

        deepsort_detections = []

        for det in detections:
            x, y, w, h = det["bbox_xywh"]
            confidence = det["confidence"]
            class_name = det["class_name"]

            if w <= 0 or h <= 0:
                continue

            # Format của deep-sort-realtime:
            # ([left, top, width, height], confidence, class_name)
            deepsort_detections.append(
                ([x, y, w, h], confidence, class_name)
            )

        raw_tracks = self.tracker.update_tracks(
            deepsort_detections,
            frame=frame
        )

        tracks = []

        for track in raw_tracks:
            if not track.is_confirmed():
                continue

            track_id = str(track.track_id)

            x1, y1, x2, y2 = track.to_ltrb()
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(width - 1, x2))
            y2 = int(min(height - 1, y2))

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            track_bbox_xyxy = [x1, y1, x2, y2]

            matched_detection = self._find_best_detection_for_track(
                track_bbox_xyxy=track_bbox_xyxy,
                detections=detections,
                iou_threshold=0.1
            )

            if matched_detection is not None:
                confidence = float(matched_detection["confidence"])
                class_id = int(matched_detection["class_id"])
                class_name = str(matched_detection["class_name"])
                visibility = 1
                matched_iou = float(matched_detection["matched_iou"])
            else:
                # Không có detection khớp ở frame hiện tại.
                # Đây có thể là track được DeepSORT dự đoán bằng motion model.
                confidence = 0.0
                class_id = -1
                class_name = "predicted"
                visibility = 0
                matched_iou = 0.0

            tracks.append({
                "track_id": track_id,
                "bbox_xyxy": track_bbox_xyxy,
                "bbox_xywh": [x1, y1, w, h],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "visibility": visibility,
                "matched_iou": matched_iou
            })

        return tracks

    def draw_tracks(self, frame, tracks: List[Dict[str, Any]]):
        """
        Vẽ bounding box và track ID lên frame.
        """

        output_frame = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track["bbox_xyxy"]
            track_id = track["track_id"]
            class_name = track["class_name"]
            confidence = track.get("confidence", 0.0)

            if track.get("visibility", 0) == 1:
                label = f"ID {track_id} | {class_name} {confidence:.2f}"
            else:
                label = f"ID {track_id} | predicted"

            color = self._get_color(track_id)

            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                output_frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return output_frame

    @staticmethod
    def _find_best_detection_for_track(
        track_bbox_xyxy: List[int],
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm detection có IoU cao nhất với bbox của track.
        Detection tìm được dùng để lấy confidence/class thật từ YOLO.
        """

        best_detection = None
        best_iou = 0.0

        for det in detections:
            det_bbox_xyxy = det["bbox_xyxy"]
            iou = DeepSORTTracker._compute_iou(track_bbox_xyxy, det_bbox_xyxy)

            if iou > best_iou:
                best_iou = iou
                best_detection = det

        if best_detection is None or best_iou < iou_threshold:
            return None

        matched_detection = dict(best_detection)
        matched_detection["matched_iou"] = best_iou

        return matched_detection

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        """
        Tính IoU giữa hai bounding box dạng [x1, y1, x2, y2].
        """

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @staticmethod
    def _get_color(track_id: str):
        """
        Tạo màu cố định theo track_id để mỗi ID có màu riêng.
        """

        try:
            track_num = int(track_id)
        except ValueError:
            track_num = sum(ord(c) for c in track_id)

        r = (37 * track_num) % 255
        g = (17 * track_num) % 255
        b = (29 * track_num) % 255

        return int(b), int(g), int(r)