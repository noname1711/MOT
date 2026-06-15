from typing import Any, Dict, List, Optional

from deep_sort_realtime.deepsort_tracker import DeepSort

from src.custom_deepsort import CustomDeepSORTTracker


class DeepSORTTracker:
    """
    Wrapper around the original deep-sort-realtime tracker.

    This tracker is kept as the baseline for comparison with the custom
    DeepSORT implementation.
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
        if frame is None:
            return []

        frame_height, frame_width = frame.shape[:2]

        # Convert project detections to deep-sort-realtime input format.
        # Example:
        #   project: {"bbox_xywh": [100,50,80,40], "confidence": 0.8, "class_name": "car"}
        #   DeepSORT: ([100,50,80,40], 0.8, "car")
        deepsort_detections = []

        for det in detections:
            x, y, w, h = det["bbox_xywh"]

            if w <= 0 or h <= 0:
                continue

            deepsort_detections.append((
                [x, y, w, h],
                float(det["confidence"]),
                str(det["class_name"])
            ))

        raw_tracks = self.tracker.update_tracks(
            deepsort_detections,
            frame=frame
        )

        tracks: List[Dict[str, Any]] = []

        for track in raw_tracks:
            if not track.is_confirmed():
                continue

            track_id = str(track.track_id)

            x1, y1, x2, y2 = track.to_ltrb()

            # Clamp coordinates to frame boundaries.
            # Example: x1=-5 -> 0; x2=1300 with width=1280 -> 1279.
            x1 = int(max(0, min(frame_width - 1, x1)))
            y1 = int(max(0, min(frame_height - 1, y1)))
            x2 = int(max(0, min(frame_width - 1, x2)))
            y2 = int(max(0, min(frame_height - 1, y2)))

            # Convert [x1,y1,x2,y2] to [x,y,w,h].
            # Example: x1=100, x2=180 -> w=80; y1=50, y2=90 -> h=40.
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            track_bbox = [x1, y1, x2, y2]

            matched_detection = self._find_best_detection_for_track(
                track_bbox_xyxy=track_bbox,
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
                confidence = 0.0
                class_id = -1
                class_name = "predicted"
                visibility = 0
                matched_iou = 0.0

            tracks.append({
                "track_id": track_id,
                "bbox_xyxy": track_bbox,
                "bbox_xywh": [x1, y1, w, h],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "visibility": visibility,
                "matched_iou": matched_iou
            })

        return tracks

    @staticmethod
    def _find_best_detection_for_track(
        track_bbox_xyxy: List[int],
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        best_detection = None
        best_iou = 0.0

        for det in detections:
            # Keep the detection with the largest overlap with this track box.
            # Example: best_iou=0.20, current_iou=0.65 -> update best_detection.
            iou = DeepSORTTracker.compute_iou(track_bbox_xyxy, det["bbox_xyxy"])

            if iou > best_iou:
                best_iou = iou
                best_detection = det

        if best_detection is None or best_iou < iou_threshold:
            return None

        matched = dict(best_detection)
        matched["matched_iou"] = best_iou

        return matched

    @staticmethod
    def compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """
        Example:
            box_a = [0,0,4,4], box_b = [2,2,6,6]
            intersection area = 4, union = 16 + 16 - 4 = 28
            IoU = 4 / 28 = 0.1429
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area


def create_tracker(
    tracker_type: str = "deepsort",
    max_age: int = 30,
    n_init: int = 3,
    max_cosine_distance: float = 0.4,
    nn_budget=None,
):
    """
    Create a tracker instance from a tracker type string.

    Supported values:
        - "deepsort": original deep-sort-realtime baseline.
        - "custom": optimized custom DeepSORT implementation.
        - "custom_deepsort": alias for the custom tracker.
    """

    normalized = str(tracker_type).strip().lower()

    if normalized in {"deepsort", "deep_sort", "original", "baseline"}:
        return DeepSORTTracker(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

    if normalized in {"custom", "custom_deepsort", "custom-deepsort"}:
        return CustomDeepSORTTracker(
            max_age=25,
            n_init=2,
            max_cosine_distance=0.50,
            max_iou_distance=0.72,
            nn_budget=nn_budget,
        )

    raise ValueError(
        f"Invalid tracker_type: {tracker_type}. "
        f"Supported values: deepsort, custom."
    )
