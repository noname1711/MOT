from typing import Any, Dict, List

import numpy as np

from src.custom_deepsort.appearance import AppearanceExtractor, cosine_distance_matrix
from src.custom_deepsort.detection import Detection
from src.custom_deepsort.iou_matching import iou, iou_distance
from src.custom_deepsort.kalman_filter import KalmanFilter
from src.custom_deepsort.linear_assignment import min_cost_matching
from src.custom_deepsort.track import Track


class CustomDeepSORTTracker:
    """
    Custom DeepSORT tracker.

    This implementation uses a combined association cost built from:
        1. Appearance distance from HSV + LAB histogram features.
        2. IoU distance from bounding box overlap.
        3. Motion distance from the Kalman Filter prediction.

    Tracks are kept internally for short occlusions, while stale predicted boxes
    are filtered from the public output to reduce noisy metrics and visualization.
    """

    def __init__(
        self,
        max_age: int = 25,
        n_init: int = 2,
        max_cosine_distance: float = 0.50,
        max_iou_distance: float = 0.72,
        nn_budget=None,
        appearance_matching: bool = True,
        appearance_weight: float = 0.35,
        iou_weight: float = 0.55,
        motion_weight: float = 0.10,
        max_combined_distance: float = 0.78,
        max_motion_distance: float = 16.0,
        min_box_area: int = 64,
        max_predicted_age_to_output: int = 8,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance = max_iou_distance
        self.nn_budget = nn_budget
        self.appearance_matching = appearance_matching

        self.appearance_weight = appearance_weight
        self.iou_weight = iou_weight
        self.motion_weight = motion_weight
        self.max_combined_distance = max_combined_distance
        self.max_motion_distance = max_motion_distance
        self.min_box_area = min_box_area
        self.max_predicted_age_to_output = max_predicted_age_to_output

        self.kf = KalmanFilter()
        self.appearance_extractor = AppearanceExtractor()

        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]], frame) -> List[Dict[str, Any]]:
        if frame is None:
            return []

        frame_height, frame_width = frame.shape[:2]

        detection_objects = self._build_detections(detections, frame)

        for track in self.tracks:
            track.predict(self.kf)

        matches, unmatched_tracks, unmatched_detections = self._match(detection_objects)

        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            detection = detection_objects[det_idx]
            matched_iou = iou(track.to_ltrb(), detection.bbox_xyxy)
            track.update(self.kf, detection, matched_iou=matched_iou)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_detections:
            self._initiate_track(detection_objects[det_idx])

        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        return self._format_output_tracks(frame_width, frame_height)

    def _build_detections(self, detections: List[Dict[str, Any]], frame) -> List[Detection]:
        if len(detections) == 0:
            return []

        features = self.appearance_extractor.extract(frame, detections)

        detection_objects = []

        for det, feature in zip(detections, features):
            x, y, w, h = det["bbox_xywh"]

            if w <= 0 or h <= 0:
                continue

            if w * h < self.min_box_area:
                continue

            detection_objects.append(Detection.from_dict(det, feature))

        return detection_objects

    def _match(self, detections: List[Detection]):
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []

        confirmed_tracks = [
            idx for idx, track in enumerate(self.tracks)
            if track.is_confirmed()
        ]

        unconfirmed_tracks = [
            idx for idx, track in enumerate(self.tracks)
            if not track.is_confirmed()
        ]

        detection_indices = list(range(len(detections)))

        matches_a = []
        unmatched_confirmed = confirmed_tracks
        unmatched_detections = detection_indices

        # Stage A: match confirmed tracks using the combined association cost.
        if self.appearance_matching and len(confirmed_tracks) > 0:
            combined_cost = self._combined_cost_matrix(
                confirmed_tracks,
                detection_indices,
                detections,
            )

            matches_a, unmatched_confirmed, unmatched_detections = min_cost_matching(
                cost_matrix=combined_cost,
                max_distance=self.max_combined_distance,
                track_indices=confirmed_tracks,
                detection_indices=detection_indices,
            )

        # Stage B: fall back to IoU matching for unconfirmed or recently unmatched tracks.
        iou_track_candidates = unconfirmed_tracks + [
            idx for idx in unmatched_confirmed
            if self.tracks[idx].time_since_update <= 2
        ]

        remaining_unmatched_confirmed = [
            idx for idx in unmatched_confirmed
            if idx not in iou_track_candidates
        ]

        matches_b = []
        unmatched_iou_tracks = iou_track_candidates

        if len(iou_track_candidates) > 0 and len(unmatched_detections) > 0:
            iou_cost = self._iou_cost_matrix(
                iou_track_candidates,
                unmatched_detections,
                detections,
            )

            matches_b, unmatched_iou_tracks, unmatched_detections = min_cost_matching(
                cost_matrix=iou_cost,
                max_distance=self.max_iou_distance,
                track_indices=iou_track_candidates,
                detection_indices=unmatched_detections,
            )

        matches = matches_a + matches_b
        unmatched_tracks = remaining_unmatched_confirmed + unmatched_iou_tracks

        return matches, unmatched_tracks, unmatched_detections

    def _appearance_cost_matrix(
        self,
        track_indices: List[int],
        detection_indices: List[int],
        detections: List[Detection],
    ) -> np.ndarray:
        track_features = [self.tracks[idx].feature for idx in track_indices]
        detection_features = [detections[idx].feature for idx in detection_indices]

        cost = cosine_distance_matrix(track_features, detection_features)

        for row, track_idx in enumerate(track_indices):
            for col, det_idx in enumerate(detection_indices):
                if self.tracks[track_idx].class_id != detections[det_idx].class_id:
                    cost[row, col] += 0.25

        return cost.astype(np.float32)

    def _iou_cost_matrix(
        self,
        track_indices: List[int],
        detection_indices: List[int],
        detections: List[Detection],
    ) -> np.ndarray:
        track_boxes = [self.tracks[idx].to_ltrb() for idx in track_indices]
        detection_boxes = [detections[idx].bbox_xyxy for idx in detection_indices]

        cost = iou_distance(track_boxes, detection_boxes)

        for row, track_idx in enumerate(track_indices):
            for col, det_idx in enumerate(detection_indices):
                if self.tracks[track_idx].class_id != detections[det_idx].class_id:
                    cost[row, col] += 0.15

        return cost.astype(np.float32)

    def _motion_cost_matrix(
        self,
        track_indices: List[int],
        detection_indices: List[int],
        detections: List[Detection],
    ) -> np.ndarray:
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return np.empty((len(track_indices), len(detection_indices)), dtype=np.float32)

        measurements = np.asarray(
            [detections[idx].to_xyah() for idx in detection_indices],
            dtype=np.float32,
        )

        cost = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)

        for row, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]

            distances = self.kf.gating_distance(
                track.mean,
                track.covariance,
                measurements,
            )

            cost[row, :] = np.minimum(distances / self.max_motion_distance, 3.0)

        return cost.astype(np.float32)

    def _combined_cost_matrix(
        self,
        track_indices: List[int],
        detection_indices: List[int],
        detections: List[Detection],
    ) -> np.ndarray:
        appearance_cost = self._appearance_cost_matrix(
            track_indices,
            detection_indices,
            detections,
        )

        iou_cost = self._iou_cost_matrix(
            track_indices,
            detection_indices,
            detections,
        )

        motion_cost = self._motion_cost_matrix(
            track_indices,
            detection_indices,
            detections,
        )

        cost = (
            self.appearance_weight * appearance_cost
            + self.iou_weight * iou_cost
            + self.motion_weight * motion_cost
        )

        # Soft gating: reject a pair only when it is both motion-inconsistent
        # and has almost no spatial overlap.
        for row in range(cost.shape[0]):
            for col in range(cost.shape[1]):
                motion_too_far = motion_cost[row, col] > 2.0
                iou_too_low = iou_cost[row, col] > 0.95

                if motion_too_far and iou_too_low:
                    cost[row, col] = 1e5

        return cost.astype(np.float32)

    def _initiate_track(self, detection: Detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())

        track = Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            detection=detection,
        )

        self.tracks.append(track)
        self._next_id += 1

    def _format_output_tracks(self, frame_width: int, frame_height: int) -> List[Dict[str, Any]]:
        output_tracks: List[Dict[str, Any]] = []
        frame_area = max(1, frame_width * frame_height)

        for track in self.tracks:
            if not track.is_confirmed():
                continue

            if track.time_since_update > self.max_age:
                continue

            # Keep stale tracks internally, but avoid exposing long-lived predictions.
            if track.time_since_update > self.max_predicted_age_to_output:
                continue

            x1, y1, x2, y2 = track.to_ltrb()

            x1 = int(max(0, min(frame_width - 1, x1)))
            y1 = int(max(0, min(frame_height - 1, y1)))
            x2 = int(max(0, min(frame_width - 1, x2)))
            y2 = int(max(0, min(frame_height - 1, y2)))

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            area = w * h

            if area < self.min_box_area:
                continue

            visibility = 1 if track.time_since_update == 0 else 0

            # Very large predicted boxes are usually caused by Kalman drift.
            if visibility == 0 and area > 0.70 * frame_area:
                continue

            if visibility == 1:
                confidence = track.confidence
                class_id = track.class_id
                class_name = track.class_name
                matched_iou = track.last_matched_iou
            else:
                confidence = 0.0
                class_id = -1
                class_name = "predicted"
                matched_iou = 0.0

            output_tracks.append({
                "track_id": str(track.track_id),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, w, h],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": str(class_name),
                "visibility": int(visibility),
                "matched_iou": float(matched_iou),
            })

        return output_tracks
