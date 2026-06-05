from typing import Optional

import numpy as np

from src.custom_deepsort.detection import Detection
from src.custom_deepsort.kalman_filter import KalmanFilter


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(
        self,
        mean,
        covariance,
        track_id: int,
        n_init: int,
        max_age: int,
        detection: Detection,
        feature_alpha: float = 0.9,
    ):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.n_init = n_init
        self.max_age = max_age

        self.feature_alpha = feature_alpha
        self.feature = detection.feature.copy()

        self.confidence = float(detection.confidence)
        self.class_id = int(detection.class_id)
        self.class_name = str(detection.class_name)

        self.last_matched_iou = 1.0

    def predict(self, kf: KalmanFilter):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilter, detection: Detection, matched_iou: float = 0.0):
        self.mean, self.covariance = kf.update(
            self.mean,
            self.covariance,
            detection.to_xyah(),
        )

        self.hits += 1
        self.time_since_update = 0

        self.confidence = float(detection.confidence)
        self.class_id = int(detection.class_id)
        self.class_name = str(detection.class_name)
        self.last_matched_iou = float(matched_iou)

        self._update_feature(detection.feature)

        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted

    def to_tlwh(self):
        cx, cy, a, h = self.mean[:4]

        h = max(float(h), 1.0)
        w = max(float(a * h), 1.0)

        x = float(cx - w / 2.0)
        y = float(cy - h / 2.0)

        return [x, y, w, h]

    def to_ltrb(self):
        x, y, w, h = self.to_tlwh()
        return [x, y, x + w, y + h]

    def _update_feature(self, new_feature: np.ndarray):
        if new_feature is None:
            return

        old = self.feature.astype(np.float32)
        new = new_feature.astype(np.float32)

        updated = self.feature_alpha * old + (1.0 - self.feature_alpha) * new

        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm

        self.feature = updated.astype(np.float32)
