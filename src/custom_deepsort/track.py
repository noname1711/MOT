import numpy as np

from src.custom_deepsort.detection import Detection
from src.custom_deepsort.kalman_filter import KalmanFilter


class TrackState:
    """
    Track lifecycle states.

    Example:
        A new detection creates a Tentative track.
        If it is matched enough times, it becomes Confirmed.
        If it is missed for too long, it becomes Deleted.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Store and update the state of one tracked object.

    One Track represents one object ID.
    Example:
        track_id = 5 means this object is being tracked as ID 5
        across multiple frames.
    """

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
        # Kalman Filter state mean and covariance.
        #
        # mean stores the estimated object state:
        #   [cx, cy, a, h, vx, vy, va, vh]
        #
        # Example:
        #   mean = [100, 50, 1.5, 160, 5, 2, 0, 1]
        #   means center=(100,50), height=160,
        #   and the object moves about (5,2) pixels per frame.
        self.mean = mean
        self.covariance = covariance

        # Unique ID assigned to this track.
        self.track_id = track_id

        # Number of successful detection matches.
        # A new track starts with 1 hit because it was created from a detection.
        self.hits = 1

        # Total number of frames since the track was created.
        self.age = 1

        # Number of frames since the last successful update.
        # 0 means this track has just matched a detection.
        self.time_since_update = 0

        # A new track starts as tentative to avoid false positives.
        self.state = TrackState.Tentative

        # Number of successful hits needed to confirm the track.
        # Example:
        #   n_init = 2
        #   after 2 successful matches, Tentative -> Confirmed.
        self.n_init = n_init

        # Maximum number of missed frames before deleting a confirmed track.
        # Example:
        #   max_age = 25
        #   if the track is missed for more than 25 frames, it is deleted.
        self.max_age = max_age

        # Exponential moving average weight for appearance feature update.
        #
        # Example:
        #   feature_alpha = 0.9
        #   updated_feature = 0.9 * old_feature + 0.1 * new_feature
        #
        # This keeps the feature stable and avoids sudden changes.
        self.feature_alpha = feature_alpha

        # Initial appearance feature comes from the first detection.
        self.feature = detection.feature.copy()

        # Store detection metadata for output.
        self.confidence = float(detection.confidence)
        self.class_id = int(detection.class_id)
        self.class_name = str(detection.class_name)

        # IoU between this track and the last matched detection.
        self.last_matched_iou = 1.0

    def predict(self, kf: KalmanFilter):
        """
        Predict the track state in the next frame using Kalman Filter.

        Example:
            If current cx = 100 and vx = 5,
            Kalman predicts next cx = 105.
        """
        # Predict the new position and uncertainty.
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

        # The track becomes older after each prediction step.
        self.age += 1

        # Since no detection has been used yet in this frame,
        # increase the missed-update counter.
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanFilter,
        detection: Detection,
        matched_iou: float = 0.0,
    ):
        """
        Update the track using a matched detection.

        Example:
            Kalman predicts the object at cx = 105.
            YOLO detection says cx = 108.
            The update step corrects the track state closer to the detection.
        """
        # Correct the Kalman prediction using the matched detection.
        self.mean, self.covariance = kf.update(
            self.mean,
            self.covariance,
            detection.to_xyah(),
        )

        # Count this successful match.
        self.hits += 1

        # The track has just been updated, so reset the missed-update counter.
        self.time_since_update = 0

        # Update metadata from the latest detection.
        self.confidence = float(detection.confidence)
        self.class_id = int(detection.class_id)
        self.class_name = str(detection.class_name)
        self.last_matched_iou = float(matched_iou)

        # Smoothly update the appearance feature.
        self._update_feature(detection.feature)

        # Confirm the track after enough successful matches.
        #
        # Example:
        #   n_init = 2
        #   hits becomes 2
        #   -> track changes from Tentative to Confirmed.
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """
        Mark this track as missed when it is not matched with any detection.

        Example:
            If YOLO misses this object in the current frame,
            this function decides whether to keep or delete the track.
        """
        # A tentative track is deleted immediately if it misses once.
        # Reason: it may be a false positive.
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted

        # A confirmed track is deleted only after being missed for too long.
        # Example:
        #   max_age = 25
        #   time_since_update = 26
        #   -> delete this track.
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def is_tentative(self) -> bool:
        """
        Check whether this track is still tentative.
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """
        Check whether this track has been confirmed.
        """
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """
        Check whether this track should be removed.
        """
        return self.state == TrackState.Deleted

    def to_tlwh(self):
        """
        Convert Kalman state [cx, cy, a, h] to [x, y, w, h].

        Kalman stores bbox as:
            cx = center x
            cy = center y
            a  = aspect ratio = width / height
            h  = height

        Output format:
            x = top-left x
            y = top-left y
            w = width
            h = height

        Formula:
            w = a * h
            x = cx - w / 2
            y = cy - h / 2

        Example:
            cx = 100, cy = 50, a = 2.0, h = 40

            w = a * h = 2.0 * 40 = 80
            x = 100 - 80 / 2 = 60
            y = 50  - 40 / 2 = 30

            result = [60, 30, 80, 40]
        """
        cx, cy, a, h = self.mean[:4]

        # Avoid invalid height or width.
        h = max(float(h), 1.0)
        w = max(float(a * h), 1.0)

        # Convert center format to top-left format.
        #
        # Example:
        #   center = (100, 50), size = (80, 40)
        #   top-left = (100 - 40, 50 - 20) = (60, 30)
        x = float(cx - w / 2.0)
        y = float(cy - h / 2.0)

        return [x, y, w, h]

    def to_ltrb(self):
        """
        Convert Kalman state to [left, top, right, bottom] format.

        Example:
            [x, y, w, h] = [60, 30, 80, 40]

            left   = 60
            top    = 30
            right  = 60 + 80 = 140
            bottom = 30 + 40 = 70

            result = [60, 30, 140, 70]
        """
        x, y, w, h = self.to_tlwh()

        return [x, y, x + w, y + h]

    def _update_feature(self, new_feature: np.ndarray):
        """
        Update appearance feature using exponential moving average.

        Formula:
            updated = alpha * old + (1 - alpha) * new

        Example:
            alpha = 0.9
            old feature = [1.0, 0.0]
            new feature = [0.0, 1.0]

            updated = 0.9 * [1.0, 0.0] + 0.1 * [0.0, 1.0]
                    = [0.9, 0.1]

        Meaning:
            The track mostly keeps the old appearance feature,
            but slowly adapts to the new detection feature.
        """
        if new_feature is None:
            return

        old = self.feature.astype(np.float32)
        new = new_feature.astype(np.float32)

        # Smooth the feature to avoid sudden appearance changes.
        #
        # Example:
        #   alpha = 0.9
        #   old = [1.0, 0.0]
        #   new = [0.0, 1.0]
        #   updated = [0.9, 0.1]
        updated = self.feature_alpha * old + (1.0 - self.feature_alpha) * new

        # Normalize the updated feature to unit length.
        #
        # Example:
        #   updated = [3, 4]
        #   norm = sqrt(3^2 + 4^2) = 5
        #   normalized = [3/5, 4/5] = [0.6, 0.8]
        #
        # This makes cosine-distance comparison more stable.
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm

        self.feature = updated.astype(np.float32)