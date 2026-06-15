from typing import Any, Dict, List

import numpy as np

from src.custom_deepsort.appearance import (
    AppearanceExtractor,
    cosine_distance_matrix,
)
from src.custom_deepsort.detection import Detection
from src.custom_deepsort.iou_matching import iou, iou_distance
from src.custom_deepsort.kalman_filter import KalmanFilter
from src.custom_deepsort.linear_assignment import min_cost_matching
from src.custom_deepsort.track import Track


class CustomDeepSORTTracker:
    """
    Custom DeepSORT tracker.

    Main idea:
        YOLO detections are associated with existing tracks using a combined cost.

    Example:
        Frame t:
            track_1 is tracking a car.

        Frame t+1:
            YOLO detects several vehicles.

        The tracker compares track_1 with each detection using:
            appearance cost + IoU cost + motion cost

        The detection with the lowest valid cost is assigned to track_1.
    """

    def __init__(
        self,
        max_age: int = 25,                  # Keep a confirmed track for at most 25 missed frames.
        n_init: int = 2,                    # Confirm a new track after 2 successful matches.
        max_cosine_distance: float = 0.50,  # Maximum appearance distance for matching.
        max_iou_distance: float = 0.72,     # Maximum IoU-based distance for fallback matching.
        nn_budget=None,                     # Optional limit for stored appearance features.

        appearance_matching: bool = True,   # Use appearance features in matching.
        appearance_weight: float = 0.35,    # Appearance contributes 35% to combined cost.
        iou_weight: float = 0.55,           # IoU contributes 55% to combined cost.
        motion_weight: float = 0.10,        # Motion contributes 10% to combined cost.

        max_combined_distance: float = 0.78,  # Reject combined matches with cost > 0.78.
        max_motion_distance: float = 16.0,    # Used to normalize Kalman motion distance.
        min_box_area: int = 64,               # Ignore boxes smaller than 64 pixels.
        max_predicted_age_to_output: int = 8, # Output predicted tracks for at most 8 missed frames.
    ):
        self.max_age = max_age
        self.n_init = n_init

        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance = max_iou_distance
        self.nn_budget = nn_budget

        self.appearance_matching = appearance_matching

        # Combined cost formula:
        #   cost = 0.35 * appearance + 0.55 * IoU + 0.10 * motion
        #
        # Example:
        #   appearance_cost = 0.2
        #   iou_cost        = 0.1
        #   motion_cost     = 0.5
        #
        #   combined = 0.35*0.2 + 0.55*0.1 + 0.10*0.5
        #            = 0.07 + 0.055 + 0.05
        #            = 0.175
        self.appearance_weight = appearance_weight
        self.iou_weight = iou_weight
        self.motion_weight = motion_weight

        self.max_combined_distance = max_combined_distance
        self.max_motion_distance = max_motion_distance

        self.min_box_area = min_box_area
        self.max_predicted_age_to_output = max_predicted_age_to_output

        # Kalman Filter predicts and corrects object motion.
        #
        # Example:
        #   old center = (100, 50), velocity = (5, 2)
        #   predicted center = (105, 52)
        self.kf = KalmanFilter()

        # Extract lightweight appearance features using HSV + LAB histograms.
        self.appearance_extractor = AppearanceExtractor()

        # Active internal tracks.
        self.tracks: List[Track] = []

        # Next ID for a new object.
        # Example:
        #   first new track  -> ID 1
        #   second new track -> ID 2
        self._next_id = 1

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame,
    ) -> List[Dict[str, Any]]:
        """
        Update tracker state for one video frame.

        Example flow:
            1. YOLO gives detections for current frame.
            2. Existing tracks predict their next positions.
            3. Predicted tracks are matched with detections.
            4. Matched tracks are updated.
            5. Unmatched detections create new tracks.
            6. Deleted tracks are removed.
            7. Confirmed tracks are returned for visualization/output.
        """
        if frame is None:
            return []

        frame_height, frame_width = frame.shape[:2]

        # Convert raw YOLO detection dictionaries into Detection objects.
        #
        # Example:
        #   raw dict:
        #       {"bbox_xywh": [10, 20, 100, 50], "confidence": 0.8}
        #
        #   Detection object:
        #       bbox + confidence + class + appearance feature
        detection_objects = self._build_detections(detections, frame)

        # Step 1: Predict the new position of every existing track.
        #
        # Example:
        #   track bbox center = (100, 50)
        #   velocity          = (5, 2)
        #   predicted center  = (105, 52)
        for track in self.tracks:
            track.predict(self.kf)

        # Step 2: Match predicted tracks with current detections.
        #
        # Example:
        #   track_0 may match detection_2
        #   track_1 may match detection_0
        matches, unmatched_tracks, unmatched_detections = self._match(
            detection_objects
        )

        # Step 3: Update matched tracks using their assigned detections.
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            detection = detection_objects[det_idx]

            # Compute IoU between predicted track box and matched detection box.
            #
            # Example:
            #   IoU = 0.75 means the two boxes overlap strongly.
            matched_iou = iou(track.to_ltrb(), detection.bbox_xyxy)

            # Correct Kalman state and update appearance/class/confidence.
            track.update(self.kf, detection, matched_iou=matched_iou)

        # Step 4: Mark unmatched tracks as missed.
        #
        # Example:
        #   If track_3 has no detection in this frame,
        #   time_since_update increases and it may be deleted later.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Step 5: Create new tentative tracks for unmatched detections.
        #
        # Example:
        #   detection_4 does not match any old track
        #   -> create new track ID.
        for det_idx in unmatched_detections:
            self._initiate_track(detection_objects[det_idx])

        # Step 6: Remove deleted tracks from the internal track list.
        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        # Step 7: Convert valid internal tracks into public output format.
        return self._format_output_tracks(frame_width, frame_height)

    def _build_detections(
        self,
        detections: List[Dict[str, Any]],
        frame,
    ) -> List[Detection]:
        """
        Convert raw YOLO detection dictionaries into Detection objects.

        Example:
            YOLO detection:
                bbox + confidence + class

            Detection object:
                bbox + confidence + class + HSV/LAB feature
        """
        if len(detections) == 0:
            return []

        # Extract one appearance feature for each detection.
        #
        # Example:
        #   3 YOLO boxes -> 3 appearance feature vectors.
        features = self.appearance_extractor.extract(frame, detections)

        detection_objects = []

        for det, feature in zip(detections, features):
            x, y, w, h = det["bbox_xywh"]

            # Skip invalid boxes.
            #
            # Example:
            #   w = 0 or h = -5 -> invalid bbox.
            if w <= 0 or h <= 0:
                continue

            # Skip very small boxes because they are often unreliable.
            #
            # Example:
            #   min_box_area = 64
            #   w = 5, h = 8 -> area = 40 < 64 -> skip.
            if w * h < self.min_box_area:
                continue

            detection_objects.append(Detection.from_dict(det, feature))

        return detection_objects

    def _match(self, detections: List[Detection]):
        """
        Match existing tracks with current detections.

        Matching is done in two stages:

        Stage A:
            Confirmed tracks are matched using combined cost:
                appearance + IoU + motion

        Stage B:
            Unconfirmed tracks and recently unmatched confirmed tracks
            are matched using IoU fallback.

        Example:
            Stage A:
                stable track uses appearance + position + motion.

            Stage B:
                new track has weak appearance history,
                so it uses IoU only as fallback.
        """
        # No existing tracks:
        # every detection will become a new track.
        #
        # Example:
        #   tracks = []
        #   detections = [det0, det1]
        #   -> det0 and det1 are unmatched detections.
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        # No detections:
        # every track is unmatched in this frame.
        #
        # Example:
        #   tracks = [track0, track1]
        #   detections = []
        #   -> both tracks are unmatched.
        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []

        # Confirmed tracks are reliable enough to use appearance features.
        confirmed_tracks = [
            idx
            for idx, track in enumerate(self.tracks)
            if track.is_confirmed()
        ]

        # Unconfirmed tracks are new and not stable yet.
        unconfirmed_tracks = [
            idx
            for idx, track in enumerate(self.tracks)
            if not track.is_confirmed()
        ]

        detection_indices = list(range(len(detections)))

        matches_a = []
        unmatched_confirmed = confirmed_tracks
        unmatched_detections = detection_indices

        # ------------------------------------------------------------
        # Stage A: Combined-cost matching for confirmed tracks.
        # ------------------------------------------------------------
        if self.appearance_matching and len(confirmed_tracks) > 0:
            combined_cost = self._combined_cost_matrix(
                confirmed_tracks,
                detection_indices,
                detections,
            )

            # Example combined cost matrix:
            #
            #              det_0   det_1
            #   track_0    0.10    0.80
            #   track_1    0.70    0.20
            #
            # Hungarian will likely choose:
            #   track_0 -> det_0
            #   track_1 -> det_1
            matches_a, unmatched_confirmed, unmatched_detections = (
                min_cost_matching(
                    cost_matrix=combined_cost,
                    max_distance=self.max_combined_distance,
                    track_indices=confirmed_tracks,
                    detection_indices=detection_indices,
                )
            )

        # ------------------------------------------------------------
        # Stage B: IoU fallback matching.
        # ------------------------------------------------------------
        #
        # Candidates:
        #   1. Unconfirmed tracks.
        #   2. Confirmed tracks that failed Stage A but were updated recently.
        #
        # Example:
        #   time_since_update = 1 or 2
        #   -> the track was lost very recently
        #   -> allow IoU fallback.
        iou_track_candidates = unconfirmed_tracks + [
            idx
            for idx in unmatched_confirmed
            if self.tracks[idx].time_since_update <= 2
        ]

        # Confirmed tracks that are not allowed to use IoU fallback remain unmatched.
        #
        # Example:
        #   time_since_update = 5
        #   -> too old for IoU fallback
        #   -> remain unmatched.
        remaining_unmatched_confirmed = [
            idx
            for idx in unmatched_confirmed
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

            matches_b, unmatched_iou_tracks, unmatched_detections = (
                min_cost_matching(
                    cost_matrix=iou_cost,
                    max_distance=self.max_iou_distance,
                    track_indices=iou_track_candidates,
                    detection_indices=unmatched_detections,
                )
            )

        # Merge matches from both stages.
        #
        # Example:
        #   matches_a = [(0, 1)]
        #   matches_b = [(2, 0)]
        #   matches   = [(0, 1), (2, 0)]
        matches = matches_a + matches_b

        # Merge all tracks that still have no matched detection.
        unmatched_tracks = remaining_unmatched_confirmed + unmatched_iou_tracks

        return matches, unmatched_tracks, unmatched_detections

    def _appearance_cost_matrix(
        self,
        track_indices: List[int],
        detection_indices: List[int],
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Build the appearance cost matrix.

        Example:
            cost = 0.05 -> very similar color appearance
            cost = 0.80 -> very different appearance
        """
        track_features = [self.tracks[idx].feature for idx in track_indices]
        detection_features = [
            detections[idx].feature for idx in detection_indices
        ]

        cost = cosine_distance_matrix(track_features, detection_features)

        # Add a small penalty if the class labels are different.
        #
        # Example:
        #   track class = car
        #   detection class = motorcycle
        #
        #   original appearance cost = 0.20
        #   after penalty = 0.20 + 0.25 = 0.45
        #
        # The pair can still match, but it becomes less favorable.
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
        """
        Build the IoU cost matrix.

        Formula:
            IoU cost = 1 - IoU

        Example:
            IoU = 0.80
            IoU cost = 1 - 0.80 = 0.20

            Higher IoU -> lower cost -> more likely match.
        """
        track_boxes = [self.tracks[idx].to_ltrb() for idx in track_indices]
        detection_boxes = [
            detections[idx].bbox_xyxy for idx in detection_indices
        ]

        cost = iou_distance(track_boxes, detection_boxes)

        # Add a small class mismatch penalty.
        #
        # Example:
        #   IoU cost = 0.20
        #   class mismatch penalty = 0.15
        #   final cost = 0.35
        #
        # This penalty is smaller because detector class labels may fluctuate.
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
        """
        Build the motion cost matrix using Kalman Filter gating distance.

        Example:
            Kalman predicts object at x = 100.
            Detection is at x = 103.
            -> small motion cost

            Detection is at x = 300.
            -> large motion cost
        """
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return np.empty(
                (len(track_indices), len(detection_indices)),
                dtype=np.float32,
            )

        # Convert detections to Kalman measurement format:
        #   [cx, cy, aspect_ratio, height]
        #
        # Example:
        #   bbox [x, y, w, h] = [60, 30, 80, 40]
        #   cx = 60 + 80/2 = 100
        #   cy = 30 + 40/2 = 50
        #   a  = 80 / 40 = 2
        #   measurement = [100, 50, 2, 40]
        measurements = np.asarray(
            [detections[idx].to_xyah() for idx in detection_indices],
            dtype=np.float32,
        )

        cost = np.zeros(
            (len(track_indices), len(detection_indices)),
            dtype=np.float32,
        )

        for row, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]

            # Mahalanobis distance between one predicted track
            # and all candidate detections.
            #
            # Example:
            #   predicted = [100, 50]
            #   det_0 = [102, 51] -> small distance
            #   det_1 = [300, 80] -> large distance
            distances = self.kf.gating_distance(
                track.mean,
                track.covariance,
                measurements,
            )

            # Normalize the motion distance and clip large values.
            #
            # Example:
            #   max_motion_distance = 16
            #   distance = 8  -> cost = 8 / 16 = 0.5
            #   distance = 80 -> cost = 80 / 16 = 5 -> clipped to 3.0
            cost[row, :] = np.minimum(
                distances / self.max_motion_distance,
                3.0,
            )

        return cost.astype(np.float32)

    def _combined_cost_matrix(
        self,
        track_indices: List[int],
        detection_indices: List[int],
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Build the final association cost matrix.

        Formula:
            combined_cost =
                appearance_weight * appearance_cost
              + iou_weight        * IoU_cost
              + motion_weight     * motion_cost

        Example:
            appearance_cost = 0.20
            iou_cost        = 0.10
            motion_cost     = 0.50

            combined =
                0.35 * 0.20 +
                0.55 * 0.10 +
                0.10 * 0.50

            combined = 0.175

        Smaller combined cost means a better track-detection match.
        """
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

        # Weighted sum of all association cues.
        cost = (
            self.appearance_weight * appearance_cost
            + self.iou_weight * iou_cost
            + self.motion_weight * motion_cost
        )

        # Soft gating:
        # reject a pair only when both motion and IoU are very bad.
        #
        # Example:
        #   motion_cost = 2.5  -> too far from Kalman prediction
        #   iou_cost    = 0.98 -> almost no bbox overlap
        #
        #   final cost = 1e5
        #
        # This almost blocks the pair from being selected by Hungarian.
        for row in range(cost.shape[0]):
            for col in range(cost.shape[1]):
                motion_too_far = motion_cost[row, col] > 2.0
                iou_too_low = iou_cost[row, col] > 0.95

                if motion_too_far and iou_too_low:
                    cost[row, col] = 1e5

        return cost.astype(np.float32)

    def _initiate_track(self, detection: Detection):
        """
        Create a new tentative track from an unmatched detection.

        Example:
            If detection_3 does not match any existing track,
            create a new track with a new ID.
        """
        # Initialize Kalman state from detection measurement.
        #
        # Example:
        #   detection.to_xyah() = [100, 50, 2.0, 40]
        #   Kalman creates initial state:
        #       [100, 50, 2.0, 40, 0, 0, 0, 0]
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

        # Increase ID counter for the next new track.
        #
        # Example:
        #   current new ID = 5
        #   next new ID becomes 6.
        self._next_id += 1

    def _format_output_tracks(
        self,
        frame_width: int,
        frame_height: int,
    ) -> List[Dict[str, Any]]:
        """
        Convert internal Track objects into public output dictionaries.

        Only valid confirmed tracks are exposed to the outside pipeline.

        Example output:
            {
                "track_id": "3",
                "bbox_xyxy": [10, 20, 110, 70],
                "bbox_xywh": [10, 20, 100, 50],
                "visibility": 1
            }
        """
        output_tracks: List[Dict[str, Any]] = []

        # Used to detect unrealistically large predicted boxes.
        #
        # Example:
        #   frame = 1280 x 720
        #   frame_area = 921600
        frame_area = max(1, frame_width * frame_height)

        for track in self.tracks:
            # Do not output tentative tracks.
            #
            # Example:
            #   a new track with only 1 hit is still tentative,
            #   so it is not displayed yet.
            if not track.is_confirmed():
                continue

            # Do not output tracks that are too old.
            #
            # Example:
            #   max_age = 25
            #   time_since_update = 30 -> skip.
            if track.time_since_update > self.max_age:
                continue

            # Keep stale tracks internally, but avoid showing long predictions.
            #
            # Example:
            #   max_predicted_age_to_output = 8
            #   time_since_update = 10 -> keep internally but do not output.
            if track.time_since_update > self.max_predicted_age_to_output:
                continue

            # Convert Kalman state to [x1, y1, x2, y2].
            #
            # Example:
            #   [x, y, w, h] = [10, 20, 100, 50]
            #   [x1, y1, x2, y2] = [10, 20, 110, 70]
            x1, y1, x2, y2 = track.to_ltrb()

            # Clamp box coordinates to stay inside the frame.
            #
            # Example:
            #   x1 = -5 -> clamp to 0
            #   x2 = 1300 with frame_width=1280 -> clamp to 1279
            x1 = int(max(0, min(frame_width - 1, x1)))
            y1 = int(max(0, min(frame_height - 1, y1)))
            x2 = int(max(0, min(frame_width - 1, x2)))
            y2 = int(max(0, min(frame_height - 1, y2)))

            w = x2 - x1
            h = y2 - y1

            # Skip invalid boxes.
            #
            # Example:
            #   x2 <= x1 -> width <= 0 -> invalid.
            if w <= 0 or h <= 0:
                continue

            area = w * h

            # Skip very small boxes.
            #
            # Example:
            #   min_box_area = 64
            #   w = 4, h = 10 -> area = 40 -> skip.
            if area < self.min_box_area:
                continue

            # visibility = 1:
            #   track was matched with a real detection in this frame.
            #
            # visibility = 0:
            #   track is only predicted by Kalman Filter in this frame.
            #
            # Example:
            #   time_since_update = 0 -> visibility = 1
            #   time_since_update = 3 -> visibility = 0
            visibility = 1 if track.time_since_update == 0 else 0

            # Very large predicted boxes are usually caused by Kalman drift.
            #
            # Example:
            #   frame_area = 921600
            #   70% frame area = 645120
            #   predicted box area = 700000 -> skip.
            if visibility == 0 and area > 0.70 * frame_area:
                continue

            if visibility == 1:
                # Real detection-backed track.
                #
                # Example:
                #   YOLO matched this track in current frame,
                #   so keep real confidence and class.
                confidence = track.confidence
                class_id = track.class_id
                class_name = track.class_name
                matched_iou = track.last_matched_iou
            else:
                # Kalman-only predicted track.
                #
                # Example:
                #   YOLO missed the object,
                #   but Kalman still predicts its position.
                #   confidence = 0 means no real detection supports it.
                confidence = 0.0
                class_id = -1
                class_name = "predicted"
                matched_iou = 0.0

            output_tracks.append(
                {
                    "track_id": str(track.track_id),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, w, h],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "visibility": int(visibility),
                    "matched_iou": float(matched_iou),
                }
            )

        return output_tracks