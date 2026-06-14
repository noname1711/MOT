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

    The combined association cost uses:
        1. Appearance distance:
            Compare HSV + LAB histogram features.

        2. IoU distance:
            Compare spatial overlap between track boxes and detection boxes.

        3. Motion distance:
            Compare detections with Kalman Filter predictions.

    This tracker keeps tracks internally during short occlusions, but filters
    stale predicted boxes from the public output to reduce noisy visualization
    and evaluation results.
    """

    def __init__(
        self,
        max_age: int = 25,                 # Maximum frames to keep a track without a matched detection.
        n_init: int = 2,                   # Number of successful matches needed to confirm a new track.
        max_cosine_distance: float = 0.50, # Maximum appearance distance for matching.
        max_iou_distance: float = 0.72,    # Maximum IoU-based distance for matching.
        nn_budget=None,                    # Optional limit for stored appearance features.

        appearance_matching: bool = True,  # Enable appearance-based matching.
        appearance_weight: float = 0.35,   # Weight of appearance cost in combined matching.
        iou_weight: float = 0.55,          # Weight of IoU cost in combined matching.
        motion_weight: float = 0.10,       # Weight of motion cost in combined matching.

        max_combined_distance: float = 0.78, # Maximum allowed combined matching cost.
        max_motion_distance: float = 16.0,   # Normalization threshold for motion distance.
        min_box_area: int = 64,              # Ignore detections with very small bounding boxes.
        max_predicted_age_to_output: int = 8, # Maximum age for outputting predicted tracks.
    ):
        # Maximum number of missed frames before deleting a confirmed track.
        self.max_age = max_age

        # Number of successful matches needed before a track becomes confirmed.
        self.n_init = n_init

        # Appearance distance threshold.
        # Kept for compatibility and parameter clarity.
        self.max_cosine_distance = max_cosine_distance

        # Maximum IoU-based cost allowed in fallback IoU matching.
        self.max_iou_distance = max_iou_distance

        # Optional feature budget, currently kept for compatibility.
        self.nn_budget = nn_budget

        # Whether to use appearance-based matching for confirmed tracks.
        self.appearance_matching = appearance_matching

        # Weights for the combined association cost:
        #   combined_cost =
        #       appearance_weight * appearance_cost
        #     + iou_weight        * iou_cost
        #     + motion_weight     * motion_cost
        self.appearance_weight = appearance_weight
        self.iou_weight = iou_weight
        self.motion_weight = motion_weight

        # Maximum allowed combined cost for Hungarian matching.
        self.max_combined_distance = max_combined_distance

        # Used to normalize Mahalanobis motion distance.
        self.max_motion_distance = max_motion_distance

        # Ignore very small boxes because they are often noisy detections.
        self.min_box_area = min_box_area

        # A track may be kept internally for max_age frames,
        # but predicted boxes are only exposed for a shorter time.
        self.max_predicted_age_to_output = max_predicted_age_to_output

        # Kalman Filter handles motion prediction and correction.
        self.kf = KalmanFilter()

        # Lightweight appearance extractor based on HSV + LAB histograms.
        self.appearance_extractor = AppearanceExtractor()

        # Active internal tracks.
        self.tracks: List[Track] = []

        # Next unique ID assigned to a newly created track.
        self._next_id = 1

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame,
    ) -> List[Dict[str, Any]]:
        """
        Update tracker state for one video frame.

        Input:
            detections:
                YOLO detections in dictionary format.

            frame:
                Current image frame.

        Output:
            A list of public track dictionaries used for visualization,
            TXT export, and metrics.
        """
        if frame is None:
            return []

        frame_height, frame_width = frame.shape[:2]

        # Convert raw detection dictionaries into Detection objects.
        # This also extracts HSV + LAB appearance features.
        detection_objects = self._build_detections(detections, frame)

        # Step 1: Predict the new position of every existing track.
        #
        # This happens before matching because we want to compare detections
        # with the predicted track positions in the current frame.
        for track in self.tracks:
            track.predict(self.kf)

        # Step 2: Match predicted tracks with current detections.
        matches, unmatched_tracks, unmatched_detections = self._match(
            detection_objects
        )

        # Step 3: Update matched tracks using their assigned detections.
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            detection = detection_objects[det_idx]

            # Store IoU for output/debugging.
            matched_iou = iou(track.to_ltrb(), detection.bbox_xyxy)

            # Correct Kalman state and update appearance/class/confidence.
            track.update(self.kf, detection, matched_iou=matched_iou)

        # Step 4: Mark unmatched tracks as missed.
        #
        # Tentative tracks may be deleted immediately.
        # Confirmed tracks are deleted only after max_age missed frames.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Step 5: Create new tentative tracks for unmatched detections.
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

        Each Detection object contains:
            - bounding box
            - confidence
            - class information
            - appearance feature
        """
        if len(detections) == 0:
            return []

        # Extract one appearance feature for each detection.
        features = self.appearance_extractor.extract(frame, detections)

        detection_objects = []

        for det, feature in zip(detections, features):
            x, y, w, h = det["bbox_xywh"]

            # Skip invalid boxes.
            if w <= 0 or h <= 0:
                continue

            # Skip very small boxes because they are often unreliable.
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
        """
        # No existing tracks:
        # every detection will become a new track.
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        # No detections:
        # every track is unmatched in this frame.
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
        # The condition time_since_update <= 2 means:
        #   only recently lost tracks are allowed to use IoU fallback.
        iou_track_candidates = unconfirmed_tracks + [
            idx
            for idx in unmatched_confirmed
            if self.tracks[idx].time_since_update <= 2
        ]

        # Confirmed tracks that are not allowed to use IoU fallback
        # remain unmatched.
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

        Cost is based on cosine distance between:
            - stored track appearance feature
            - current detection appearance feature

        Smaller cost means more similar appearance.
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
        # They still may be matched if other cues are strong,
        # but the assignment becomes less favorable.
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

        Smaller cost means stronger spatial overlap.
        """
        track_boxes = [self.tracks[idx].to_ltrb() for idx in track_indices]
        detection_boxes = [
            detections[idx].bbox_xyxy for idx in detection_indices
        ]

        cost = iou_distance(track_boxes, detection_boxes)

        # Add a small class mismatch penalty.
        #
        # The penalty here is smaller than appearance penalty because
        # detector class predictions may sometimes fluctuate between
        # similar vehicle classes such as car, bus, and truck.
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

        The Kalman Filter predicts where a track should be.
        A detection far from that prediction receives a larger motion cost.
        """
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return np.empty(
                (len(track_indices), len(detection_indices)),
                dtype=np.float32,
            )

        # Convert detections to Kalman measurement format:
        #   [cx, cy, aspect_ratio, height]
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

            # Mahalanobis distance between the predicted track state
            # and all candidate detections.
            distances = self.kf.gating_distance(
                track.mean,
                track.covariance,
                measurements,
            )

            # Normalize the motion distance and clip large values.
            #
            # Clipping prevents one extreme distance from dominating
            # the combined cost too strongly.
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
                0.35 * appearance_cost
              + 0.55 * IoU_cost
              + 0.10 * motion_cost

        The actual weights come from:
            self.appearance_weight
            self.iou_weight
            self.motion_weight
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
        # Reject a pair only when both conditions are bad:
        #   1. motion is too far from Kalman prediction
        #   2. spatial overlap is almost zero
        #
        # This is softer than rejecting by motion alone or IoU alone.
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
        """
        # Initialize Kalman state from detection measurement.
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
        self._next_id += 1

    def _format_output_tracks(
        self,
        frame_width: int,
        frame_height: int,
    ) -> List[Dict[str, Any]]:
        """
        Convert internal Track objects into public output dictionaries.

        Only valid confirmed tracks are exposed to the outside pipeline.
        """
        output_tracks: List[Dict[str, Any]] = []

        # Used to detect unrealistically large predicted boxes.
        frame_area = max(1, frame_width * frame_height)

        for track in self.tracks:
            # Do not output tentative tracks.
            if not track.is_confirmed():
                continue

            # Do not output tracks that are too old.
            if track.time_since_update > self.max_age:
                continue

            # Keep stale tracks internally, but avoid showing long predictions.
            if track.time_since_update > self.max_predicted_age_to_output:
                continue

            # Convert Kalman state to [x1, y1, x2, y2].
            x1, y1, x2, y2 = track.to_ltrb()

            # Clamp box coordinates to stay inside the frame.
            x1 = int(max(0, min(frame_width - 1, x1)))
            y1 = int(max(0, min(frame_height - 1, y1)))
            x2 = int(max(0, min(frame_width - 1, x2)))
            y2 = int(max(0, min(frame_height - 1, y2)))

            w = x2 - x1
            h = y2 - y1

            # Skip invalid boxes.
            if w <= 0 or h <= 0:
                continue

            area = w * h

            # Skip very small boxes.
            if area < self.min_box_area:
                continue

            # visibility = 1:
            #   track was matched with a real detection in this frame.
            #
            # visibility = 0:
            #   track is only predicted by Kalman Filter in this frame.
            visibility = 1 if track.time_since_update == 0 else 0

            # Very large predicted boxes are usually caused by Kalman drift.
            # They are filtered out to avoid noisy visualization and metrics.
            if visibility == 0 and area > 0.70 * frame_area:
                continue

            if visibility == 1:
                # Real detection-backed track.
                confidence = track.confidence
                class_id = track.class_id
                class_name = track.class_name
                matched_iou = track.last_matched_iou
            else:
                # Kalman-only predicted track.
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
