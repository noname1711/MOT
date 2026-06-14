from typing import List

import cv2
import numpy as np


class AppearanceExtractor:
    """
    Extract lightweight appearance features for object association.

    The feature vector combines HSV and LAB color histograms from a resized
    object crop. This keeps the custom tracker CPU-friendly while providing
    stronger appearance cues than a single color space.
    """

    def __init__(
        self,
        output_size=(64, 128),
        hsv_bins=(8, 8, 4),
        lab_bins=(8, 8, 4),
    ):
        # Resize each object crop to a fixed size before feature extraction.
        # Format: (width, height).
        self.output_size = output_size

        # HSV bins: (Hue, Saturation, Value).
        # Hue = color type, Saturation = color intensity, Value = brightness.
        self.hsv_bins = hsv_bins

        # LAB bins: (Lightness, A, B).
        # Lightness = brightness, A = green-red, B = blue-yellow.
        self.lab_bins = lab_bins

    def extract(self, frame, detections: List[dict]) -> List[np.ndarray]:
        """
        Extract one appearance feature vector for each detection.
        """
        features = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]

            # Clamp bounding box coordinates to stay inside the frame.
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(frame.shape[1] - 1, x2))
            y2 = int(min(frame.shape[0] - 1, y2))

            # Crop the detected object region.
            crop = frame[y1:y2, x1:x2]

            # Use an empty feature if the crop is invalid.
            if crop.size == 0:
                features.append(self._empty_feature())
                continue

            features.append(self._extract_feature(crop))

        return features

    def _extract_feature(self, crop) -> np.ndarray:
        """
        Build a normalized HSV + LAB histogram feature from one object crop.
        """
        # Resize all crops to the same size for stable histogram extraction.
        crop = cv2.resize(
            crop,
            self.output_size,
            interpolation=cv2.INTER_AREA,
        )

        # Extract HSV color histogram.
        hsv_feature = self._hist_feature(
            image=cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
            channels=[0, 1, 2],
            bins=self.hsv_bins,
            ranges=[0, 180, 0, 256, 0, 256],
        )

        # Extract LAB color histogram.
        lab_feature = self._hist_feature(
            image=cv2.cvtColor(crop, cv2.COLOR_BGR2LAB),
            channels=[0, 1, 2],
            bins=self.lab_bins,
            ranges=[0, 256, 0, 256, 0, 256],
        )

        # Combine both color descriptors into one appearance vector.
        feature = np.concatenate([hsv_feature, lab_feature]).astype(np.float32)

        # Normalize the final feature for cosine-distance comparison.
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm

        return feature.astype(np.float32)

    def _hist_feature(self, image, channels, bins, ranges) -> np.ndarray:
        """
        Compute and normalize a color histogram feature.
        """
        hist = cv2.calcHist(
            [image],
            channels,
            None,
            bins,
            ranges,
        )

        # Flatten the histogram into a 1D feature vector.
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

        # Apply L2 normalization for stable distance calculation.
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm

        return hist

    def _empty_feature(self) -> np.ndarray:
        """
        Return a zero feature vector when the object crop is invalid.
        """
        hsv_size = int(self.hsv_bins[0] * self.hsv_bins[1] * self.hsv_bins[2])
        lab_size = int(self.lab_bins[0] * self.lab_bins[1] * self.lab_bins[2])

        return np.zeros(hsv_size + lab_size, dtype=np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine distance between two appearance feature vectors.

    Smaller distance means more similar appearance.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # Invalid or empty feature vectors are treated as completely different.
    if a_norm <= 0 or b_norm <= 0:
        return 1.0

    similarity = float(np.dot(a, b) / (a_norm * b_norm))

    # Clamp the value to avoid numerical precision issues.
    similarity = max(-1.0, min(1.0, similarity))

    return 1.0 - similarity


def cosine_distance_matrix(
    track_features: List[np.ndarray],
    detection_features: List[np.ndarray],
) -> np.ndarray:
    """
    Build a cost matrix between track features and detection features.
    """
    if len(track_features) == 0 or len(detection_features) == 0:
        return np.empty(
            (len(track_features), len(detection_features)),
            dtype=np.float32,
        )

    cost = np.zeros(
        (len(track_features), len(detection_features)),
        dtype=np.float32,
    )

    # Each cell stores the appearance distance between one track and one detection.
    for i, track_feature in enumerate(track_features):
        for j, det_feature in enumerate(detection_features):
            cost[i, j] = cosine_distance(track_feature, det_feature)

    return cost