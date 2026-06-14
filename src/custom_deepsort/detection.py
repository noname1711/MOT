from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class Detection:
    """
    Store one detector output together with its appearance feature.

    This object is used by the custom DeepSORT tracker during matching
    and Kalman Filter updates.
    """

    # Bounding box in [x1, y1, x2, y2] format.
    bbox_xyxy: List[float]

    # Bounding box in [x, y, w, h] format.
    bbox_xywh: List[float]

    # Detection confidence from the object detector.
    confidence: float

    # Class ID and class name predicted by the detector.
    class_id: int
    class_name: str

    # Appearance feature extracted from the detected object crop.
    feature: np.ndarray

    @classmethod
    def from_dict(cls, det: Dict[str, Any], feature: np.ndarray):
        """
        Create a Detection object from a detector output dictionary.
        """
        return cls(
            bbox_xyxy=list(map(float, det["bbox_xyxy"])),
            bbox_xywh=list(map(float, det["bbox_xywh"])),
            confidence=float(det["confidence"]),
            class_id=int(det["class_id"]),
            class_name=str(det["class_name"]),
            feature=feature.astype(np.float32),
        )

    def to_xyah(self) -> np.ndarray:
        """
        Convert [x, y, w, h] to [center_x, center_y, aspect_ratio, height].

        This is the measurement format used by the Kalman Filter.
        """
        x, y, w, h = self.bbox_xywh

        # Avoid division by zero when computing the aspect ratio.
        if h <= 0:
            h = 1.0

        # Convert top-left box format to center-based box format.
        cx = x + w / 2.0
        cy = y + h / 2.0
        a = w / h

        return np.array([cx, cy, a, h], dtype=np.float32)