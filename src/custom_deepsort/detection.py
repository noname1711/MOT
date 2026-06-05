from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class Detection:
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    confidence: float
    class_id: int
    class_name: str
    feature: np.ndarray

    @classmethod
    def from_dict(cls, det: Dict[str, Any], feature: np.ndarray):
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

        if h <= 0:
            h = 1.0

        cx = x + w / 2.0
        cy = y + h / 2.0
        a = w / h

        return np.array([cx, cy, a, h], dtype=np.float32)
