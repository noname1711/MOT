from typing import List

import numpy as np


def iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Each box is expected in [x1, y1, x2, y2] format.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Compute the coordinates of the intersection rectangle.
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Compute intersection width and height.
    # If boxes do not overlap, width or height becomes 0.
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    # Compute the area of each bounding box.
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # Union = area A + area B - intersection area.
    union = area_a + area_b - inter_area

    # Avoid division by zero for invalid boxes.
    if union <= 0:
        return 0.0

    return inter_area / union


def iou_distance(
    track_boxes: List[List[float]],
    detection_boxes: List[List[float]],
) -> np.ndarray:
    """
    Build an IoU-based cost matrix between tracks and detections.

    Cost is computed as:
        cost = 1 - IoU

    Smaller cost means better spatial match.
    """
    if len(track_boxes) == 0 or len(detection_boxes) == 0:
        return np.empty(
            (len(track_boxes), len(detection_boxes)),
            dtype=np.float32,
        )

    cost = np.zeros(
        (len(track_boxes), len(detection_boxes)),
        dtype=np.float32,
    )

    # Each cell stores the spatial distance between one track and one detection.
    for i, track_box in enumerate(track_boxes):
        for j, det_box in enumerate(detection_boxes):
            cost[i, j] = 1.0 - iou(track_box, det_box)

    return cost