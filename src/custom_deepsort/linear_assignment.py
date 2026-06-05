from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


INFTY_COST = 1e5


def min_cost_matching(
    cost_matrix: np.ndarray,
    max_distance: float,
    track_indices: List[int],
    detection_indices: List[int],
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve track-detection assignment using the Hungarian algorithm.

    Args:
        cost_matrix: Cost matrix with shape
            [len(track_indices), len(detection_indices)].
        max_distance: Maximum allowed assignment cost.
        track_indices: Original indices of candidate tracks.
        detection_indices: Original indices of candidate detections.

    Returns:
        matches: List of (track_idx, detection_idx) pairs.
        unmatched_tracks: Tracks that were not assigned to any detection.
        unmatched_detections: Detections that were not assigned to any track.
    """

    if len(track_indices) == 0:
        return [], [], detection_indices.copy()

    if len(detection_indices) == 0:
        return [], track_indices.copy(), []

    if cost_matrix.size == 0:
        return [], track_indices.copy(), detection_indices.copy()

    gated_cost = cost_matrix.copy()
    gated_cost[gated_cost > max_distance] = INFTY_COST

    row_indices, col_indices = linear_sum_assignment(gated_cost)

    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    matched_rows = set(row_indices.tolist())
    matched_cols = set(col_indices.tolist())

    for row, track_idx in enumerate(track_indices):
        if row not in matched_rows:
            unmatched_tracks.append(track_idx)

    for col, detection_idx in enumerate(detection_indices):
        if col not in matched_cols:
            unmatched_detections.append(detection_idx)

    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]

        if gated_cost[row, col] >= INFTY_COST:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections
