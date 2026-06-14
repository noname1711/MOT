from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# A very large cost used to block invalid assignments.
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
        cost_matrix:
            Cost matrix with shape:
                [len(track_indices), len(detection_indices)]

            Each cell cost_matrix[i, j] means:
                how bad it is to match track_indices[i]
                with detection_indices[j].

        max_distance:
            Maximum allowed assignment cost.
            Assignments with cost greater than this threshold are rejected.

        track_indices:
            Original indices of candidate tracks.

        detection_indices:
            Original indices of candidate detections.

    Returns:
        matches:
            List of valid (track_idx, detection_idx) pairs.

        unmatched_tracks:
            Tracks that were not assigned to any detection.

        unmatched_detections:
            Detections that were not assigned to any track.
    """

    # No tracks means every detection is unmatched.
    if len(track_indices) == 0:
        return [], [], detection_indices.copy()

    # No detections means every track is unmatched.
    if len(detection_indices) == 0:
        return [], track_indices.copy(), []

    # Empty cost matrix means no valid assignment can be solved.
    if cost_matrix.size == 0:
        return [], track_indices.copy(), detection_indices.copy()

    # Copy the cost matrix so the original input is not modified.
    gated_cost = cost_matrix.copy()

    # Reject assignments whose cost is too large.
    # They are replaced by a very large value so the Hungarian algorithm
    # will avoid them unless there is no valid alternative.
    gated_cost[gated_cost > max_distance] = INFTY_COST

    # Hungarian algorithm:
    # Find row-column pairs that minimize the total assignment cost.
    row_indices, col_indices = linear_sum_assignment(gated_cost)

    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    # Rows and columns selected by the Hungarian algorithm.
    matched_rows = set(row_indices.tolist())
    matched_cols = set(col_indices.tolist())

    # Tracks whose rows were not selected are unmatched.
    for row, track_idx in enumerate(track_indices):
        if row not in matched_rows:
            unmatched_tracks.append(track_idx)

    # Detections whose columns were not selected are unmatched.
    for col, detection_idx in enumerate(detection_indices):
        if col not in matched_cols:
            unmatched_detections.append(detection_idx)

    # Convert selected row-column pairs back to original track/detection indices.
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]

        # If the selected pair has infinite cost, reject it.
        if gated_cost[row, col] >= INFTY_COST:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections