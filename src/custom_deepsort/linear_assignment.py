from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# A very large cost used to block invalid assignments.
# Example:
#   normal cost = 0.2 or 0.7
#   invalid cost = 1e5
# Hungarian will almost never choose a pair with cost 1e5.
INFTY_COST = 1e5


def min_cost_matching(
    cost_matrix: np.ndarray,
    max_distance: float,
    track_indices: List[int],
    detection_indices: List[int],
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve track-detection assignment using the Hungarian algorithm.

    Each row is a track.
    Each column is a detection.
    Each cell is the matching cost between them.

    Example:
        cost_matrix = [
            [0.10, 0.80],
            [0.70, 0.20],
        ]

        Hungarian will choose:
            track_0 -> detection_0  cost = 0.10
            track_1 -> detection_1  cost = 0.20

        because this gives the smallest total cost:
            0.10 + 0.20 = 0.30

    Returns:
        matches:
            Valid (track_idx, detection_idx) pairs.

        unmatched_tracks:
            Tracks that do not match any detection.

        unmatched_detections:
            Detections that do not match any track.
    """

    # If there are no tracks, all detections are unmatched.
    # Example:
    #   tracks = []
    #   detections = [0, 1, 2]
    #   -> unmatched_detections = [0, 1, 2]
    if len(track_indices) == 0:
        return [], [], detection_indices.copy()

    # If there are no detections, all tracks are unmatched.
    # Example:
    #   tracks = [0, 1]
    #   detections = []
    #   -> unmatched_tracks = [0, 1]
    if len(detection_indices) == 0:
        return [], track_indices.copy(), []

    # If the matrix is empty, no assignment can be solved.
    if cost_matrix.size == 0:
        return [], track_indices.copy(), detection_indices.copy()

    # Copy the cost matrix so the original matrix is not changed.
    gated_cost = cost_matrix.copy()

    # Reject pairs whose cost is greater than max_distance.
    #
    # Example:
    #   max_distance = 0.7
    #   cost = 0.9  -> too large -> replace with 1e5
    #
    # This means the pair is considered invalid for matching.
    gated_cost[gated_cost > max_distance] = INFTY_COST

    # Hungarian algorithm finds row-column pairs with the smallest total cost.
    #
    # Example:
    #   gated_cost = [
    #       [0.10, 1e5],
    #       [0.70, 0.20],
    #   ]
    #
    # Possible good result:
    #   row_indices = [0, 1]
    #   col_indices = [0, 1]
    #
    # Meaning:
    #   row 0 -> col 0
    #   row 1 -> col 1
    row_indices, col_indices = linear_sum_assignment(gated_cost)

    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    # Rows and columns selected by Hungarian.
    matched_rows = set(row_indices.tolist())
    matched_cols = set(col_indices.tolist())

    # Tracks whose rows were not selected are unmatched.
    #
    # Example:
    #   track_indices = [3, 5, 8]
    #   matched_rows = {0, 2}
    #   row 1 is not matched -> track_indices[1] = 5 is unmatched
    for row, track_idx in enumerate(track_indices):
        if row not in matched_rows:
            unmatched_tracks.append(track_idx)

    # Detections whose columns were not selected are unmatched.
    #
    # Example:
    #   detection_indices = [10, 11, 12]
    #   matched_cols = {0, 2}
    #   col 1 is not matched -> detection_indices[1] = 11 is unmatched
    for col, detection_idx in enumerate(detection_indices):
        if col not in matched_cols:
            unmatched_detections.append(detection_idx)

    # Convert selected row-column pairs back to original track/detection indices.
    #
    # Example:
    #   track_indices = [3, 5]
    #   detection_indices = [10, 11]
    #
    #   row = 0, col = 1
    #   -> track_idx = track_indices[0] = 3
    #   -> detection_idx = detection_indices[1] = 11
    #   -> match = (3, 11)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]

        # If Hungarian selected an invalid pair with cost 1e5,
        # reject it and mark both as unmatched.
        #
        # Example:
        #   gated_cost[row, col] = 1e5
        #   -> this pair was blocked before
        #   -> do not add it to matches
        if gated_cost[row, col] >= INFTY_COST:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections