import numpy as np


class KalmanFilter:
    """
    Kalman Filter used by the custom DeepSORT tracker.

    State vector:
        [cx, cy, a, h, vx, vy, va, vh]

    Measurement vector:
        [cx, cy, a, h]

    where:
        cx, cy: bounding box center
        a: aspect ratio
        h: bounding box height
    """

    def __init__(self):
        ndim = 4
        dt = 1.0

        self.motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self.motion_mat[i, ndim + i] = dt

        self.update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        self.std_weight_position = 1.0 / 20.0
        self.std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray):
        mean_pos = measurement.astype(np.float32)
        mean_vel = np.zeros_like(mean_pos, dtype=np.float32)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)

        h = max(float(measurement[3]), 1.0)

        std = np.array([
            2 * self.std_weight_position * h,
            2 * self.std_weight_position * h,
            1e-2,
            2 * self.std_weight_position * h,
            10 * self.std_weight_velocity * h,
            10 * self.std_weight_velocity * h,
            1e-5,
            10 * self.std_weight_velocity * h,
        ], dtype=np.float32)

        covariance = np.diag(std * std).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = max(float(mean[3]), 1.0)

        std_pos = np.array([
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
        ], dtype=np.float32)

        std_vel = np.array([
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ], dtype=np.float32)

        motion_cov = np.diag(np.r_[std_pos, std_vel] ** 2).astype(np.float32)

        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov

        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        h = max(float(mean[3]), 1.0)

        std = np.array([
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ], dtype=np.float32)

        innovation_cov = np.diag(std * std).astype(np.float32)

        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T
        projected_cov = projected_cov + innovation_cov

        return projected_mean.astype(np.float32), projected_cov.astype(np.float32)

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        projected_mean, projected_cov = self.project(mean, covariance)

        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean.astype(np.float32), new_covariance.astype(np.float32)

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray):
        """
        Compute Mahalanobis distances between the predicted state and detections.

        These distances are used as a motion consistency term during matching.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        d = measurements - projected_mean
        inv_cov = np.linalg.inv(projected_cov)

        return np.sum(d @ inv_cov * d, axis=1)
