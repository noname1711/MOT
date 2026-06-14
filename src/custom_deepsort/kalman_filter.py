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
        a: aspect ratio, computed as width / height
        h: bounding box height
        vx, vy, va, vh: velocities of cx, cy, a, h

    The filter follows a constant-velocity motion model:
        new_position = old_position + velocity
        new_velocity = old_velocity
    """

    def __init__(self):
        # Number of measured variables: cx, cy, aspect ratio, height.
        ndim = 4

        # Time step between two consecutive frames.
        # Since we process frame by frame, dt is set to 1.
        dt = 1.0

        # Motion matrix F.
        #
        # It maps the previous state to the predicted next state:
        #   cx' = cx + vx * dt
        #   cy' = cy + vy * dt
        #   a'  = a  + va * dt
        #   h'  = h  + vh * dt
        #
        # State:
        #   [cx, cy, a, h, vx, vy, va, vh]
        self.motion_mat = np.eye(2 * ndim, dtype=np.float32)

        # Add velocity contribution to the position part of the state.
        for i in range(ndim):
            self.motion_mat[i, ndim + i] = dt

        # Update matrix H.
        #
        # It projects the full 8D state into the 4D measurement space:
        #   [cx, cy, a, h, vx, vy, va, vh] -> [cx, cy, a, h]
        self.update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Noise weights are scaled by bounding box height.
        # Larger objects can tolerate larger pixel-level uncertainty.
        self.std_weight_position = 1.0 / 20.0
        self.std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray):
        """
        Create a new Kalman state from the first detection.

        Args:
            measurement:
                Detection box in [cx, cy, a, h] format.

        Returns:
            mean:
                Initial 8D state vector.
            covariance:
                Initial uncertainty matrix.
        """
        # Initial position comes directly from the detection measurement.
        mean_pos = measurement.astype(np.float32)

        # Initial velocity is unknown, so it is set to zero.
        mean_vel = np.zeros_like(mean_pos, dtype=np.float32)

        # Combine position and velocity into one 8D state vector:
        #   [cx, cy, a, h, 0, 0, 0, 0]
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)

        # Use object height to scale uncertainty.
        h = max(float(measurement[3]), 1.0)

        # Initial standard deviations for position and velocity.
        #
        # Position uncertainty is moderate because the detection gives us
        # a visible bounding box.
        #
        # Velocity uncertainty is larger because we do not know how the object
        # is moving when it first appears.
        std = np.array(
            [
                2 * self.std_weight_position * h,    # cx uncertainty
                2 * self.std_weight_position * h,    # cy uncertainty
                1e-2,                                # aspect ratio uncertainty
                2 * self.std_weight_position * h,    # height uncertainty
                10 * self.std_weight_velocity * h,   # vx uncertainty
                10 * self.std_weight_velocity * h,   # vy uncertainty
                1e-5,                                # va uncertainty
                10 * self.std_weight_velocity * h,   # vh uncertainty
            ],
            dtype=np.float32,
        )

        # Covariance matrix P.
        # It stores how uncertain each state variable is.
        covariance = np.diag(std * std).astype(np.float32)

        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Predict the next state before seeing the current frame detection.

        Formula:
            x' = F x
            P' = F P F^T + Q

        where:
            x: current state mean
            P: current covariance
            F: motion matrix
            Q: motion noise covariance
        """
        # Scale process noise according to the current object height.
        h = max(float(mean[3]), 1.0)

        # Standard deviation for position noise.
        std_pos = np.array(
            [
                self.std_weight_position * h,  # cx noise
                self.std_weight_position * h,  # cy noise
                1e-2,                          # aspect ratio noise
                self.std_weight_position * h,  # height noise
            ],
            dtype=np.float32,
        )

        # Standard deviation for velocity noise.
        std_vel = np.array(
            [
                self.std_weight_velocity * h,  # vx noise
                self.std_weight_velocity * h,  # vy noise
                1e-5,                          # va noise
                self.std_weight_velocity * h,  # vh noise
            ],
            dtype=np.float32,
        )

        # Motion noise covariance Q.
        motion_cov = np.diag(np.r_[std_pos, std_vel] ** 2).astype(np.float32)

        # Predict next state using the constant-velocity model.
        mean = self.motion_mat @ mean

        # Predict next uncertainty.
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov

        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Project the 8D state distribution into the 4D measurement space.

        This is needed because detections only provide:
            [cx, cy, a, h]

        while the Kalman state stores:
            [cx, cy, a, h, vx, vy, va, vh]
        """
        h = max(float(mean[3]), 1.0)

        # Measurement noise.
        # It represents uncertainty in detector measurements.
        std = np.array(
            [
                self.std_weight_position * h,  # cx measurement noise
                self.std_weight_position * h,  # cy measurement noise
                1e-1,                          # aspect ratio measurement noise
                self.std_weight_position * h,  # height measurement noise
            ],
            dtype=np.float32,
        )

        # Measurement noise covariance R.
        innovation_cov = np.diag(std * std).astype(np.float32)

        # Project state mean:
        #   [cx, cy, a, h, vx, vy, va, vh] -> [cx, cy, a, h]
        projected_mean = self.update_mat @ mean

        # Project covariance into measurement space.
        projected_cov = self.update_mat @ covariance @ self.update_mat.T

        # Add measurement noise.
        projected_cov = projected_cov + innovation_cov

        return projected_mean.astype(np.float32), projected_cov.astype(np.float32)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ):
        """
        Correct the predicted state using a matched detection.

        Formula:
            K = P H^T (H P H^T + R)^(-1)
            x_new = x + K (z - Hx)
            P_new = P - K S K^T

        where:
            K: Kalman gain
            z: detection measurement
            Hx: predicted measurement
            z - Hx: innovation, or prediction error
        """
        # Convert the predicted state into measurement space.
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain decides how much we trust the detection
        # compared with the prediction.
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)

        # Difference between actual detection and predicted detection.
        innovation = measurement - projected_mean

        # Correct the predicted state using the innovation.
        new_mean = mean + kalman_gain @ innovation

        # Reduce uncertainty after using the new measurement.
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean.astype(np.float32), new_covariance.astype(np.float32)

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
    ):
        """
        Compute Mahalanobis distances between one predicted track and detections.

        This is used as a motion consistency cost during matching.

        A small distance means:
            the detection is consistent with the predicted track motion.

        A large distance means:
            the detection is far from where the Kalman Filter expected the object.
        """
        # Project the predicted state into measurement space.
        projected_mean, projected_cov = self.project(mean, covariance)

        # Difference between each detection and the predicted measurement.
        d = measurements - projected_mean

        # Inverse covariance is used to normalize the distance by uncertainty.
        inv_cov = np.linalg.inv(projected_cov)

        # Mahalanobis distance:
        #   d^2 = (z - mean)^T S^(-1) (z - mean)
        #
        # This is better than plain Euclidean distance because it considers
        # the uncertainty of the prediction.
        return np.sum(d @ inv_cov * d, axis=1)

