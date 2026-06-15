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
        Predict the next track state before matching with detections.

        The Kalman Filter uses a constant-velocity model:
            new_position = old_position + velocity

        State vector:
            x = [cx, cy, a, h, vx, vy, va, vh]

        where:
            cx, cy : center point of the bounding box
            a      : aspect ratio = width / height
            h      : bounding box height
            vx, vy : velocity of cx and cy
            va     : velocity of aspect ratio
            vh     : velocity of height

        Formula:
            x' = F x
            P' = F P F^T + Q

        Example:
            If current state is:
                cx = 100, cy = 50, h = 160
                vx = 5,   vy = 2

            Then prediction gives:
                cx' = 100 + 5 = 105
                cy' = 50  + 2 = 52

            So before seeing the current frame detection, the tracker predicts
            that the object moves from (100, 50) to around (105, 52).
        """

        # Use current bounding box height to scale the prediction noise.
        # Example:
        #   if h = 160:
        #       position noise = h / 20  = 8 pixels
        #       velocity noise = h / 160 = 1 pixel/frame
        #
        # Larger boxes allow larger pixel-level uncertainty.
        h = max(float(mean[3]), 1.0)

        # Standard deviation for position-related state values: [cx, cy, a, h].
        # Example with h = 160:
        #   cx noise = 160 / 20 = 8
        #   cy noise = 160 / 20 = 8
        #   a noise  = 0.01, kept very small because aspect ratio should be stable
        #   h noise  = 160 / 20 = 8
        std_pos = np.array(
            [
                self.std_weight_position * h,  # cx noise
                self.std_weight_position * h,  # cy noise
                1e-2,                          # aspect ratio noise
                self.std_weight_position * h,  # height noise
            ],
            dtype=np.float32,
        )

        # Standard deviation for velocity-related state values: [vx, vy, va, vh].
        # Example with h = 160:
        #   vx noise = 160 / 160 = 1
        #   vy noise = 160 / 160 = 1
        #   va noise = 0.00001, kept very small
        #   vh noise = 160 / 160 = 1
        #
        # Velocity noise is smaller than position noise to keep motion smooth.
        std_vel = np.array(
            [
                self.std_weight_velocity * h,  # vx noise
                self.std_weight_velocity * h,  # vy noise
                1e-5,                          # aspect ratio velocity noise
                self.std_weight_velocity * h,  # height velocity noise
            ],
            dtype=np.float32,
        )

        # Build motion noise covariance Q.
        # Covariance uses variance, so standard deviation must be squared.
        #
        # Example:
        #   std = [8, 8, 0.01, 8, 1, 1, 0.00001, 1]
        #   variance = std^2
        #
        # np.diag(...) puts these variance values on the diagonal of Q.
        motion_cov = np.diag(np.r_[std_pos, std_vel] ** 2).astype(np.float32)

        # Predict the next state using:
        #   x' = F x
        #
        # Example:
        #   cx' = cx + vx
        #   cy' = cy + vy
        #
        # This estimates the next object position before using YOLO detection.
        mean = self.motion_mat @ mean

        # Predict the next uncertainty using:
        #   P' = F P F^T + Q
        #
        # F P F^T:
        #   moves the old uncertainty through the motion model.
        #
        # + Q:
        #   adds motion noise because the prediction is not perfectly reliable.
        #
        # Meaning:
        #   after prediction, uncertainty usually increases because no detection
        #   has corrected the state yet.
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

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Project the 8D Kalman state into the 4D detection space.

        Kalman state has 8 values:
            [cx, cy, a, h, vx, vy, va, vh]

        YOLO detection only has 4 values:
            [cx, cy, a, h]

        Therefore, this function keeps only [cx, cy, a, h]
        so the Kalman prediction can be compared with a detection.

        Formula:
            projected_mean = H x
            projected_cov  = H P H^T + R

        Simple example:
            mean = [100, 50, 1.5, 160, 5, 2, 0, 1]

            After projection:
            projected_mean = [100, 50, 1.5, 160]

            The velocity part [5, 2, 0, 1] is removed because
            detections do not provide velocity.
        """

        # Use current bbox height to scale measurement noise.
        #
        # Example:
        #   if h = 160:
        #       cx noise = 160 / 20 = 8 pixels
        #       cy noise = 160 / 20 = 8 pixels
        #       h noise  = 160 / 20 = 8 pixels
        #
        # This means larger boxes allow larger pixel-level detector error.
        h = max(float(mean[3]), 1.0)

        # Standard deviation of measurement noise for [cx, cy, a, h].
        #
        # Example with h = 160:
        #   std = [8, 8, 0.1, 8]
        #
        # cx, cy, and h noise depend on bbox height.
        # Aspect ratio noise is fixed because it is a scale-independent value.
        std = np.array(
            [
                self.std_weight_position * h,  # cx measurement noise
                self.std_weight_position * h,  # cy measurement noise
                1e-1,                          # aspect ratio measurement noise
                self.std_weight_position * h,  # height measurement noise
            ],
            dtype=np.float32,
        )

        # Build measurement noise covariance R.
        #
        # Covariance stores variance, so we square each std value.
        #
        # Example:
        #   std = [8, 8, 0.1, 8]
        #   std * std = [64, 64, 0.01, 64]
        #
        # np.diag(...) puts these values on the diagonal of R.
        innovation_cov = np.diag(std * std).astype(np.float32)

        # Project state mean using:
        #   projected_mean = H x
        #
        # Example:
        #   [100, 50, 1.5, 160, 5, 2, 0, 1]
        #        -> [100, 50, 1.5, 160]
        #
        # This removes velocity because detection has no velocity values.
        projected_mean = self.update_mat @ mean

        # Project covariance into detection space using:
        #   projected_cov = H P H^T
        #   keep only the uncertainty related to [cx, cy, a, h].
        projected_cov = self.update_mat @ covariance @ self.update_mat.T

        # Add measurement noise R:
        #   projected_cov = H P H^T + R
        #   YOLO detection is not perfectly accurate, so we add detector noise.
        projected_cov = projected_cov + innovation_cov

        return projected_mean.astype(np.float32), projected_cov.astype(np.float32)

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
    ):
        """
        Compute Mahalanobis distances between one predicted track and detections.

        Example:
            predicted = [100, 50, 1.5, 160]
            det_0     = [102, 51, 1.5, 158] -> small distance
            det_1     = [300, 80, 1.2, 140] -> large distance

        Smaller distance means the detection is more consistent with the prediction.
        """
        # Convert 8D state to 4D measurement space: [cx, cy, a, h].
        projected_mean, projected_cov = self.project(mean, covariance)

        # Error between each detection and the predicted measurement.
        # Example: [102, 51, 1.5, 158] - [100, 50, 1.5, 160] = [2, 1, 0, -2].
        d = measurements - projected_mean

        # Compute motion cost between the Kalman prediction and each detection.
        # Simple idea:
        #   distance = prediction_error^2 / prediction_uncertainty
        # Example:
        #   predicted = [100, 50]
        #   detection = [105, 52]
        #   error d   = [5, 2]
        #
        # If this error is small compared with Kalman uncertainty,
        # the distance is small -> likely match.
        #
        # If this error is large compared with Kalman uncertainty,
        # the distance is large -> unlikely match.
        inv_cov = np.linalg.inv(projected_cov)

        # Vectorized Mahalanobis distance for all detections.
        # It returns one motion distance for each detection.
        return np.sum(d @ inv_cov * d, axis=1)
