"""
Kalman Filter for object tracking.

Implements a constant-velocity Kalman filter for bounding box tracking.
State: [x, y, s, r, vx, vy, vs]
  - x, y: center position
  - s: area (scale)
  - r: aspect ratio
  - vx, vy, vs: velocities
"""
import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    Kalman filter tracker for bounding boxes.

    Tracks objects using bounding box coordinates and predicts future positions.
    """
    count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initialize tracker with bounding box.

        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1],  # s = s + vs
            [0, 0, 0, 1, 0, 0, 0],  # r = r (constant)
            [0, 0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1],  # vs = vs
        ])

        # Measurement matrix (we observe x, y, s, r)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Initial covariance
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.kf.P *= 10.0

        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state from bbox
        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Store original detection for appearance features
        self.last_detection = bbox

    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x_center, y_center, scale, aspect_ratio]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h  # scale (area)
        r = w / (h + 1e-6)  # aspect ratio
        return np.array([[x], [y], [s], [r]])

    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """Convert [x_center, y_center, scale, aspect_ratio] to [x1, y1, x2, y2]"""
        x, y, s, r = z.flatten()[:4]

        # Ensure positive values
        s = max(1e-6, s)
        r = max(1e-6, r)

        w = np.sqrt(s * r)
        h = s / (w + 1e-6)

        return np.array([
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2
        ])

    def update(self, bbox: np.ndarray):
        """
        Update state with observed bounding box.

        Args:
            bbox: [x1, y1, x2, y2] detected bounding box
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
        self.last_detection = bbox

    def predict(self) -> np.ndarray:
        """
        Advance state and return predicted bounding box.

        Returns:
            Predicted [x1, y1, x2, y2] bounding box
        """
        # Handle negative area
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(self._z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """Get current bounding box estimate."""
        return self._z_to_bbox(self.kf.x)


def reset_tracker_count():
    """Reset global tracker ID counter (useful for testing)"""
    KalmanBoxTracker.count = 0
