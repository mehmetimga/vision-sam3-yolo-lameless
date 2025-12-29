"""
Cow Tracking Module

Implements ByteTrack multi-object tracking for cow detection and tracking
in video sequences. Uses Kalman filtering for motion prediction and
supports appearance-based matching.
"""

from .bytetrack import ByteTracker, Detection
from .kalman import KalmanBoxTracker, reset_tracker_count
from .matching import iou_batch, cosine_distance, associate_detections_to_tracks
from .track import Track, TrackManager, TrackState

__all__ = [
    "ByteTracker",
    "Detection",
    "KalmanBoxTracker",
    "reset_tracker_count",
    "iou_batch",
    "cosine_distance",
    "associate_detections_to_tracks",
    "Track",
    "TrackManager",
    "TrackState"
]
