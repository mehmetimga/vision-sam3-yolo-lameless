"""
ByteTrack Multi-Object Tracker for Cow Tracking.

Implements the ByteTrack algorithm which performs two-stage association:
1. High-confidence detections matched to existing tracks
2. Low-confidence detections matched to unmatched tracks

This approach reduces ID switches by utilizing low-confidence detections
that might otherwise be discarded.

Reference: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .kalman import KalmanBoxTracker, reset_tracker_count
from .matching import (
    iou_batch,
    associate_detections_to_tracks,
    linear_assignment
)
from .track import Track, TrackManager, TrackState


@dataclass
class Detection:
    """Single detection from YOLO"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0  # 0 for cow
    embedding: Optional[np.ndarray] = None


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Two-stage association:
    1. Match high-confidence (>high_thresh) detections to tracks
    2. Match low-confidence (low_thresh to high_thresh) detections to remaining tracks

    Args:
        high_thresh: Confidence threshold for primary association (default: 0.6)
        low_thresh: Minimum confidence for secondary association (default: 0.1)
        match_thresh: IoU threshold for matching (default: 0.8)
        track_buffer: Maximum frames to keep lost tracks (default: 30)
        use_appearance: Whether to use appearance features (default: True)
        appearance_weight: Weight for appearance vs IoU (default: 0.5)
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        use_appearance: bool = True,
        appearance_weight: float = 0.5
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.use_appearance = use_appearance
        self.appearance_weight = appearance_weight

        # Track management
        self.track_manager = TrackManager(max_tracks=100)
        self.kalman_trackers: Dict[int, KalmanBoxTracker] = {}

        # Statistics
        self.frame_id = 0

    def update(
        self,
        detections: List[Detection],
        frame_idx: Optional[int] = None
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of Detection objects
            frame_idx: Optional frame index (defaults to internal counter)

        Returns:
            List of confirmed tracks after update
        """
        if frame_idx is None:
            frame_idx = self.frame_id
        self.frame_id = frame_idx + 1

        if len(detections) == 0:
            # No detections - just predict
            self._predict_all()
            self._mark_all_missed()
            return self.track_manager.get_active_tracks()

        # Split detections by confidence
        high_conf_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_conf_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        # Get active tracks
        active_tracks = self.track_manager.get_all_tracks()

        # Predict new positions
        self._predict_all()

        # Stage 1: Match high-confidence detections to tracks
        matched_h, unmatched_dets_h, unmatched_tracks_h = self._first_stage_association(
            high_conf_dets, active_tracks
        )

        # Update matched tracks
        for det_idx, track_idx in matched_h:
            det = high_conf_dets[det_idx]
            track = active_tracks[track_idx]
            self._update_track(track, det, frame_idx)

        # Stage 2: Match low-confidence detections to unmatched tracks
        unmatched_tracks = [active_tracks[i] for i in unmatched_tracks_h]
        matched_l, unmatched_dets_l, remaining_unmatched_tracks = self._second_stage_association(
            low_conf_dets, unmatched_tracks
        )

        # Update matched tracks from second stage
        for det_idx, track_idx in matched_l:
            det = low_conf_dets[det_idx]
            track = unmatched_tracks[track_idx]
            self._update_track(track, det, frame_idx)

        # Stage 3: Try to reactivate lost tracks with remaining high-conf detections
        lost_tracks = [t for t in self.track_manager.tracks if t.state == TrackState.LOST]
        unmatched_high_dets = [high_conf_dets[i] for i in unmatched_dets_h]

        matched_r, still_unmatched_dets, _ = self._reactivation_association(
            unmatched_high_dets, lost_tracks
        )

        for det_idx, track_idx in matched_r:
            det = unmatched_high_dets[det_idx]
            track = lost_tracks[track_idx]
            self._update_track(track, det, frame_idx)

        # Mark remaining unmatched tracks as missed
        for track in unmatched_tracks:
            if track not in [t for t in lost_tracks if any(m[1] == lost_tracks.index(t) for m in matched_r)]:
                track.mark_missed()

        # Create new tracks from remaining unmatched high-confidence detections
        final_unmatched_dets = [unmatched_high_dets[i] for i in still_unmatched_dets]
        for det in final_unmatched_dets:
            self._create_track(det, frame_idx)

        # Cleanup deleted tracks
        self.track_manager.cleanup()

        return self.track_manager.get_active_tracks()

    def _first_stage_association(
        self,
        detections: List[Detection],
        tracks: List[Track]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        First-stage association with high-confidence detections.

        Uses both IoU and appearance (if available).
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(tracks))

        det_bboxes = np.array([d.bbox for d in detections])
        track_bboxes = np.array([t.bbox for t in tracks])

        # Get features if using appearance
        det_features = None
        track_features = None
        if self.use_appearance:
            det_features = np.array([d.embedding for d in detections if d.embedding is not None])
            track_features = np.array([t.get_feature() for t in tracks if t.get_feature() is not None])

            if len(det_features) != len(detections) or len(track_features) != len(tracks):
                # Fall back to IoU only
                det_features = None
                track_features = None

        return associate_detections_to_tracks(
            det_bboxes, track_bboxes,
            iou_threshold=self.match_thresh,
            detection_features=det_features,
            track_features=track_features,
            appearance_weight=self.appearance_weight
        )

    def _second_stage_association(
        self,
        detections: List[Detection],
        tracks: List[Track]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Second-stage association with low-confidence detections.

        Uses IoU only (low-confidence detections may have unreliable features).
        """
        if len(detections) == 0 or len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(tracks))

        det_bboxes = np.array([d.bbox for d in detections])
        track_bboxes = np.array([t.bbox for t in tracks])

        # IoU only for low-confidence
        return associate_detections_to_tracks(
            det_bboxes, track_bboxes,
            iou_threshold=0.5,  # Lower threshold for second stage
            detection_features=None,
            track_features=None
        )

    def _reactivation_association(
        self,
        detections: List[Detection],
        lost_tracks: List[Track]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Attempt to reactivate lost tracks with unmatched detections.

        Uses both IoU and appearance for better recovery.
        """
        if len(detections) == 0 or len(lost_tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(lost_tracks))

        det_bboxes = np.array([d.bbox for d in detections])
        track_bboxes = np.array([t.bbox for t in lost_tracks])

        # Use appearance for reactivation
        det_features = None
        track_features = None
        if self.use_appearance:
            det_features = np.array([d.embedding for d in detections if d.embedding is not None])
            track_features = np.array([t.get_feature() for t in lost_tracks if t.get_feature() is not None])

            if len(det_features) != len(detections) or len(track_features) != len(lost_tracks):
                det_features = None
                track_features = None

        return associate_detections_to_tracks(
            det_bboxes, track_bboxes,
            iou_threshold=0.3,  # Lower threshold for reactivation
            detection_features=det_features,
            track_features=track_features,
            appearance_weight=0.7  # Higher weight on appearance for reactivation
        )

    def _predict_all(self):
        """Predict new positions for all tracks using Kalman filter"""
        for track in self.track_manager.tracks:
            if track.track_id in self.kalman_trackers:
                kf = self.kalman_trackers[track.track_id]
                predicted_bbox = kf.predict()
                track.predict(predicted_bbox)

    def _mark_all_missed(self):
        """Mark all tracks as missed"""
        for track in self.track_manager.tracks:
            track.mark_missed()

    def _update_track(self, track: Track, detection: Detection, frame_idx: int):
        """Update track with detection"""
        track.update(
            bbox=detection.bbox,
            confidence=detection.confidence,
            embedding=detection.embedding,
            frame_idx=frame_idx
        )

        # Update Kalman filter
        if track.track_id in self.kalman_trackers:
            self.kalman_trackers[track.track_id].update(detection.bbox)

    def _create_track(self, detection: Detection, frame_idx: int) -> Track:
        """Create new track from detection"""
        track = self.track_manager.create_track(
            bbox=detection.bbox,
            confidence=detection.confidence,
            embedding=detection.embedding,
            frame_idx=frame_idx
        )

        # Create Kalman filter
        self.kalman_trackers[track.track_id] = KalmanBoxTracker(detection.bbox)

        return track

    def reset(self):
        """Reset tracker state"""
        self.track_manager.reset()
        self.kalman_trackers.clear()
        self.frame_id = 0
        reset_tracker_count()

    def get_tracks_for_frame(self) -> List[dict]:
        """Get all active tracks as dictionaries for current frame"""
        return [t.to_dict() for t in self.track_manager.get_active_tracks()]

    def get_statistics(self) -> dict:
        """Get tracking statistics"""
        stats = self.track_manager.get_statistics()
        stats.update({
            "frame_id": self.frame_id,
            "high_thresh": self.high_thresh,
            "low_thresh": self.low_thresh,
            "use_appearance": self.use_appearance
        })
        return stats
