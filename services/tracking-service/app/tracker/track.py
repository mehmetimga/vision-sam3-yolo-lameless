"""
Track state management for multi-object tracking.

Represents the state of a tracked object including its Kalman filter,
appearance features, and tracking statistics.
"""
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum


class TrackState(Enum):
    """Track lifecycle states"""
    TENTATIVE = 1  # Newly created, not yet confirmed
    CONFIRMED = 2  # Actively tracked
    LOST = 3       # Lost but may be recovered
    DELETED = 4    # To be removed


@dataclass
class Track:
    """
    Represents a tracked object.

    Attributes:
        track_id: Unique identifier
        bbox: Current bounding box [x1, y1, x2, y2]
        confidence: Detection confidence
        embedding: Appearance embedding vector
        state: Current track state
        age: Total frames since creation
        hits: Number of successful matches
        time_since_update: Frames since last update
        frame_history: History of frame indices where track was detected
        bbox_history: History of bounding boxes
    """
    track_id: int
    bbox: np.ndarray
    confidence: float = 0.0
    embedding: Optional[np.ndarray] = None
    state: TrackState = TrackState.TENTATIVE
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    frame_history: List[int] = field(default_factory=list)
    bbox_history: List[np.ndarray] = field(default_factory=list)
    smoothed_embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize histories"""
        if not self.bbox_history:
            self.bbox_history = [self.bbox.copy()]
        if not self.frame_history:
            self.frame_history = [0]
        if self.embedding is not None:
            self.smoothed_embedding = self.embedding.copy()

    def update(self, bbox: np.ndarray, confidence: float,
               embedding: Optional[np.ndarray] = None, frame_idx: int = 0):
        """
        Update track with new detection.

        Args:
            bbox: New bounding box
            confidence: Detection confidence
            embedding: New appearance embedding
            frame_idx: Current frame index
        """
        self.bbox = bbox.copy()
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.bbox_history.append(bbox.copy())
        self.frame_history.append(frame_idx)

        # Update embedding with momentum smoothing
        if embedding is not None:
            if self.smoothed_embedding is None:
                self.smoothed_embedding = embedding.copy()
            else:
                # Exponential moving average
                momentum = 0.9
                self.smoothed_embedding = momentum * self.smoothed_embedding + (1 - momentum) * embedding
            self.embedding = embedding

        # Transition states
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        """Mark track as having no detection this frame"""
        self.age += 1
        self.time_since_update += 1

        # State transitions
        if self.state == TrackState.CONFIRMED and self.time_since_update > 30:
            self.state = TrackState.LOST
        elif self.state == TrackState.TENTATIVE and self.time_since_update > 3:
            self.state = TrackState.DELETED
        elif self.state == TrackState.LOST and self.time_since_update > 90:
            self.state = TrackState.DELETED

    def predict(self, predicted_bbox: np.ndarray):
        """Update with predicted position (no detection)"""
        self.bbox = predicted_bbox
        self.age += 1

    def is_confirmed(self) -> bool:
        """Check if track is confirmed"""
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        """Check if track should be deleted"""
        return self.state == TrackState.DELETED

    def get_feature(self) -> Optional[np.ndarray]:
        """Get smoothed appearance feature"""
        return self.smoothed_embedding

    def get_velocity(self) -> np.ndarray:
        """Estimate velocity from recent positions"""
        if len(self.bbox_history) < 2:
            return np.zeros(2)

        # Use last two positions
        prev = self.bbox_history[-2]
        curr = self.bbox_history[-1]

        prev_center = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
        curr_center = np.array([(curr[0] + curr[2]) / 2, (curr[1] + curr[3]) / 2])

        return curr_center - prev_center

    def get_area(self) -> float:
        """Get current bounding box area"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.tolist(),
            "confidence": float(self.confidence),
            "state": self.state.name,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "start_frame": self.frame_history[0] if self.frame_history else 0,
            "end_frame": self.frame_history[-1] if self.frame_history else 0,
            "has_embedding": self.embedding is not None
        }


class TrackManager:
    """
    Manages multiple tracks.

    Handles track creation, deletion, and ID assignment.
    """

    def __init__(self, max_tracks: int = 100):
        """
        Initialize track manager.

        Args:
            max_tracks: Maximum number of tracks to maintain
        """
        self.tracks: List[Track] = []
        self.next_id: int = 0
        self.max_tracks = max_tracks
        self.track_count = 0

    def create_track(self, bbox: np.ndarray, confidence: float,
                     embedding: Optional[np.ndarray] = None,
                     frame_idx: int = 0) -> Track:
        """
        Create a new track.

        Args:
            bbox: Initial bounding box
            confidence: Detection confidence
            embedding: Appearance embedding
            frame_idx: Frame index

        Returns:
            New Track object
        """
        track = Track(
            track_id=self.next_id,
            bbox=bbox,
            confidence=confidence,
            embedding=embedding,
            frame_history=[frame_idx],
            bbox_history=[bbox.copy()]
        )
        self.next_id += 1
        self.tracks.append(track)
        self.track_count += 1
        return track

    def delete_track(self, track: Track):
        """Remove a track"""
        if track in self.tracks:
            self.tracks.remove(track)

    def get_active_tracks(self) -> List[Track]:
        """Get confirmed tracks"""
        return [t for t in self.tracks if t.is_confirmed()]

    def get_all_tracks(self) -> List[Track]:
        """Get all non-deleted tracks"""
        return [t for t in self.tracks if not t.is_deleted()]

    def cleanup(self):
        """Remove deleted tracks"""
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Limit total tracks
        if len(self.tracks) > self.max_tracks:
            # Keep most recently updated tracks
            self.tracks.sort(key=lambda t: t.time_since_update)
            self.tracks = self.tracks[:self.max_tracks]

    def reset(self):
        """Clear all tracks"""
        self.tracks = []
        self.next_id = 0
        self.track_count = 0

    def get_statistics(self) -> dict:
        """Get tracking statistics"""
        return {
            "total_tracks": self.track_count,
            "active_tracks": len(self.get_active_tracks()),
            "confirmed": len([t for t in self.tracks if t.state == TrackState.CONFIRMED]),
            "tentative": len([t for t in self.tracks if t.state == TrackState.TENTATIVE]),
            "lost": len([t for t in self.tracks if t.state == TrackState.LOST])
        }
