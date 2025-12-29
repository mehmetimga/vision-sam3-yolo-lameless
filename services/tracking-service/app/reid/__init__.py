"""
Cow Re-Identification Module

Provides cross-video cow identification using vector similarity search
with DINOv3 embeddings stored in Qdrant.
"""

from .matcher import CowReIDMatcher, CowIdentity, ReIDMatch

__all__ = [
    "CowReIDMatcher",
    "CowIdentity",
    "ReIDMatch"
]
