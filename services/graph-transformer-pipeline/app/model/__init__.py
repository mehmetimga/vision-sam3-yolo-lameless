"""
Graph Transformer Model Module

Implements Graphormer-style architecture for cow lameness detection.
"""

from .encodings import (
    CentralityEncoding,
    SpatialEncoding,
    TemporalEncoding,
    EdgeEncoding,
    GraphormerEncodings
)
from .attention import GraphBiasedMultiHeadAttention, VirtualNodeAttention
from .layers import GraphormerLayer, GraphormerEncoder, GraphLevelReadout
from .graphormer import CowLamenessGraphormer, GraphormerGraphBuilder

__all__ = [
    # Encodings
    "CentralityEncoding",
    "SpatialEncoding",
    "TemporalEncoding",
    "EdgeEncoding",
    "GraphormerEncodings",
    # Attention
    "GraphBiasedMultiHeadAttention",
    "VirtualNodeAttention",
    # Layers
    "GraphormerLayer",
    "GraphormerEncoder",
    "GraphLevelReadout",
    # Model
    "CowLamenessGraphormer",
    "GraphormerGraphBuilder"
]
