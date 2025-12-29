"""
Graphormer-style Graph Encodings for True Graph Transformers.

Implements:
1. Centrality Encoding - Node importance based on degree
2. Spatial Encoding - Shortest path distances between nodes
3. Temporal Encoding - Time-based positional encoding for videos
4. Edge Encoding - Encodes edge features as attention biases
"""
import math
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from typing import Optional, Tuple


class CentralityEncoding(nn.Module):
    """
    Centrality Encoding from Graphormer.

    Adds learnable embeddings based on node degree (in-degree and out-degree).
    This captures the importance of each node in the graph structure.

    For undirected graphs (like our cow similarity graph), in-degree = out-degree.
    """

    def __init__(self, max_degree: int = 100, hidden_dim: int = 128):
        """
        Args:
            max_degree: Maximum node degree to encode (clipped)
            hidden_dim: Output dimension
        """
        super().__init__()
        self.max_degree = max_degree

        # Learnable embedding per degree
        self.degree_encoder = nn.Embedding(max_degree + 1, hidden_dim)

        # For directed graphs, separate in/out degree
        # For our use case, we combine them
        self.out_degree_encoder = nn.Embedding(max_degree + 1, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.degree_encoder.weight, std=0.02)
        nn.init.normal_(self.out_degree_encoder.weight, std=0.02)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute centrality encoding for all nodes.

        Args:
            edge_index: (2, E) edge indices
            num_nodes: Number of nodes

        Returns:
            (N, hidden_dim) centrality encodings
        """
        device = edge_index.device

        # Compute in-degree (number of incoming edges)
        in_degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        in_degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long, device=device))

        # Compute out-degree (number of outgoing edges)
        out_degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        out_degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=device))

        # Clip to max_degree
        in_degree = torch.clamp(in_degree, max=self.max_degree)
        out_degree = torch.clamp(out_degree, max=self.max_degree)

        # Get embeddings
        in_enc = self.degree_encoder(in_degree)
        out_enc = self.out_degree_encoder(out_degree)

        # Combine (add for undirected, could concatenate for directed)
        return in_enc + out_enc


class SpatialEncoding(nn.Module):
    """
    Spatial Encoding from Graphormer.

    Encodes the shortest path distance (SPD) between node pairs as an attention bias.
    Nodes that are closer in the graph structure have stronger attention bias.

    This replaces the need for explicit positional encodings from Laplacian eigenvectors.
    """

    def __init__(self, max_spd: int = 10, num_heads: int = 8):
        """
        Args:
            max_spd: Maximum shortest path distance to encode
            num_heads: Number of attention heads
        """
        super().__init__()
        self.max_spd = max_spd
        self.num_heads = num_heads

        # Learnable bias per SPD per head
        # +2 for: unreachable (-1 -> 0), self-loop (0 -> 1), SPD 1-max_spd (2 to max_spd+1)
        self.spd_bias = nn.Embedding(max_spd + 2, num_heads)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.spd_bias.weight)

    def compute_shortest_paths(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute all-pairs shortest path distances using NetworkX.

        Args:
            edge_index: (2, E) edge indices
            num_nodes: Number of nodes

        Returns:
            (N, N) SPD matrix
        """
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        edge_index_np = edge_index.cpu().numpy()
        edges = list(zip(edge_index_np[0], edge_index_np[1]))
        G.add_edges_from(edges)

        # Compute all-pairs shortest paths
        spd_matrix = np.full((num_nodes, num_nodes), self.max_spd + 1, dtype=np.int64)

        # Use Floyd-Warshall for dense graphs, BFS for sparse
        if num_nodes <= 500:
            # Dense: compute all paths
            for i in range(num_nodes):
                lengths = nx.single_source_shortest_path_length(G, i, cutoff=self.max_spd)
                for j, dist in lengths.items():
                    spd_matrix[i, j] = min(dist, self.max_spd)
        else:
            # For very large graphs, sample or use approximation
            # Here we use BFS with cutoff
            for i in range(num_nodes):
                lengths = nx.single_source_shortest_path_length(G, i, cutoff=self.max_spd)
                for j, dist in lengths.items():
                    spd_matrix[i, j] = min(dist, self.max_spd)

        return torch.tensor(spd_matrix, dtype=torch.long)

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute spatial encoding bias for attention.

        Args:
            edge_index: (2, E) edge indices
            num_nodes: Number of nodes

        Returns:
            (N, N, num_heads) attention bias matrix
        """
        device = edge_index.device

        # Compute SPD matrix
        spd = self.compute_shortest_paths(edge_index, num_nodes)
        spd = spd.to(device)

        # Shift: -1 (unreachable) -> 0, 0 (self) -> 1, etc.
        spd_shifted = spd + 1
        spd_shifted = torch.clamp(spd_shifted, max=self.max_spd + 1)

        # Get bias per node pair
        bias = self.spd_bias(spd_shifted)  # (N, N, num_heads)

        return bias


class TemporalEncoding(nn.Module):
    """
    Temporal Encoding for video-based graphs.

    Encodes the time difference between videos using sinusoidal encoding
    (similar to positional encoding in transformers).

    For cows, this captures:
    - Same-day videos (high similarity expected)
    - Multi-day gaps (lameness may change)
    - Long-term trends (chronic vs acute lameness)
    """

    def __init__(self, hidden_dim: int = 128, max_time_days: float = 365.0):
        """
        Args:
            hidden_dim: Output dimension
            max_time_days: Maximum time difference to encode (in days)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_time_days = max_time_days

        # Learnable projection
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        # Create sinusoidal frequencies
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal encoding for each node based on timestamp.

        Args:
            timestamps: (N,) timestamps in seconds from epoch

        Returns:
            (N, hidden_dim) temporal encodings
        """
        device = timestamps.device

        # Normalize timestamps to days from first timestamp
        if timestamps.numel() == 0:
            return torch.zeros(0, self.hidden_dim, device=device)

        min_time = timestamps.min()
        time_days = (timestamps - min_time) / 86400.0  # Convert to days
        time_days = torch.clamp(time_days, max=self.max_time_days)

        # Sinusoidal encoding
        pe = torch.zeros(len(timestamps), self.hidden_dim, device=device)
        time_days = time_days.unsqueeze(1)  # (N, 1)

        pe[:, 0::2] = torch.sin(time_days * self.div_term)
        pe[:, 1::2] = torch.cos(time_days * self.div_term)

        return self.time_proj(pe)


class EdgeEncoding(nn.Module):
    """
    Edge Encoding for Graphormer.

    Converts edge features into attention biases.
    For our cow graph:
    - Similarity score (from DINOv3)
    - Temporal distance (days between videos)
    - Edge type (kNN vs temporal connection)
    """

    def __init__(self, edge_dim: int = 3, num_heads: int = 8):
        """
        Args:
            edge_dim: Dimension of edge features
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads

        # Project edge features to attention bias per head
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, num_heads * 2),
            nn.ReLU(),
            nn.Linear(num_heads * 2, num_heads)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.edge_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute edge-based attention bias.

        Args:
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
            num_nodes: Number of nodes

        Returns:
            (N, N, num_heads) attention bias from edges
        """
        device = edge_attr.device

        # Initialize bias matrix with zeros (no bias for non-edges)
        bias = torch.zeros(num_nodes, num_nodes, self.num_heads, device=device)

        # Project edge features to bias
        edge_bias = self.edge_proj(edge_attr)  # (E, num_heads)

        # Scatter into bias matrix
        src, dst = edge_index
        bias[src, dst] = edge_bias

        return bias


class GraphormerEncodings(nn.Module):
    """
    Combined Graphormer-style encodings.

    Aggregates all encoding types:
    1. Centrality (node degree importance)
    2. Spatial (shortest path distances)
    3. Temporal (time-based encoding)
    4. Edge (edge feature biases)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        max_degree: int = 50,
        max_spd: int = 10,
        edge_dim: int = 3,
        use_temporal: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_temporal = use_temporal

        # Encodings
        self.centrality_enc = CentralityEncoding(max_degree, hidden_dim)
        self.spatial_enc = SpatialEncoding(max_spd, num_heads)
        self.edge_enc = EdgeEncoding(edge_dim, num_heads)

        if use_temporal:
            self.temporal_enc = TemporalEncoding(hidden_dim)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        num_nodes: int,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute all Graphormer encodings.

        Args:
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
            num_nodes: Number of nodes
            timestamps: (N,) optional timestamps

        Returns:
            node_encoding: (N, hidden_dim) node-level encodings
            attention_bias: (N, N, num_heads) attention biases
        """
        # Node-level encodings
        centrality = self.centrality_enc(edge_index, num_nodes)

        if self.use_temporal and timestamps is not None:
            temporal = self.temporal_enc(timestamps)
            node_encoding = centrality + temporal
        else:
            node_encoding = centrality

        # Attention biases
        spatial_bias = self.spatial_enc(edge_index, num_nodes)

        if edge_attr is not None:
            edge_bias = self.edge_enc(edge_index, edge_attr, num_nodes)
            attention_bias = spatial_bias + edge_bias
        else:
            attention_bias = spatial_bias

        return node_encoding, attention_bias
