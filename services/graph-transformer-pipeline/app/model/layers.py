"""
Transformer Layers for Graph Transformer.

Implements the full Graphormer layer stack including:
- Self-attention with graph biases
- Feed-forward networks
- Layer normalization
- Residual connections
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import GraphBiasedMultiHeadAttention, VirtualNodeAttention


class GraphormerLayer(nn.Module):
    """
    Single Graphormer Transformer Layer.

    Architecture (Pre-LN):
    1. LayerNorm -> Multi-Head Attention (with graph bias) -> Residual
    2. LayerNorm -> Feed-Forward Network -> Residual
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Args:
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network hidden dimension
            dropout: Dropout rate
            activation: Activation function ("gelu" or "relu")
        """
        super().__init__()

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Self-attention
        self.self_attn = GraphBiasedMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-forward network
        if activation == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (N, hidden_dim) node features
            attention_bias: (N, N, num_heads) structural bias
            return_attention: Whether to return attention weights

        Returns:
            x: (N, hidden_dim) updated features
            attn_weights: Optional attention weights
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        if return_attention:
            x, attn_weights = self.self_attn(x, attention_bias, return_attention=True)
        else:
            x = self.self_attn(x, attention_bias)
            attn_weights = None

        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weights


class GraphormerEncoder(nn.Module):
    """
    Graphormer Encoder: Stack of Graphormer layers.

    Features:
    - Multiple transformer layers with graph biases
    - Optional virtual node for graph-level aggregation
    - Post-encoder normalization
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_virtual_node: bool = True
    ):
        """
        Args:
            hidden_dim: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ffn_dim: FFN hidden dimension
            dropout: Dropout rate
            use_virtual_node: Whether to use virtual node aggregation
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node

        # Transformer layers
        self.layers = nn.ModuleList([
            GraphormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Virtual node attention (applied every layer)
        if use_virtual_node:
            self.virtual_node_layers = nn.ModuleList([
                VirtualNodeAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])

        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through encoder.

        Args:
            x: (N, hidden_dim) node features
            attention_bias: (N, N, num_heads) structural bias
            return_attention: Whether to return attention weights

        Returns:
            x: (N, hidden_dim) encoded node features
            vn: (1, hidden_dim) virtual node embedding (if use_virtual_node)
            all_attention: List of attention weights (if return_attention)
        """
        all_attention = [] if return_attention else None
        vn = None

        for i, layer in enumerate(self.layers):
            # Apply transformer layer
            x, attn_weights = layer(x, attention_bias, return_attention)

            if return_attention and attn_weights is not None:
                all_attention.append(attn_weights)

            # Apply virtual node
            if self.use_virtual_node:
                x, vn = self.virtual_node_layers[i](x, attention_bias)

        # Final normalization
        x = self.final_norm(x)

        return x, vn, all_attention


class GraphLevelReadout(nn.Module):
    """
    Graph-level readout for prediction.

    Aggregates node features into a single graph representation.
    Supports multiple aggregation strategies:
    1. Virtual node (from Graphormer)
    2. Mean pooling
    3. Attention-weighted pooling
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        use_virtual_node: bool = True,
        use_attention_pool: bool = True
    ):
        super().__init__()

        self.use_virtual_node = use_virtual_node
        self.use_attention_pool = use_attention_pool

        # Attention pooling
        if use_attention_pool:
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

        # Combine different pooling outputs
        pool_dim = hidden_dim  # Mean pool always included
        if use_virtual_node:
            pool_dim += hidden_dim
        if use_attention_pool:
            pool_dim += hidden_dim

        self.combine = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        vn: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute graph-level representation.

        Args:
            x: (N, hidden_dim) node features
            vn: (1, hidden_dim) virtual node embedding

        Returns:
            (1, hidden_dim) graph representation
        """
        pools = []

        # Mean pooling
        mean_pool = x.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        pools.append(mean_pool)

        # Virtual node
        if self.use_virtual_node and vn is not None:
            pools.append(vn)

        # Attention pooling
        if self.use_attention_pool:
            attn_scores = self.attention_pool(x)  # (N, 1)
            attn_weights = torch.softmax(attn_scores, dim=0)
            attn_pool = (attn_weights * x).sum(dim=0, keepdim=True)  # (1, hidden_dim)
            pools.append(attn_pool)

        # Concatenate and combine
        combined = torch.cat(pools, dim=-1)
        graph_repr = self.combine(combined)

        return graph_repr
