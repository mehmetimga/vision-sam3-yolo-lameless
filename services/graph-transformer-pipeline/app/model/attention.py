"""
Graph-Biased Multi-Head Attention for True Graph Transformers.

Implements the core attention mechanism from Graphormer where attention
scores are biased by structural information (spatial encoding, edge features).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphBiasedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with graph structural biases.

    Standard transformer attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

    Graphormer attention (with bias):
        Attention(Q, K, V) = softmax(QK^T / sqrt(d) + bias) V

    Where bias encodes:
    - Spatial encoding (shortest path distances)
    - Edge encoding (edge features)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            dropout: Attention dropout rate
            bias: Whether to use bias in projections
        """
        super().__init__()

        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        """
        Forward pass with optional graph structural bias.

        Args:
            x: (N, hidden_dim) node features
            attention_bias: (N, N, num_heads) structural bias for attention
            key_padding_mask: (N,) mask for invalid nodes
            return_attention: Whether to return attention weights

        Returns:
            output: (N, hidden_dim) attended features
            attention_weights: (num_heads, N, N) if return_attention=True
        """
        N = x.size(0)

        # Project to Q, K, V
        q = self.q_proj(x)  # (N, hidden_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (N, hidden_dim) -> (N, num_heads, head_dim) -> (num_heads, N, head_dim)
        q = q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # Compute attention scores: (num_heads, N, N)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        # Add graph structural bias
        if attention_bias is not None:
            # attention_bias: (N, N, num_heads) -> (num_heads, N, N)
            bias = attention_bias.permute(2, 0, 1)
            attn_scores = attn_scores + bias

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (N,) -> (1, 1, N) for broadcasting
            mask = key_padding_mask.unsqueeze(0).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # (num_heads, N, N) @ (num_heads, N, head_dim) -> (num_heads, N, head_dim)
        out = torch.bmm(attn_weights, v)

        # Reshape back: (num_heads, N, head_dim) -> (N, num_heads, head_dim) -> (N, hidden_dim)
        out = out.transpose(0, 1).contiguous().view(N, self.hidden_dim)

        # Output projection
        out = self.out_proj(out)
        out = self.out_dropout(out)

        if return_attention:
            return out, attn_weights
        return out


class VirtualNodeAttention(nn.Module):
    """
    Virtual Node for graph-level information aggregation.

    Graphormer uses a virtual node that attends to all nodes in the graph.
    This provides a learnable graph-level representation that can be used
    for graph classification.

    The virtual node:
    1. Attends to all real nodes (receives graph-level info)
    2. All real nodes attend to it (broadcasts global context)
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Learnable virtual node embedding
        self.virtual_node = nn.Parameter(torch.zeros(1, hidden_dim))

        # Attention for virtual node
        self.vn_attention = GraphBiasedMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Update network for virtual node
        self.vn_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        nn.init.normal_(self.virtual_node, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None
    ):
        """
        Update nodes and virtual node.

        Args:
            x: (N, hidden_dim) node features
            attention_bias: (N, N, num_heads) structural bias

        Returns:
            x_updated: (N, hidden_dim) updated node features
            vn: (1, hidden_dim) virtual node embedding
        """
        N = x.size(0)
        device = x.device

        # Expand virtual node
        vn = self.virtual_node.expand(1, -1).to(device)

        # Concatenate virtual node with real nodes: (N+1, hidden_dim)
        x_with_vn = torch.cat([vn, x], dim=0)

        # Extend attention bias for virtual node
        if attention_bias is not None:
            # Add virtual node connections (zero bias - no structural info)
            num_heads = attention_bias.size(2)
            vn_bias = torch.zeros(1, N + 1, num_heads, device=device)
            extended_bias = torch.zeros(N + 1, N + 1, num_heads, device=device)
            extended_bias[1:, 1:] = attention_bias
            attention_bias = extended_bias
        else:
            attention_bias = None

        # Apply attention
        x_attended = self.vn_attention(x_with_vn, attention_bias)

        # Separate virtual node and real nodes
        vn_out = x_attended[0:1]
        x_out = x_attended[1:]

        # Update virtual node
        vn_out = self.vn_update(vn_out)

        return x_out, vn_out
