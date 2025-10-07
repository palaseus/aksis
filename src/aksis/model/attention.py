"""Multi-head attention implementation for Aksis."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in "Attention Is All You Need".

    This implementation supports both self-attention and cross-attention with
    causal masking for decoder-only models.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            dropout: Dropout probability for attention weights.

        Raises:
            ValueError: If d_model is not divisible by num_heads.
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scale factor for attention scores
        self.scale = math.sqrt(self.d_k)

        logger.info(
            f"Initialized MultiHeadAttention: d_model={d_model}, "
            f"num_heads={num_heads}, d_k={self.d_k}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape [batch_size, seq_len, d_model].
            key: Key tensor of shape [batch_size, seq_len, d_model].
            value: Value tensor of shape [batch_size, seq_len, d_model].
            mask: Attention mask of shape [batch_size, num_heads, seq_len, seq_len].
            padding_mask: Padding mask of shape [batch_size, seq_len].

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size, seq_len, d_model = query.size()

        # Linear projections
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)  # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]

        # Reshape for multi-head attention
        q_seq_len = Q.size(1)
        k_seq_len = K.size(1)
        v_seq_len = V.size(1)

        Q = Q.view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, v_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: [batch_size, num_heads, seq_len, d_k]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: [batch_size, num_heads, seq_len, seq_len]

        # Apply masks
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        if padding_mask is not None:
            # Expand padding mask to match attention scores shape
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # Shape: [batch_size, num_heads, seq_len, d_k]

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        # Shape: [batch_size, seq_len, d_model]

        # Final linear projection
        output = self.w_o(context)

        return output  # type: ignore

    def get_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights.

        Args:
            query: Query tensor of shape [batch_size, seq_len, d_model].
            key: Key tensor of shape [batch_size, seq_len, d_model].
            value: Value tensor of shape [batch_size, seq_len, d_model].
            mask: Attention mask of shape [batch_size, num_heads, seq_len, seq_len].
            padding_mask: Padding mask of shape [batch_size, seq_len].

        Returns:
            Tuple of (output, attention_weights).
        """
        batch_size, seq_len, d_model = query.size()

        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply masks
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Final linear projection
        output = self.w_o(context)

        return output, attention_weights
