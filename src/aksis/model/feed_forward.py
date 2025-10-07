"""Feed-forward network implementation for Aksis."""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    """
    Feed-forward network as described in "Attention Is All You Need".

    This implementation uses two linear transformations with ReLU activation
    and dropout in between.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension.
            d_ff: Feed-forward dimension (defaults to 4 * d_model).
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        # Linear transformations
        self.linear1 = nn.Linear(d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, d_model)

        # Activation function
        self.activation = nn.ReLU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"Initialized FeedForward: d_model={d_model}, d_ff={self.d_ff}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model].
        """
        # First linear transformation
        x = self.linear1(x)

        # ReLU activation
        x = self.activation(x)

        # Dropout
        x = self.dropout(x)

        # Second linear transformation
        x = self.linear2(x)

        return x
