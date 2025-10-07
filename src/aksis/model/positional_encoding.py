"""Sinusoidal positional encoding implementation for Aksis."""

import math
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need".

    This implementation creates fixed positional encodings using sine and cosine
    functions of different frequencies.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length for positional encoding.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create div_term for the sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)

        logger.info(
            f"Initialized SinusoidalPositionalEncoding: d_model={d_model}, "
            f"max_len={max_len}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of positional encoding.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model].
        """
        batch_size, seq_len, d_model = x.size()

        # Handle sequences longer than max_len
        if seq_len > self.max_len:
            # Extend positional encoding for longer sequences
            extended_pe = self._extend_positional_encoding(seq_len)
            pe = extended_pe[:seq_len, :]
        else:
            pe = self.pe[:seq_len, :]

        # Add positional encoding to input (ensure same dtype)
        pe = pe.to(x.dtype)
        x = x + pe

        # Apply dropout
        return self.dropout(x)  # type: ignore

    def _extend_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Extend positional encoding for sequences longer than max_len.

        Args:
            seq_len: Required sequence length.

        Returns:
            Extended positional encoding tensor.
        """
        # Create extended positional encoding
        pe = torch.zeros(seq_len, self.d_model, device=self.pe.device)
        position = torch.arange(
            0, seq_len, dtype=torch.float, device=self.pe.device
        ).unsqueeze(1)

        # Create div_term for the sinusoidal functions
        div_term = torch.exp(
            torch.arange(
                0, self.d_model, 2, dtype=torch.float, device=self.pe.device
            )
            * (-math.log(10000.0) / self.d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding for a given sequence length.

        Args:
            seq_len: Sequence length.

        Returns:
            Positional encoding tensor of shape [1, seq_len, d_model].
        """
        if seq_len <= self.max_len:
            return self.pe[:seq_len, :]  # type: ignore
        else:
            return self._extend_positional_encoding(seq_len)
