"""Transformer decoder implementation for Aksis."""

import math
import torch
import torch.nn as nn
from typing import Optional
import logging

from aksis.model.attention import MultiHeadAttention
from aksis.model.positional_encoding import SinusoidalPositionalEncoding
from aksis.model.feed_forward import FeedForward

logger = logging.getLogger(__name__)


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer.

    This layer contains self-attention, feed-forward network, and layer normalization
    with residual connections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize transformer decoder layer.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of transformer decoder layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].
            mask: Attention mask of shape [batch_size, num_heads, seq_len, seq_len].
            padding_mask: Padding mask of shape [batch_size, seq_len].

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model].
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(
            x, x, x, mask=mask, padding_mask=padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder-only model for language modeling.

    This implementation follows the architecture described in "Attention Is All You Need"
    but adapted for decoder-only (GPT-style) language modeling.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: Optional[int] = None,
        max_len: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize transformer decoder.

        Args:
            vocab_size: Vocabulary size.
            d_model: Model dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            d_ff: Feed-forward dimension (defaults to 4 * d_model).
            max_len: Maximum sequence length.
            dropout: Dropout probability.

        Raises:
            ValueError: If any parameter is invalid.
        """
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, max_len, dropout
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, num_heads, self.d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Initialized TransformerDecoder: vocab_size={vocab_size}, "
            f"d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}, "
            f"d_ff={self.d_ff}, max_len={max_len}"
        )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of transformer decoder.

        Args:
            x: Input token IDs of shape [batch_size, seq_len].
            mask: Attention mask of shape [batch_size, num_heads, seq_len, seq_len].
            padding_mask: Padding mask of shape [batch_size, seq_len].

        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size].
        """
        # Token embedding
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask, padding_mask=padding_mask)

        # Final layer normalization
        x = self.norm(x)

        # Output projection
        logits = self.output_projection(x)

        return logits  # type: ignore

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create causal mask for decoder-only attention.

        Args:
            seq_len: Sequence length.

        Returns:
            Causal mask tensor of shape [1, 1, seq_len, seq_len].
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Generate text using the transformer decoder.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len].
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            pad_token_id: Padding token ID.

        Returns:
            Generated token IDs of shape [batch_size, max_length].
        """
        self.eval()
        device = input_ids.device

        # Initialize generation
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get current sequence length
                current_len = generated.size(1)

                # Create causal mask
                mask = self.create_causal_mask(current_len).to(device)

                # Forward pass
                logits = self.forward(generated, mask=mask)

                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature

                if do_sample:
                    # Sample from the distribution
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(
                        next_token_logits, dim=-1, keepdim=True
                    )

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def get_num_parameters(self) -> int:
        """
        Get the total number of parameters in the model.

        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def get_model_size_mb(self) -> float:
        """
        Get the model size in megabytes.

        Returns:
            Model size in MB.
        """
        param_size = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
