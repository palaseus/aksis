"""Tests for MultiHeadAttention component."""

import pytest
import torch
import torch.nn as nn
import math

from aksis.model.attention import MultiHeadAttention


class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention class."""

    def test_attention_initialization(self) -> None:
        """Test attention module initialization."""
        d_model = 512
        num_heads = 8
        dropout = 0.1

        attention = MultiHeadAttention(d_model, num_heads, dropout)

        assert attention.d_model == d_model
        assert attention.num_heads == num_heads
        assert attention.d_k == d_model // num_heads
        assert attention.dropout.p == dropout
        assert isinstance(attention.w_q, nn.Linear)
        assert isinstance(attention.w_k, nn.Linear)
        assert isinstance(attention.w_v, nn.Linear)
        assert isinstance(attention.w_o, nn.Linear)

    def test_attention_initialization_invalid_heads(self) -> None:
        """Test attention initialization with invalid number of heads."""
        d_model = 512
        num_heads = 7  # Not divisible by d_model

        with pytest.raises(
            ValueError, match="d_model must be divisible by num_heads"
        ):
            MultiHeadAttention(d_model, num_heads)

    def test_attention_forward_shape(self) -> None:
        """Test attention forward pass output shapes."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert isinstance(output, torch.Tensor)

    def test_attention_forward_different_shapes(self) -> None:
        """Test attention with different query, key, value shapes."""
        batch_size = 2
        q_len = 5
        k_len = 8
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        q = torch.randn(batch_size, q_len, d_model)
        k = torch.randn(batch_size, k_len, d_model)
        v = torch.randn(batch_size, k_len, d_model)

        output = attention(q, k, v)

        assert output.shape == (batch_size, q_len, d_model)

    def test_attention_with_mask(self) -> None:
        """Test attention with causal mask."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        output = attention(x, x, x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_attention_with_padding_mask(self) -> None:
        """Test attention with padding mask."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Create padding mask (0s for padding tokens)
        padding_mask = torch.ones(batch_size, seq_len)
        padding_mask[:, -2:] = 0  # Last 2 tokens are padding

        output = attention(x, x, x, padding_mask=padding_mask)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_attention_scores_normalization(self) -> None:
        """Test that attention scores are properly normalized before dropout."""
        batch_size = 1
        seq_len = 5
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(
            d_model, num_heads, dropout=0.0
        )  # No dropout for testing
        x = torch.randn(batch_size, seq_len, d_model)

        # Use the get_attention_weights method to get attention scores
        output, attention_weights = attention.get_attention_weights(x, x, x)

        # Check that attention scores sum to 1 along the last dimension
        scores_sum = attention_weights.sum(dim=-1)
        assert torch.allclose(
            scores_sum, torch.ones_like(scores_sum), atol=1e-6
        )

    def test_attention_gradient_flow(self) -> None:
        """Test that gradients flow properly through attention."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = attention(x, x, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_attention_cuda_compatibility(self) -> None:
        """Test attention on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads).cuda()
        x = torch.randn(batch_size, seq_len, d_model).cuda()

        output = attention(x, x, x)

        assert output.device.type == "cuda"
        assert output.shape == (batch_size, seq_len, d_model)

    def test_attention_mixed_precision(self) -> None:
        """Test attention with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision")

        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads).cuda()
        x = torch.randn(batch_size, seq_len, d_model).cuda()

        with torch.amp.autocast("cuda"):
            output = attention(x, x, x)

        assert output.dtype == torch.float16
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_attention_empty_sequence(self) -> None:
        """Test attention with empty sequence."""
        batch_size = 2
        seq_len = 0
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_attention_single_token(self) -> None:
        """Test attention with single token."""
        batch_size = 2
        seq_len = 1
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_attention_large_sequence(self) -> None:
        """Test attention with large sequence length."""
        batch_size = 1
        seq_len = 1024
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_attention_dropout_training(self) -> None:
        """Test attention dropout during training."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        dropout = 0.5

        attention = MultiHeadAttention(d_model, num_heads, dropout)
        attention.train()  # Enable dropout

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times to check for dropout variation
        outputs = []
        for _ in range(5):
            output = attention(x, x, x)
            outputs.append(output)

        # Check that outputs are different (due to dropout)
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_attention_dropout_eval(self) -> None:
        """Test attention without dropout during evaluation."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        dropout = 0.5

        attention = MultiHeadAttention(d_model, num_heads, dropout)
        attention.eval()  # Disable dropout

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times to check for consistency
        outputs = []
        for _ in range(5):
            output = attention(x, x, x)
            outputs.append(output)

        # Check that outputs are identical (no dropout)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_attention_parameter_count(self) -> None:
        """Test that attention has correct number of parameters."""
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)

        # Expected parameters: 4 linear layers (Q, K, V, O) each with d_model x d_model
        expected_params = 4 * d_model * d_model
        actual_params = sum(p.numel() for p in attention.parameters())

        assert actual_params == expected_params

    def test_attention_attention_weights_shape(self) -> None:
        """Test attention weights have correct shape."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # We need to modify the forward method to return attention weights
        # For now, we'll test the internal computation
        q = attention.w_q(x)
        k = attention.w_k(x)
        v = attention.w_v(x)

        # Reshape for multi-head attention
        batch_size, seq_len, d_model = q.size()
        q = q.view(
            batch_size, seq_len, num_heads, d_model // num_heads
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, num_heads, d_model // num_heads
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, num_heads, d_model // num_heads
        ).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            d_model // num_heads
        )

        assert scores.shape == (batch_size, num_heads, seq_len, seq_len)
