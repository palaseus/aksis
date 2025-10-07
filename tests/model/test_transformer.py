"""Tests for TransformerDecoder component."""

import pytest
import torch
import torch.nn as nn

from aksis.model.transformer import TransformerDecoder
from aksis.model.attention import MultiHeadAttention
from aksis.model.positional_encoding import SinusoidalPositionalEncoding
from aksis.model.feed_forward import FeedForward


class TestTransformerDecoder:
    """Test cases for TransformerDecoder class."""

    def test_transformer_initialization(self) -> None:
        """Test transformer decoder initialization."""
        vocab_size = 10000
        d_model = 512
        num_heads = 8
        num_layers = 6
        d_ff = 2048
        max_len = 1000
        dropout = 0.1

        transformer = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
        )

        assert transformer.vocab_size == vocab_size
        assert transformer.d_model == d_model
        assert transformer.num_heads == num_heads
        assert transformer.num_layers == num_layers
        assert transformer.d_ff == d_ff
        assert transformer.max_len == max_len
        assert transformer.dropout.p == dropout

        # Check components
        assert isinstance(transformer.embedding, nn.Embedding)
        assert isinstance(transformer.pos_encoding, SinusoidalPositionalEncoding)
        assert isinstance(transformer.dropout, nn.Dropout)
        assert len(transformer.layers) == num_layers

        # Check each layer
        for layer in transformer.layers:
            assert isinstance(layer.self_attn, MultiHeadAttention)
            assert isinstance(layer.feed_forward, FeedForward)
            assert isinstance(layer.norm1, nn.LayerNorm)
            assert isinstance(layer.norm2, nn.LayerNorm)
            assert isinstance(layer.dropout, nn.Dropout)

    def test_transformer_initialization_defaults(self) -> None:
        """Test transformer decoder initialization with default parameters."""
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)

        assert transformer.vocab_size == vocab_size
        assert transformer.d_model == d_model
        assert transformer.num_heads == 8  # default
        assert transformer.num_layers == 6  # default
        assert transformer.d_ff == 4 * d_model  # default
        assert transformer.max_len == 1000  # default
        assert transformer.dropout.p == 0.1  # default

    def test_transformer_forward_shape(self) -> None:
        """Test transformer decoder forward pass output shapes."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = transformer(x)

        assert output.shape == (batch_size, seq_len, vocab_size)
        assert isinstance(output, torch.Tensor)

    def test_transformer_forward_with_mask(self) -> None:
        """Test transformer decoder forward pass with attention mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        output = transformer(x, mask=mask)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_transformer_forward_with_padding_mask(self) -> None:
        """Test transformer decoder forward pass with padding mask."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create padding mask (0s for padding tokens)
        padding_mask = torch.ones(batch_size, seq_len)
        padding_mask[:, -2:] = 0  # Last 2 tokens are padding

        output = transformer(x, padding_mask=padding_mask)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_transformer_embedding_layer(self) -> None:
        """Test transformer embedding layer properties."""
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)

        assert transformer.embedding.num_embeddings == vocab_size
        assert transformer.embedding.embedding_dim == d_model

        # Test embedding lookup
        batch_size = 2
        seq_len = 10
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        embedded = transformer.embedding(x)
        assert embedded.shape == (batch_size, seq_len, d_model)

    def test_transformer_positional_encoding(self) -> None:
        """Test transformer positional encoding integration."""
        vocab_size = 10000
        d_model = 512
        max_len = 1000

        transformer = TransformerDecoder(
            vocab_size=vocab_size, d_model=d_model, max_len=max_len
        )

        assert transformer.pos_encoding.d_model == d_model
        assert transformer.pos_encoding.max_len == max_len

        # Test positional encoding
        batch_size = 2
        seq_len = 10
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        embedded = transformer.embedding(x)
        pos_encoded = transformer.pos_encoding(embedded)

        assert pos_encoded.shape == (batch_size, seq_len, d_model)

    def test_transformer_layer_norm_properties(self) -> None:
        """Test transformer layer normalization properties."""
        vocab_size = 10000
        d_model = 512
        num_layers = 6

        transformer = TransformerDecoder(
            vocab_size=vocab_size, d_model=d_model, num_layers=num_layers
        )

        for i, layer in enumerate(transformer.layers):
            assert layer.norm1.normalized_shape == (d_model,)
            assert layer.norm2.normalized_shape == (d_model,)

    def test_transformer_dropout_training(self) -> None:
        """Test transformer dropout during training."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512
        dropout = 0.5

        transformer = TransformerDecoder(
            vocab_size=vocab_size, d_model=d_model, dropout=dropout
        )
        transformer.train()  # Enable dropout

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Run multiple times to check for dropout variation
        outputs = []
        for _ in range(5):
            output = transformer(x)
            outputs.append(output)

        # Check that outputs are different (due to dropout)
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_transformer_dropout_eval(self) -> None:
        """Test transformer without dropout during evaluation."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512
        dropout = 0.5

        transformer = TransformerDecoder(
            vocab_size=vocab_size, d_model=d_model, dropout=dropout
        )
        transformer.eval()  # Disable dropout

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Run multiple times to check for consistency
        outputs = []
        for _ in range(5):
            output = transformer(x)
            outputs.append(output)

        # Check that outputs are identical (no dropout)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_transformer_cuda_compatibility(self) -> None:
        """Test transformer on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model).cuda()
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()

        output = transformer(x)

        assert output.device.type == "cuda"
        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_transformer_mixed_precision(self) -> None:
        """Test transformer with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision")

        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model).cuda()
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()

        with torch.amp.autocast("cuda"):
            output = transformer(x)

        assert output.dtype == torch.float16
        assert output.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(output).any()

    def test_transformer_empty_sequence(self) -> None:
        """Test transformer with empty sequence."""
        batch_size = 2
        seq_len = 0
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = transformer(x)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_transformer_single_token(self) -> None:
        """Test transformer with single token."""
        batch_size = 2
        seq_len = 1
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = transformer(x)

        assert output.shape == (batch_size, seq_len, vocab_size)

    def test_transformer_large_sequence(self) -> None:
        """Test transformer with large sequence length."""
        batch_size = 1
        seq_len = 1024
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = transformer(x)

        assert output.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(output).any()

    def test_transformer_gradient_flow(self) -> None:
        """Test that gradients flow properly through transformer."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = transformer(x)
        loss = output.sum()
        loss.backward()

        # Check that all parameters have gradients
        for param in transformer.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_transformer_parameter_count(self) -> None:
        """Test that transformer has correct number of parameters."""
        vocab_size = 10000
        d_model = 512
        num_heads = 8
        num_layers = 6
        d_ff = 2048

        transformer = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
        )

        # Count parameters manually
        total_params = sum(p.numel() for p in transformer.parameters())

        # Expected parameters:
        # - Embedding: vocab_size * d_model
        # - Per layer: 4 * d_model * d_model (attention) + 2 * d_model * d_ff + d_ff * d_model (ffn) + 2 * d_model (layer norms)
        # - Output projection: d_model * vocab_size
        # - No parameters for positional encoding and dropout

        expected_embedding = vocab_size * d_model
        expected_per_layer = (
            4 * d_model * d_model  # attention (no bias)
            + 2 * d_model * d_ff
            + d_ff
            + d_model  # ffn linear layers (with bias)
            + 2 * d_model
            + 2 * d_model  # layer norms (weight + bias)
        )
        expected_output_proj = d_model * vocab_size + vocab_size  # weight + bias
        expected_final_norm = d_model + d_model  # weight + bias
        expected_total = (
            expected_embedding
            + num_layers * expected_per_layer
            + expected_output_proj
            + expected_final_norm
        )

        assert total_params == expected_total

    def test_transformer_different_configurations(self) -> None:
        """Test transformer with different configurations."""
        test_cases = [
            (1000, 256, 4, 2, 1024),
            (5000, 768, 12, 4, 3072),
            (20000, 1024, 16, 8, 4096),
        ]

        for vocab_size, d_model, num_heads, num_layers, d_ff in test_cases:
            batch_size = 2
            seq_len = 10

            transformer = TransformerDecoder(
                vocab_size=vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
            )
            x = torch.randint(0, vocab_size, (batch_size, seq_len))

            output = transformer(x)

            assert output.shape == (batch_size, seq_len, vocab_size)
            assert not torch.isnan(output).any()

    def test_transformer_invalid_vocab_size(self) -> None:
        """Test transformer with invalid vocabulary size."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            TransformerDecoder(vocab_size=0, d_model=512)

    def test_transformer_invalid_d_model(self) -> None:
        """Test transformer with invalid model dimension."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            TransformerDecoder(vocab_size=10000, d_model=0)

    def test_transformer_invalid_num_heads(self) -> None:
        """Test transformer with invalid number of heads."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            TransformerDecoder(vocab_size=10000, d_model=512, num_heads=0)

    def test_transformer_invalid_num_layers(self) -> None:
        """Test transformer with invalid number of layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            TransformerDecoder(vocab_size=10000, d_model=512, num_layers=0)

    def test_transformer_consistency(self) -> None:
        """Test that transformer is consistent across calls."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512

        transformer = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        transformer.eval()  # Disable dropout for consistency
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Run multiple times
        outputs = []
        for _ in range(3):
            output = transformer(x)
            outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_transformer_device_consistency(self) -> None:
        """Test that transformer maintains device consistency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        seq_len = 10
        vocab_size = 10000
        d_model = 512

        # Test CPU
        transformer_cpu = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)
        x_cpu = torch.randint(0, vocab_size, (batch_size, seq_len))
        output_cpu = transformer_cpu(x_cpu)
        assert output_cpu.device.type == "cpu"

        # Test CUDA
        transformer_cuda = TransformerDecoder(
            vocab_size=vocab_size, d_model=d_model
        ).cuda()
        x_cuda = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
        output_cuda = transformer_cuda(x_cuda)
        assert output_cuda.device.type == "cuda"
