"""Tests for SinusoidalPositionalEncoding component."""

import pytest
import torch

from aksis.model.positional_encoding import SinusoidalPositionalEncoding


class TestSinusoidalPositionalEncoding:
    """Test cases for SinusoidalPositionalEncoding class."""

    def test_positional_encoding_initialization(self) -> None:
        """Test positional encoding initialization."""
        d_model = 512
        max_len = 1000
        dropout = 0.1

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        assert pos_enc.d_model == d_model
        assert pos_enc.max_len == max_len
        assert pos_enc.dropout.p == dropout
        assert pos_enc.pe.shape == (max_len, d_model)

    def test_positional_encoding_forward_shape(self) -> None:
        """Test positional encoding forward pass output shapes."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert isinstance(output, torch.Tensor)

    def test_positional_encoding_values(self) -> None:
        """Test that positional encoding values are correct."""
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        # Test first few positions manually
        pe = pos_enc.pe

        # Position 0: sin(0) = 0, cos(0) = 1
        assert torch.allclose(pe[0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(pe[0, 1], torch.tensor(1.0), atol=1e-6)

        # Position 1: sin(1/10000^0) = sin(1), cos(1/10000^0) = cos(1)
        assert torch.allclose(pe[1, 0], torch.sin(torch.tensor(1.0)), atol=1e-6)
        assert torch.allclose(pe[1, 1], torch.cos(torch.tensor(1.0)), atol=1e-6)

    def test_positional_encoding_sinusoidal_property(self) -> None:
        """Test that positional encoding follows sinusoidal pattern."""
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        pe = pos_enc.pe

        # Check that even dimensions use sin and odd dimensions use cos
        for pos in range(min(10, max_len)):
            for i in range(0, d_model, 2):
                if i + 1 < d_model:
                    # Even dimension should be sin
                    expected_sin = torch.sin(
                        torch.tensor(pos) / (10000 ** (i / d_model))
                    )
                    assert torch.allclose(pe[pos, i], expected_sin, atol=1e-6)

                    # Odd dimension should be cos
                    expected_cos = torch.cos(
                        torch.tensor(pos) / (10000 ** (i / d_model))
                    )
                    assert torch.allclose(pe[pos, i + 1], expected_cos, atol=1e-6)

    def test_positional_encoding_additive(self) -> None:
        """Test that positional encoding is added to input."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len, dropout=0.0
        )  # No dropout for testing
        x = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(x)

        # Output should be input + positional encoding
        expected = x + pos_enc.pe[:seq_len, :].unsqueeze(0)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_positional_encoding_dropout_training(self) -> None:
        """Test positional encoding dropout during training."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000
        dropout = 0.5

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        pos_enc.train()  # Enable dropout

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times to check for dropout variation
        outputs = []
        for _ in range(5):
            output = pos_enc(x)
            outputs.append(output)

        # Check that outputs are different (due to dropout)
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_positional_encoding_dropout_eval(self) -> None:
        """Test positional encoding without dropout during evaluation."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000
        dropout = 0.5

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        pos_enc.eval()  # Disable dropout

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times to check for consistency
        outputs = []
        for _ in range(5):
            output = pos_enc(x)
            outputs.append(output)

        # Check that outputs are identical (no dropout)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_positional_encoding_cuda_compatibility(self) -> None:
        """Test positional encoding on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len).cuda()
        x = torch.randn(batch_size, seq_len, d_model).cuda()

        output = pos_enc(x)

        assert output.device.type == "cuda"
        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding_mixed_precision(self) -> None:
        """Test positional encoding with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision")

        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len, dropout=0.0
        ).cuda()  # No dropout for mixed precision test

        with torch.amp.autocast("cuda"):
            x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float16).cuda()
            output = pos_enc(x)

        # Mixed precision may not always produce float16 due to dropout
        # Just check that it's not float32 (should be float16 or bfloat16)
        assert output.dtype in [torch.float16, torch.bfloat16]
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_positional_encoding_long_sequence(self) -> None:
        """Test positional encoding with sequence longer than max_len."""
        batch_size = 2
        seq_len = 1500  # Longer than max_len
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)

        # Should handle sequences longer than max_len
        output = pos_enc(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_positional_encoding_empty_sequence(self) -> None:
        """Test positional encoding with empty sequence."""
        batch_size = 2
        seq_len = 0
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding_single_token(self) -> None:
        """Test positional encoding with single token."""
        batch_size = 2
        seq_len = 1
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding_gradient_flow(self) -> None:
        """Test that gradients flow properly through positional encoding."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = pos_enc(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_positional_encoding_different_dimensions(self) -> None:
        """Test positional encoding with different model dimensions."""
        test_cases = [
            (256, 1000),
            (768, 2000),
            (1024, 500),
        ]

        for d_model, max_len in test_cases:
            batch_size = 2
            seq_len = 10

            pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
            x = torch.randn(batch_size, seq_len, d_model)

            output = pos_enc(x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.isnan(output).any()

    def test_positional_encoding_consistency(self) -> None:
        """Test that positional encoding is consistent across calls."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len, dropout=0.0
        )  # No dropout for consistency
        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times
        outputs = []
        for _ in range(3):
            output = pos_enc(x)
            outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_positional_encoding_parameter_count(self) -> None:
        """Test that positional encoding has correct number of parameters."""
        d_model = 512
        max_len = 1000

        pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        # Should have no trainable parameters (only dropout)
        trainable_params = sum(
            p.numel() for p in pos_enc.parameters() if p.requires_grad
        )
        assert trainable_params == 0

        # Total parameters should be 0 (no trainable parameters)
        total_params = sum(p.numel() for p in pos_enc.parameters())
        assert total_params == 0

    def test_positional_encoding_device_consistency(self) -> None:
        """Test that positional encoding maintains device consistency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        seq_len = 10
        d_model = 512
        max_len = 1000

        # Test CPU
        pos_enc_cpu = SinusoidalPositionalEncoding(d_model, max_len)
        x_cpu = torch.randn(batch_size, seq_len, d_model)
        output_cpu = pos_enc_cpu(x_cpu)
        assert output_cpu.device.type == "cpu"

        # Test CUDA
        pos_enc_cuda = SinusoidalPositionalEncoding(d_model, max_len).cuda()
        x_cuda = torch.randn(batch_size, seq_len, d_model).cuda()
        output_cuda = pos_enc_cuda(x_cuda)
        assert output_cuda.device.type == "cuda"
