"""Tests for FeedForward component."""

import pytest
import torch
import torch.nn as nn

from aksis.model.feed_forward import FeedForward


class TestFeedForward:
    """Test cases for FeedForward class."""

    def test_feed_forward_initialization(self) -> None:
        """Test feed forward network initialization."""
        d_model = 512
        d_ff = 2048
        dropout = 0.1

        ff = FeedForward(d_model, d_ff, dropout)

        assert ff.d_model == d_model
        assert ff.d_ff == d_ff
        assert ff.dropout.p == dropout
        assert isinstance(ff.linear1, nn.Linear)
        assert isinstance(ff.linear2, nn.Linear)
        assert isinstance(ff.dropout, nn.Dropout)
        assert isinstance(ff.activation, nn.ReLU)

    def test_feed_forward_initialization_default_d_ff(self) -> None:
        """Test feed forward network with default d_ff (4 * d_model)."""
        d_model = 512
        dropout = 0.1

        ff = FeedForward(d_model, dropout=dropout)

        assert ff.d_model == d_model
        assert ff.d_ff == 4 * d_model
        assert ff.dropout.p == dropout

    def test_feed_forward_forward_shape(self) -> None:
        """Test feed forward network forward pass output shapes."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert isinstance(output, torch.Tensor)

    def test_feed_forward_forward_process(self) -> None:
        """Test feed forward network forward process step by step."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff, dropout=0.0)  # No dropout for testing
        x = torch.randn(batch_size, seq_len, d_model)

        # Manual forward pass
        linear1_out = ff.linear1(x)
        assert linear1_out.shape == (batch_size, seq_len, d_ff)

        relu_out = ff.activation(linear1_out)
        assert relu_out.shape == (batch_size, seq_len, d_ff)
        assert (relu_out >= 0).all()  # ReLU should be non-negative

        dropout_out = ff.dropout(relu_out)
        assert dropout_out.shape == (batch_size, seq_len, d_ff)

        linear2_out = ff.linear2(dropout_out)
        assert linear2_out.shape == (batch_size, seq_len, d_model)

        # Compare with full forward pass
        full_output = ff(x)
        assert torch.allclose(full_output, linear2_out, atol=1e-6)

    def test_feed_forward_dropout_training(self) -> None:
        """Test feed forward network dropout during training."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048
        dropout = 0.5

        ff = FeedForward(d_model, d_ff, dropout)
        ff.train()  # Enable dropout

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times to check for dropout variation
        outputs = []
        for _ in range(5):
            output = ff(x)
            outputs.append(output)

        # Check that outputs are different (due to dropout)
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_feed_forward_dropout_eval(self) -> None:
        """Test feed forward network without dropout during evaluation."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048
        dropout = 0.5

        ff = FeedForward(d_model, d_ff, dropout)
        ff.eval()  # Disable dropout

        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times to check for consistency
        outputs = []
        for _ in range(5):
            output = ff(x)
            outputs.append(output)

        # Check that outputs are identical (no dropout)
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_feed_forward_cuda_compatibility(self) -> None:
        """Test feed forward network on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff).cuda()
        x = torch.randn(batch_size, seq_len, d_model).cuda()

        output = ff(x)

        assert output.device.type == "cuda"
        assert output.shape == (batch_size, seq_len, d_model)

    def test_feed_forward_mixed_precision(self) -> None:
        """Test feed forward network with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision")

        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff).cuda()
        x = torch.randn(batch_size, seq_len, d_model).cuda()

        with torch.amp.autocast("cuda"):
            output = ff(x)

        assert output.dtype == torch.float16
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_feed_forward_empty_sequence(self) -> None:
        """Test feed forward network with empty sequence."""
        batch_size = 2
        seq_len = 0
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_feed_forward_single_token(self) -> None:
        """Test feed forward network with single token."""
        batch_size = 2
        seq_len = 1
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_feed_forward_large_sequence(self) -> None:
        """Test feed forward network with large sequence length."""
        batch_size = 1
        seq_len = 1024
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_feed_forward_gradient_flow(self) -> None:
        """Test that gradients flow properly through feed forward network."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = ff(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that all parameters have gradients
        for param in ff.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_feed_forward_parameter_count(self) -> None:
        """Test that feed forward network has correct number of parameters."""
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)

        # Expected parameters: linear1 (d_model * d_ff + d_ff) + linear2 (d_ff * d_model + d_model)
        expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
        actual_params = sum(p.numel() for p in ff.parameters())

        assert actual_params == expected_params

    def test_feed_forward_different_dimensions(self) -> None:
        """Test feed forward network with different dimensions."""
        test_cases = [
            (256, 1024),
            (768, 3072),
            (1024, 4096),
        ]

        for d_model, d_ff in test_cases:
            batch_size = 2
            seq_len = 10

            ff = FeedForward(d_model, d_ff)
            x = torch.randn(batch_size, seq_len, d_model)

            output = ff(x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert not torch.isnan(output).any()

    def test_feed_forward_activation_function(self) -> None:
        """Test that ReLU activation is applied correctly."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)

        # Create input that will produce negative values after linear1
        x_negative = torch.randn(batch_size, seq_len, d_model) - 2.0

        output = ff(x_negative)

        # The output should not be NaN even with negative inputs
        assert not torch.isnan(output).any()
        assert output.shape == (batch_size, seq_len, d_model)

    def test_feed_forward_zero_input(self) -> None:
        """Test feed forward network with zero input."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)
        x = torch.zeros(batch_size, seq_len, d_model)

        output = ff(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_feed_forward_consistency(self) -> None:
        """Test that feed forward network is consistent across calls."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)
        ff.eval()  # Disable dropout for consistency
        x = torch.randn(batch_size, seq_len, d_model)

        # Run multiple times
        outputs = []
        for _ in range(3):
            output = ff(x)
            outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_feed_forward_device_consistency(self) -> None:
        """Test that feed forward network maintains device consistency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        # Test CPU
        ff_cpu = FeedForward(d_model, d_ff)
        x_cpu = torch.randn(batch_size, seq_len, d_model)
        output_cpu = ff_cpu(x_cpu)
        assert output_cpu.device.type == "cpu"

        # Test CUDA
        ff_cuda = FeedForward(d_model, d_ff).cuda()
        x_cuda = torch.randn(batch_size, seq_len, d_model).cuda()
        output_cuda = ff_cuda(x_cuda)
        assert output_cuda.device.type == "cuda"

    def test_feed_forward_linear_layer_properties(self) -> None:
        """Test properties of linear layers in feed forward network."""
        d_model = 512
        d_ff = 2048

        ff = FeedForward(d_model, d_ff)

        # Check linear1 properties
        assert ff.linear1.in_features == d_model
        assert ff.linear1.out_features == d_ff

        # Check linear2 properties
        assert ff.linear2.in_features == d_ff
        assert ff.linear2.out_features == d_model
