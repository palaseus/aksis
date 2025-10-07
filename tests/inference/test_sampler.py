"""Tests for sampling methods."""

import pytest
import torch
from unittest.mock import patch

from aksis.inference.sampler import (
    GreedySampler,
    BeamSearchSampler,
    TopKSampler,
    TopPSampler,
    TemperatureSampler,
)


class TestGreedySampler:
    """Test greedy sampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = GreedySampler()

    def test_greedy_sampler_initialization(self):
        """Test greedy sampler initialization."""
        assert isinstance(self.sampler, GreedySampler)

    def test_greedy_sample_basic(self):
        """Test basic greedy sampling."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = self.sampler.sample(logits)

        assert result.shape == (1, 1)
        assert result.item() == 3  # argmax of [1, 2, 3, 4]

    def test_greedy_sample_batch(self):
        """Test greedy sampling with batch."""
        logits = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        result = self.sampler.sample(logits)

        assert result.shape == (2, 1)
        assert result[0].item() == 3
        assert result[1].item() == 0

    def test_greedy_sample_invalid_logits(self):
        """Test greedy sampling with invalid logits."""
        with pytest.raises(ValueError, match="logits must be a 2D tensor"):
            self.sampler.sample(torch.tensor([1.0, 2.0, 3.0]))

    def test_greedy_sample_empty_logits(self):
        """Test greedy sampling with empty logits."""
        logits = torch.tensor([]).reshape(0, 0)

        with pytest.raises(ValueError, match="logits cannot be empty"):
            self.sampler.sample(logits)

    def test_greedy_sample_cuda(self):
        """Test greedy sampling on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).cuda()
        result = self.sampler.sample(logits)

        assert result.device.type == "cuda"
        assert result.item() == 3


class TestBeamSearchSampler:
    """Test beam search sampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = BeamSearchSampler(beam_width=4)

    def test_beam_search_initialization(self):
        """Test beam search sampler initialization."""
        assert self.sampler.beam_width == 4

    def test_beam_search_initialization_invalid_width(self):
        """Test beam search sampler with invalid beam width."""
        with pytest.raises(ValueError, match="beam_width must be positive"):
            BeamSearchSampler(beam_width=0)

    def test_beam_search_sample_basic(self):
        """Test basic beam search sampling."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = self.sampler.sample(logits)

        assert result.shape == (1, 1)
        assert result.item() == 3

    def test_beam_search_sample_batch(self):
        """Test beam search sampling with batch."""
        logits = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        result = self.sampler.sample(logits)

        assert result.shape == (2, 1)
        assert result[0].item() == 3
        assert result[1].item() == 0

    def test_beam_search_sample_multiple_steps(self):
        """Test beam search sampling over multiple steps."""
        # Mock the beam search process
        with patch.object(self.sampler, "_beam_search") as mock_beam_search:
            mock_beam_search.return_value = torch.tensor([[3, 2, 1]])

            logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            result = self.sampler.sample(logits, num_steps=3)

            assert result.shape == (1, 3)
            mock_beam_search.assert_called_once()

    def test_beam_search_invalid_logits(self):
        """Test beam search with invalid logits."""
        with pytest.raises(ValueError, match="logits must be a 2D tensor"):
            self.sampler.sample(torch.tensor([1.0, 2.0, 3.0]))


class TestTopKSampler:
    """Test top-k sampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = TopKSampler(k=2)

    def test_top_k_initialization(self):
        """Test top-k sampler initialization."""
        assert self.sampler.k == 2

    def test_top_k_initialization_invalid_k(self):
        """Test top-k sampler with invalid k."""
        with pytest.raises(ValueError, match="k must be positive"):
            TopKSampler(k=0)

    def test_top_k_sample_basic(self):
        """Test basic top-k sampling."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = self.sampler.sample(logits)

        assert result.shape == (1, 1)
        # Should be one of the top 2 tokens (indices 2 or 3)
        assert result.item() in [2, 3]

    def test_top_k_sample_batch(self):
        """Test top-k sampling with batch."""
        logits = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        result = self.sampler.sample(logits)

        assert result.shape == (2, 1)
        # First batch: top 2 are indices 2, 3
        assert result[0].item() in [2, 3]
        # Second batch: top 2 are indices 0, 1
        assert result[1].item() in [0, 1]

    def test_top_k_sample_k_larger_than_vocab(self):
        """Test top-k sampling when k is larger than vocabulary size."""
        sampler = TopKSampler(k=10)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = sampler.sample(logits)

        assert result.shape == (1, 1)
        assert result.item() in [0, 1, 2, 3]

    def test_top_k_sample_invalid_logits(self):
        """Test top-k sampling with invalid logits."""
        with pytest.raises(ValueError, match="logits must be a 2D tensor"):
            self.sampler.sample(torch.tensor([1.0, 2.0, 3.0]))


class TestTopPSampler:
    """Test top-p (nucleus) sampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = TopPSampler(p=0.8)

    def test_top_p_initialization(self):
        """Test top-p sampler initialization."""
        assert self.sampler.p == 0.8

    def test_top_p_initialization_invalid_p(self):
        """Test top-p sampler with invalid p."""
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            TopPSampler(p=1.5)

    def test_top_p_sample_basic(self):
        """Test basic top-p sampling."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = self.sampler.sample(logits)

        assert result.shape == (1, 1)
        assert result.item() in [0, 1, 2, 3]

    def test_top_p_sample_batch(self):
        """Test top-p sampling with batch."""
        logits = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        result = self.sampler.sample(logits)

        assert result.shape == (2, 1)
        assert result[0].item() in [0, 1, 2, 3]
        assert result[1].item() in [0, 1, 2, 3]

    def test_top_p_sample_p_zero(self):
        """Test top-p sampling with p=0 (greedy)."""
        sampler = TopPSampler(p=0.0)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = sampler.sample(logits)

        assert result.shape == (1, 1)
        assert result.item() == 3  # argmax

    def test_top_p_sample_p_one(self):
        """Test top-p sampling with p=1 (uniform)."""
        sampler = TopPSampler(p=1.0)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = sampler.sample(logits)

        assert result.shape == (1, 1)
        assert result.item() in [0, 1, 2, 3]

    def test_top_p_sample_invalid_logits(self):
        """Test top-p sampling with invalid logits."""
        with pytest.raises(ValueError, match="logits must be a 2D tensor"):
            self.sampler.sample(torch.tensor([1.0, 2.0, 3.0]))


class TestTemperatureSampler:
    """Test temperature sampling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = TemperatureSampler(temperature=0.7)

    def test_temperature_initialization(self):
        """Test temperature sampler initialization."""
        assert self.sampler.temperature == 0.7

    def test_temperature_initialization_invalid_temp(self):
        """Test temperature sampler with invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            TemperatureSampler(temperature=-1.0)

    def test_temperature_sample_basic(self):
        """Test basic temperature sampling."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = self.sampler.sample(logits)

        assert result.shape == (1, 1)
        assert result.item() in [0, 1, 2, 3]

    def test_temperature_sample_batch(self):
        """Test temperature sampling with batch."""
        logits = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        result = self.sampler.sample(logits)

        assert result.shape == (2, 1)
        assert result[0].item() in [0, 1, 2, 3]
        assert result[1].item() in [0, 1, 2, 3]

    def test_temperature_sample_temp_zero(self):
        """Test temperature sampling with low temperature (near greedy)."""
        sampler = TemperatureSampler(temperature=0.001)  # Very low temp
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = sampler.sample(logits)

        assert result.shape == (1, 1)
        # With very low temperature, should be very close to argmax
        assert result.item() == 3  # argmax

    def test_temperature_sample_temp_high(self):
        """Test temperature sampling with high temperature."""
        sampler = TemperatureSampler(temperature=10.0)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = sampler.sample(logits)

        assert result.shape == (1, 1)
        assert result.item() in [0, 1, 2, 3]

    def test_temperature_sample_invalid_logits(self):
        """Test temperature sampling with invalid logits."""
        with pytest.raises(ValueError, match="logits must be a 2D tensor"):
            self.sampler.sample(torch.tensor([1.0, 2.0, 3.0]))


class TestSamplerIntegration:
    """Test integration between different samplers."""

    def test_sampler_consistency(self):
        """Test that samplers produce consistent results with same seed."""
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        # Test that greedy sampler is deterministic
        greedy = GreedySampler()
        result1 = greedy.sample(logits)
        result2 = greedy.sample(logits)
        assert torch.equal(result1, result2)

    def test_sampler_diversity(self):
        """Test that different samplers produce different results."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        greedy = GreedySampler()
        top_k = TopKSampler(k=2)
        top_p = TopPSampler(p=0.8)
        temperature = TemperatureSampler(temperature=0.7)

        results = []
        for _ in range(10):
            results.append(greedy.sample(logits).item())
            results.append(top_k.sample(logits).item())
            results.append(top_p.sample(logits).item())
            results.append(temperature.sample(logits).item())

        # Greedy should always be the same
        greedy_results = results[::4]
        assert all(r == greedy_results[0] for r in greedy_results)

        # Other samplers should have some diversity
        top_k_results = results[1::4]
        top_p_results = results[2::4]
        temperature_results = results[3::4]

        # At least one should have diversity (not all the same)
        assert (
            len(set(top_k_results)) > 1
            or len(set(top_p_results)) > 1
            or len(set(temperature_results)) > 1
        )

    def test_sampler_cuda_compatibility(self):
        """Test that samplers work on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).cuda()

        samplers = [
            GreedySampler(),
            TopKSampler(k=2),
            TopPSampler(p=0.8),
            TemperatureSampler(temperature=0.7),
        ]

        for sampler in samplers:
            result = sampler.sample(logits)
            assert result.device.type == "cuda"
            assert result.shape == (1, 1)

    def test_sampler_edge_cases(self):
        """Test samplers with edge cases."""
        # Single token vocabulary
        logits = torch.tensor([[5.0]])

        samplers = [
            GreedySampler(),
            TopKSampler(k=1),
            TopPSampler(p=0.5),
            TemperatureSampler(temperature=0.7),
        ]

        for sampler in samplers:
            result = sampler.sample(logits)
            assert result.shape == (1, 1)
            assert result.item() == 0

    def test_sampler_numerical_stability(self):
        """Test samplers with extreme logit values."""
        # Very large logits
        logits = torch.tensor([[1000.0, 1001.0, 1002.0, 1003.0]])

        samplers = [
            GreedySampler(),
            TopKSampler(k=2),
            TopPSampler(p=0.8),
            TemperatureSampler(temperature=0.7),
        ]

        for sampler in samplers:
            result = sampler.sample(logits)
            assert result.shape == (1, 1)
            assert result.item() in [0, 1, 2, 3]
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()

        # Very small logits
        logits = torch.tensor([[-1000.0, -1001.0, -1002.0, -1003.0]])

        for sampler in samplers:
            result = sampler.sample(logits)
            assert result.shape == (1, 1)
            assert result.item() in [0, 1, 2, 3]
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()
