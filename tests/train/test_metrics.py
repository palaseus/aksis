"""Tests for training metrics."""

import pytest
import torch
import math

from aksis.train.metrics import MetricsTracker, compute_perplexity


class TestComputePerplexity:
    """Test perplexity computation."""

    def test_perplexity_basic(self):
        """Test basic perplexity computation."""
        # Create logits and targets
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        perplexity = compute_perplexity(logits, targets)

        assert isinstance(perplexity, float)
        assert perplexity > 0
        assert not math.isnan(perplexity)
        assert not math.isinf(perplexity)

    def test_perplexity_with_mask(self):
        """Test perplexity computation with attention mask."""
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len)
        mask[:, -2:] = 0  # Last 2 tokens are padding

        perplexity = compute_perplexity(logits, targets, mask)

        assert isinstance(perplexity, float)
        assert perplexity > 0
        assert not math.isnan(perplexity)
        assert not math.isinf(perplexity)

    def test_perplexity_single_token(self):
        """Test perplexity computation with single token."""
        batch_size, seq_len, vocab_size = 1, 1, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        perplexity = compute_perplexity(logits, targets)

        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_perplexity_cuda(self):
        """Test perplexity computation on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size).cuda()
        targets = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()

        perplexity = compute_perplexity(logits, targets)

        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_perplexity_mixed_precision(self):
        """Test perplexity computation with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision")

        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size).cuda()
        targets = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()

        with torch.amp.autocast("cuda"):
            perplexity = compute_perplexity(logits, targets)

        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_perplexity_edge_cases(self):
        """Test perplexity computation edge cases."""
        # Test with very small logits
        batch_size, seq_len, vocab_size = 1, 1, 10
        logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
        targets = torch.tensor([[0]])

        perplexity = compute_perplexity(logits, targets)
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_perplexity_invalid_inputs(self):
        """Test perplexity computation with invalid inputs."""
        batch_size, seq_len, vocab_size = 2, 10, 1000

        # Test with mismatched shapes
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len + 1))

        with pytest.raises(RuntimeError):
            compute_perplexity(logits, targets)


class TestMetricsTracker:
    """Test metrics tracking."""

    def test_metrics_tracker_initialization(self):
        """Test metrics tracker initialization."""
        tracker = MetricsTracker()

        assert tracker.train_losses == []
        assert tracker.val_losses == []
        assert tracker.train_perplexities == []
        assert tracker.val_perplexities == []
        assert tracker.epoch_times == []

    def test_metrics_tracker_update_train(self):
        """Test updating training metrics."""
        tracker = MetricsTracker()

        # Update training metrics
        tracker.update_train(loss=1.5, perplexity=4.5, epoch_time=120.0)

        assert len(tracker.train_losses) == 1
        assert len(tracker.train_perplexities) == 1
        assert len(tracker.epoch_times) == 1
        assert tracker.train_losses[0] == 1.5
        assert tracker.train_perplexities[0] == 4.5
        assert tracker.epoch_times[0] == 120.0

    def test_metrics_tracker_update_val(self):
        """Test updating validation metrics."""
        tracker = MetricsTracker()

        # Update validation metrics
        tracker.update_val(loss=1.2, perplexity=3.3)

        assert len(tracker.val_losses) == 1
        assert len(tracker.val_perplexities) == 1
        assert tracker.val_losses[0] == 1.2
        assert tracker.val_perplexities[0] == 3.3

    def test_metrics_tracker_multiple_updates(self):
        """Test multiple metric updates."""
        tracker = MetricsTracker()

        # Update multiple times
        for i in range(3):
            tracker.update_train(
                loss=1.0 + i, perplexity=2.0 + i, epoch_time=100.0 + i
            )
            tracker.update_val(loss=0.8 + i, perplexity=1.5 + i)

        assert len(tracker.train_losses) == 3
        assert len(tracker.val_losses) == 3
        assert len(tracker.train_perplexities) == 3
        assert len(tracker.val_perplexities) == 3
        assert len(tracker.epoch_times) == 3

    def test_metrics_tracker_get_latest(self):
        """Test getting latest metrics."""
        tracker = MetricsTracker()

        # Update metrics
        tracker.update_train(loss=1.5, perplexity=4.5, epoch_time=120.0)
        tracker.update_val(loss=1.2, perplexity=3.3)

        latest = tracker.get_latest()

        assert latest["train_loss"] == 1.5
        assert latest["train_perplexity"] == 4.5
        assert latest["val_loss"] == 1.2
        assert latest["val_perplexity"] == 3.3
        assert latest["epoch_time"] == 120.0

    def test_metrics_tracker_get_latest_empty(self):
        """Test getting latest metrics when empty."""
        tracker = MetricsTracker()

        latest = tracker.get_latest()

        assert latest["train_loss"] is None
        assert latest["train_perplexity"] is None
        assert latest["val_loss"] is None
        assert latest["val_perplexity"] is None
        assert latest["epoch_time"] is None

    def test_metrics_tracker_get_averages(self):
        """Test getting average metrics."""
        tracker = MetricsTracker()

        # Update metrics multiple times
        tracker.update_train(loss=1.0, perplexity=2.0, epoch_time=100.0)
        tracker.update_train(loss=2.0, perplexity=4.0, epoch_time=110.0)
        tracker.update_train(loss=3.0, perplexity=6.0, epoch_time=120.0)

        tracker.update_val(loss=0.8, perplexity=1.6)
        tracker.update_val(loss=1.2, perplexity=2.4)

        averages = tracker.get_averages()

        assert averages["avg_train_loss"] == 2.0
        assert averages["avg_train_perplexity"] == 4.0
        assert averages["avg_val_loss"] == 1.0
        assert averages["avg_val_perplexity"] == 2.0
        assert averages["avg_epoch_time"] == 110.0

    def test_metrics_tracker_get_averages_empty(self):
        """Test getting average metrics when empty."""
        tracker = MetricsTracker()

        averages = tracker.get_averages()

        assert averages["avg_train_loss"] is None
        assert averages["avg_train_perplexity"] is None
        assert averages["avg_val_loss"] is None
        assert averages["avg_val_perplexity"] is None
        assert averages["avg_epoch_time"] is None

    def test_metrics_tracker_reset(self):
        """Test resetting metrics tracker."""
        tracker = MetricsTracker()

        # Add some metrics
        tracker.update_train(loss=1.5, perplexity=4.5, epoch_time=120.0)
        tracker.update_val(loss=1.2, perplexity=3.3)

        # Reset
        tracker.reset()

        assert tracker.train_losses == []
        assert tracker.val_losses == []
        assert tracker.train_perplexities == []
        assert tracker.val_perplexities == []
        assert tracker.epoch_times == []

    def test_metrics_tracker_to_dict(self):
        """Test converting metrics to dictionary."""
        tracker = MetricsTracker()

        # Add some metrics
        tracker.update_train(loss=1.5, perplexity=4.5, epoch_time=120.0)
        tracker.update_val(loss=1.2, perplexity=3.3)

        metrics_dict = tracker.to_dict()

        assert "train_losses" in metrics_dict
        assert "val_losses" in metrics_dict
        assert "train_perplexities" in metrics_dict
        assert "val_perplexities" in metrics_dict
        assert "epoch_times" in metrics_dict
        assert metrics_dict["train_losses"] == [1.5]
        assert metrics_dict["val_losses"] == [1.2]

    def test_metrics_tracker_from_dict(self):
        """Test creating metrics tracker from dictionary."""
        metrics_dict = {
            "train_losses": [1.0, 2.0],
            "val_losses": [0.8, 1.2],
            "train_perplexities": [2.0, 4.0],
            "val_perplexities": [1.6, 2.4],
            "epoch_times": [100.0, 110.0],
        }

        tracker = MetricsTracker.from_dict(metrics_dict)

        assert tracker.train_losses == [1.0, 2.0]
        assert tracker.val_losses == [0.8, 1.2]
        assert tracker.train_perplexities == [2.0, 4.0]
        assert tracker.val_perplexities == [1.6, 2.4]
        assert tracker.epoch_times == [100.0, 110.0]

    def test_metrics_tracker_invalid_values(self):
        """Test metrics tracker with invalid values."""
        tracker = MetricsTracker()

        # Test with NaN values
        with pytest.raises(ValueError, match="Loss cannot be NaN"):
            tracker.update_train(
                loss=float("nan"), perplexity=4.5, epoch_time=120.0
            )

        with pytest.raises(ValueError, match="Perplexity cannot be NaN"):
            tracker.update_train(
                loss=1.5, perplexity=float("nan"), epoch_time=120.0
            )

        # Test with negative values
        with pytest.raises(ValueError, match="Loss must be non-negative"):
            tracker.update_train(loss=-1.0, perplexity=4.5, epoch_time=120.0)

        with pytest.raises(ValueError, match="Perplexity must be positive"):
            tracker.update_train(loss=1.5, perplexity=0.0, epoch_time=120.0)
