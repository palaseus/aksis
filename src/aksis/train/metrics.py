"""Training metrics and evaluation utilities."""

import math
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from logits and targets.

    Args:
        logits: Model output logits of shape [batch_size, seq_len, vocab_size].
        targets: Target token IDs of shape [batch_size, seq_len].
        mask: Optional attention mask of shape [batch_size, seq_len].
        ignore_index: Index to ignore in loss computation.

    Returns:
        Perplexity value.

    Raises:
        RuntimeError: If shapes don't match or computation fails.
    """
    if logits.dim() != 3 or targets.dim() != 2:
        raise RuntimeError(
            f"Expected logits shape [batch_size, seq_len, vocab_size] and "
            f"targets shape [batch_size, seq_len], got {logits.shape} and "
            f"{targets.shape}"
        )

    if logits.size(0) != targets.size(0) or logits.size(1) != targets.size(1):
        raise RuntimeError(
            f"Batch size and sequence length must match between logits "
            f"{logits.shape} and targets {targets.shape}"
        )

    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )

    # Apply mask if provided
    if mask is not None:
        if mask.shape != targets.shape:
            raise RuntimeError(
                f"Mask shape {mask.shape} must match targets shape "
                f"{targets.shape}"
            )
        # Reshape loss to match mask
        loss = loss.view(targets.shape)
        # Apply mask (1 for valid tokens, 0 for padding)
        loss = loss * mask
        # Count valid tokens
        valid_tokens = mask.sum().item()
    else:
        # Count non-ignored tokens
        valid_tokens = (targets != ignore_index).sum().item()

    if valid_tokens == 0:
        logger.warning("No valid tokens found for perplexity computation")
        return float("inf")

    # Compute average loss
    avg_loss = loss.sum().item() / valid_tokens

    # Compute perplexity
    perplexity = math.exp(avg_loss)

    # Check for numerical issues
    if math.isnan(perplexity) or math.isinf(perplexity):
        logger.warning(f"Invalid perplexity computed: {perplexity}")
        return float("inf")

    return perplexity


class MetricsTracker:
    """Track training and validation metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_perplexities: List[float] = []
        self.val_perplexities: List[float] = []
        self.epoch_times: List[float] = []

    def update_train(
        self, loss: float, perplexity: float, epoch_time: float
    ) -> None:
        """
        Update training metrics.

        Args:
            loss: Training loss.
            perplexity: Training perplexity.
            epoch_time: Time taken for the epoch in seconds.

        Raises:
            ValueError: If any metric is invalid.
        """
        if math.isnan(loss) or math.isinf(loss):
            raise ValueError("Loss cannot be NaN or infinite")
        if loss < 0:
            raise ValueError("Loss must be non-negative")

        if math.isnan(perplexity) or math.isinf(perplexity):
            raise ValueError("Perplexity cannot be NaN or infinite")
        if perplexity <= 0:
            raise ValueError("Perplexity must be positive")

        if math.isnan(epoch_time) or math.isinf(epoch_time):
            raise ValueError("Epoch time cannot be NaN or infinite")
        if epoch_time < 0:
            raise ValueError("Epoch time must be non-negative")

        self.train_losses.append(loss)
        self.train_perplexities.append(perplexity)
        self.epoch_times.append(epoch_time)

        logger.debug(
            f"Updated training metrics: loss={loss:.4f}, "
            f"perplexity={perplexity:.4f}"
        )

    def update_val(self, loss: float, perplexity: float) -> None:
        """
        Update validation metrics.

        Args:
            loss: Validation loss.
            perplexity: Validation perplexity.

        Raises:
            ValueError: If any metric is invalid.
        """
        if math.isnan(loss) or math.isinf(loss):
            raise ValueError("Loss cannot be NaN or infinite")
        if loss < 0:
            raise ValueError("Loss must be non-negative")

        if math.isnan(perplexity) or math.isinf(perplexity):
            raise ValueError("Perplexity cannot be NaN or infinite")
        if perplexity <= 0:
            raise ValueError("Perplexity must be positive")

        self.val_losses.append(loss)
        self.val_perplexities.append(perplexity)

        logger.debug(
            f"Updated validation metrics: loss={loss:.4f}, "
            f"perplexity={perplexity:.4f}"
        )

    def get_latest(self) -> Dict[str, Optional[float]]:
        """
        Get the latest metrics.

        Returns:
            Dictionary containing the latest metrics.
        """
        return {
            "train_loss": self.train_losses[-1] if self.train_losses else None,
            "train_perplexity": (
                self.train_perplexities[-1]
                if self.train_perplexities
                else None
            ),
            "val_loss": self.val_losses[-1] if self.val_losses else None,
            "val_perplexity": (
                self.val_perplexities[-1] if self.val_perplexities else None
            ),
            "epoch_time": self.epoch_times[-1] if self.epoch_times else None,
        }

    def get_averages(self) -> Dict[str, Optional[float]]:
        """
        Get average metrics across all epochs.

        Returns:
            Dictionary containing average metrics.
        """
        return {
            "avg_train_loss": (
                sum(self.train_losses) / len(self.train_losses)
                if self.train_losses
                else None
            ),
            "avg_train_perplexity": (
                sum(self.train_perplexities) / len(self.train_perplexities)
                if self.train_perplexities
                else None
            ),
            "avg_val_loss": (
                sum(self.val_losses) / len(self.val_losses)
                if self.val_losses
                else None
            ),
            "avg_val_perplexity": (
                sum(self.val_perplexities) / len(self.val_perplexities)
                if self.val_perplexities
                else None
            ),
            "avg_epoch_time": (
                sum(self.epoch_times) / len(self.epoch_times)
                if self.epoch_times
                else None
            ),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_perplexities.clear()
        self.val_perplexities.clear()
        self.epoch_times.clear()

        logger.debug("Reset all metrics")

    def to_dict(self) -> Dict[str, List[float]]:
        """
        Convert metrics to dictionary.

        Returns:
            Dictionary containing all metrics.
        """
        return {
            "train_losses": self.train_losses.copy(),
            "val_losses": self.val_losses.copy(),
            "train_perplexities": self.train_perplexities.copy(),
            "val_perplexities": self.val_perplexities.copy(),
            "epoch_times": self.epoch_times.copy(),
        }

    @classmethod
    def from_dict(
        cls, metrics_dict: Dict[str, List[float]]
    ) -> "MetricsTracker":
        """
        Create metrics tracker from dictionary.

        Args:
            metrics_dict: Dictionary containing metrics.

        Returns:
            MetricsTracker instance.
        """
        tracker = cls()
        tracker.train_losses = metrics_dict.get("train_losses", []).copy()
        tracker.val_losses = metrics_dict.get("val_losses", []).copy()
        tracker.train_perplexities = metrics_dict.get(
            "train_perplexities", []
        ).copy()
        tracker.val_perplexities = metrics_dict.get(
            "val_perplexities", []
        ).copy()
        tracker.epoch_times = metrics_dict.get("epoch_times", []).copy()

        return tracker

    def __len__(self) -> int:
        """Return number of recorded epochs."""
        return len(self.train_losses)

    def __repr__(self) -> str:
        """String representation of metrics tracker."""
        latest = self.get_latest()
        return (
            f"MetricsTracker(epochs={len(self)}, "
            f"latest_train_loss={latest['train_loss']:.4f}, "
            f"latest_val_loss={latest['val_loss']:.4f})"
        )
