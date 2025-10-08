"""Checkpoint management for saving and loading model states."""

import os
import math
import logging
from typing import Dict, Any, Optional, List, Union

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints for saving and loading."""

    def __init__(
        self,
        checkpoint_dir: str,
        best_loss: Optional[float] = None,
        max_checkpoints: int = 5,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            best_loss: Initial best loss value.
            max_checkpoints: Maximum number of checkpoints to keep.
        """
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = best_loss
        self.best_epoch: Optional[int] = None
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history: List[str] = []

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"Initialized CheckpointManager in {checkpoint_dir}")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        epoch: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            scheduler: Scheduler to save (optional).
            epoch: Current epoch number.
            loss: Current loss value.
            metrics: Additional metrics to save.
            metadata: Additional metadata to save.
            is_best: Whether this is the best checkpoint so far.

        Returns:
            Path to the saved checkpoint.

        Raises:
            ValueError: If epoch or loss is invalid.
        """
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")

        if math.isnan(loss) or math.isinf(loss):
            raise ValueError("Loss cannot be NaN or infinite")

        if loss < 0:
            raise ValueError("Loss must be non-negative")

        # Determine if this is the best checkpoint
        if self.best_loss is None:
            # First checkpoint - set as best but don't save as best.pt
            self.best_loss = loss
            self.best_epoch = epoch
            is_best = False  # Save as epoch_X.pt, not best.pt
        elif loss < self.best_loss:
            # Better than previous best
            self.best_loss = loss
            self.best_epoch = epoch
            is_best = True
        else:
            # Not better than previous best
            is_best = False

        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
        }

        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        # Add metrics if provided
        if metrics is not None:
            checkpoint_data["metrics"] = metrics

        # Add metadata if provided
        if metadata is not None:
            checkpoint_data["metadata"] = metadata

        # Determine checkpoint filename
        if is_best:
            filename = "best.pt"
        else:
            filename = f"epoch_{epoch}.pt"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Update checkpoint history
            if checkpoint_path not in self.checkpoint_history:
                self.checkpoint_history.append(checkpoint_path)

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            model: Model to load state into (optional).
            optimizer: Optimizer to load state into (optional).
            scheduler: Scheduler to load state into (optional).
            map_location: Device to map tensors to.

        Returns:
            Dictionary containing checkpoint data.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            Exception: If loading fails.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Load checkpoint
            if map_location is None:
                map_location = "cpu"  # Default to CPU for safety
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")

            # Load model state if model is provided
            if model is not None and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded model state")

            # Load optimizer state if optimizer is provided
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Loaded optimizer state")

            # Load scheduler state if scheduler is provided
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Loaded scheduler state")

            # Update best loss and epoch if available
            if "best_loss" in checkpoint:
                self.best_loss = checkpoint["best_loss"]
            if "best_epoch" in checkpoint:
                self.best_epoch = checkpoint["best_epoch"]

            # Update checkpoint history
            if checkpoint_path not in self.checkpoint_history:
                self.checkpoint_history.append(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

        return checkpoint  # type: ignore

    def get_best_checkpoint_path(self) -> Optional[str]:
        """
        Get the path to the best checkpoint.

        Returns:
            Path to the best checkpoint, or None if no checkpoints exist.
        """
        best_path = os.path.join(self.checkpoint_dir, "best.pt")
        if os.path.exists(best_path):
            return best_path
        return None

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist.
        """
        if not self.checkpoint_history:
            return None

        # Return the most recently added checkpoint
        return self.checkpoint_history[-1]

    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint paths.
        """
        checkpoints = []

        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith(".pt"):
                    checkpoints.append(
                        os.path.join(self.checkpoint_dir, filename)
                    )

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        return checkpoints

    def delete_checkpoint(self, checkpoint_path: str) -> None:
        """
        Delete a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint to delete.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            os.remove(checkpoint_path)
            logger.info(f"Deleted checkpoint: {checkpoint_path}")

            # Remove from history
            if checkpoint_path in self.checkpoint_history:
                self.checkpoint_history.remove(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            raise

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints to maintain max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return

        checkpoints = self.list_checkpoints()

        # Don't delete the best checkpoint
        best_path = self.get_best_checkpoint_path()
        if best_path and best_path in checkpoints:
            checkpoints.remove(best_path)

        # Delete excess checkpoints
        while (
            len(checkpoints) > self.max_checkpoints - 1
        ):  # -1 for best checkpoint
            oldest_checkpoint = checkpoints.pop()  # Remove from end (oldest)
            try:
                self.delete_checkpoint(oldest_checkpoint)
            except Exception as e:
                logger.warning(
                    f"Failed to delete old checkpoint {oldest_checkpoint}: {e}"
                )

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading it.

        Args:
            checkpoint_path: Path to the checkpoint.

        Returns:
            Dictionary containing checkpoint information.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Load only the metadata
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if not isinstance(checkpoint, dict):
                raise ValueError("Invalid checkpoint format")

            info = {
                "path": checkpoint_path,
                "epoch": checkpoint.get("epoch", "unknown"),
                "loss": checkpoint.get("loss", "unknown"),
                "best_loss": checkpoint.get("best_loss", "unknown"),
                "best_epoch": checkpoint.get("best_epoch", "unknown"),
                "metrics": checkpoint.get("metrics", {}),
                "metadata": checkpoint.get("metadata", {}),
                "file_size_mb": os.path.getsize(checkpoint_path)
                / (1024 * 1024),
                "modified_time": os.path.getmtime(checkpoint_path),
            }

            return info

        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of checkpoint manager."""
        return (
            f"CheckpointManager(dir={self.checkpoint_dir}, "
            f"best_loss={self.best_loss}, best_epoch={self.best_epoch})"
        )
