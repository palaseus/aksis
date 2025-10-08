"""Training loop and trainer implementation."""

import math
import time
import logging
from typing import Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn

from aksis.train.optimizer import (
    create_optimizer,
    create_scheduler,
    get_learning_rate,
)
from aksis.train.metrics import MetricsTracker, compute_perplexity
from aksis.train.checkpoint import CheckpointManager
from aksis.utils.device import get_device

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for transformer models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,  # DataLoader
        val_loader: Any,  # DataLoader
        checkpoint_dir: str,
        device: Optional[Union[str, torch.device]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_mixed_precision: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            checkpoint_dir: Directory to save checkpoints.
            device: Device to use for training.
            optimizer: Optimizer (created automatically if None).
            scheduler: Learning rate scheduler (created automatically if None).
            criterion: Loss function (CrossEntropyLoss if None).
            epochs: Number of training epochs.
            gradient_accumulation_steps: Number of steps to accumulate
                gradients.
            max_grad_norm: Maximum gradient norm for clipping.
            use_mixed_precision: Whether to use mixed precision training.
            **kwargs: Additional arguments for optimizer and scheduler
                creation.
        """
        # Validate inputs
        if epochs <= 0:
            raise ValueError("Epochs must be positive")

        if gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")

        if max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")

        # Set device
        device_str = str(device) if device is not None else None
        self.device = get_device(device_str)
        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = model.to(self.device)

        # Set data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set training parameters
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision

        # Initialize mixed precision scaler if needed
        if self.use_mixed_precision:
            if not torch.cuda.is_available():
                logger.warning(
                    "Mixed precision requested but CUDA not available, "
                    "disabling"
                )
                self.use_mixed_precision = False
            else:
                self.scaler = torch.amp.GradScaler()
                logger.info("Initialized mixed precision training")

        # Set up optimizer
        if optimizer is None:
            self.optimizer = create_optimizer(self.model, **kwargs)
        else:
            self.optimizer = optimizer

        # Set up scheduler
        if scheduler is None:
            # Filter out optimizer-specific parameters for scheduler
            scheduler_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "lr",
                    "weight_decay",
                    "betas",
                    "eps",
                    "momentum",
                    "nesterov",
                ]
            }
            self.scheduler = create_scheduler(
                self.optimizer, **scheduler_kwargs
            )
        else:
            self.scheduler = scheduler

        # Set up loss function
        if criterion is None:
            self.criterion: nn.Module = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.criterion = criterion

        # Initialize metrics tracker
        self.metrics = MetricsTracker()

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        logger.info(f"Initialized trainer with {epochs} epochs")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, average_perplexity).
        """
        self.model.train()

        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0

        logger.info(f"Starting training epoch {epoch}")

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                loss, perplexity = self._train_step(batch)

                total_loss += loss
                total_perplexity += perplexity
                num_batches += 1

                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}/"
                        f"{len(self.train_loader)}, "
                        f"Loss: {loss:.4f}, Perplexity: {perplexity:.4f}"
                    )

            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                raise

        # Compute averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_perplexity = (
            total_perplexity / num_batches if num_batches > 0 else 0.0
        )

        logger.info(
            f"Epoch {epoch} training completed - Loss: {avg_loss:.4f}, "
            f"Perplexity: {avg_perplexity:.4f}"
        )

        return avg_loss, avg_perplexity

    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, average_perplexity).
        """
        self.model.eval()

        total_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0

        logger.info(f"Starting validation epoch {epoch}")

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    loss, perplexity = self._val_step(batch)

                    total_loss += loss
                    total_perplexity += perplexity
                    num_batches += 1

                except Exception as e:
                    logger.error(f"Error in validation step {batch_idx}: {e}")
                    raise

        # Compute averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_perplexity = (
            total_perplexity / num_batches if num_batches > 0 else 0.0
        )

        logger.info(
            f"Epoch {epoch} validation completed - Loss: {avg_loss:.4f}, "
            f"Perplexity: {avg_perplexity:.4f}"
        )

        return avg_loss, avg_perplexity

    def _train_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """
        Perform one training step.

        Args:
            batch: Batch of training data.

        Returns:
            Tuple of (loss, perplexity).

        Raises:
            RuntimeError: If loss is NaN or infinite.
            ValueError: If batch is empty.
        """
        # Check for empty batch
        if batch["input_ids"].size(0) == 0:
            raise ValueError("Empty batch")

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        
        # Handle attention_mask if present
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"].to(self.device)
        else:
            # Create attention mask (all ones for non-padding tokens)
            attention_mask = (input_ids != 0).float()

        # Create targets (shifted input_ids)
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        # Forward pass
        if self.use_mixed_precision:
            with torch.amp.autocast(device_type=self.device.type):
                logits = self.model(input_ids, padding_mask=attention_mask)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
        else:
            logits = self.model(input_ids, padding_mask=attention_mask)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        # Check for numerical issues
        if torch.isnan(loss).any():
            raise RuntimeError("NaN loss detected")
        if torch.isinf(loss).any():
            raise RuntimeError("Infinite loss detected")

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights
        if (
            self.optimizer.param_groups[0].get("step", 0) + 1
        ) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()

            self.optimizer.zero_grad()

        # Compute perplexity using the same loss that was used for training
        # This ensures consistency between loss and perplexity
        perplexity = math.exp(loss.item())

        # Return the unscaled loss for consistency with perplexity calculation
        return loss.item(), perplexity

    def _val_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Perform one validation step.

        Args:
            batch: Batch of validation data.

        Returns:
            Tuple of (loss, perplexity).
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        
        # Handle attention_mask if present
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"].to(self.device)
        else:
            # Create attention mask (all ones for non-padding tokens)
            attention_mask = (input_ids != 0).float()

        # Create targets (shifted input_ids)
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        # Forward pass
        if self.use_mixed_precision:
            with torch.amp.autocast(device_type=self.device.type):
                logits = self.model(input_ids, padding_mask=attention_mask)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
        else:
            logits = self.model(input_ids, padding_mask=attention_mask)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        # Compute perplexity using the same loss that was used for validation
        # This ensures consistency between loss and perplexity
        perplexity = math.exp(loss.item())

        return loss.item(), perplexity

    def train(self) -> Dict[str, Any]:
        """
        Train the model for the specified number of epochs.

        Returns:
            Dictionary containing training results.
        """
        logger.info(f"Starting training for {self.epochs} epochs")

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # Training
            train_loss, train_perplexity = self.train_epoch(epoch)

            # Validation
            val_loss, val_perplexity = self.validate_epoch(epoch)

            # Update scheduler
            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Record metrics
            epoch_time = time.time() - epoch_start_time
            self.metrics.update_train(train_loss, train_perplexity, epoch_time)
            self.metrics.update_val(val_loss, val_perplexity)

            # Save checkpoint
            self.save_checkpoint(
                epoch=epoch,
                loss=val_loss,
                metrics={
                    "train_loss": train_loss,
                    "train_perplexity": train_perplexity,
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity,
                    "learning_rate": get_learning_rate(self.optimizer),
                },
            )

            # Log epoch summary
            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s - "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Perplexity: {train_perplexity:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Perplexity: {val_perplexity:.4f}, "
                f"LR: {get_learning_rate(self.optimizer):.2e}"
            )

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        return {
            "total_time": total_time,
            "metrics": self.metrics.to_dict(),
            "best_loss": self.checkpoint_manager.best_loss,
            "best_epoch": self.checkpoint_manager.best_epoch,
        }

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch.
            loss: Current loss.
            metrics: Additional metrics.
            metadata: Additional metadata.

        Returns:
            Path to saved checkpoint.
        """
        return self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            loss=loss,
            metrics=metrics,
            metadata=metadata,
        )

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.

        Returns:
            Checkpoint data.
        """
        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return get_learning_rate(self.optimizer)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Estimate model size in MB
        param_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size() for b in self.model.buffers()
        )
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "device": str(self.device),
            "mixed_precision": self.use_mixed_precision,
        }
