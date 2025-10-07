"""Fine-tuning pipeline for Aksis AI chatbot/LLM."""

import logging
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from tqdm import tqdm

from aksis.model.transformer import TransformerDecoder
from aksis.data.tokenizer import Tokenizer
from aksis.train.trainer import Trainer

logger = logging.getLogger(__name__)


class FineTuner(Trainer):
    """Fine-tunes a pre-trained model on chatbot datasets."""

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: Tokenizer,
        device: torch.device,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        max_epochs: int = 10,
        early_stopping_patience: int = 3,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        """
        Initialize the FineTuner.

        Args:
            model: The pre-trained model to fine-tune.
            tokenizer: The tokenizer for encoding/decoding.
            device: The device (CPU or CUDA) to run on.
            learning_rate: Learning rate for fine-tuning.
            batch_size: Batch size for training.
            max_epochs: Maximum number of epochs.
            early_stopping_patience: Patience for early stopping.
            use_mixed_precision: Whether to use mixed precision training.
            gradient_accumulation_steps: Number of steps to accumulate
                gradients.
            warmup_steps: Number of warmup steps for learning rate
                scheduling.
            weight_decay: Weight decay for optimizer.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        if not isinstance(model, TransformerDecoder):
            raise ValueError("model must be a TransformerDecoder")
        if not hasattr(tokenizer, "encode") or not hasattr(
            tokenizer, "decode"
        ):
            raise ValueError("tokenizer must have encode and decode methods")
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device object")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")

        # Initialize parent class with default values for required parameters
        # We'll override the training logic in fine_tune method
        super().__init__(
            model=model,
            train_loader=None,  # Will be provided in fine_tune method
            val_loader=None,  # Will be provided in fine_tune method
            checkpoint_dir="checkpoints",  # Default checkpoint directory
            device=device,
            epochs=max_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            use_mixed_precision=use_mixed_precision,
        )

        # Store fine-tuning specific parameters
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.fine_tuning_history: List[Dict[str, float]] = []
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        logger.info(
            f"FineTuner initialized with learning_rate={learning_rate}"
        )

    def fine_tune(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        output_dir: Union[str, Path] = "checkpoints",
        save_best_only: bool = True,
        save_frequency: int = 1,
    ) -> Dict[str, Any]:
        """
        Fine-tune the model on the provided dataset.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            output_dir: Directory to save checkpoints.
            save_best_only: Whether to save only the best checkpoint.
            save_frequency: Frequency to save checkpoints (in epochs).

        Returns:
            Dictionary containing training history and best metrics.
        """
        if not train_dataloader:
            raise ValueError("train_dataloader cannot be None")
        if val_dataloader is None and save_best_only:
            raise ValueError(
                "val_dataloader required when save_best_only=True"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting fine-tuning...")
        train_samples = len(train_dataloader.dataset)  # type: ignore
        logger.info(f"Training samples: {train_samples}")
        if val_dataloader:
            val_samples = len(val_dataloader.dataset)  # type: ignore
            logger.info(f"Validation samples: {val_samples}")

        # Reset training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.fine_tuning_history = []

        for epoch in range(self.max_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs}")

            # Training phase
            train_metrics = self._train_epoch(train_dataloader, epoch)
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")

            # Validation phase
            val_metrics = None
            if val_dataloader:
                val_metrics = self._validate_epoch(val_dataloader, epoch)
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}")

            # Record metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_perplexity": train_metrics["perplexity"],
            }
            if val_metrics:
                epoch_metrics.update(
                    {
                        "val_loss": val_metrics["loss"],
                        "val_perplexity": val_metrics["perplexity"],
                    }
                )

            self.fine_tuning_history.append(epoch_metrics)

            # Save checkpoint
            should_save = False
            if save_best_only and val_metrics:
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    should_save = True
                    logger.info(
                        f"New best validation loss: {self.best_val_loss:.4f}"
                    )
                else:
                    self.patience_counter += 1
            elif not save_best_only and (epoch + 1) % save_frequency == 0:
                should_save = True

            if should_save:
                checkpoint_path = (
                    output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                )
                self._save_checkpoint(checkpoint_path, epoch_metrics)

            # Early stopping
            if (
                val_dataloader
                and self.patience_counter >= self.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs"
                )
                break

        # Load best checkpoint if using early stopping
        if save_best_only and val_dataloader:
            best_checkpoint = self._find_best_checkpoint(output_dir)
            if best_checkpoint:
                logger.info(f"Loading best checkpoint: {best_checkpoint}")
                self._load_checkpoint(best_checkpoint)

        logger.info("Fine-tuning completed")

        return {
            "history": self.fine_tuning_history,
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.fine_tuning_history),
        }

    def _train_epoch(
        self, dataloader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1} - Training",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare batch
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                else:
                    input_ids = batch.to(self.device)
                    attention_mask = None

                # Forward pass
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_mixed_precision,
                ):
                    logits = self.model(input_ids)
                    loss = self._compute_loss(
                        logits, input_ids, attention_mask
                    )

                # Backward pass
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()

                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()

                # Accumulate metrics
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_tokens += input_ids.numel()
                num_batches += 1

                # Update progress bar
                loss_display = loss.item() * self.gradient_accumulation_steps
                lr_display = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss_display:.4f}",
                        "lr": f"{lr_display:.2e}",
                    }
                )

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue

        # Compute epoch metrics
        avg_loss = (
            total_loss / num_batches if num_batches > 0 else float("inf")
        )
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {"loss": avg_loss, "perplexity": perplexity}

    def _validate_epoch(
        self, dataloader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1} - Validation",
            leave=False,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Prepare batch
                    if isinstance(batch, dict):
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch.get("attention_mask", None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                    else:
                        input_ids = batch.to(self.device)
                        attention_mask = None

                    # Forward pass
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=self.use_mixed_precision,
                    ):
                        logits = self.model(input_ids)
                        loss = self._compute_loss(
                            logits, input_ids, attention_mask
                        )

                    # Accumulate metrics
                    total_loss += loss.item()
                    total_tokens += input_ids.numel()
                    num_batches += 1

                    # Update progress bar
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                        }
                    )

                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

        # Compute epoch metrics
        avg_loss = (
            total_loss / num_batches if num_batches > 0 else float("inf")
        )
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {"loss": avg_loss, "perplexity": perplexity}

    def _compute_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss for the batch."""
        # Shift logits and targets for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Flatten for loss computation
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )

        return loss

    def _save_checkpoint(
        self, checkpoint_path: Path, metrics: Dict[str, Any]
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "fine_tuning_history": self.fine_tuning_history,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load scaler state
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load training history
        if "fine_tuning_history" in checkpoint:
            self.fine_tuning_history = checkpoint["fine_tuning_history"]

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def _find_best_checkpoint(self, output_dir: Path) -> Optional[Path]:
        """Find the best checkpoint based on validation loss."""
        checkpoint_files = list(output_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None

        best_checkpoint = None
        best_loss = float("inf")

        for checkpoint_file in checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_file, map_location="cpu")
                if (
                    "metrics" in checkpoint
                    and "val_loss" in checkpoint["metrics"]
                ):
                    val_loss = checkpoint["metrics"]["val_loss"]
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_checkpoint = checkpoint_file
            except Exception as e:
                logger.warning(
                    f"Error loading checkpoint {checkpoint_file}: {e}"
                )
                continue

        return best_checkpoint

    def hyperparameter_search(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        param_grid: Dict[str, List[Any]],
        output_dir: Union[str, Path] = "hyperparameter_search",
        max_trials: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search for fine-tuning.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            param_grid: Dictionary of parameter names to lists of values.
            output_dir: Directory to save search results.
            max_trials: Maximum number of trials to run.

        Returns:
            Dictionary containing search results and best parameters.
        """
        if not param_grid:
            raise ValueError("param_grid cannot be empty")
        if max_trials <= 0:
            raise ValueError("max_trials must be positive")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting hyperparameter search with {max_trials} trials")

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = self._generate_param_combinations(
            param_values, max_trials
        )

        results = []
        best_score = float("inf")
        best_params = None

        for trial_idx, param_combo in enumerate(param_combinations):
            logger.info(f"Trial {trial_idx + 1}/{len(param_combinations)}")
            logger.info(f"Parameters: {dict(zip(param_names, param_combo))}")

            try:
                # Create new FineTuner with current parameters
                trial_finetuner = FineTuner(
                    model=self.model,  # type: ignore
                    tokenizer=self.tokenizer,
                    device=self.device,
                    learning_rate=(
                        param_combo[param_names.index("learning_rate")]
                        if "learning_rate" in param_names
                        else self.learning_rate
                    ),
                    batch_size=(
                        param_combo[param_names.index("batch_size")]
                        if "batch_size" in param_names
                        else self.batch_size
                    ),
                    max_epochs=(
                        param_combo[param_names.index("max_epochs")]
                        if "max_epochs" in param_names
                        else self.max_epochs
                    ),
                    early_stopping_patience=self.early_stopping_patience,
                    use_mixed_precision=self.use_mixed_precision,
                    gradient_accumulation_steps=(
                        self.gradient_accumulation_steps
                    ),
                    warmup_steps=self.warmup_steps,
                    weight_decay=self.weight_decay,
                    max_grad_norm=self.max_grad_norm,
                )

                # Fine-tune with current parameters
                trial_output_dir = output_dir / f"trial_{trial_idx + 1}"
                trial_results = trial_finetuner.fine_tune(
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    output_dir=trial_output_dir,
                    save_best_only=True,
                )

                # Record results
                trial_result = {
                    "trial": trial_idx + 1,
                    "parameters": dict(zip(param_names, param_combo)),
                    "best_val_loss": trial_results["best_val_loss"],
                    "total_epochs": trial_results["total_epochs"],
                }
                results.append(trial_result)

                # Update best parameters
                if trial_results["best_val_loss"] < best_score:
                    best_score = trial_results["best_val_loss"]
                    best_params = dict(zip(param_names, param_combo))

                logger.info(
                    f"Trial {trial_idx + 1} completed. "
                    f"Best val loss: {trial_results['best_val_loss']:.4f}"
                )

            except Exception as e:
                logger.error(f"Trial {trial_idx + 1} failed: {e}")
                continue

        # Save search results
        search_results = {
            "best_parameters": best_params,
            "best_score": best_score,
            "all_results": results,
        }

        results_path = output_dir / "hyperparameter_search_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(search_results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Hyperparameter search completed. Best score: {best_score:.4f}"
        )
        logger.info(f"Best parameters: {best_params}")

        return search_results

    def _generate_param_combinations(
        self, param_values: List[List[Any]], max_trials: int
    ) -> List[List[Any]]:
        """Generate parameter combinations for hyperparameter search."""
        import itertools
        import random

        # Generate all possible combinations
        all_combinations = list(itertools.product(*param_values))

        # Sample if we have too many combinations
        if len(all_combinations) > max_trials:
            all_combinations = random.sample(all_combinations, max_trials)

        return [list(combo) for combo in all_combinations]

    def get_training_history(self) -> List[Dict[str, float]]:
        """Get the training history."""
        return self.fine_tuning_history.copy()

    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best validation metrics."""
        if not self.fine_tuning_history:
            return {}

        best_epoch = min(
            self.fine_tuning_history,
            key=lambda x: x.get("val_loss", float("inf")),
        )

        return best_epoch
