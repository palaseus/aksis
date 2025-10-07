"""Tests for fine-tuning pipeline."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from aksis.eval.fine_tuner import FineTuner
from aksis.model.transformer import TransformerDecoder
from aksis.data.tokenizer import Tokenizer
from aksis.train.trainer import Trainer


class TestFineTuner:
    """Test fine-tuning functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
        # Mock tokenizer
        self.tokenizer = Mock(spec=Tokenizer)
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Hello world")
        self.tokenizer.vocab_size_with_special = 1000
        
        # Mock model
        self.model = Mock(spec=TransformerDecoder)
        self.model.eval = Mock()
        self.model.train = Mock()
        self.model.to = Mock(return_value=self.model)
        # Mock parameters for optimizer creation
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        self.model.parameters = Mock(return_value=[mock_param])
        self.model.state_dict = Mock(return_value={})
        self.model.load_state_dict = Mock()
        
        # Mock data loaders
        self.train_loader = Mock()
        self.val_loader = Mock()
        self.train_loader.__iter__ = Mock(return_value=iter([]))
        self.val_loader.__iter__ = Mock(return_value=iter([]))
        
        # Sample fine-tuning data
        self.fine_tune_data = [
            {"input": "Hello", "target": "Hello! How can I help you?"},
            {"input": "What's the weather?", "target": "I don't have access to weather data."},
            {"input": "Tell me a joke", "target": "Why did the chicken cross the road?"},
        ]
        
        self.fine_tuner = FineTuner(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def test_fine_tuner_initialization(self) -> None:
        """Test fine-tuner initialization."""
        assert self.fine_tuner.model == self.model
        assert self.fine_tuner.tokenizer == self.tokenizer
        assert self.fine_tuner.device == self.device
        assert self.fine_tuner.learning_rate == 1e-4  # Default value
        assert self.fine_tuner.batch_size == 16  # Default value
        assert self.fine_tuner.max_epochs == 10  # Default value

    def test_fine_tuner_initialization_invalid_model(self) -> None:
        """Test fine-tuner initialization with invalid model."""
        with pytest.raises(ValueError, match="model must be a TransformerDecoder"):
            FineTuner(
                model="invalid_model",
                tokenizer=self.tokenizer,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
            )

    def test_fine_tuner_initialization_invalid_tokenizer(self) -> None:
        """Test fine-tuner initialization with invalid tokenizer."""
        with pytest.raises(
            ValueError, match="tokenizer must have encode and decode methods"
        ):
            FineTuner(
                model=self.model,
                tokenizer="invalid_tokenizer",
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
            )

    def test_fine_tuner_initialization_invalid_device(self) -> None:
        """Test fine-tuner initialization with invalid device."""
        with pytest.raises(ValueError, match="device must be a torch.device"):
            FineTuner(
                model=self.model,
                tokenizer=self.tokenizer,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device="invalid_device",
            )

    def test_fine_tune_single_epoch(self) -> None:
        """Test fine-tuning for a single epoch."""
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                results = self.fine_tuner.fine_tune(
                    epochs=1,
                    learning_rate=1e-5,
                    save_dir="./test_checkpoints",
                )
        
        assert isinstance(results, dict)
        assert "epoch" in results
        assert "train_loss" in results
        assert "val_loss" in results
        assert results["epoch"] == 1

    def test_fine_tune_multiple_epochs(self) -> None:
        """Test fine-tuning for multiple epochs."""
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock validation data
        mock_val_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.val_loader.__iter__ = Mock(return_value=iter([mock_val_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                results = self.fine_tuner.fine_tune(
                    epochs=3,
                    learning_rate=1e-5,
                    save_dir="./test_checkpoints",
                )
        
        assert isinstance(results, dict)
        assert results["epoch"] == 3

    def test_fine_tune_hyperparameter_tuning(self) -> None:
        """Test hyperparameter tuning."""
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        hyperparams = {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "batch_size": [8, 16, 32],
        }
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                results = self.fine_tuner.hyperparameter_tuning(
                    hyperparams=hyperparams,
                    epochs=1,
                    save_dir="./test_checkpoints",
                )
        
        assert isinstance(results, dict)
        assert "best_params" in results
        assert "best_score" in results
        assert "all_results" in results

    def test_fine_tune_early_stopping(self) -> None:
        """Test early stopping functionality."""
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock validation data
        mock_val_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.val_loader.__iter__ = Mock(return_value=iter([mock_val_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss that increases (should trigger early stopping)
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(side_effect=[2.5, 3.0, 3.5, 4.0])
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                results = self.fine_tuner.fine_tune(
                    epochs=10,
                    learning_rate=1e-5,
                    save_dir="./test_checkpoints",
                    early_stopping_patience=2,
                )
        
        assert isinstance(results, dict)
        # Should stop early due to increasing loss
        assert results["epoch"] < 10

    def test_fine_tune_save_checkpoint(self) -> None:
        """Test checkpoint saving during fine-tuning."""
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        mock_optimizer.state_dict = Mock(return_value={})
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                with patch("torch.save") as mock_save:
                    results = self.fine_tuner.fine_tune(
                        epochs=1,
                        learning_rate=1e-5,
                        save_dir="./test_checkpoints",
                    )
        
        # Should save checkpoint
        mock_save.assert_called()

    def test_fine_tune_load_checkpoint(self) -> None:
        """Test loading checkpoint for fine-tuning."""
        # Mock checkpoint
        mock_checkpoint = {
            "model_state_dict": {"embedding.weight": torch.randn(1000, 512)},
            "optimizer_state_dict": {},
            "epoch": 5,
            "loss": 2.0,
        }
        
        with patch("torch.load", return_value=mock_checkpoint):
            self.fine_tuner.load_checkpoint("test_checkpoint.pt")
        
        # Should load model state dict
        self.model.load_state_dict.assert_called_once()

    def test_fine_tune_load_checkpoint_file_not_found(self) -> None:
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            self.fine_tuner.load_checkpoint("nonexistent.pt")

    def test_fine_tune_validation_loss(self) -> None:
        """Test validation loss computation."""
        # Mock validation data
        mock_val_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.val_loader.__iter__ = Mock(return_value=iter([mock_val_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            val_loss = self.fine_tuner.compute_validation_loss()
        
        assert isinstance(val_loss, float)
        assert val_loss == 2.5

    def test_fine_tune_cuda_compatibility(self) -> None:
        """Test CUDA compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        cuda_device = torch.device("cuda")
        fine_tuner = FineTuner(
            model=self.model,
            tokenizer=self.tokenizer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=cuda_device,
        )
        
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]).to(cuda_device),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]).to(cuda_device),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special).to(
            cuda_device
        )
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                results = fine_tuner.fine_tune(
                    epochs=1,
                    learning_rate=1e-5,
                    save_dir="./test_checkpoints",
                )
        
        assert isinstance(results, dict)

    def test_fine_tune_mixed_precision(self) -> None:
        """Test fine-tuning with mixed precision."""
        fine_tuner = FineTuner(
            model=self.model,
            tokenizer=self.tokenizer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            use_mixed_precision=True,
        )
        
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                results = fine_tuner.fine_tune(
                    epochs=1,
                    learning_rate=1e-5,
                    save_dir="./test_checkpoints",
                )
        
        assert isinstance(results, dict)

    def test_fine_tune_gradient_clipping(self) -> None:
        """Test gradient clipping during fine-tuning."""
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
                    results = self.fine_tuner.fine_tune(
                        epochs=1,
                        learning_rate=1e-5,
                        save_dir="./test_checkpoints",
                        max_grad_norm=1.0,
                    )
        
        # Should apply gradient clipping
        mock_clip.assert_called()

    def test_fine_tune_performance(self) -> None:
        """Test fine-tuning performance."""
        import time
        
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                start_time = time.time()
                results = self.fine_tuner.fine_tune(
                    epochs=1,
                    learning_rate=1e-5,
                    save_dir="./test_checkpoints",
                )
                end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 10.0
        assert isinstance(results, dict)

    def test_fine_tune_error_handling(self) -> None:
        """Test error handling during fine-tuning."""
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model to raise an error
        self.model.side_effect = RuntimeError("Model error")
        
        with pytest.raises(RuntimeError, match="Model error"):
            self.fine_tuner.fine_tune(
                epochs=1,
                learning_rate=1e-5,
                save_dir="./test_checkpoints",
            )

    def test_fine_tune_metadata_saving(self) -> None:
        """Test saving metadata with checkpoints."""
        # Mock training data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        }
        self.train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.zero_grad = Mock()
        mock_optimizer.step = Mock()
        mock_optimizer.state_dict = Mock(return_value={})
        
        # Mock loss
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=2.5)
        
        with patch("torch.nn.functional.cross_entropy", return_value=mock_loss):
            with patch("torch.optim.Adam", return_value=mock_optimizer):
                with patch("torch.save") as mock_save:
                    results = self.fine_tuner.fine_tune(
                        epochs=1,
                        learning_rate=1e-5,
                        save_dir="./test_checkpoints",
                    )
        
        # Check that metadata is saved
        call_args = mock_save.call_args[0]
        checkpoint = call_args[0]
        assert "epoch" in checkpoint
        assert "loss" in checkpoint
        assert "learning_rate" in checkpoint
        assert "hyperparameters" in checkpoint
