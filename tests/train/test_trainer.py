"""Tests for trainer functionality."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os

from aksis.train.trainer import Trainer


class TestTrainer:
    """Test trainer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        model.buffers.return_value = [torch.randn(5, 5)]
        model.train.return_value = None
        model.eval.return_value = None
        model.to.return_value = model

        # Mock the forward method to return logits with correct shape
        def mock_forward(input_ids, padding_mask=None):
            batch_size, seq_len = input_ids.shape
            vocab_size = 1000
            return torch.randn(
                batch_size,
                seq_len,
                vocab_size,
                device=input_ids.device,
                requires_grad=True,
            )

        model.side_effect = mock_forward
        return model

    def create_mock_dataloader(self):
        """Create a mock dataloader for testing."""
        dataloader = Mock()
        # Create a proper iterator
        batch_data = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        dataloader.__iter__ = Mock(return_value=iter([batch_data]))
        dataloader.__len__ = Mock(return_value=1)
        return dataloader

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        assert trainer.model == model
        assert trainer.train_loader == train_loader
        assert trainer.val_loader == val_loader
        assert trainer.checkpoint_manager.checkpoint_dir == self.checkpoint_dir
        assert trainer.device.type in [
            "cpu",
            "cuda",
        ]  # Default device (CUDA if available)

    def test_trainer_initialization_with_device(self):
        """Test trainer initialization with specific device."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            device="cpu",
        )

        assert trainer.device.type == "cpu"

    def test_trainer_initialization_cuda(self):
        """Test trainer initialization with CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            device="cuda",
        )

        assert trainer.device.type == "cuda"

    def test_trainer_initialization_with_optimizer(self):
        """Test trainer initialization with custom optimizer."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            optimizer=optimizer,
        )

        assert trainer.optimizer == optimizer

    def test_trainer_initialization_with_scheduler(self):
        """Test trainer initialization with custom scheduler."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        assert trainer.scheduler == scheduler

    def test_trainer_initialization_with_criterion(self):
        """Test trainer initialization with custom criterion."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            criterion=criterion,
        )

        assert trainer.criterion == criterion

    def test_trainer_initialization_with_gradient_accumulation(self):
        """Test trainer initialization with gradient accumulation."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            gradient_accumulation_steps=4,
        )

        assert trainer.gradient_accumulation_steps == 4

    def test_trainer_initialization_with_mixed_precision(self):
        """Test trainer initialization with mixed precision."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            use_mixed_precision=True,
        )

        assert trainer.use_mixed_precision is True
        assert trainer.scaler is not None

    def test_trainer_initialization_invalid_epochs(self):
        """Test trainer initialization with invalid epochs."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        with pytest.raises(ValueError, match="Epochs must be positive"):
            Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=self.checkpoint_dir,
                epochs=0,
            )

    def test_trainer_initialization_invalid_gradient_accumulation(self):
        """Test trainer initialization with invalid gradient accumulation."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        with pytest.raises(
            ValueError, match="Gradient accumulation steps must be positive"
        ):
            Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=self.checkpoint_dir,
                gradient_accumulation_steps=0,
            )

    def test_trainer_initialization_invalid_max_grad_norm(self):
        """Test trainer initialization with invalid max grad norm."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        with pytest.raises(
            ValueError, match="Max gradient norm must be positive"
        ):
            Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=self.checkpoint_dir,
                max_grad_norm=0,
            )

    @patch("aksis.train.trainer.create_optimizer")
    @patch("aksis.train.trainer.create_scheduler")
    def test_trainer_setup_optimizer_scheduler(
        self, mock_scheduler, mock_optimizer
    ):
        """Test trainer optimizer and scheduler setup."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        mock_optimizer.return_value = torch.optim.AdamW(
            model.parameters(), lr=1e-3
        )
        mock_scheduler.return_value = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                mock_optimizer.return_value, T_max=100
            )
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        mock_optimizer.assert_called_once()
        mock_scheduler.assert_called_once()
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_trainer_train_epoch(self):
        """Test training for one epoch."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Mock the training step
        with patch.object(trainer, "_train_step") as mock_step:
            mock_step.return_value = (1.5, 4.5)  # loss, perplexity

            loss, perplexity = trainer.train_epoch(epoch=0)

            assert loss == 1.5
            assert perplexity == 4.5
            mock_step.assert_called_once()

    def test_trainer_validate_epoch(self):
        """Test validation for one epoch."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Mock the validation step
        with patch.object(trainer, "_val_step") as mock_step:
            mock_step.return_value = (1.2, 3.3)  # loss, perplexity

            loss, perplexity = trainer.validate_epoch(epoch=0)

            assert loss == 1.2
            assert perplexity == 3.3
            mock_step.assert_called_once()

    def test_trainer_train_step(self):
        """Test training step."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Mock model forward pass
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }

        # Mock model output
        model_output = torch.randn(2, 10, 1000)
        model.return_value = model_output

        # Mock criterion
        criterion_output = torch.tensor(1.5, requires_grad=True)
        trainer.criterion.return_value = criterion_output

        loss, perplexity = trainer._train_step(batch)

        assert isinstance(loss, float)
        assert isinstance(perplexity, float)
        assert loss > 0
        assert perplexity > 0

    def test_trainer_val_step(self):
        """Test validation step."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Mock model forward pass
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }

        # Mock model output
        model_output = torch.randn(2, 10, 1000)
        model.return_value = model_output

        # Mock criterion
        criterion_output = torch.tensor(1.2)
        trainer.criterion.return_value = criterion_output

        loss, perplexity = trainer._val_step(batch)

        assert isinstance(loss, float)
        assert isinstance(perplexity, float)
        assert loss > 0
        assert perplexity > 0

    def test_trainer_train_step_with_gradient_accumulation(self):
        """Test training step with gradient accumulation."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            gradient_accumulation_steps=2,
        )

        # Mock model forward pass
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }

        # Mock model output
        model_output = torch.randn(2, 10, 1000)
        model.return_value = model_output

        # Mock criterion
        criterion_output = torch.tensor(1.5, requires_grad=True)
        trainer.criterion.return_value = criterion_output

        loss, perplexity = trainer._train_step(batch)

        assert isinstance(loss, float)
        assert isinstance(perplexity, float)

    def test_trainer_train_step_with_mixed_precision(self):
        """Test training step with mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision")

        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            use_mixed_precision=True,
            device="cuda",
        )

        # Mock model forward pass
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)).cuda(),
            "attention_mask": torch.ones(2, 10).cuda(),
        }

        # Mock criterion
        criterion_output = torch.tensor(1.5, requires_grad=True).cuda()
        trainer.criterion.return_value = criterion_output

        # Mock optimizer step for mixed precision
        trainer.optimizer.step = Mock()
        trainer.optimizer.zero_grad = Mock()

        # Mock scaler for mixed precision
        trainer.scaler.step = Mock()
        trainer.scaler.update = Mock()
        trainer.scaler.scale = Mock(side_effect=lambda x: x)
        trainer.scaler.unscale_ = Mock()

        loss, perplexity = trainer._train_step(batch)

        assert isinstance(loss, float)
        assert isinstance(perplexity, float)

    def test_trainer_train_step_with_gradient_clipping(self):
        """Test training step with gradient clipping."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
            max_grad_norm=1.0,
        )

        # Mock model forward pass
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }

        # Mock model output
        model_output = torch.randn(2, 10, 1000)
        model.return_value = model_output

        # Mock criterion
        criterion_output = torch.tensor(1.5, requires_grad=True)
        trainer.criterion.return_value = criterion_output

        loss, perplexity = trainer._train_step(batch)

        assert isinstance(loss, float)
        assert isinstance(perplexity, float)

    def test_trainer_train_step_nan_loss(self):
        """Test training step with NaN loss."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Mock model forward pass
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }

        # Mock criterion with NaN output
        criterion_output = torch.tensor(float("nan"), requires_grad=True)
        mock_criterion = Mock(return_value=criterion_output)
        trainer.criterion = mock_criterion

        with pytest.raises(RuntimeError, match="NaN loss detected"):
            trainer._train_step(batch)

    def test_trainer_train_step_inf_loss(self):
        """Test training step with infinite loss."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Mock model forward pass
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }

        # Mock criterion with infinite output
        criterion_output = torch.tensor(float("inf"), requires_grad=True)
        mock_criterion = Mock(return_value=criterion_output)
        trainer.criterion = mock_criterion

        with pytest.raises(RuntimeError, match="Infinite loss detected"):
            trainer._train_step(batch)

    def test_trainer_train_step_empty_batch(self):
        """Test training step with empty batch."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Empty batch
        batch = {
            "input_ids": torch.empty(0, 10, dtype=torch.long),
            "attention_mask": torch.empty(0, 10, dtype=torch.long),
        }

        with pytest.raises(ValueError, match="Empty batch"):
            trainer._train_step(batch)

    def test_trainer_save_checkpoint(self):
        """Test saving checkpoint."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Mock checkpoint manager instance
        mock_manager = Mock()
        mock_manager.save_checkpoint.return_value = "checkpoint.pt"
        trainer.checkpoint_manager = mock_manager

        trainer.save_checkpoint(epoch=1, loss=1.5, metrics={"perplexity": 4.5})

        mock_manager.save_checkpoint.assert_called_once()

    def test_trainer_load_checkpoint(self):
        """Test loading checkpoint."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Mock checkpoint manager instance
        mock_manager = Mock()
        mock_manager.load_checkpoint.return_value = {
            "model_state_dict": {"weight": torch.randn(10, 10)},
            "optimizer_state_dict": {"param_groups": []},
            "scheduler_state_dict": {"last_epoch": 0},
            "epoch": 1,
            "loss": 1.5,
        }
        trainer.checkpoint_manager = mock_manager

        trainer.load_checkpoint("checkpoint.pt")

        mock_manager.load_checkpoint.assert_called_once_with(
            "checkpoint.pt",
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
        )

    def test_trainer_get_learning_rate(self):
        """Test getting current learning rate."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        lr = trainer.get_learning_rate()

        assert isinstance(lr, float)
        assert lr > 0

    def test_trainer_get_model_info(self):
        """Test getting model information."""
        model = self.create_mock_model()
        train_loader = self.create_mock_dataloader()
        val_loader = self.create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=self.checkpoint_dir,
        )

        info = trainer.get_model_info()

        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "model_size_mb" in info
        assert info["total_parameters"] > 0
