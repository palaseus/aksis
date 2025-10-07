"""Tests for checkpointing functionality."""

import pytest
import torch
import tempfile
import os
import shutil
from unittest.mock import Mock

from aksis.train.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Test checkpoint manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_manager_initialization(self):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(self.checkpoint_dir)

        assert manager.checkpoint_dir == self.checkpoint_dir
        assert manager.best_loss is None
        assert manager.best_epoch is None
        assert manager.checkpoint_history == []

    def test_checkpoint_manager_initialization_with_best_loss(self):
        """Test checkpoint manager initialization with best loss."""
        manager = CheckpointManager(self.checkpoint_dir, best_loss=1.5)

        assert manager.best_loss == 1.5

    def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock model, optimizer, and scheduler
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        # Mock state dicts
        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=1.5,
            metrics={"perplexity": 4.5},
        )

        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith("epoch_1.pt")
        assert manager.best_loss == 1.5
        assert manager.best_epoch == 1
        assert len(manager.checkpoint_history) == 1

    def test_save_checkpoint_with_metadata(self):
        """Test saving checkpoint with metadata."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save checkpoint with metadata
        metadata = {
            "model_config": {"d_model": 512, "num_heads": 8},
            "training_config": {"batch_size": 32, "lr": 1e-3},
        }

        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=1.5,
            metrics={"perplexity": 4.5},
            metadata=metadata,
        )

        # Load and verify metadata
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert "metadata" in checkpoint
        assert checkpoint["metadata"] == metadata

    def test_save_checkpoint_improved_loss(self):
        """Test saving checkpoint with improved loss."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save first checkpoint
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=2.0,
            metrics={"perplexity": 7.4},
        )

        # Save second checkpoint with better loss
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            loss=1.5,
            metrics={"perplexity": 4.5},
        )

        assert manager.best_loss == 1.5
        assert manager.best_epoch == 2
        assert checkpoint_path.endswith("best.pt")

    def test_save_checkpoint_worse_loss(self):
        """Test saving checkpoint with worse loss."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save first checkpoint
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=1.5,
            metrics={"perplexity": 4.5},
        )

        # Save second checkpoint with worse loss
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            loss=2.0,
            metrics={"perplexity": 7.4},
        )

        assert manager.best_loss == 1.5
        assert manager.best_epoch == 1
        assert checkpoint_path.endswith("epoch_2.pt")

    def test_load_checkpoint_basic(self):
        """Test basic checkpoint loading."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=1.5,
            metrics={"perplexity": 4.5},
        )

        # Load checkpoint
        loaded_data = manager.load_checkpoint(checkpoint_path)

        assert "model_state_dict" in loaded_data
        assert "optimizer_state_dict" in loaded_data
        assert "scheduler_state_dict" in loaded_data
        assert loaded_data["epoch"] == 1
        assert loaded_data["loss"] == 1.5
        assert loaded_data["metrics"]["perplexity"] == 4.5

    def test_load_checkpoint_nonexistent(self):
        """Test loading nonexistent checkpoint."""
        manager = CheckpointManager(self.checkpoint_dir)

        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint("nonexistent.pt")

    def test_load_checkpoint_corrupted(self):
        """Test loading corrupted checkpoint."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create a corrupted file
        corrupted_path = os.path.join(self.checkpoint_dir, "corrupted.pt")
        with open(corrupted_path, "w") as f:
            f.write("corrupted data")

        with pytest.raises(Exception):
            manager.load_checkpoint(corrupted_path)

    def test_load_best_checkpoint(self):
        """Test loading best checkpoint."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save multiple checkpoints
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=2.0,
            metrics={"perplexity": 7.4},
        )
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            loss=1.5,
            metrics={"perplexity": 4.5},
        )
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            loss=1.8,
            metrics={"perplexity": 6.0},
        )

        # Load best checkpoint
        best_path = manager.get_best_checkpoint_path()
        loaded_data = manager.load_checkpoint(best_path)

        assert loaded_data["epoch"] == 2
        assert loaded_data["loss"] == 1.5

    def test_load_latest_checkpoint(self):
        """Test loading latest checkpoint."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save multiple checkpoints
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=2.0,
            metrics={"perplexity": 7.4},
        )
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            loss=1.5,
            metrics={"perplexity": 4.5},
        )

        # Load latest checkpoint
        latest_path = manager.get_latest_checkpoint_path()
        loaded_data = manager.load_checkpoint(latest_path)

        assert loaded_data["epoch"] == 2

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save multiple checkpoints
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=2.0,
            metrics={"perplexity": 7.4},
        )
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            loss=1.5,
            metrics={"perplexity": 4.5},
        )

        # List checkpoints
        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 2
        assert any("epoch_1.pt" in cp for cp in checkpoints)
        assert any("best.pt" in cp for cp in checkpoints)

    def test_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoints."""
        manager = CheckpointManager(self.checkpoint_dir, max_checkpoints=2)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save multiple checkpoints
        for epoch in range(5):
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=2.0 - epoch * 0.1,
                metrics={"perplexity": 7.4},
            )

        # Check that only the latest checkpoints are kept
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 2

    def test_checkpoint_manager_cuda(self):
        """Test checkpoint manager with CUDA tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        # Create CUDA tensors
        model.state_dict.return_value = {"weight": torch.randn(10, 10).cuda()}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            loss=1.5,
            metrics={"perplexity": 4.5},
        )

        # Load checkpoint
        loaded_data = manager.load_checkpoint(checkpoint_path)

        assert "model_state_dict" in loaded_data
        # Check that tensors are loaded on CPU by default
        assert loaded_data["model_state_dict"]["weight"].device.type == "cpu"

    def test_checkpoint_manager_invalid_epoch(self):
        """Test checkpoint manager with invalid epoch."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        with pytest.raises(ValueError, match="Epoch must be non-negative"):
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=-1,
                loss=1.5,
                metrics={"perplexity": 4.5},
            )

    def test_checkpoint_manager_invalid_loss(self):
        """Test checkpoint manager with invalid loss."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        with pytest.raises(ValueError, match="Loss must be non-negative"):
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                loss=-1.5,
                metrics={"perplexity": 4.5},
            )

    def test_checkpoint_manager_nan_loss(self):
        """Test checkpoint manager with NaN loss."""
        manager = CheckpointManager(self.checkpoint_dir)

        # Create mock components
        model = Mock()
        optimizer = Mock()
        scheduler = Mock()

        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        optimizer.state_dict.return_value = {"param_groups": []}
        scheduler.state_dict.return_value = {"last_epoch": 0}

        with pytest.raises(ValueError, match="Loss cannot be NaN"):
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                loss=float("nan"),
                metrics={"perplexity": 4.5},
            )
