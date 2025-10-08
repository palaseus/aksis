"""Tests for distributed training and scaling functionality."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from aksis.deploy.scaler import DistributedTrainer, ScalingConfig, ScalingError


class TestDistributedTrainer:
    """Test distributed training functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = ScalingConfig(
            num_gpus=1,  # Use 1 GPU for testing
            batch_size=32,
            learning_rate=0.001,
            use_mixed_precision=True,
        )
        self.trainer = DistributedTrainer(self.config)

    def test_scaling_config_initialization(self) -> None:
        """Test scaling configuration initialization."""
        config = ScalingConfig(
            num_gpus=4,
            batch_size=64,
            learning_rate=0.0005,
            use_mixed_precision=True,
        )
        assert config.num_gpus == 4
        assert config.batch_size == 64
        assert config.learning_rate == 0.0005
        assert config.use_mixed_precision is True

    def test_scaling_config_defaults(self) -> None:
        """Test scaling configuration with default values."""
        config = ScalingConfig()
        assert config.num_gpus == 1
        assert config.batch_size == 16
        assert config.learning_rate == 0.001
        assert config.use_mixed_precision is False

    def test_distributed_trainer_initialization(self) -> None:
        """Test distributed trainer initialization."""
        assert isinstance(self.trainer, DistributedTrainer)
        assert self.trainer.config == self.config
        assert self.trainer.device is not None

    @patch("torch.cuda.is_available")
    def test_device_detection_cuda(self, mock_cuda_available):
        """Test CUDA device detection."""
        mock_cuda_available.return_value = True

        trainer = DistributedTrainer(self.config)
        assert trainer.device.type == "cuda"

    @patch("torch.cuda.is_available")
    def test_device_detection_cpu(self, mock_cuda_available):
        """Test CPU device detection when CUDA unavailable."""
        mock_cuda_available.return_value = False

        config = ScalingConfig(num_gpus=0)  # Use 0 GPUs for CPU test
        trainer = DistributedTrainer(config)
        assert trainer.device.type == "cpu"

    @patch("torch.cuda.device_count")
    @patch("torch.cuda.is_available")
    def test_gpu_count_detection(self, mock_cuda_available, mock_device_count):
        """Test GPU count detection."""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 4

        trainer = DistributedTrainer(self.config)
        assert trainer.available_gpus == 4

    @patch("torch.cuda.is_available")
    def test_gpu_count_no_cuda(self, mock_cuda_available):
        """Test GPU count when CUDA unavailable."""
        mock_cuda_available.return_value = False

        config = ScalingConfig(num_gpus=0)  # Use 0 GPUs for this test
        trainer = DistributedTrainer(config)
        assert trainer.available_gpus == 0

    def test_validate_config_valid(self) -> None:
        """Test configuration validation with valid config."""
        result = self.trainer.validate_config()
        assert result is True

    def test_validate_config_invalid_gpus(self) -> None:
        """Test configuration validation with invalid GPU count."""
        config = ScalingConfig(num_gpus=-1)
        with pytest.raises(ScalingError, match="Invalid number of GPUs"):
            DistributedTrainer(config)

    def test_validate_config_invalid_batch_size(self) -> None:
        """Test configuration validation with invalid batch size."""
        config = ScalingConfig(batch_size=0)
        with pytest.raises(ScalingError, match="Invalid batch size"):
            DistributedTrainer(config)

    def test_validate_config_invalid_learning_rate(self) -> None:
        """Test configuration validation with invalid learning rate."""
        config = ScalingConfig(learning_rate=-0.001)
        with pytest.raises(ScalingError, match="Invalid learning rate"):
            DistributedTrainer(config)

    def test_validate_config_insufficient_gpus(self) -> None:
        """Test configuration validation with insufficient GPUs."""
        config = ScalingConfig(num_gpus=8)
        with pytest.raises(ScalingError, match="Insufficient GPUs available"):
            DistributedTrainer(config)

    @patch("torch.nn.DataParallel")
    def test_wrap_model_dataparallel(self, mock_dataparallel):
        """Test model wrapping with DataParallel."""
        mock_model = Mock()
        mock_model.to.return_value = mock_model  # Mock .to() to return self
        mock_wrapped_model = Mock()
        mock_dataparallel.return_value = mock_wrapped_model

        # Create a trainer with multiple GPUs - mock the GPU count
        with patch("torch.cuda.device_count", return_value=2):
            config = ScalingConfig(num_gpus=2)
            trainer = DistributedTrainer(config)

        result = trainer.wrap_model(mock_model)
        assert result == mock_wrapped_model
        mock_dataparallel.assert_called_once_with(mock_model)

    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.is_available")
    def test_wrap_model_distributed(
        self, mock_dist_available, mock_init_process_group
    ):
        """Test model wrapping with distributed training."""
        mock_dist_available.return_value = True
        mock_model = Mock()

        # Mock distributed model wrapping
        with patch("torch.nn.parallel.DistributedDataParallel") as mock_ddp:
            mock_wrapped_model = Mock()
            mock_ddp.return_value = mock_wrapped_model

            result = self.trainer.wrap_model(mock_model, use_distributed=True)
            assert result == mock_wrapped_model

    def test_wrap_model_single_gpu(self) -> None:
        """Test model wrapping for single GPU."""
        mock_model = Mock()
        mock_model.to.return_value = mock_model  # Mock .to() to return self
        result = self.trainer.wrap_model(mock_model, use_distributed=False)
        assert result == mock_model

    @patch("torch.cuda.amp.autocast")
    def test_mixed_precision_context(self, mock_autocast):
        """Test mixed precision context manager."""
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=None)
        mock_context.__exit__ = Mock(return_value=None)
        mock_autocast.return_value = mock_context

        with self.trainer.mixed_precision_context():
            pass

        if self.config.use_mixed_precision:
            mock_autocast.assert_called_once()

    def test_mixed_precision_context_disabled(self) -> None:
        """Test mixed precision context when disabled."""
        self.config.use_mixed_precision = False

        with patch("torch.cuda.amp.autocast") as mock_autocast:
            with self.trainer.mixed_precision_context():
                pass

            mock_autocast.assert_not_called()

    def test_calculate_effective_batch_size(self) -> None:
        """Test effective batch size calculation."""
        self.config.num_gpus = 4
        self.config.batch_size = 32

        effective_batch_size = self.trainer.calculate_effective_batch_size()
        assert effective_batch_size == 128  # 4 * 32

    def test_calculate_effective_batch_size_single_gpu(self) -> None:
        """Test effective batch size calculation for single GPU."""
        self.config.num_gpus = 1
        self.config.batch_size = 16

        effective_batch_size = self.trainer.calculate_effective_batch_size()
        assert effective_batch_size == 16

    @patch("torch.cuda.amp.GradScaler")
    def test_grad_scaler_initialization(self, mock_grad_scaler):
        """Test gradient scaler initialization."""
        mock_scaler = Mock()
        mock_grad_scaler.return_value = mock_scaler

        scaler = self.trainer.get_grad_scaler()
        assert scaler == mock_scaler

    def test_grad_scaler_no_mixed_precision(self) -> None:
        """Test gradient scaler when mixed precision disabled."""
        self.config.use_mixed_precision = False

        scaler = self.trainer.get_grad_scaler()
        assert scaler is None

    def test_sync_gradients(self) -> None:
        """Test gradient synchronization."""
        mock_model = Mock()
        mock_parameter = Mock()
        mock_parameter.grad = torch.tensor([1.0, 2.0, 3.0])
        mock_model.parameters.return_value = [mock_parameter]

        # Should not raise an exception
        self.trainer.sync_gradients(mock_model)

    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.is_initialized")
    def test_sync_gradients_distributed(
        self, mock_is_initialized, mock_all_reduce, mock_world_size
    ):
        """Test gradient synchronization in distributed mode."""
        mock_is_initialized.return_value = True
        mock_world_size.return_value = 2
        mock_model = Mock()
        mock_parameter = Mock()
        mock_parameter.grad = torch.tensor([1.0, 2.0, 3.0])
        mock_model.parameters.return_value = [mock_parameter]

        self.trainer.sync_gradients(mock_model)
        mock_all_reduce.assert_called()

    def test_scale_loss(self) -> None:
        """Test loss scaling for mixed precision."""
        mock_scaler = Mock()
        mock_scaler.scale.return_value = torch.tensor(2.0)

        loss = torch.tensor(1.0)
        scaled_loss = self.trainer.scale_loss(loss, mock_scaler)
        assert scaled_loss == torch.tensor(2.0)
        mock_scaler.scale.assert_called_once_with(loss)

    def test_scale_loss_no_scaler(self) -> None:
        """Test loss scaling without scaler."""
        loss = torch.tensor(1.0)
        scaled_loss = self.trainer.scale_loss(loss, None)
        assert scaled_loss == loss

    def test_unscale_gradients(self) -> None:
        """Test gradient unscaling."""
        mock_scaler = Mock()
        mock_model = Mock()

        self.trainer.unscale_gradients(mock_model, mock_scaler)
        mock_scaler.unscale_.assert_called_once()

    def test_unscale_gradients_no_scaler(self) -> None:
        """Test gradient unscaling without scaler."""
        mock_model = Mock()

        # Should not raise an exception
        self.trainer.unscale_gradients(mock_model, None)

    def test_update_scaler(self) -> None:
        """Test scaler update."""
        mock_scaler = Mock()

        self.trainer.update_scaler(mock_scaler)
        mock_scaler.update.assert_called_once()

    def test_update_scaler_no_scaler(self) -> None:
        """Test scaler update without scaler."""
        # Should not raise an exception
        self.trainer.update_scaler(None)

    def test_get_learning_rate(self) -> None:
        """Test learning rate retrieval."""
        lr = self.trainer.get_learning_rate()
        assert lr == self.config.learning_rate

    def test_set_learning_rate(self) -> None:
        """Test learning rate setting."""
        new_lr = 0.0005
        self.trainer.set_learning_rate(new_lr)
        assert self.config.learning_rate == new_lr

    def test_get_batch_size(self) -> None:
        """Test batch size retrieval."""
        batch_size = self.trainer.get_batch_size()
        assert batch_size == self.config.batch_size

    def test_set_batch_size(self) -> None:
        """Test batch size setting."""
        new_batch_size = 64
        self.trainer.set_batch_size(new_batch_size)
        assert self.config.batch_size == new_batch_size

    def test_get_num_gpus(self) -> None:
        """Test GPU count retrieval."""
        num_gpus = self.trainer.get_num_gpus()
        assert num_gpus == self.config.num_gpus

    def test_set_num_gpus(self) -> None:
        """Test GPU count setting."""
        new_num_gpus = 1  # Use 1 GPU which is available
        self.trainer.set_num_gpus(new_num_gpus)
        assert self.config.num_gpus == new_num_gpus

    def test_is_mixed_precision_enabled(self) -> None:
        """Test mixed precision status check."""
        enabled = self.trainer.is_mixed_precision_enabled()
        assert enabled == self.config.use_mixed_precision

    def test_enable_mixed_precision(self) -> None:
        """Test enabling mixed precision."""
        self.trainer.enable_mixed_precision()
        assert self.config.use_mixed_precision is True

    def test_disable_mixed_precision(self) -> None:
        """Test disabling mixed precision."""
        self.trainer.disable_mixed_precision()
        assert self.config.use_mixed_precision is False

    def test_get_device_info(self) -> None:
        """Test device information retrieval."""
        info = self.trainer.get_device_info()
        assert "device_type" in info
        assert "device_count" in info
        assert "available_gpus" in info

    def test_get_memory_info(self) -> None:
        """Test memory information retrieval."""
        info = self.trainer.get_memory_info()
        assert "total_memory" in info
        assert "available_memory" in info
        assert "memory_usage" in info

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.max_memory_allocated")
    def test_get_gpu_memory_info(self, mock_max_memory, mock_allocated):
        """Test GPU memory information retrieval."""
        mock_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_max_memory.return_value = 2048 * 1024 * 1024  # 2GB

        info = self.trainer.get_gpu_memory_info()
        assert "allocated_memory" in info
        assert "max_allocated_memory" in info
        assert "memory_usage_percent" in info

    def test_get_gpu_memory_info_no_cuda(self) -> None:
        """Test GPU memory info when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            info = self.trainer.get_gpu_memory_info()
            assert info == {}

    def test_clear_gpu_cache(self) -> None:
        """Test GPU cache clearing."""
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            self.trainer.clear_gpu_cache()
            mock_empty_cache.assert_called_once()

    def test_clear_gpu_cache_no_cuda(self) -> None:
        """Test GPU cache clearing when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                self.trainer.clear_gpu_cache()
                mock_empty_cache.assert_not_called()

    def test_optimize_for_inference(self) -> None:
        """Test model optimization for inference."""
        mock_model = Mock()

        with patch("torch.jit.optimize_for_inference") as mock_optimize:
            mock_optimized_model = Mock()
            mock_optimize.return_value = mock_optimized_model

            result = self.trainer.optimize_for_inference(mock_model)
            assert result == mock_optimized_model
            mock_optimize.assert_called_once_with(mock_model)

    def test_optimize_for_inference_no_jit(self) -> None:
        """Test model optimization when JIT unavailable."""
        mock_model = Mock()

        with patch(
            "torch.jit.optimize_for_inference",
            side_effect=Exception("JIT not available"),
        ):
            result = self.trainer.optimize_for_inference(mock_model)
            assert result == mock_model

    def test_get_scaling_stats(self) -> None:
        """Test scaling statistics retrieval."""
        stats = self.trainer.get_scaling_stats()
        assert "num_gpus" in stats
        assert "batch_size" in stats
        assert "effective_batch_size" in stats
        assert "learning_rate" in stats
        assert "mixed_precision" in stats

    def test_scaling_error_initialization(self) -> None:
        """Test ScalingError initialization."""
        error = ScalingError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_scaling_error_with_details(self) -> None:
        """Test ScalingError with additional details."""
        error = ScalingError(
            "Test error", details={"gpu_count": 2, "batch_size": 32}
        )
        assert str(error) == "Test error"
        assert hasattr(error, "details")
        assert error.details["gpu_count"] == 2
        assert error.details["batch_size"] == 32
