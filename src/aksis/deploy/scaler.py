"""Distributed training and scaling functionality."""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for distributed training and scaling."""
    
    num_gpus: int = 1
    batch_size: int = 16
    learning_rate: float = 0.001
    use_mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    weight_decay: float = 0.01


class ScalingError(Exception):
    """Exception raised for scaling-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize scaling error.
        
        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(message)
        self.details = details or {}


class DistributedTrainer:
    """Distributed training and scaling manager."""
    
    def __init__(self, config: Optional[ScalingConfig] = None) -> None:
        """Initialize distributed trainer.
        
        Args:
            config: Scaling configuration. If None, uses default config.
        """
        self.config = config or ScalingConfig()
        self.device = self._get_device()
        self.available_gpus = self._get_available_gpus()
        self._validate_config()
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for training.
        
        Returns:
            PyTorch device.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _get_available_gpus(self) -> int:
        """Get the number of available GPUs.
        
        Returns:
            Number of available GPUs.
        """
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    
    def _validate_config(self) -> None:
        """Validate scaling configuration."""
        if self.config.num_gpus < 0:
            raise ScalingError("Invalid number of GPUs")
        
        if self.config.batch_size <= 0:
            raise ScalingError("Invalid batch size")
        
        if self.config.learning_rate <= 0:
            raise ScalingError("Invalid learning rate")
        
        if self.config.num_gpus > self.available_gpus:
            raise ScalingError(
                f"Insufficient GPUs available. Requested: {self.config.num_gpus}, "
                f"Available: {self.available_gpus}"
            )
    
    def validate_config(self) -> bool:
        """Validate scaling configuration.
        
        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            self._validate_config()
            return True
        except ScalingError:
            return False
    
    def wrap_model(
        self,
        model: nn.Module,
        use_distributed: bool = False
    ) -> nn.Module:
        """Wrap model for distributed training.
        
        Args:
            model: PyTorch model to wrap.
            use_distributed: Whether to use distributed training.
            
        Returns:
            Wrapped model.
        """
        if use_distributed and torch.distributed.is_available():
            # Use DistributedDataParallel for true distributed training
            model = model.to(self.device)
            model = nn.parallel.DistributedDataParallel(model)
            logger.info("Model wrapped with DistributedDataParallel")
        elif self.config.num_gpus > 1:
            # Use DataParallel for multi-GPU training on single node
            model = model.to(self.device)
            model = nn.DataParallel(model)
            logger.info("Model wrapped with DataParallel")
        else:
            # Single GPU or CPU
            model = model.to(self.device)
            logger.info("Model moved to device")
        
        return model
    
    def mixed_precision_context(self):
        """Get mixed precision context manager.
        
        Returns:
            Mixed precision context manager.
        """
        if self.config.use_mixed_precision and torch.cuda.is_available():
            return torch.cuda.amp.autocast()
        else:
            # Return a dummy context manager that does nothing
            from contextlib import nullcontext
            return nullcontext()
    
    def get_grad_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        """Get gradient scaler for mixed precision training.
        
        Returns:
            Gradient scaler if mixed precision enabled, None otherwise.
        """
        if self.config.use_mixed_precision and torch.cuda.is_available():
            return torch.cuda.amp.GradScaler()
        else:
            return None
    
    def calculate_effective_batch_size(self) -> int:
        """Calculate effective batch size across all GPUs.
        
        Returns:
            Effective batch size.
        """
        return self.config.batch_size * self.config.num_gpus * self.config.gradient_accumulation_steps
    
    def sync_gradients(self, model: nn.Module) -> None:
        """Synchronize gradients across GPUs.
        
        Args:
            model: PyTorch model.
        """
        if torch.distributed.is_initialized():
            # Synchronize gradients in distributed training
            for param in model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                    param.grad.data /= torch.distributed.get_world_size()
    
    def scale_loss(self, loss: torch.Tensor, scaler: Optional[torch.cuda.amp.GradScaler]) -> torch.Tensor:
        """Scale loss for mixed precision training.
        
        Args:
            loss: Loss tensor.
            scaler: Gradient scaler.
            
        Returns:
            Scaled loss tensor.
        """
        if scaler is not None:
            return scaler.scale(loss)
        else:
            return loss
    
    def unscale_gradients(self, model: nn.Module, scaler: Optional[torch.cuda.amp.GradScaler]) -> None:
        """Unscale gradients for mixed precision training.
        
        Args:
            model: PyTorch model.
            scaler: Gradient scaler.
        """
        if scaler is not None:
            scaler.unscale_(model.parameters())
    
    def update_scaler(self, scaler: Optional[torch.cuda.amp.GradScaler]) -> None:
        """Update gradient scaler.
        
        Args:
            scaler: Gradient scaler.
        """
        if scaler is not None:
            scaler.update()
    
    def get_learning_rate(self) -> float:
        """Get current learning rate.
        
        Returns:
            Current learning rate.
        """
        return self.config.learning_rate
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate.
        
        Args:
            learning_rate: New learning rate.
        """
        if learning_rate <= 0:
            raise ScalingError("Learning rate must be positive")
        self.config.learning_rate = learning_rate
    
    def get_batch_size(self) -> int:
        """Get current batch size.
        
        Returns:
            Current batch size.
        """
        return self.config.batch_size
    
    def set_batch_size(self, batch_size: int) -> None:
        """Set batch size.
        
        Args:
            batch_size: New batch size.
        """
        if batch_size <= 0:
            raise ScalingError("Batch size must be positive")
        self.config.batch_size = batch_size
    
    def get_num_gpus(self) -> int:
        """Get number of GPUs.
        
        Returns:
            Number of GPUs.
        """
        return self.config.num_gpus
    
    def set_num_gpus(self, num_gpus: int) -> None:
        """Set number of GPUs.
        
        Args:
            num_gpus: New number of GPUs.
        """
        if num_gpus < 0:
            raise ScalingError("Number of GPUs must be non-negative")
        if num_gpus > self.available_gpus:
            raise ScalingError(
                f"Insufficient GPUs available. Requested: {num_gpus}, "
                f"Available: {self.available_gpus}"
            )
        self.config.num_gpus = num_gpus
    
    def is_mixed_precision_enabled(self) -> bool:
        """Check if mixed precision is enabled.
        
        Returns:
            True if mixed precision is enabled, False otherwise.
        """
        return self.config.use_mixed_precision
    
    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training."""
        self.config.use_mixed_precision = True
    
    def disable_mixed_precision(self) -> None:
        """Disable mixed precision training."""
        self.config.use_mixed_precision = False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information.
        
        Returns:
            Dictionary containing device information.
        """
        info = {
            "device_type": self.device.type,
            "device_count": self.available_gpus,
            "available_gpus": self.available_gpus
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
        
        return info
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information.
        
        Returns:
            Dictionary containing memory information.
        """
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            available_memory = total_memory - allocated_memory
            
            return {
                "total_memory": total_memory,
                "allocated_memory": allocated_memory,
                "available_memory": available_memory,
                "memory_usage": allocated_memory / total_memory
            }
        else:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_memory": memory.total,
                "available_memory": memory.available,
                "memory_usage": memory.percent / 100
            }
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information.
        
        Returns:
            Dictionary containing GPU memory information.
        """
        if not torch.cuda.is_available():
            return {}
        
        allocated_memory = torch.cuda.memory_allocated()
        max_allocated_memory = torch.cuda.max_memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "allocated_memory": allocated_memory,
            "max_allocated_memory": max_allocated_memory,
            "total_memory": total_memory,
            "memory_usage_percent": (allocated_memory / total_memory) * 100
        }
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: PyTorch model to optimize.
            
        Returns:
            Optimized model.
        """
        try:
            # Use JIT optimization if available
            model = torch.jit.optimize_for_inference(model)
            logger.info("Model optimized for inference with JIT")
        except Exception as e:
            logger.warning(f"JIT optimization failed: {e}")
        
        return model
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics.
        
        Returns:
            Dictionary containing scaling statistics.
        """
        return {
            "num_gpus": self.config.num_gpus,
            "batch_size": self.config.batch_size,
            "effective_batch_size": self.calculate_effective_batch_size(),
            "learning_rate": self.config.learning_rate,
            "mixed_precision": self.config.use_mixed_precision,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "max_grad_norm": self.config.max_grad_norm,
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay
        }
