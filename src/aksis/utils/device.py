"""Device utilities for CUDA/CPU detection and management."""

import logging
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for tensor operations.

    Args:
        device: Optional device specification. If None, auto-detect.

    Returns:
        torch.device: The device to use for computations.

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")

    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    return torch.device(device)


def get_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        int: Number of CUDA devices available.
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def get_gpu_memory_info(device: torch.device) -> Dict[str, Any]:
    """
    Get GPU memory information.

    Args:
        device: The GPU device to get memory info for.

    Returns:
        Dictionary containing GPU memory information.
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        return {"error": "CUDA not available or not a CUDA device"}
    
    try:
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        memory_total = torch.cuda.get_device_properties(device).total_memory
        
        return {
            "gpu_memory_allocated": memory_allocated,
            "gpu_memory_reserved": memory_reserved,
            "gpu_memory_total": memory_total,
            "gpu_memory_free": memory_total - memory_reserved,
            "gpu_memory_used": memory_allocated,
            "gpu_memory_utilization": (memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {e}")
        return {"error": str(e)}


def get_system_memory_info() -> Dict[str, Any]:
    """
    Get system memory information.

    Returns:
        Dictionary containing system memory information.
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "system_memory_total": memory.total,
            "system_memory_available": memory.available,
            "system_memory_used": memory.used,
            "system_memory_percent": memory.percent,
            "system_swap_total": swap.total,
            "system_swap_used": swap.used,
            "system_swap_percent": swap.percent,
        }
    except ImportError:
        logger.warning("psutil not available for system memory info")
        return {"error": "psutil not available"}
    except Exception as e:
        logger.error(f"Error getting system memory info: {e}")
        return {"error": str(e)}
