"""Device utilities for CUDA/CPU detection and management."""

import logging
from typing import Optional

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
