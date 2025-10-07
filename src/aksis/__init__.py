"""
Aksis - AI Chatbot/LLM from Scratch

A complete transformer-based language model built from scratch using Python and PyTorch.
"""

__version__ = "0.1.0"
__author__ = "Aksis Team"
__email__ = "team@aksis.ai"

# Import main components for easy access
from aksis.utils.device import get_device
from aksis.utils.logging import setup_logging

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "get_device",
    "setup_logging",
]
