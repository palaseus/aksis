"""Tests for utility modules."""

import pytest
import torch
import logging
import tempfile
import os

# No additional imports needed

from aksis.utils.device import get_device, get_device_count, is_cuda_available
from aksis.utils.logging import setup_logging, get_logger


class TestDeviceUtils:
    """Test cases for device utilities."""

    def test_get_device_default(self) -> None:
        """Test get_device with default parameters."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]

    def test_get_device_cpu(self) -> None:
        """Test get_device with CPU specification."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_cuda_available(self) -> None:
        """Test get_device with CUDA when available."""
        if torch.cuda.is_available():
            device = get_device("cuda")
            assert device.type == "cuda"
        else:
            # Should raise RuntimeError if CUDA is requested but not available
            with pytest.raises(RuntimeError, match="CUDA requested but not available"):
                get_device("cuda")

    def test_get_device_cuda_unavailable(self) -> None:
        """Test get_device behavior when CUDA is unavailable."""
        # This test assumes CUDA might not be available
        # The function should handle this gracefully
        device = get_device()
        assert device.type in ["cpu", "cuda"]

    def test_get_device_count(self) -> None:
        """Test get_device_count function."""
        count = get_device_count()
        assert isinstance(count, int)
        assert count >= 0

        if torch.cuda.is_available():
            assert count == torch.cuda.device_count()
        else:
            assert count == 0

    def test_is_cuda_available(self) -> None:
        """Test is_cuda_available function."""
        available = is_cuda_available()
        assert isinstance(available, bool)
        assert available == torch.cuda.is_available()


class TestLoggingUtils:
    """Test cases for logging utilities."""

    def test_setup_logging_default(self) -> None:
        """Test setup_logging with default parameters."""
        # This should not raise any exceptions
        setup_logging()

        # Check that root logger is configured
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0

    def test_setup_logging_custom_level(self) -> None:
        """Test setup_logging with custom level."""
        setup_logging(level="DEBUG")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_custom_format(self) -> None:
        """Test setup_logging with custom format."""
        custom_format = "%(levelname)s: %(message)s"
        setup_logging(format_string=custom_format)

        root_logger = logging.getLogger()
        # Check that a handler with the custom format exists
        for handler in root_logger.handlers:
            if hasattr(handler, "formatter") and handler.formatter:
                assert handler.formatter._fmt == custom_format

    def test_setup_logging_with_file(self) -> None:
        """Test setup_logging with log file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_file = f.name

        try:
            setup_logging(log_file=log_file)

            # Check that file handler was added
            root_logger = logging.getLogger()
            file_handlers = [
                h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) > 0

            # Check that log file exists
            assert os.path.exists(log_file)

        finally:
            # Clean up
            if os.path.exists(log_file):
                os.unlink(log_file)

    def test_get_logger(self) -> None:
        """Test get_logger function."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_logging_integration(self) -> None:
        """Test logging integration with setup_logging."""
        setup_logging(level="INFO")

        logger = get_logger("test_integration")

        # This should not raise any exceptions
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")

    def test_logging_levels(self) -> None:
        """Test different logging levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            setup_logging(level=level)
            root_logger = logging.getLogger()
            expected_level = getattr(logging, level)
            assert root_logger.level == expected_level
