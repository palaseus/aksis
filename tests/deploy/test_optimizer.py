"""Tests for model optimization utilities."""

import pytest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn

from aksis.deploy.optimizer import ModelOptimizer, OptimizationConfig, OptimizationError


class TestModelOptimizer:
    """Test model optimization functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = OptimizationConfig(
            quantization_type="dynamic",
            target_device="cpu",
            onnx_export=True,
            onnx_opset_version=11,
            optimize_for_inference=True,
            preserve_accuracy=True,
            compression_ratio=0.5
        )
        self.optimizer = ModelOptimizer(self.config)

    def test_optimization_config_initialization(self) -> None:
        """Test optimization configuration initialization."""
        config = OptimizationConfig(
            quantization_type="static",
            target_device="cuda",
            onnx_export=False,
            onnx_opset_version=12,
            optimize_for_inference=False,
            preserve_accuracy=False,
            compression_ratio=0.3
        )
        assert config.quantization_type == "static"
        assert config.target_device == "cuda"
        assert config.onnx_export is False
        assert config.onnx_opset_version == 12
        assert config.optimize_for_inference is False
        assert config.preserve_accuracy is False
        assert config.compression_ratio == 0.3

    def test_optimization_config_defaults(self) -> None:
        """Test optimization configuration with default values."""
        config = OptimizationConfig()
        assert config.quantization_type == "dynamic"
        assert config.target_device == "cpu"
        assert config.onnx_export is False
        assert config.onnx_opset_version == 11
        assert config.optimize_for_inference is True
        assert config.preserve_accuracy is True
        assert config.compression_ratio == 0.5

    def test_model_optimizer_initialization(self) -> None:
        """Test model optimizer initialization."""
        assert isinstance(self.optimizer, ModelOptimizer)
        assert self.optimizer.config == self.config

    def test_validate_config_invalid_quantization(self) -> None:
        """Test configuration validation with invalid quantization type."""
        config = OptimizationConfig(quantization_type="invalid")
        with pytest.raises(OptimizationError, match="Invalid quantization type"):
            ModelOptimizer(config)

    def test_validate_config_invalid_target_device(self) -> None:
        """Test configuration validation with invalid target device."""
        config = OptimizationConfig(target_device="invalid")
        with pytest.raises(OptimizationError, match="Invalid target device"):
            ModelOptimizer(config)

    def test_validate_config_invalid_compression_ratio(self) -> None:
        """Test configuration validation with invalid compression ratio."""
        with pytest.raises(Exception):  # Pydantic validation error
            OptimizationConfig(compression_ratio=1.5)

    @patch('torch.quantization.quantize_dynamic')
    def test_quantize_model_dynamic(self, mock_quantize_dynamic):
        """Test dynamic quantization."""
        mock_model = Mock()
        mock_quantized_model = Mock()
        mock_quantize_dynamic.return_value = mock_quantized_model
        
        result = self.optimizer.quantize_model(mock_model, "dynamic")
        assert result == mock_quantized_model
        mock_quantize_dynamic.assert_called_once()

    @patch('torch.quantization.get_default_qconfig')
    @patch('torch.quantization.prepare')
    @patch('torch.quantization.convert')
    def test_quantize_model_static(self, mock_convert, mock_prepare, mock_qconfig):
        """Test static quantization."""
        mock_model = Mock()
        mock_prepared_model = Mock()
        mock_quantized_model = Mock()
        
        mock_qconfig.return_value = Mock()
        mock_prepare.return_value = mock_prepared_model
        mock_convert.return_value = mock_quantized_model
        
        result = self.optimizer.quantize_model(mock_model, "static")
        assert result == mock_quantized_model
        mock_prepare.assert_called_once()
        mock_convert.assert_called_once()

    @patch('torch.quantization.convert')
    def test_quantize_model_qat(self, mock_convert):
        """Test quantization aware training."""
        mock_model = Mock()
        mock_quantized_model = Mock()
        mock_convert.return_value = mock_quantized_model
        
        result = self.optimizer.quantize_model(mock_model, "qat")
        assert result == mock_quantized_model
        mock_convert.assert_called_once()

    def test_quantize_model_invalid_type(self) -> None:
        """Test quantization with invalid type."""
        mock_model = Mock()
        with pytest.raises(OptimizationError, match="Unsupported quantization type"):
            self.optimizer.quantize_model(mock_model, "invalid")

    @patch('torch.onnx.export')
    def test_export_to_onnx(self, mock_export):
        """Test ONNX export."""
        mock_model = Mock()
        mock_input = torch.randn(1, 512)
        
        result = self.optimizer.export_to_onnx(mock_model, "test_model.onnx", input_shape=(1, 512))
        assert result is True
        mock_export.assert_called_once()

    @patch('torch.onnx.export')
    def test_export_to_onnx_failure(self, mock_export):
        """Test ONNX export failure."""
        mock_model = Mock()
        mock_export.side_effect = Exception("Export failed")
        
        result = self.optimizer.export_to_onnx(mock_model, "test_model.onnx")
        assert result is False

    @patch('torch.jit.script')
    def test_optimize_for_inference_script(self, mock_script):
        """Test inference optimization with TorchScript."""
        mock_model = Mock()
        mock_scripted_model = Mock()
        mock_script.return_value = mock_scripted_model
        
        result = self.optimizer.optimize_for_inference(mock_model)
        assert result == mock_scripted_model
        mock_script.assert_called_once()

    @patch('torch.jit.script')
    @patch('torch.jit.trace')
    def test_optimize_for_inference_trace_fallback(self, mock_trace, mock_script):
        """Test inference optimization with tracing fallback."""
        mock_model = Mock()
        mock_scripted_model = Mock()
        mock_traced_model = Mock()
        
        mock_script.side_effect = Exception("Script failed")
        mock_trace.return_value = mock_traced_model
        
        result = self.optimizer.optimize_for_inference(mock_model)
        assert result == mock_traced_model
        mock_trace.assert_called_once()

    def test_get_model_size(self) -> None:
        """Test model size calculation."""
        mock_model = Mock()
        mock_parameter = Mock()
        mock_parameter.numel.return_value = 1000
        mock_parameter.element_size.return_value = 4  # 4 bytes for float32
        mock_model.parameters.return_value = [mock_parameter]
        mock_model.buffers.return_value = []
        
        size_info = self.optimizer.get_model_size(mock_model)
        assert "total_parameters" in size_info
        assert "model_size_mb" in size_info
        assert size_info["total_parameters"] > 0

    def test_compare_models(self) -> None:
        """Test model comparison."""
        mock_original = Mock()
        mock_optimized = Mock()
        
        # Mock the get_model_size method
        with patch.object(self.optimizer, 'get_model_size') as mock_get_size:
            mock_get_size.side_effect = [
                {"model_size_mb": 100.0},  # original
                {"model_size_mb": 50.0}    # optimized
            ]
            
            result = self.optimizer.compare_models(mock_original, mock_optimized)
            assert "compression_ratio" in result
            assert result["compression_ratio"] == 0.5

    @patch('torch.save')
    @patch.object(ModelOptimizer, 'get_model_size')
    @patch.object(ModelOptimizer, 'optimize_for_inference')
    @patch.object(ModelOptimizer, 'quantize_model')
    @patch.object(ModelOptimizer, 'export_to_onnx')
    def test_optimize_model(self, mock_export, mock_quantize, mock_optimize, mock_get_size, mock_save):
        """Test comprehensive model optimization."""
        mock_model = Mock()
        mock_optimized_model = Mock()
        mock_quantized_model = Mock()
        
        mock_get_size.side_effect = [
            {"model_size_mb": 100.0},  # original
            {"model_size_mb": 50.0}    # final
        ]
        mock_optimize.return_value = mock_optimized_model
        mock_quantize.return_value = mock_quantized_model
        mock_export.return_value = True
        
        result = self.optimizer.optimize_model(mock_model, "output.pt", save_onnx=True)
        
        assert "original_size" in result
        assert "final_size" in result
        assert "compression_ratio" in result
        assert result["compression_ratio"] == 0.5
        assert result["quantized"] is True
        assert result["onnx_exported"] is True

    def test_get_optimization_stats(self) -> None:
        """Test optimization statistics retrieval."""
        stats = self.optimizer.get_optimization_stats()
        assert "quantization_type" in stats
        assert "target_device" in stats
        assert "onnx_export" in stats
        assert "onnx_opset_version" in stats
        assert "optimize_for_inference" in stats
        assert "preserve_accuracy" in stats
        assert "compression_ratio" in stats

    def test_optimization_error_initialization(self) -> None:
        """Test OptimizationError initialization."""
        error = OptimizationError("Test error")
        assert str(error) == "Test error"