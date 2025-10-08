import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from aksis.utils.device import get_device

logger = logging.getLogger(__name__)


class OptimizationConfig(BaseModel):
    """Configuration for model optimization."""

    quantization_type: str = Field(
        "dynamic", description="Type of quantization (dynamic, static, qat)"
    )
    target_device: str = Field(
        "cpu", description="Target device for optimization (cpu, cuda, mobile)"
    )
    onnx_export: bool = Field(False, description="Export model to ONNX format")
    onnx_opset_version: int = Field(
        11, ge=7, le=17, description="ONNX opset version"
    )
    optimize_for_inference: bool = Field(
        True, description="Optimize model for inference"
    )
    preserve_accuracy: bool = Field(
        True, description="Preserve model accuracy during optimization"
    )
    compression_ratio: float = Field(
        0.5, ge=0.1, le=0.9, description="Target compression ratio"
    )

    model_config = {"protected_namespaces": ()}


class OptimizationError(Exception):
    """Custom exception for optimization-related errors."""

    pass


class ModelOptimizer:
    """Manages model optimization including quantization and ONNX export."""

    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        """
        Initialize ModelOptimizer.

        Args:
            config: Optimization configuration. If None, uses default config.
        """
        self.config = config or OptimizationConfig()
        self.device = get_device()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate optimization configuration."""
        if self.config.quantization_type not in ["dynamic", "static", "qat"]:
            raise OptimizationError(
                f"Invalid quantization type: {self.config.quantization_type}"
            )

        if self.config.target_device not in ["cpu", "cuda", "mobile"]:
            raise OptimizationError(
                f"Invalid target device: {self.config.target_device}"
            )

        if (
            self.config.compression_ratio <= 0
            or self.config.compression_ratio >= 1
        ):
            raise OptimizationError(
                "Compression ratio must be between 0 and 1"
            )

    def quantize_model(
        self,
        model: nn.Module,
        quantization_type: Optional[str] = None,
        target_device: Optional[str] = None,
    ) -> nn.Module:
        """
        Quantize a PyTorch model.

        Args:
            model: The PyTorch model to quantize.
            quantization_type: Type of quantization. If None, uses config.
            target_device: Target device. If None, uses config.

        Returns:
            The quantized model.
        """
        qtype = quantization_type or self.config.quantization_type
        device = target_device or self.config.target_device

        logger.info(f"Quantizing model with {qtype} quantization for {device}")

        try:
            if qtype == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
            elif qtype == "static":
                # For static quantization, we need to prepare the model first
                model.eval()
                model.qconfig = torch.quantization.get_default_qconfig(
                    "fbgemm"
                )
                prepared_model = torch.quantization.prepare(model)
                # Note: In practice, you'd need calibration data for static quantization
                quantized_model = torch.quantization.convert(prepared_model)
            elif qtype == "qat":
                # Quantization Aware Training - model should already be prepared
                quantized_model = torch.quantization.convert(model)
            else:
                raise OptimizationError(
                    f"Unsupported quantization type: {qtype}"
                )

            logger.info(
                f"Model quantized successfully with {qtype} quantization"
            )
            return quantized_model

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise OptimizationError(f"Quantization failed: {e}")

    def export_to_onnx(
        self,
        model: nn.Module,
        output_path: Union[str, Path],
        input_shape: tuple = (1, 512),
        opset_version: Optional[int] = None,
    ) -> bool:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: The PyTorch model to export.
            output_path: Path to save the ONNX model.
            input_shape: Input shape for the model.
            opset_version: ONNX opset version. If None, uses config.

        Returns:
            True if export successful, False otherwise.
        """
        opset = opset_version or self.config.onnx_opset_version
        output_path = Path(output_path)

        logger.info(f"Exporting model to ONNX format at {output_path}")

        try:
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(input_shape)
            if self.device.type == "cuda":
                dummy_input = dummy_input.to(self.device)
                model = model.to(self.device)

            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )

            logger.info(
                f"Model exported to ONNX successfully at {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for inference.

        Args:
            model: The PyTorch model to optimize.

        Returns:
            The optimized model.
        """
        logger.info("Optimizing model for inference")

        try:
            model.eval()

            # Use TorchScript JIT compilation
            if self.device.type == "cuda":
                model = model.to(self.device)

            # Try to script the model
            try:
                scripted_model = torch.jit.script(model)
                logger.info("Model optimized with TorchScript JIT compilation")
                return scripted_model
            except Exception as e:
                logger.warning(f"TorchScript JIT compilation failed: {e}")
                # Fallback to tracing
                dummy_input = torch.randn(1, 512)
                if self.device.type == "cuda":
                    dummy_input = dummy_input.to(self.device)

                traced_model = torch.jit.trace(model, dummy_input)
                logger.info("Model optimized with TorchScript tracing")
                return traced_model

        except Exception as e:
            logger.error(f"Inference optimization failed: {e}")
            raise OptimizationError(f"Inference optimization failed: {e}")

    def get_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get model size information.

        Args:
            model: The PyTorch model.

        Returns:
            Dictionary containing size information.
        """
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            # Estimate memory usage
            param_size = sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
            buffer_size = sum(
                b.numel() * b.element_size() for b in model.buffers()
            )
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)

            size_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": model_size_mb,
                "model_size_gb": model_size_mb / 1024,
            }

            logger.info(
                f"Model size: {total_params:,} parameters, {model_size_mb:.2f} MB"
            )
            return size_info

        except Exception as e:
            logger.error(f"Failed to get model size: {e}")
            return {"error": str(e)}

    def compare_models(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        test_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compare original and optimized models.

        Args:
            original_model: The original model.
            optimized_model: The optimized model.
            test_input: Test input for comparison. If None, creates dummy input.

        Returns:
            Dictionary containing comparison results.
        """
        logger.info("Comparing original and optimized models")

        try:
            # Get model sizes
            original_size = self.get_model_size(original_model)
            optimized_size = self.get_model_size(optimized_model)

            # Calculate compression ratio
            if (
                "model_size_mb" in original_size
                and "model_size_mb" in optimized_size
            ):
                compression_ratio = (
                    optimized_size["model_size_mb"]
                    / original_size["model_size_mb"]
                )
            else:
                compression_ratio = 0.0

            # Test inference speed (basic comparison)
            if test_input is None:
                test_input = torch.randn(1, 512)
                if self.device.type == "cuda":
                    test_input = test_input.to(self.device)

            # Time original model
            original_model.eval()
            start_time = (
                torch.cuda.Event(enable_timing=True)
                if self.device.type == "cuda"
                else None
            )
            end_time = (
                torch.cuda.Event(enable_timing=True)
                if self.device.type == "cuda"
                else None
            )

            if start_time:
                start_time.record()
            else:
                import time

                start_time = time.time()

            with torch.no_grad():
                _ = original_model(test_input)

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                original_time = (
                    start_time.elapsed_time(end_time) / 1000.0
                )  # Convert to seconds
            else:
                original_time = time.time() - start_time

            # Time optimized model
            optimized_model.eval()
            if start_time:
                start_time.record()
            else:
                start_time = time.time()

            with torch.no_grad():
                _ = optimized_model(test_input)

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                optimized_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                optimized_time = time.time() - start_time

            comparison = {
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": compression_ratio,
                "original_inference_time": original_time,
                "optimized_inference_time": optimized_time,
                "speedup_ratio": original_time / optimized_time
                if optimized_time > 0
                else 0,
            }

            logger.info(
                f"Model comparison completed. Compression ratio: {compression_ratio:.2f}"
            )
            return comparison

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {"error": str(e)}

    def optimize_model(
        self,
        model: nn.Module,
        output_path: Optional[Union[str, Path]] = None,
        save_onnx: bool = False,
    ) -> Dict[str, Any]:
        """
        Comprehensive model optimization.

        Args:
            model: The PyTorch model to optimize.
            output_path: Path to save optimized model. If None, doesn't save.
            save_onnx: Whether to save ONNX version.

        Returns:
            Dictionary containing optimization results.
        """
        logger.info("Starting comprehensive model optimization")

        try:
            results = {}

            # Get original model size
            original_size = self.get_model_size(model)
            results["original_size"] = original_size

            # Optimize for inference
            if self.config.optimize_for_inference:
                optimized_model = self.optimize_for_inference(model)
                results["inference_optimized"] = True
            else:
                optimized_model = model
                results["inference_optimized"] = False

            # Quantize if requested
            if self.config.quantization_type != "none":
                quantized_model = self.quantize_model(optimized_model)
                results["quantized"] = True
                final_model = quantized_model
            else:
                results["quantized"] = False
                final_model = optimized_model

            # Get final model size
            final_size = self.get_model_size(final_model)
            results["final_size"] = final_size

            # Calculate compression ratio
            if (
                "model_size_mb" in original_size
                and "model_size_mb" in final_size
            ):
                compression_ratio = (
                    final_size["model_size_mb"]
                    / original_size["model_size_mb"]
                )
                results["compression_ratio"] = compression_ratio

            # Save optimized model
            if output_path:
                output_path = Path(output_path)
                torch.save(final_model.state_dict(), output_path)
                results["saved_path"] = str(output_path)
                logger.info(f"Optimized model saved to {output_path}")

            # Export to ONNX if requested
            if save_onnx or self.config.onnx_export:
                onnx_path = (
                    output_path.with_suffix(".onnx")
                    if output_path
                    else Path("optimized_model.onnx")
                )
                onnx_success = self.export_to_onnx(final_model, onnx_path)
                results["onnx_exported"] = onnx_success
                if onnx_success:
                    results["onnx_path"] = str(onnx_path)

            logger.info("Model optimization completed successfully")
            return results

        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise OptimizationError(f"Model optimization failed: {e}")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization configuration and stats."""
        return {
            "quantization_type": self.config.quantization_type,
            "target_device": self.config.target_device,
            "onnx_export": self.config.onnx_export,
            "onnx_opset_version": self.config.onnx_opset_version,
            "optimize_for_inference": self.config.optimize_for_inference,
            "preserve_accuracy": self.config.preserve_accuracy,
            "compression_ratio": self.config.compression_ratio,
        }
