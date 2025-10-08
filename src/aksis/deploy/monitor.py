import logging
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import psutil
import torch
from pydantic import BaseModel, Field

from aksis.utils.device import (
    get_device,
    get_gpu_memory_info,
    get_system_memory_info,
)

logger = logging.getLogger(__name__)


class MonitoringConfig(BaseModel):
    """Configuration for performance monitoring."""

    collect_interval: float = Field(
        1.0,
        ge=0.1,
        description="Interval in seconds between metric collections",
    )
    enable_gpu_monitoring: bool = Field(
        True, description="Enable GPU monitoring"
    )
    enable_cpu_monitoring: bool = Field(
        True, description="Enable CPU monitoring"
    )
    enable_memory_monitoring: bool = Field(
        True, description="Enable memory monitoring"
    )
    enable_disk_monitoring: bool = Field(
        False, description="Enable disk monitoring"
    )
    enable_network_monitoring: bool = Field(
        False, description="Enable network monitoring"
    )
    log_file: Optional[Union[str, Path]] = Field(
        None, description="Path to log file for metrics"
    )
    max_log_size: int = Field(
        100 * 1024 * 1024, description="Maximum log file size in bytes"
    )
    backup_count: int = Field(
        5, description="Number of backup log files to keep"
    )

    model_config = {"protected_namespaces": ()}


class MonitoringError(Exception):
    """Custom exception for monitoring-related errors."""

    pass


class PerformanceMonitor:
    """Monitors system and model performance metrics."""

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        """
        Initialize PerformanceMonitor.

        Args:
            config: Monitoring configuration. If None, uses default config.
        """
        self.config = config or MonitoringConfig()
        self.device = get_device()
        self.start_time = time.time()
        self.metrics_history: List[Dict[str, Any]] = []
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate monitoring configuration."""
        if self.config.collect_interval <= 0:
            raise MonitoringError("Collection interval must be positive")

        if self.config.max_log_size <= 0:
            raise MonitoringError("Maximum log size must be positive")

        if self.config.backup_count < 0:
            raise MonitoringError("Backup count cannot be negative")

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.

        Returns:
            Dictionary containing system metrics.
        """
        metrics = {
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
        }

        try:
            if self.config.enable_cpu_monitoring:
                metrics.update(self._get_cpu_metrics())

            if self.config.enable_memory_monitoring:
                metrics.update(self._get_memory_metrics())

            if self.config.enable_disk_monitoring:
                metrics.update(self._get_disk_metrics())

            if self.config.enable_network_monitoring:
                metrics.update(self._get_network_metrics())

            if self.config.enable_gpu_monitoring and torch.cuda.is_available():
                metrics.update(self._get_gpu_metrics())

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            metrics["error"] = str(e)

        return metrics

    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict()
            if psutil.cpu_freq()
            else None,
            "load_avg": psutil.getloadavg()
            if hasattr(psutil, "getloadavg")
            else None,
        }

    def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory metrics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_percent": swap.percent,
        }

    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk metrics."""
        disk_usage = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()

        metrics = {
            "disk_total": disk_usage.total,
            "disk_used": disk_usage.used,
            "disk_free": disk_usage.free,
            "disk_percent": (disk_usage.used / disk_usage.total) * 100,
        }

        if disk_io:
            metrics.update(
                {
                    "disk_read_bytes": disk_io.read_bytes,
                    "disk_write_bytes": disk_io.write_bytes,
                    "disk_read_count": disk_io.read_count,
                    "disk_write_count": disk_io.write_count,
                }
            )

        return metrics

    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network metrics."""
        network_io = psutil.net_io_counters()

        if network_io:
            return {
                "network_bytes_sent": network_io.bytes_sent,
                "network_bytes_recv": network_io.bytes_recv,
                "network_packets_sent": network_io.packets_sent,
                "network_packets_recv": network_io.packets_recv,
            }
        return {}

    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        gpu_metrics = {"gpu_available": True}

        try:
            # Get GPU memory info
            gpu_mem_info = get_gpu_memory_info(self.device)
            gpu_metrics.update(gpu_mem_info)

            # Get GPU utilization (if available)
            gpu_metrics["gpu_count"] = torch.cuda.device_count()
            gpu_metrics["current_device"] = torch.cuda.current_device()

        except Exception as e:
            logger.warning(f"Error getting GPU metrics: {e}")
            gpu_metrics["gpu_error"] = str(e)

        return gpu_metrics

    def get_model_metrics(
        self, model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Get model-specific metrics.

        Args:
            model: PyTorch model to analyze.

        Returns:
            Dictionary containing model metrics.
        """
        metrics = {
            "timestamp": time.time(),
        }

        try:
            if model is not None:
                # Model size metrics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )

                metrics.update(
                    {
                        "model_total_parameters": total_params,
                        "model_trainable_parameters": trainable_params,
                        "model_parameter_ratio": trainable_params
                        / total_params
                        if total_params > 0
                        else 0,
                    }
                )

                # Model memory usage
                if self.device.type == "cuda":
                    model_memory = torch.cuda.memory_allocated(self.device)
                    model_memory_cached = torch.cuda.memory_reserved(
                        self.device
                    )
                    metrics.update(
                        {
                            "model_memory_allocated": model_memory,
                            "model_memory_cached": model_memory_cached,
                        }
                    )

            # PyTorch memory info
            if torch.cuda.is_available():
                metrics.update(
                    {
                        "torch_cuda_available": True,
                        "torch_cuda_device_count": torch.cuda.device_count(),
                        "torch_cuda_current_device": torch.cuda.current_device(),
                    }
                )
            else:
                metrics["torch_cuda_available"] = False

        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
            metrics["error"] = str(e)

        return metrics

    def collect_metrics(
        self, model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Collect all available metrics.

        Args:
            model: PyTorch model to analyze.

        Returns:
            Dictionary containing all collected metrics.
        """
        all_metrics = {
            "system": self.get_system_metrics(),
            "model": self.get_model_metrics(model),
        }

        # Store in history
        self.metrics_history.append(all_metrics)

        # Limit history size to prevent memory issues
        max_history = 1000
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]

        return all_metrics

    def get_metrics_summary(
        self, duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for metrics over a time period.

        Args:
            duration: Time period in seconds. If None, uses all available data.

        Returns:
            Dictionary containing summary statistics.
        """
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}

        # Filter by duration if specified
        current_time = time.time()
        if duration:
            cutoff_time = current_time - duration
            filtered_history = [
                m
                for m in self.metrics_history
                if m["system"]["timestamp"] >= cutoff_time
            ]
        else:
            filtered_history = self.metrics_history

        if not filtered_history:
            return {"error": "No metrics in specified duration"}

        summary = {
            "total_samples": len(filtered_history),
            "duration_seconds": filtered_history[-1]["system"]["timestamp"]
            - filtered_history[0]["system"]["timestamp"],
            "collection_interval": self.config.collect_interval,
        }

        # Calculate averages for key metrics
        try:
            cpu_values = [
                m["system"].get("cpu_percent", 0) for m in filtered_history
            ]
            memory_values = [
                m["system"].get("memory_percent", 0) for m in filtered_history
            ]

            summary.update(
                {
                    "avg_cpu_percent": sum(cpu_values) / len(cpu_values)
                    if cpu_values
                    else 0,
                    "max_cpu_percent": max(cpu_values) if cpu_values else 0,
                    "min_cpu_percent": min(cpu_values) if cpu_values else 0,
                    "avg_memory_percent": sum(memory_values)
                    / len(memory_values)
                    if memory_values
                    else 0,
                    "max_memory_percent": max(memory_values)
                    if memory_values
                    else 0,
                    "min_memory_percent": min(memory_values)
                    if memory_values
                    else 0,
                }
            )

            # GPU metrics if available
            gpu_memory_values = [
                m["system"].get("gpu_memory_used", 0)
                for m in filtered_history
                if "gpu_memory_used" in m["system"]
            ]
            if gpu_memory_values:
                summary.update(
                    {
                        "avg_gpu_memory_used": sum(gpu_memory_values)
                        / len(gpu_memory_values),
                        "max_gpu_memory_used": max(gpu_memory_values),
                        "min_gpu_memory_used": min(gpu_memory_values),
                    }
                )

        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            summary["error"] = str(e)

        return summary

    def clear_history(self) -> None:
        """Clear metrics history."""
        self.metrics_history.clear()
        logger.info("Metrics history cleared")

    def export_metrics(
        self, file_path: Union[str, Path], format: str = "json"
    ) -> bool:
        """
        Export metrics to file.

        Args:
            file_path: Path to export file.
            format: Export format ("json" or "csv").

        Returns:
            True if export successful, False otherwise.
        """
        try:
            file_path = Path(file_path)

            if format.lower() == "json":
                import json

                with open(file_path, "w") as f:
                    json.dump(self.metrics_history, f, indent=2)

            elif format.lower() == "csv":
                import csv

                if not self.metrics_history:
                    logger.warning("No metrics to export")
                    return False

                # Flatten the nested structure for CSV
                flattened_data = []
                for entry in self.metrics_history:
                    flat_entry = {}
                    for category, metrics in entry.items():
                        for key, value in metrics.items():
                            flat_entry[f"{category}_{key}"] = value
                    flattened_data.append(flat_entry)

                if flattened_data:
                    with open(file_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f, fieldnames=flattened_data[0].keys()
                        )
                        writer.writeheader()
                        writer.writerows(flattened_data)

            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Metrics exported to {file_path} in {format} format")
            return True

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring configuration and statistics."""
        return {
            "config": self.config.model_dump(),
            "uptime": time.time() - self.start_time,
            "total_metrics_collected": len(self.metrics_history),
            "collection_interval": self.config.collect_interval,
            "gpu_available": torch.cuda.is_available(),
            "device": str(self.device),
        }
