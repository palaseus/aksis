"""Tests for performance monitoring utilities."""

import pytest
from unittest.mock import Mock, patch
import torch
import torch.nn as nn
import tempfile
import json
import csv
from pathlib import Path

from aksis.deploy.monitor import PerformanceMonitor, MonitoringConfig, MonitoringError


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.config = MonitoringConfig(
            collect_interval=1.0,
            enable_gpu_monitoring=True,
            enable_cpu_monitoring=True,
            enable_memory_monitoring=True,
            enable_disk_monitoring=False,
            enable_network_monitoring=False,
        )
        self.monitor = PerformanceMonitor(self.config)

    def test_monitoring_config_initialization(self) -> None:
        """Test monitoring configuration initialization."""
        config = MonitoringConfig(
            collect_interval=2.0,
            enable_gpu_monitoring=False,
            enable_cpu_monitoring=True,
            enable_memory_monitoring=True,
            enable_disk_monitoring=True,
            enable_network_monitoring=True,
            log_file="test.log",
            max_log_size=50 * 1024 * 1024,
            backup_count=3,
        )
        assert config.collect_interval == 2.0
        assert config.enable_gpu_monitoring is False
        assert config.enable_cpu_monitoring is True
        assert config.enable_memory_monitoring is True
        assert config.enable_disk_monitoring is True
        assert config.enable_network_monitoring is True
        assert config.log_file == "test.log"
        assert config.max_log_size == 50 * 1024 * 1024
        assert config.backup_count == 3

    def test_monitoring_config_defaults(self) -> None:
        """Test monitoring configuration with default values."""
        config = MonitoringConfig()
        assert config.collect_interval == 1.0
        assert config.enable_gpu_monitoring is True
        assert config.enable_cpu_monitoring is True
        assert config.enable_memory_monitoring is True
        assert config.enable_disk_monitoring is False
        assert config.enable_network_monitoring is False
        assert config.log_file is None
        assert config.max_log_size == 100 * 1024 * 1024
        assert config.backup_count == 5

    def test_performance_monitor_initialization(self) -> None:
        """Test performance monitor initialization."""
        assert isinstance(self.monitor, PerformanceMonitor)
        assert self.monitor.config == self.config

    def test_validate_config_invalid_interval(self) -> None:
        """Test configuration validation with invalid interval."""
        with pytest.raises(Exception):  # Pydantic validation error
            MonitoringConfig(collect_interval=0.0)

    def test_validate_config_invalid_max_log_size(self) -> None:
        """Test configuration validation with invalid max log size."""
        config = MonitoringConfig(max_log_size=0)
        with pytest.raises(MonitoringError, match="Maximum log size must be positive"):
            PerformanceMonitor(config)

    def test_validate_config_invalid_backup_count(self) -> None:
        """Test configuration validation with invalid backup count."""
        config = MonitoringConfig(backup_count=-1)
        with pytest.raises(MonitoringError, match="Backup count cannot be negative"):
            PerformanceMonitor(config)

    @patch('psutil.cpu_percent')
    @patch('psutil.cpu_count')
    def test_get_system_metrics_cpu(self, mock_cpu_count, mock_cpu_percent):
        """Test getting CPU metrics."""
        mock_cpu_percent.return_value = 25.5
        mock_cpu_count.return_value = 8
        
        metrics = self.monitor.get_system_metrics()
        
        assert "timestamp" in metrics
        assert "uptime" in metrics
        assert "cpu_percent" in metrics
        assert metrics["cpu_percent"] == 25.5
        assert metrics["cpu_count"] == 8

    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_get_system_metrics_memory(self, mock_swap, mock_memory):
        """Test getting memory metrics."""
        mock_memory.return_value = Mock(
            total=8589934592,  # 8GB
            available=4294967296,  # 4GB
            used=4294967296,  # 4GB
            percent=50.0
        )
        mock_swap.return_value = Mock(
            total=2147483648,  # 2GB
            used=1073741824,  # 1GB
            percent=50.0
        )
        
        metrics = self.monitor.get_system_metrics()
        
        assert "memory_total" in metrics
        assert "memory_available" in metrics
        assert "memory_used" in metrics
        assert "memory_percent" in metrics
        assert "swap_total" in metrics
        assert "swap_used" in metrics
        assert "swap_percent" in metrics

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    def test_get_system_metrics_gpu(self, mock_props, mock_reserved, mock_allocated, mock_available):
        """Test getting GPU metrics."""
        mock_allocated.return_value = 1073741824  # 1GB
        mock_reserved.return_value = 2147483648   # 2GB
        mock_props.return_value = Mock(total_memory=8589934592)  # 8GB
        
        metrics = self.monitor.get_system_metrics()
        
        assert "gpu_available" in metrics
        assert metrics["gpu_available"] is True
        assert "gpu_memory_allocated" in metrics
        assert "gpu_memory_reserved" in metrics
        assert "gpu_memory_total" in metrics

    def test_get_model_metrics(self) -> None:
        """Test getting model metrics."""
        mock_model = Mock(spec=nn.Module)
        mock_param1 = Mock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True
        mock_param2 = Mock()
        mock_param2.numel.return_value = 500
        mock_param2.requires_grad = False
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        
        metrics = self.monitor.get_model_metrics(mock_model)
        
        assert "timestamp" in metrics
        assert "model_total_parameters" in metrics
        assert "model_trainable_parameters" in metrics
        assert "model_parameter_ratio" in metrics
        assert metrics["model_total_parameters"] == 1500
        assert metrics["model_trainable_parameters"] == 1000
        assert metrics["model_parameter_ratio"] == 1000 / 1500

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_get_model_metrics_gpu(self, mock_reserved, mock_allocated, mock_available):
        """Test getting model metrics with GPU."""
        mock_allocated.return_value = 1073741824  # 1GB
        mock_reserved.return_value = 2147483648   # 2GB
        
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = []
        
        metrics = self.monitor.get_model_metrics(mock_model)
        
        assert "model_memory_allocated" in metrics
        assert "model_memory_cached" in metrics
        assert metrics["model_memory_allocated"] == 1073741824
        assert metrics["model_memory_cached"] == 2147483648

    def test_collect_metrics(self) -> None:
        """Test collecting all metrics."""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = []
        
        with patch.object(self.monitor, 'get_system_metrics') as mock_system, \
             patch.object(self.monitor, 'get_model_metrics') as mock_model_metrics:
            
            mock_system.return_value = {"cpu_percent": 25.0}
            mock_model_metrics.return_value = {"total_parameters": 1000}
            
            result = self.monitor.collect_metrics(mock_model)
            
            assert "system" in result
            assert "model" in result
            assert len(self.monitor.metrics_history) == 1

    def test_get_metrics_summary(self) -> None:
        """Test getting metrics summary."""
        # Add some mock metrics to history
        self.monitor.metrics_history = [
            {"system": {"timestamp": 1000, "cpu_percent": 20.0, "memory_percent": 50.0}},
            {"system": {"timestamp": 1001, "cpu_percent": 30.0, "memory_percent": 60.0}},
            {"system": {"timestamp": 1002, "cpu_percent": 25.0, "memory_percent": 55.0}},
        ]
        
        summary = self.monitor.get_metrics_summary()
        
        assert "total_samples" in summary
        assert "duration_seconds" in summary
        assert "avg_cpu_percent" in summary
        assert "max_cpu_percent" in summary
        assert "min_cpu_percent" in summary
        assert summary["total_samples"] == 3
        assert summary["avg_cpu_percent"] == 25.0
        assert summary["max_cpu_percent"] == 30.0
        assert summary["min_cpu_percent"] == 20.0

    def test_get_metrics_summary_with_duration(self) -> None:
        """Test getting metrics summary with duration filter."""
        current_time = 1000
        with patch('time.time', return_value=current_time):
            # Add metrics with different timestamps
            self.monitor.metrics_history = [
                {"system": {"timestamp": 995, "cpu_percent": 20.0}},  # 5 seconds ago
                {"system": {"timestamp": 998, "cpu_percent": 30.0}},  # 2 seconds ago
                {"system": {"timestamp": 1000, "cpu_percent": 25.0}}, # now
            ]
            
            summary = self.monitor.get_metrics_summary(duration=3.0)  # Last 3 seconds
            
            assert summary["total_samples"] == 2  # Only last 2 metrics
            assert summary["avg_cpu_percent"] == 27.5  # (30.0 + 25.0) / 2

    def test_clear_history(self) -> None:
        """Test clearing metrics history."""
        self.monitor.metrics_history = [{"test": "data"}]
        self.monitor.clear_history()
        assert len(self.monitor.metrics_history) == 0

    def test_export_metrics_json(self) -> None:
        """Test exporting metrics to JSON."""
        self.monitor.metrics_history = [
            {"system": {"cpu_percent": 25.0}, "model": {"parameters": 1000}}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            result = self.monitor.export_metrics(temp_path, "json")
            assert result is True
            
            # Verify the file was created and contains correct data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["system"]["cpu_percent"] == 25.0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_metrics_csv(self) -> None:
        """Test exporting metrics to CSV."""
        self.monitor.metrics_history = [
            {"system": {"cpu_percent": 25.0}, "model": {"parameters": 1000}}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            result = self.monitor.export_metrics(temp_path, "csv")
            assert result is True
            
            # Verify the file was created and contains correct data
            with open(temp_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["system_cpu_percent"] == "25.0"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_metrics_invalid_format(self) -> None:
        """Test exporting metrics with invalid format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            result = self.monitor.export_metrics(temp_path, "invalid")
            assert result is False
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_metrics_empty_history(self) -> None:
        """Test exporting metrics with empty history."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            result = self.monitor.export_metrics(temp_path, "csv")
            assert result is False
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_get_monitoring_stats(self) -> None:
        """Test getting monitoring statistics."""
        stats = self.monitor.get_monitoring_stats()
        
        assert "config" in stats
        assert "uptime" in stats
        assert "total_metrics_collected" in stats
        assert "collection_interval" in stats
        assert "gpu_available" in stats
        assert "device" in stats
        assert stats["total_metrics_collected"] == 0
        assert stats["collection_interval"] == 1.0

    def test_monitoring_error_initialization(self) -> None:
        """Test MonitoringError initialization."""
        error = MonitoringError("Test error")
        assert str(error) == "Test error"