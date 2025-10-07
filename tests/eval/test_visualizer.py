"""Tests for visualization components."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import os

from aksis.eval.visualizer import Visualizer


class TestVisualizer:
    """Test visualization functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.visualizer = Visualizer()
        
        # Sample training data
        self.training_data = {
            "epochs": [1, 2, 3, 4, 5],
            "train_loss": [2.5, 2.2, 2.0, 1.8, 1.6],
            "val_loss": [2.6, 2.3, 2.1, 1.9, 1.7],
            "train_perplexity": [12.2, 9.0, 7.4, 6.0, 5.0],
            "val_perplexity": [13.5, 10.0, 8.2, 6.7, 5.5],
        }
        
        # Sample evaluation data
        self.evaluation_data = {
            "checkpoints": ["epoch_1.pt", "epoch_2.pt", "epoch_3.pt"],
            "bleu_1": [0.65, 0.72, 0.78],
            "bleu_2": [0.45, 0.52, 0.58],
            "bleu_3": [0.32, 0.38, 0.44],
            "bleu_4": [0.25, 0.30, 0.35],
            "rouge_1": [0.70, 0.75, 0.80],
            "rouge_2": [0.50, 0.55, 0.60],
            "rouge_l": [0.68, 0.73, 0.78],
            "perplexity": [15.2, 12.8, 10.5],
        }

    def test_visualizer_initialization(self) -> None:
        """Test visualizer initialization."""
        assert isinstance(self.visualizer, Visualizer)

    def test_plot_training_curves(self) -> None:
        """Test plotting training curves."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "training_curves.png")
            
            self.visualizer.plot_training_curves(
                self.training_data, output_path=output_path
            )
            
            # Check that file was created
            assert os.path.exists(output_path)

    def test_plot_training_curves_with_custom_title(self) -> None:
        """Test plotting training curves with custom title."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "training_curves.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                title="Custom Training Progress",
            )
            
            assert os.path.exists(output_path)

    def test_plot_training_curves_invalid_data(self) -> None:
        """Test plotting training curves with invalid data."""
        invalid_data = {"epochs": [1, 2, 3]}  # Missing required keys
        
        with pytest.raises(ValueError, match="Missing required keys in training data"):
            self.visualizer.plot_training_curves(invalid_data, "test.png")

    def test_plot_training_curves_empty_data(self) -> None:
        """Test plotting training curves with empty data."""
        empty_data = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "train_perplexity": [],
            "val_perplexity": [],
        }
        
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            self.visualizer.plot_training_curves(empty_data, "test.png")

    def test_plot_evaluation_metrics(self) -> None:
        """Test plotting evaluation metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "evaluation_metrics.png")
            
            self.visualizer.plot_evaluation_metrics(
                self.evaluation_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_evaluation_metrics_bleu_only(self) -> None:
        """Test plotting evaluation metrics with BLEU scores only."""
        bleu_data = {
            "checkpoints": ["epoch_1.pt", "epoch_2.pt", "epoch_3.pt"],
            "bleu_1": [0.65, 0.72, 0.78],
            "bleu_2": [0.45, 0.52, 0.58],
            "bleu_3": [0.32, 0.38, 0.44],
            "bleu_4": [0.25, 0.30, 0.35],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "bleu_metrics.png")
            
            self.visualizer.plot_evaluation_metrics(
                bleu_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_evaluation_metrics_rouge_only(self) -> None:
        """Test plotting evaluation metrics with ROUGE scores only."""
        rouge_data = {
            "checkpoints": ["epoch_1.pt", "epoch_2.pt", "epoch_3.pt"],
            "rouge_1": [0.70, 0.75, 0.80],
            "rouge_2": [0.50, 0.55, 0.60],
            "rouge_l": [0.68, 0.73, 0.78],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "rouge_metrics.png")
            
            self.visualizer.plot_evaluation_metrics(
                rouge_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_evaluation_metrics_invalid_data(self) -> None:
        """Test plotting evaluation metrics with invalid data."""
        invalid_data = {"checkpoints": ["epoch_1.pt"]}  # Missing metric data
        
        with pytest.raises(ValueError, match="No metric data found"):
            self.visualizer.plot_evaluation_metrics(invalid_data, "test.png")

    def test_plot_evaluation_metrics_empty_data(self) -> None:
        """Test plotting evaluation metrics with empty data."""
        empty_data = {
            "checkpoints": [],
            "bleu_1": [],
            "rouge_1": [],
        }
        
        with pytest.raises(ValueError, match="Evaluation data cannot be empty"):
            self.visualizer.plot_evaluation_metrics(empty_data, "test.png")

    def test_plot_hyperparameter_comparison(self) -> None:
        """Test plotting hyperparameter comparison."""
        hyperparam_data = {
            "learning_rates": [1e-5, 5e-5, 1e-4],
            "batch_sizes": [8, 16, 32],
            "scores": [
                [0.65, 0.70, 0.68],  # lr=1e-5
                [0.72, 0.75, 0.73],  # lr=5e-5
                [0.68, 0.71, 0.69],  # lr=1e-4
            ],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "hyperparam_comparison.png")
            
            self.visualizer.plot_hyperparameter_comparison(
                hyperparam_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_hyperparameter_comparison_invalid_data(self) -> None:
        """Test plotting hyperparameter comparison with invalid data."""
        invalid_data = {
            "learning_rates": [1e-5, 5e-5],
            "batch_sizes": [8, 16, 32],
            "scores": [[0.65, 0.70]],  # Mismatched dimensions
        }
        
        with pytest.raises(ValueError, match="Data dimensions do not match"):
            self.visualizer.plot_hyperparameter_comparison(
                invalid_data, "test.png"
            )

    def test_plot_loss_comparison(self) -> None:
        """Test plotting loss comparison."""
        loss_data = {
            "epochs": [1, 2, 3, 4, 5],
            "baseline_loss": [2.5, 2.2, 2.0, 1.8, 1.6],
            "fine_tuned_loss": [2.3, 2.0, 1.8, 1.6, 1.4],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "loss_comparison.png")
            
            self.visualizer.plot_loss_comparison(
                loss_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_loss_comparison_invalid_data(self) -> None:
        """Test plotting loss comparison with invalid data."""
        invalid_data = {
            "epochs": [1, 2, 3],
            "baseline_loss": [2.5, 2.2, 2.0],
            "fine_tuned_loss": [2.3, 2.0],  # Mismatched length
        }
        
        with pytest.raises(ValueError, match="All arrays must have the same length"):
            self.visualizer.plot_loss_comparison(invalid_data, "test.png")

    def test_create_summary_plot(self) -> None:
        """Test creating summary plot with multiple subplots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "summary_plot.png")
            
            self.visualizer.create_summary_plot(
                training_data=self.training_data,
                evaluation_data=self.evaluation_data,
                output_path=output_path,
            )
            
            assert os.path.exists(output_path)

    def test_create_summary_plot_training_only(self) -> None:
        """Test creating summary plot with training data only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "training_summary.png")
            
            self.visualizer.create_summary_plot(
                training_data=self.training_data,
                output_path=output_path,
            )
            
            assert os.path.exists(output_path)

    def test_create_summary_plot_evaluation_only(self) -> None:
        """Test creating summary plot with evaluation data only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "evaluation_summary.png")
            
            self.visualizer.create_summary_plot(
                evaluation_data=self.evaluation_data,
                output_path=output_path,
            )
            
            assert os.path.exists(output_path)

    def test_create_summary_plot_no_data(self) -> None:
        """Test creating summary plot with no data."""
        with pytest.raises(ValueError, match="At least one dataset must be provided"):
            self.visualizer.create_summary_plot(output_path="test.png")

    def test_save_plot_with_custom_format(self) -> None:
        """Test saving plot with custom format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "plot.pdf")
            
            self.visualizer.plot_training_curves(
                self.training_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_with_custom_style(self) -> None:
        """Test plotting with custom style."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "styled_plot.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                style="seaborn-v0_8",
            )
            
            assert os.path.exists(output_path)

    def test_plot_with_custom_colors(self) -> None:
        """Test plotting with custom colors."""
        custom_colors = {
            "train_loss": "blue",
            "val_loss": "red",
            "train_perplexity": "green",
            "val_perplexity": "orange",
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "colored_plot.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                colors=custom_colors,
            )
            
            assert os.path.exists(output_path)

    def test_plot_with_custom_figure_size(self) -> None:
        """Test plotting with custom figure size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "large_plot.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                figsize=(12, 8),
            )
            
            assert os.path.exists(output_path)

    def test_plot_with_legend(self) -> None:
        """Test plotting with legend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "legend_plot.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                show_legend=True,
            )
            
            assert os.path.exists(output_path)

    def test_plot_without_legend(self) -> None:
        """Test plotting without legend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "no_legend_plot.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                show_legend=False,
            )
            
            assert os.path.exists(output_path)

    def test_plot_with_grid(self) -> None:
        """Test plotting with grid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "grid_plot.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                show_grid=True,
            )
            
            assert os.path.exists(output_path)

    def test_plot_without_grid(self) -> None:
        """Test plotting without grid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "no_grid_plot.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                show_grid=False,
            )
            
            assert os.path.exists(output_path)

    def test_plot_with_custom_dpi(self) -> None:
        """Test plotting with custom DPI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "high_dpi_plot.png")
            
            self.visualizer.plot_training_curves(
                self.training_data,
                output_path=output_path,
                dpi=300,
            )
            
            assert os.path.exists(output_path)

    def test_plot_error_handling(self) -> None:
        """Test error handling in plotting."""
        # Test with invalid output path
        with pytest.raises(OSError):
            self.visualizer.plot_training_curves(
                self.training_data, output_path="/invalid/path/plot.png"
            )

    def test_plot_data_integrity(self) -> None:
        """Test data integrity in plots."""
        # Test with NaN values
        corrupted_data = self.training_data.copy()
        corrupted_data["train_loss"] = [2.5, np.nan, 2.0, 1.8, 1.6]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "corrupted_plot.png")
            
            # Should handle NaN values gracefully
            self.visualizer.plot_training_curves(
                corrupted_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_with_single_data_point(self) -> None:
        """Test plotting with single data point."""
        single_point_data = {
            "epochs": [1],
            "train_loss": [2.5],
            "val_loss": [2.6],
            "train_perplexity": [12.2],
            "val_perplexity": [13.5],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "single_point_plot.png")
            
            self.visualizer.plot_training_curves(
                single_point_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_with_large_dataset(self) -> None:
        """Test plotting with large dataset."""
        large_data = {
            "epochs": list(range(1, 101)),
            "train_loss": [2.5 - 0.01 * i for i in range(100)],
            "val_loss": [2.6 - 0.01 * i for i in range(100)],
            "train_perplexity": [12.2 - 0.1 * i for i in range(100)],
            "val_perplexity": [13.5 - 0.1 * i for i in range(100)],
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "large_dataset_plot.png")
            
            self.visualizer.plot_training_curves(
                large_data, output_path=output_path
            )
            
            assert os.path.exists(output_path)

    def test_plot_performance(self) -> None:
        """Test plotting performance."""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "performance_plot.png")
            
            start_time = time.time()
            self.visualizer.plot_training_curves(
                self.training_data, output_path=output_path
            )
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 5.0
            assert os.path.exists(output_path)

    def test_plot_with_matplotlib_backend(self) -> None:
        """Test plotting with different matplotlib backends."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "backend_plot.png")
            
            # Test with Agg backend (non-interactive)
            with patch("matplotlib.pyplot.switch_backend"):
                self.visualizer.plot_training_curves(
                    self.training_data, output_path=output_path
                )
            
            assert os.path.exists(output_path)
