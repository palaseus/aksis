"""Visualization utilities for Aksis AI chatbot/LLM evaluation."""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class Visualizer:
    """Creates visualizations for training and evaluation metrics."""

    def __init__(
        self,
        output_dir: Union[str, Path] = "results/plots",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        style: str = "seaborn-v0_8",
    ) -> None:
        """
        Initialize the Visualizer.

        Args:
            output_dir: Directory to save plots.
            figsize: Figure size (width, height) in inches.
            dpi: Dots per inch for saved plots.
            style: Matplotlib style to use.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi

        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            logger.warning(f"Style {style} not found, using default")
            plt.style.use("default")

        logger.info(
            f"Visualizer initialized with output directory: {self.output_dir}"
        )

    def plot_loss_perplexity(
        self,
        history: Union[List[Dict[str, float]], Dict[str, List[float]]],
        title: str = "Training Loss and Perplexity",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
        style: Optional[str] = None,
        colors: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        show_legend: bool = True,
        show_grid: bool = True,
        dpi: int = 100,
    ) -> Path:
        """
        Plot training loss and perplexity curves.

        Args:
            history: List of dictionaries or dictionary with lists containing training metrics.
            title: Title for the plot.
            save_path: Path to save the plot. If None, auto-generates.
            show_plot: Whether to display the plot.
            style: Matplotlib style to use.
            colors: List of colors for the plots.
            figsize: Figure size as (width, height).
            show_legend: Whether to show legend.
            show_grid: Whether to show grid.
            dpi: DPI for the saved plot.

        Returns:
            Path to the saved plot.
        """
        if not history:
            raise ValueError("History cannot be empty")

        # Handle both data formats
        if isinstance(history, dict):
            # Dictionary format: {"epochs": [1,2,3], "train_loss": [1,2,3], ...}
            epochs = history.get("epochs", [])
            train_losses = history.get("train_loss", [])
            val_losses = history.get("val_loss", [])
            train_perplexities = history.get("train_perplexity", [])
            val_perplexities = history.get("val_perplexity", [])

            # Validate required keys
            required_keys = [
                "epochs",
                "train_loss",
                "val_loss",
                "train_perplexity",
                "val_perplexity",
            ]
            missing_keys = [key for key in required_keys if key not in history]
            if missing_keys:
                raise ValueError(
                    f"Missing required keys in training data: {missing_keys}"
                )

            if not epochs:
                raise ValueError("Training data cannot be empty")
        else:
            # List format: [{"epoch": 1, "train_loss": 1, ...}, ...]
            epochs = [h["epoch"] for h in history]
            train_losses = [h.get("train_loss", 0) for h in history]
            val_losses = [h.get("val_loss", None) for h in history]
            train_perplexities = [
                h.get("train_perplexity", 0) for h in history
            ]
            val_perplexities = [h.get("val_perplexity", None) for h in history]

        # Set style if provided
        if style:
            plt.style.use(style)

        # Use custom figure size if provided
        plot_figsize = figsize if figsize else self.figsize

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=plot_figsize, sharex=True, dpi=dpi
        )

        # Set colors if provided
        if isinstance(colors, dict):
            train_color = colors.get(
                "train_loss", colors.get("train_perplexity")
            )
            val_color = colors.get("val_loss", colors.get("val_perplexity"))
        elif isinstance(colors, list):
            train_color = colors[0] if colors and len(colors) > 0 else None
            val_color = colors[1] if colors and len(colors) > 1 else None
        else:
            train_color = None
            val_color = None

        # Plot loss
        ax1.plot(
            epochs,
            train_losses,
            label="Train Loss",
            marker="o",
            linewidth=2,
            color=train_color,
        )
        if any(v is not None for v in val_losses):
            val_epochs = [
                e for e, v in zip(epochs, val_losses) if v is not None
            ]
            val_losses_clean = [v for v in val_losses if v is not None]
            ax1.plot(
                val_epochs,
                val_losses_clean,
                label="Val Loss",
                marker="s",
                linewidth=2,
                color=val_color,
            )

        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        if show_legend:
            ax1.legend()
        if show_grid:
            ax1.grid(True, alpha=0.3)

        # Plot perplexity
        ax2.plot(
            epochs,
            train_perplexities,
            label="Train Perplexity",
            marker="o",
            linewidth=2,
            color=train_color,
        )
        if any(v is not None for v in val_perplexities):
            val_epochs = [
                e for e, v in zip(epochs, val_perplexities) if v is not None
            ]
            val_perplexities_clean = [
                v for v in val_perplexities if v is not None
            ]
            ax2.plot(
                val_epochs,
                val_perplexities_clean,
                label="Val Perplexity",
                marker="s",
                linewidth=2,
                color=val_color,
            )

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Perplexity")
        ax2.set_title("Training and Validation Perplexity")
        if show_legend:
            ax2.legend()
        if show_grid:
            ax2.grid(True, alpha=0.3)

        # Set overall title
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Adjust layout
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "loss_perplexity_curves.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Loss and perplexity plot saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_evaluation_metrics(
        self,
        metrics: Union[Dict[str, float], Dict[str, List[float]]],
        title: str = "Evaluation Metrics",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
    ) -> Path:
        """
        Plot evaluation metrics as a bar chart.

        Args:
            metrics: Dictionary containing evaluation metrics (single values or lists).
            title: Title for the plot.
            save_path: Path to save the plot. If None, auto-generates.
            show_plot: Whether to display the plot.

        Returns:
            Path to the saved plot.
        """
        if not metrics:
            raise ValueError("Metrics cannot be empty")

        # Handle both single values and lists
        if any(isinstance(v, list) for v in metrics.values()):
            # List format - take the last values
            numeric_metrics = {}
            for k, v in metrics.items():
                if k == "checkpoints":  # Skip non-numeric keys
                    continue
                if isinstance(v, list) and v:
                    numeric_metrics[k] = v[-1]  # Take last value
                elif isinstance(v, (int, float)):
                    numeric_metrics[k] = v
        else:
            # Single value format
            numeric_metrics = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "checkpoints"
            }
        if not numeric_metrics:
            raise ValueError("No numeric metrics found")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Prepare data
        metric_names = list(numeric_metrics.keys())
        metric_values = list(numeric_metrics.values())

        # Create bar plot with proper labels
        x_pos = range(len(metric_names))
        bars = ax.bar(
            x_pos,
            metric_values,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names)

        # Customize plot
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if they're too long
        if max(len(name) for name in metric_names) > 10:
            plt.xticks(rotation=45, ha="right")

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(metric_values),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Adjust layout
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "evaluation_metrics.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Evaluation metrics plot saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_metric_comparison(
        self,
        results: Union[List[Dict[str, Any]], Dict[str, List[float]]],
        metric_name: str,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
    ) -> Path:
        """
        Plot comparison of a metric across different models/checkpoints.

        Args:
            results: List of result dictionaries or dictionary with lists.
            metric_name: Name of the metric to compare.
            title: Title for the plot. If None, auto-generates.
            save_path: Path to save the plot. If None, auto-generates.
            show_plot: Whether to display the plot.

        Returns:
            Path to the saved plot.
        """
        if not results:
            raise ValueError("Results cannot be empty")
        if not metric_name:
            raise ValueError("Metric name cannot be empty")

        # Handle both data formats
        if isinstance(results, dict):
            # Dictionary format: {"epochs": [1,2,3], "baseline_loss": [1,2,3], ...}
            if "epochs" in results:
                # Look for metrics that contain the metric_name
                matching_metrics = [
                    k
                    for k in results.keys()
                    if metric_name in k and k != "epochs"
                ]
                if matching_metrics:
                    # Validate all matching metrics have the same length as epochs
                    epochs_length = len(results["epochs"])
                    for metric_key in matching_metrics:
                        if len(results[metric_key]) != epochs_length:
                            raise ValueError(
                                "All arrays must have the same length"
                            )

                    # Use the first matching metric
                    metric_key = matching_metrics[0]
                    labels = [f"Epoch {epoch}" for epoch in results["epochs"]]
                    values = results[metric_key]
                else:
                    raise ValueError(
                        f"Metric '{metric_name}' not found in results"
                    )
            else:
                raise ValueError("No epochs found in results")
        else:
            # List format: [{"epoch": 1, "loss": 1, ...}, ...]
            labels: List[str] = []
            values: List[float] = []
            for result in results:
                # Try to extract a meaningful label
                if "checkpoint_path" in result:
                    label = Path(result["checkpoint_path"]).stem
                elif "epoch" in result:
                    label = f"Epoch {result['epoch']}"
                else:
                    label = f"Model {len(labels) + 1}"

                if metric_name in result:
                    labels.append(label)
                    values.append(result[metric_name])

        if not values:
            raise ValueError(f"Metric '{metric_name}' not found in results")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create bar plot
        bars = ax.bar(
            labels, values, alpha=0.7, edgecolor="black", linewidth=1
        )

        # Customize plot
        if title is None:
            title = f"Comparison of {metric_name.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis="y")

        # Rotate x-axis labels if they're too long
        if max(len(label) for label in labels) > 10:
            plt.xticks(rotation=45, ha="right")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(values),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Adjust layout
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"{metric_name}_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Metric comparison plot saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_hyperparameter_search(
        self,
        search_results: Dict[str, Any],
        title: str = "Hyperparameter Search Results",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
    ) -> Path:
        """
        Plot hyperparameter search results.

        Args:
            search_results: Dictionary containing search results.
            title: Title for the plot.
            save_path: Path to save the plot. If None, auto-generates.
            show_plot: Whether to display the plot.

        Returns:
            Path to the saved plot.
        """
        # Handle different data formats
        if "all_results" in search_results:
            # Original format
            results = search_results["all_results"]
            if not results:
                raise ValueError("No search results found")
            trials = [r["trial"] for r in results]
            scores = [r["best_val_loss"] for r in results]
        elif "learning_rates" in search_results and "scores" in search_results:
            # New format with learning rates and scores
            learning_rates = search_results["learning_rates"]
            scores_matrix = search_results["scores"]
            if not learning_rates or not scores_matrix:
                raise ValueError("No search results found")

            # Validate dimensions
            if len(learning_rates) != len(scores_matrix):
                raise ValueError("Data dimensions do not match")

            # Flatten the data for plotting
            trials = []
            scores = []
            for i, lr in enumerate(learning_rates):
                if i >= len(scores_matrix):
                    raise ValueError("Data dimensions do not match")
                for j, score in enumerate(scores_matrix[i]):
                    trials.append(f"LR={lr:.0e}, BS={j+1}")
                    scores.append(score)
        else:
            raise ValueError("Invalid search results format")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create scatter plot
        ax.scatter(
            trials, scores, alpha=0.7, s=100, edgecolors="black", linewidth=1
        )

        # Highlight best result
        if "all_results" in search_results:
            best_trial = min(results, key=lambda x: x["best_val_loss"])
            ax.scatter(
                best_trial["trial"],
                best_trial["best_val_loss"],
                color="red",
                s=200,
                edgecolors="black",
                linewidth=2,
                label="Best",
            )
        else:
            # For new format, find best score
            best_idx = scores.index(min(scores))
            ax.scatter(
                trials[best_idx],
                scores[best_idx],
                color="red",
                s=200,
                edgecolors="black",
                linewidth=2,
                label="Best",
            )

        # Customize plot
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best Validation Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add value labels
        for trial, score in zip(trials, scores):
            ax.annotate(
                f"{score:.4f}",
                (trial, score),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

        # Adjust layout
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "hyperparameter_search_results.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Hyperparameter search plot saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_learning_rate_schedule(
        self,
        history: List[Dict[str, float]],
        title: str = "Learning Rate Schedule",
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
    ) -> Path:
        """
        Plot learning rate schedule during training.

        Args:
            history: List of dictionaries containing training metrics.
            title: Title for the plot.
            save_path: Path to save the plot. If None, auto-generates.
            show_plot: Whether to display the plot.

        Returns:
            Path to the saved plot.
        """
        if not history:
            raise ValueError("History cannot be empty")

        # Extract data
        epochs = [h["epoch"] for h in history]
        learning_rates = [h.get("learning_rate", None) for h in history]

        # Filter out None values
        valid_data = [
            (e, lr) for e, lr in zip(epochs, learning_rates) if lr is not None
        ]
        if not valid_data:
            raise ValueError("No learning rate data found in history")

        epochs_clean, lr_clean = zip(*valid_data)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot learning rate
        ax.plot(epochs_clean, lr_clean, marker="o", linewidth=2, markersize=6)

        # Customize plot
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        # Adjust layout
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "learning_rate_schedule.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Learning rate schedule plot saved: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return save_path

    def create_summary_report(
        self,
        evaluation_results: Optional[Dict[str, Any]] = None,
        training_history: Optional[List[Dict[str, float]]] = None,
        training_data: Optional[Dict[str, List[float]]] = None,
        evaluation_data: Optional[Dict[str, List[float]]] = None,
        output_path: Optional[Union[str, Path]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Create a comprehensive summary report with multiple plots.

        Args:
            evaluation_results: Dictionary containing evaluation results.
            training_history: List of dictionaries containing training metrics.
            training_data: Dictionary with training data in list format.
            evaluation_data: Dictionary with evaluation data in list format.
            output_path: Path to save the report. If None, auto-generates.
            save_path: Alternative name for output_path.

        Returns:
            Path to the saved report.
        """
        # Handle save_path as alternative to output_path
        if save_path is not None:
            output_path = save_path

        # Convert training_data to training_history format if needed
        if training_data is not None and training_history is None:
            training_history = []
            for i in range(len(training_data.get("epochs", []))):
                training_history.append(
                    {
                        "epoch": training_data["epochs"][i],
                        "train_loss": training_data["train_loss"][i],
                        "val_loss": training_data["val_loss"][i],
                        "train_perplexity": training_data["train_perplexity"][
                            i
                        ],
                        "val_perplexity": training_data["val_perplexity"][i],
                    }
                )

        # Convert evaluation_data to evaluation_results format if needed
        if evaluation_data is not None and evaluation_results is None:
            evaluation_results = {}
            for key, values in evaluation_data.items():
                if values:  # Take last value if it's a list
                    evaluation_results[key] = (
                        values[-1] if isinstance(values, list) else values
                    )

        if not evaluation_results and not training_history:
            raise ValueError("At least one dataset must be provided")

        if output_path is None:
            output_path = self.output_dir / "summary_report.png"
        else:
            output_path = Path(output_path)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Plot 1: Training curves (if available)
        if training_history:
            ax1 = plt.subplot(2, 2, 1)
            epochs = [h["epoch"] for h in training_history]
            train_losses = [h.get("train_loss", 0) for h in training_history]
            val_losses = [h.get("val_loss", None) for h in training_history]

            ax1.plot(
                epochs,
                train_losses,
                label="Train Loss",
                marker="o",
                linewidth=2,
            )
            if any(v is not None for v in val_losses):
                val_epochs = [
                    e for e, v in zip(epochs, val_losses) if v is not None
                ]
                val_losses_clean = [v for v in val_losses if v is not None]
                ax1.plot(
                    val_epochs,
                    val_losses_clean,
                    label="Val Loss",
                    marker="s",
                    linewidth=2,
                )

            ax1.set_title("Training and Validation Loss", fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Evaluation metrics (if available)
        if evaluation_results:
            ax2 = plt.subplot(2, 2, 2)
            numeric_metrics = {
                k: v
                for k, v in evaluation_results.items()
                if isinstance(v, (int, float))
            }
            if numeric_metrics:
                metric_names = list(numeric_metrics.keys())
                metric_values = list(numeric_metrics.values())
                ax2.bar(
                    metric_names,
                    metric_values,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=1,
                )
                ax2.set_title("Evaluation Metrics", fontweight="bold")
                ax2.set_ylabel("Score")
                ax2.grid(True, alpha=0.3, axis="y")
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

        # Plot 3: Perplexity curves (if available)
        if training_history:
            ax3 = plt.subplot(2, 2, 3)
            train_perplexities = [
                h.get("train_perplexity", 0) for h in training_history
            ]
            val_perplexities = [
                h.get("val_perplexity", None) for h in training_history
            ]

            ax3.plot(
                epochs,
                train_perplexities,
                label="Train Perplexity",
                marker="o",
                linewidth=2,
            )
            if any(v is not None for v in val_perplexities):
                val_epochs = [
                    e
                    for e, v in zip(epochs, val_perplexities)
                    if v is not None
                ]
                val_perplexities_clean = [
                    v for v in val_perplexities if v is not None
                ]
                ax3.plot(
                    val_epochs,
                    val_perplexities_clean,
                    label="Val Perplexity",
                    marker="s",
                    linewidth=2,
                )

            ax3.set_title(
                "Training and Validation Perplexity", fontweight="bold"
            )
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Perplexity")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Summary statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis("off")

        # Create summary text
        summary_text = "Summary Statistics\n\n"
        if training_history:
            best_epoch = min(
                training_history, key=lambda x: x.get("val_loss", float("inf"))
            )
            summary_text += f"Best Epoch: {best_epoch['epoch']}\n"
            summary_text += (
                f"Best Train Loss: {best_epoch.get('train_loss', 0):.4f}\n"
            )
            if "val_loss" in best_epoch:
                summary_text += (
                    f"Best Val Loss: {best_epoch['val_loss']:.4f}\n"
                )
            train_perplexity = best_epoch.get("train_perplexity", 0)
            summary_text += f"Best Train Perplexity: {train_perplexity:.2f}\n"
            if "val_perplexity" in best_epoch:
                val_perplexity = best_epoch["val_perplexity"]
                summary_text += f"Best Val Perplexity: {val_perplexity:.2f}\n"

        if evaluation_results:
            summary_text += "\nEvaluation Results:\n"
            for metric, value in evaluation_results.items():
                if isinstance(value, (int, float)):
                    summary_text += f"{metric}: {value:.4f}\n"

        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # Set overall title
        fig.suptitle(
            "Training and Evaluation Summary Report",
            fontsize=18,
            fontweight="bold",
        )

        # Adjust layout
        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Summary report saved: {output_path}")

        plt.close()

        return output_path

    def save_plot_data(
        self,
        data: Dict[str, Any],
        filename: str = "plot_data.json",
    ) -> Path:
        """
        Save plot data to JSON file for later use.

        Args:
            data: Dictionary containing plot data.
            filename: Name of the file to save.

        Returns:
            Path to the saved file.
        """
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Plot data saved: {output_path}")
        return output_path
