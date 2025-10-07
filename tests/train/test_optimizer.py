"""Tests for optimizer and scheduler."""

import pytest
import torch
import torch.optim as optim
from unittest.mock import Mock

from aksis.train.optimizer import create_optimizer, create_scheduler


class TestCreateOptimizer:
    """Test optimizer creation."""

    def test_create_optimizer_adamw(self):
        """Test creating AdamW optimizer."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        optimizer = create_optimizer(model, optimizer_type="adamw", lr=1e-3)

        assert isinstance(optimizer, optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_create_optimizer_adam(self):
        """Test creating Adam optimizer."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        optimizer = create_optimizer(model, optimizer_type="adam", lr=1e-3)

        assert isinstance(optimizer, optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_create_optimizer_sgd(self):
        """Test creating SGD optimizer."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        optimizer = create_optimizer(model, optimizer_type="sgd", lr=1e-3)

        assert isinstance(optimizer, optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_create_optimizer_default(self):
        """Test creating default optimizer (AdamW)."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        optimizer = create_optimizer(model)

        assert isinstance(optimizer, optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 5e-5  # Default learning rate

    def test_create_optimizer_with_weight_decay(self):
        """Test creating optimizer with weight decay."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        optimizer = create_optimizer(model, weight_decay=0.01)

        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_create_optimizer_with_betas(self):
        """Test creating optimizer with custom betas."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        optimizer = create_optimizer(model, betas=(0.9, 0.999))

        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    def test_create_optimizer_with_eps(self):
        """Test creating optimizer with custom eps."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        optimizer = create_optimizer(model, eps=1e-8)

        assert optimizer.param_groups[0]["eps"] == 1e-8

    def test_create_optimizer_invalid_type(self):
        """Test creating optimizer with invalid type."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        with pytest.raises(ValueError, match="Unsupported optimizer type"):
            create_optimizer(model, optimizer_type="invalid")

    def test_create_optimizer_invalid_lr(self):
        """Test creating optimizer with invalid learning rate."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            create_optimizer(model, lr=-1e-3)

    def test_create_optimizer_invalid_weight_decay(self):
        """Test creating optimizer with invalid weight decay."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]

        with pytest.raises(
            ValueError, match="Weight decay must be non-negative"
        ):
            create_optimizer(model, weight_decay=-0.01)


class TestCreateScheduler:
    """Test scheduler creation."""

    def test_create_scheduler_cosine(self):
        """Test creating cosine annealing scheduler."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_scheduler(
            optimizer, scheduler_type="cosine", T_max=100
        )

        assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100

    def test_create_scheduler_step(self):
        """Test creating step scheduler."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_scheduler(
            optimizer, scheduler_type="step", step_size=10
        )

        assert isinstance(scheduler, optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 10

    def test_create_scheduler_exponential(self):
        """Test creating exponential scheduler."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_scheduler(
            optimizer, scheduler_type="exponential", gamma=0.9
        )

        assert isinstance(scheduler, optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == 0.9

    def test_create_scheduler_plateau(self):
        """Test creating reduce on plateau scheduler."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_scheduler(
            optimizer, scheduler_type="plateau", patience=5
        )

        assert isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.patience == 5

    def test_create_scheduler_default(self):
        """Test creating default scheduler (cosine)."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_scheduler(optimizer)

        assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100  # Default T_max

    def test_create_scheduler_with_eta_min(self):
        """Test creating cosine scheduler with eta_min."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_scheduler(optimizer, eta_min=1e-6)

        assert scheduler.eta_min == 1e-6

    def test_create_scheduler_with_gamma(self):
        """Test creating exponential scheduler with gamma."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = create_scheduler(
            optimizer, scheduler_type="exponential", gamma=0.95
        )

        assert scheduler.gamma == 0.95

    def test_create_scheduler_invalid_type(self):
        """Test creating scheduler with invalid type."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            create_scheduler(optimizer, scheduler_type="invalid")

    def test_create_scheduler_invalid_t_max(self):
        """Test creating scheduler with invalid T_max."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match="T_max must be positive"):
            create_scheduler(optimizer, T_max=-1)

    def test_create_scheduler_invalid_step_size(self):
        """Test creating scheduler with invalid step_size."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match="Step size must be positive"):
            create_scheduler(optimizer, scheduler_type="step", step_size=-1)

    def test_create_scheduler_invalid_gamma(self):
        """Test creating scheduler with invalid gamma."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match="Gamma must be positive"):
            create_scheduler(
                optimizer, scheduler_type="exponential", gamma=-0.1
            )

    def test_create_scheduler_invalid_patience(self):
        """Test creating scheduler with invalid patience."""
        model = Mock()
        model.parameters.return_value = [
            torch.randn(10, 10, requires_grad=True)
        ]
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match="Patience must be positive"):
            create_scheduler(optimizer, scheduler_type="plateau", patience=-1)


class TestOptimizerSchedulerIntegration:
    """Test optimizer and scheduler integration."""

    def test_optimizer_scheduler_workflow(self):
        """Test optimizer and scheduler working together."""
        # Create a simple model
        model = torch.nn.Linear(10, 1)

        # Create optimizer
        optimizer = create_optimizer(model, lr=1e-3)

        # Create scheduler
        scheduler = create_scheduler(optimizer, T_max=10)

        # Simulate training steps
        initial_lr = optimizer.param_groups[0]["lr"]

        for epoch in range(5):
            # Simulate training step
            loss = torch.tensor(1.0, requires_grad=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Step scheduler
            scheduler.step()

            # Check that learning rate is changing
            current_lr = optimizer.param_groups[0]["lr"]
            assert current_lr > 0
            assert current_lr <= initial_lr

    def test_plateau_scheduler_workflow(self):
        """Test plateau scheduler workflow."""
        # Create a simple model
        model = torch.nn.Linear(10, 1)

        # Create optimizer
        optimizer = create_optimizer(model, lr=1e-3)

        # Create plateau scheduler
        scheduler = create_scheduler(
            optimizer, scheduler_type="plateau", patience=2
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Simulate training with decreasing loss
        for epoch in range(5):
            # Simulate training step
            loss = torch.tensor(1.0 - epoch * 0.1, requires_grad=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Step scheduler with loss
            scheduler.step(loss.item())

            current_lr = optimizer.param_groups[0]["lr"]
            assert current_lr > 0
            assert current_lr <= initial_lr

    def test_scheduler_state_dict(self):
        """Test scheduler state dict save/load."""
        model = torch.nn.Linear(10, 1)
        optimizer = create_optimizer(model, lr=1e-3)
        scheduler = create_scheduler(optimizer, T_max=10)

        # Step scheduler a few times
        for _ in range(3):
            scheduler.step()

        # Save state
        state_dict = scheduler.state_dict()

        # Create new scheduler and load state
        new_scheduler = create_scheduler(optimizer, T_max=10)
        new_scheduler.load_state_dict(state_dict)

        # Check that states match
        assert scheduler.state_dict() == new_scheduler.state_dict()
