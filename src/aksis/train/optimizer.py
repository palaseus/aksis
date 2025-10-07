"""Optimizer and scheduler utilities for training."""

import logging
from typing import Tuple

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.

    Args:
        model: PyTorch model to optimize.
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd').
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Beta parameters for Adam/AdamW.
        eps: Epsilon parameter for numerical stability.
        **kwargs: Additional optimizer-specific parameters.

    Returns:
        Configured optimizer.

    Raises:
        ValueError: If optimizer type is unsupported or parameters are invalid.
    """
    if lr <= 0:
        raise ValueError("Learning rate must be positive")

    if weight_decay < 0:
        raise ValueError("Weight decay must be non-negative")

    if betas[0] < 0 or betas[1] < 0:
        raise ValueError("Beta parameters must be non-negative")

    if eps <= 0:
        raise ValueError("Epsilon must be positive")

    # Get model parameters
    params = list(model.parameters())
    if not params:
        raise ValueError("Model has no parameters to optimize")

    # Create optimizer based on type
    if optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs,
        )
        logger.info(
            f"Created AdamW optimizer with lr={lr}, "
            f"weight_decay={weight_decay}"
        )

    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs,
        )
        logger.info(
            f"Created Adam optimizer with lr={lr}, weight_decay={weight_decay}"
        )

    elif optimizer_type.lower() == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", False)
        optimizer = optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["momentum", "nesterov"]
            },
        )
        logger.info(f"Created SGD optimizer with lr={lr}, momentum={momentum}")

    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    T_max: int = 100,
    eta_min: float = 0.0,
    step_size: int = 10,
    gamma: float = 0.1,
    patience: int = 5,
    factor: float = 0.5,
    min_lr: float = 1e-7,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        scheduler_type: Type of scheduler ('cosine', 'step', 'exponential',
                                          'plateau').
        T_max: Maximum number of epochs for cosine annealing.
        eta_min: Minimum learning rate for cosine annealing.
        step_size: Step size for step scheduler.
        gamma: Multiplicative factor for step/exponential schedulers.
        patience: Patience for plateau scheduler.
        factor: Factor for plateau scheduler.
        min_lr: Minimum learning rate for plateau scheduler.
        **kwargs: Additional scheduler-specific parameters.

    Returns:
        Configured scheduler.

    Raises:
        ValueError: If scheduler type is unsupported or parameters are invalid.
    """
    if T_max <= 0:
        raise ValueError("T_max must be positive")

    if eta_min < 0:
        raise ValueError("eta_min must be non-negative")

    if step_size <= 0:
        raise ValueError("Step size must be positive")

    if gamma <= 0:
        raise ValueError("Gamma must be positive")

    if patience <= 0:
        raise ValueError("Patience must be positive")

    if factor <= 0 or factor >= 1:
        raise ValueError("Factor must be between 0 and 1")

    if min_lr <= 0:
        raise ValueError("min_lr must be positive")

    # Create scheduler based on type
    if scheduler_type.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min, **kwargs
        )
        logger.info(
            f"Created CosineAnnealingLR scheduler with T_max={T_max}, "
            f"eta_min={eta_min}"
        )

    elif scheduler_type.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma, **kwargs
        )
        logger.info(
            f"Created StepLR scheduler with step_size={step_size}, "
            f"gamma={gamma}"
        )

    elif scheduler_type.lower() == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma, **kwargs
        )
        logger.info(f"Created ExponentialLR scheduler with gamma={gamma}")

    elif scheduler_type.lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            **kwargs,
        )
        logger.info(
            f"Created ReduceLROnPlateau scheduler with patience={patience}, "
            f"factor={factor}"
        )

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from the optimizer.

    Args:
        optimizer: Optimizer to get learning rate from.

    Returns:
        Current learning rate.
    """
    return optimizer.param_groups[0]["lr"]


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """
    Set the learning rate for the optimizer.

    Args:
        optimizer: Optimizer to set learning rate for.
        lr: New learning rate.

    Raises:
        ValueError: If learning rate is invalid.
    """
    if lr <= 0:
        raise ValueError("Learning rate must be positive")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logger.info(f"Set learning rate to {lr}")


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    warmup_factor: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a warmup learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        warmup_epochs: Number of warmup epochs.
        warmup_factor: Factor to multiply initial learning rate by.

    Returns:
        Warmup scheduler.

    Raises:
        ValueError: If parameters are invalid.
    """
    if warmup_epochs <= 0:
        raise ValueError("Warmup epochs must be positive")

    if warmup_factor <= 0 or warmup_factor >= 1:
        raise ValueError("Warmup factor must be between 0 and 1")

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Created warmup scheduler with {warmup_epochs} warmup epochs")

    return scheduler


def create_optimizer_with_scheduler(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    scheduler_type: str = "cosine",
    lr: float = 5e-5,
    **kwargs,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Create an optimizer and scheduler together.

    Args:
        model: PyTorch model to optimize.
        optimizer_type: Type of optimizer.
        scheduler_type: Type of scheduler.
        lr: Learning rate.
        **kwargs: Additional parameters for optimizer and scheduler.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    # Extract optimizer-specific parameters
    optimizer_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ["weight_decay", "betas", "eps", "momentum", "nesterov"]
    }

    # Extract scheduler-specific parameters
    scheduler_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in [
            "T_max",
            "eta_min",
            "step_size",
            "gamma",
            "patience",
            "factor",
            "min_lr",
        ]
    }

    # Create optimizer
    optimizer = create_optimizer(
        model, optimizer_type=optimizer_type, lr=lr, **optimizer_kwargs
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer, scheduler_type=scheduler_type, **scheduler_kwargs
    )

    return optimizer, scheduler
