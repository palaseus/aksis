"""Sampling methods for text generation."""

import logging
import torch
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


class BaseSampler:
    """Base class for all samplers."""
    
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample tokens from logits. Must be implemented by subclasses."""
        raise NotImplementedError


class GreedySampler(BaseSampler):
    """Greedy sampling (always select the most likely token)."""

    def __init__(self) -> None:
        """Initialize greedy sampler."""
        pass

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using greedy selection.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).

        Returns:
            Sampled token indices of shape (batch_size, 1).
        """
        if logits.dim() != 2:
            raise ValueError("logits must be a 2D tensor")

        if logits.size(0) == 0 or logits.size(1) == 0:
            raise ValueError("logits cannot be empty")

        # Greedy selection: argmax
        sampled = torch.argmax(logits, dim=-1, keepdim=True)

        return sampled


class BeamSearchSampler(BaseSampler):
    """Beam search sampling for better quality generation."""

    def __init__(self, beam_width: int = 4):
        """
        Initialize beam search sampler.

        Args:
            beam_width: Number of beams to maintain.
        """
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")

        self.beam_width = beam_width

    def sample(self, logits: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """
        Sample tokens using beam search.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).
            num_steps: Number of steps to generate (for multi-step generation).

        Returns:
            Sampled token indices of shape (batch_size, num_steps).
        """
        if logits.dim() != 2:
            raise ValueError("logits must be a 2D tensor")

        if logits.size(0) == 0 or logits.size(1) == 0:
            raise ValueError("logits cannot be empty")

        if num_steps == 1:
            # For single step, use greedy selection
            return torch.argmax(logits, dim=-1, keepdim=True)

        # For multi-step generation, use beam search
        return self._beam_search(logits, num_steps)

    def _beam_search(
        self, logits: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """
        Perform beam search generation.

        Args:
            logits: Initial logits tensor.
            num_steps: Number of steps to generate.

        Returns:
            Generated token sequence.
        """
        batch_size, vocab_size = logits.shape

        # Initialize beams
        beams = torch.zeros(
            batch_size, self.beam_width, num_steps, dtype=torch.long
        )
        beam_scores = torch.zeros(batch_size, self.beam_width)

        # First step: select top-k tokens
        top_k_logits, top_k_indices = torch.topk(
            logits, self.beam_width, dim=-1
        )

        for batch_idx in range(batch_size):
            beams[batch_idx, :, 0] = top_k_indices[batch_idx]
            beam_scores[batch_idx] = top_k_logits[batch_idx]

        # Subsequent steps: expand beams
        for step in range(1, num_steps):
            # For simplicity, we'll use the same logits for all steps
            # In practice, you'd want to use the model to get new logits
            new_logits = logits.unsqueeze(1).expand(-1, self.beam_width, -1)

            # Add beam scores to logits
            new_logits = new_logits + beam_scores.unsqueeze(-1)

            # Reshape for top-k selection
            new_logits = new_logits.view(batch_size, -1)

            # Select top-k from all possible combinations
            top_k_logits, top_k_indices = torch.topk(
                new_logits, self.beam_width, dim=-1
            )

            # Update beams and scores
            for batch_idx in range(batch_size):
                for beam_idx in range(self.beam_width):
                    # Get the original beam and token indices
                    beam_idx_old = (
                        top_k_indices[batch_idx, beam_idx] // vocab_size
                    )
                    token_idx = top_k_indices[batch_idx, beam_idx] % vocab_size

                    # Update beam
                    beams[batch_idx, beam_idx, :step] = beams[
                        batch_idx, beam_idx_old, :step
                    ]
                    beams[batch_idx, beam_idx, step] = token_idx
                    beam_scores[batch_idx, beam_idx] = top_k_logits[
                        batch_idx, beam_idx
                    ]

        # Return the best beam for each batch
        best_beams = beams[:, 0, :]  # Take the first (best) beam

        return best_beams


class TopKSampler(BaseSampler):
    """Top-k sampling (sample from the k most likely tokens)."""

    def __init__(self, k: int = 50):
        """
        Initialize top-k sampler.

        Args:
            k: Number of top tokens to consider.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        self.k = k

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using top-k selection.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).

        Returns:
            Sampled token indices of shape (batch_size, 1).
        """
        if logits.dim() != 2:
            raise ValueError("logits must be a 2D tensor")

        if logits.size(0) == 0 or logits.size(1) == 0:
            raise ValueError("logits cannot be empty")

        # Get top-k tokens
        k = min(self.k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

        # Apply softmax to get probabilities
        probs = F.softmax(top_k_logits, dim=-1)

        # Sample from the distribution
        sampled_indices = torch.multinomial(probs, 1)

        # Get the actual token indices
        batch_size = logits.size(0)
        batch_indices = torch.arange(
            batch_size, device=logits.device
        ).unsqueeze(1)
        sampled_tokens = top_k_indices[batch_indices, sampled_indices]

        return sampled_tokens


class TopPSampler(BaseSampler):
    """Top-p (nucleus) sampling (sample from tokens with cumulative prob p)."""

    def __init__(self, p: float = 0.95):
        """
        Initialize top-p sampler.

        Args:
            p: Cumulative probability threshold (0 < p <= 1).
        """
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")

        self.p = p

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using top-p (nucleus) selection.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).

        Returns:
            Sampled token indices of shape (batch_size, 1).
        """
        if logits.dim() != 2:
            raise ValueError("logits must be a 2D tensor")

        if logits.size(0) == 0 or logits.size(1) == 0:
            raise ValueError("logits cannot be empty")

        # Handle special case p=0 (greedy)
        if self.p == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(
            probs, dim=-1, descending=True
        )

        # Calculate cumulative probabilities
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the cutoff point where cumulative probability exceeds p
        cutoff = cumsum_probs > self.p

        # Set probabilities to 0 for tokens beyond the cutoff
        # Keep at least one token
        cutoff[:, 0] = False

        # Set cutoff probabilities to 0
        sorted_probs[cutoff] = 0

        # Renormalize probabilities
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample from the distribution
        sampled_indices = torch.multinomial(sorted_probs, 1)

        # Get the actual token indices
        batch_size = logits.size(0)
        batch_indices = torch.arange(
            batch_size, device=logits.device
        ).unsqueeze(1)
        sampled_tokens = sorted_indices[batch_indices, sampled_indices]

        return sampled_tokens


class TemperatureSampler(BaseSampler):
    """Temperature sampling (control randomness with temperature parameter)."""

    def __init__(self, temperature: float = 0.7):
        """
        Initialize temperature sampler.

        Args:
            temperature: Temperature parameter (higher = more random).
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using temperature scaling.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).

        Returns:
            Sampled token indices of shape (batch_size, 1).
        """
        if logits.dim() != 2:
            raise ValueError("logits must be a 2D tensor")

        if logits.size(0) == 0 or logits.size(1) == 0:
            raise ValueError("logits cannot be empty")

        # Note: temperature=0 is not allowed in init, but handle it gracefully
        if self.temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Apply softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Sample from the distribution
        sampled = torch.multinomial(probs, 1)

        return sampled


class CombinedSampler(BaseSampler):
    """Combined sampling strategy (e.g., top-k + top-p + temperature)."""

    def __init__(
        self,
        k: Optional[int] = None,
        p: Optional[float] = None,
        temperature: float = 1.0,
    ):
        """
        Initialize combined sampler.

        Args:
            k: Top-k parameter (if None, no top-k filtering).
            p: Top-p parameter (if None, no top-p filtering).
            temperature: Temperature parameter.
        """
        if k is not None and k <= 0:
            raise ValueError("k must be positive")

        if p is not None and not 0 < p <= 1:
            raise ValueError("p must be between 0 and 1")

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.k = k
        self.p = p
        self.temperature = temperature

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens using combined strategy.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).

        Returns:
            Sampled token indices of shape (batch_size, 1).
        """
        if logits.dim() != 2:
            raise ValueError("logits must be a 2D tensor")

        if logits.size(0) == 0 or logits.size(1) == 0:
            raise ValueError("logits cannot be empty")

        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Apply top-k filtering if specified
        if self.k is not None:
            k = min(self.k, scaled_logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(scaled_logits, k, dim=-1)

            # Create new logits tensor with only top-k tokens
            new_logits = torch.full_like(scaled_logits, float("-inf"))
            batch_size = scaled_logits.size(0)
            batch_indices = torch.arange(
                batch_size, device=scaled_logits.device
            ).unsqueeze(1)
            new_logits[batch_indices, top_k_indices] = top_k_logits
            scaled_logits = new_logits

        # Apply top-p filtering if specified
        if self.p is not None:
            probs = F.softmax(scaled_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(
                probs, dim=-1, descending=True
            )
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum_probs > self.p
            cutoff[:, 0] = False  # Keep at least one token
            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(
                dim=-1, keepdim=True
            )

            # Sample from the distribution
            sampled_indices = torch.multinomial(sorted_probs, 1)
            batch_size = scaled_logits.size(0)
            batch_indices = torch.arange(
                batch_size, device=scaled_logits.device
            ).unsqueeze(1)
            sampled_tokens = sorted_indices[batch_indices, sampled_indices]

            return sampled_tokens

        # If no top-p filtering, use standard temperature sampling
        probs = F.softmax(scaled_logits, dim=-1)
        sampled = torch.multinomial(probs, 1)

        return sampled
