"""Interactive chatbot interface for Aksis AI chatbot/LLM."""

import logging
from typing import List, Dict, Any, Optional
import time

from aksis.inference.inference import Generator
from aksis.inference.context_manager import ContextManager

logger = logging.getLogger(__name__)


class ChatBot:
    """Interactive chatbot interface."""

    def __init__(
        self,
        generator: Generator,
        context_manager: ContextManager,
        sampler: Any,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 100,
        stop_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize the chatbot.

        Args:
            generator: Text generation engine.
            context_manager: Context management system.
            sampler: Sampling strategy for generation.
            system_prompt: System prompt to set the chatbot's behavior.
            max_new_tokens: Maximum number of new tokens to generate.
            stop_tokens: List of stop tokens to halt generation.
        """
        if not isinstance(generator, Generator):
            raise ValueError("generator must be a Generator")

        if not isinstance(context_manager, ContextManager):
            raise ValueError("context_manager must be a ContextManager")

        if not hasattr(sampler, "sample"):
            raise ValueError("sampler must have a sample method")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        if stop_tokens is not None and not isinstance(stop_tokens, list):
            raise ValueError("stop_tokens must be a list")

        self.generator = generator
        self.context_manager = context_manager
        self.sampler = sampler
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.stop_tokens = stop_tokens or ["<|endoftext|>", "<|end|>"]

        # Add system prompt to context if provided
        if self.system_prompt:
            self.context_manager.add_message(self.system_prompt, role="system")

        logger.info("ChatBot initialized")

    def chat(
        self,
        user_input: str,
        max_new_tokens: Optional[int] = None,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """
        Process a user input and generate a response.

        Args:
            user_input: User's input message.
            max_new_tokens: Maximum number of new tokens to generate.
            stop_tokens: List of stop tokens to halt generation.

        Returns:
            Generated response.
        """
        if user_input is None:
            raise ValueError("Input cannot be None")

        if not user_input or not user_input.strip():
            raise ValueError("Input cannot be empty")

        # Use provided parameters or defaults
        max_tokens = max_new_tokens or self.max_new_tokens
        stop_tokens = stop_tokens or self.stop_tokens

        logger.debug(f"Processing user input: {user_input[:50]}...")

        try:
            # Add user message to context
            self.context_manager.add_message(user_input, role="user")

            # Get context for generation
            context_tokens = self.context_manager.get_context_tokens()

            # Convert context tokens to string for generation
            if context_tokens:
                context_text = self.generator.tokenizer.decode(context_tokens)
            else:
                context_text = user_input

            # Generate response
            start_time = time.time()
            response = self.generator.generate(
                prompt=context_text,
                sampler=self.sampler,
                max_new_tokens=max_tokens,
                stop_tokens=stop_tokens,
            )
            generation_time = time.time() - start_time

            # Validate response
            if response is None:
                raise RuntimeError("Generator returned None")

            # Handle empty or whitespace-only responses
            if not response or not response.strip():
                response = "I'm sorry, I couldn't generate a response. Please try again."
                logger.warning("Generated empty response, using fallback")

            # Add assistant response to context
            self.context_manager.add_message(response, role="assistant")

            logger.debug(f"Generated response in {generation_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"Error during chat: {e}")
            raise

    def clear_context(self) -> None:
        """Clear the conversation context."""
        self.context_manager.clear_context()

        # Re-add system prompt if it exists
        if self.system_prompt:
            self.context_manager.add_message(self.system_prompt, role="system")

        logger.debug("Context cleared")

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.

        Returns:
            Dictionary containing context statistics.
        """
        return self.context_manager.get_context_summary()

    def get_context_as_string(
        self,
        format_string: str = "{role}: {content}\n",
    ) -> str:
        """
        Get the context as a formatted string.

        Args:
            format_string: Format string for each message.

        Returns:
            Formatted context string.
        """
        return self.context_manager.get_context_as_string(format_string)

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        """
        Set or update the system prompt.

        Args:
            prompt: New system prompt (None to remove).
        """
        # Remove old system prompt if it exists
        if self.system_prompt:
            # Find and remove system messages
            system_messages = self.context_manager.get_messages_by_role(
                "system"
            )
            for _ in system_messages:
                self.context_manager.remove_last_message()

        # Set new system prompt
        self.system_prompt = prompt

        # Add new system prompt to context if provided
        if self.system_prompt:
            self.context_manager.add_message(self.system_prompt, role="system")

        logger.debug("System prompt updated")

    def set_max_new_tokens(self, max_tokens: int) -> None:
        """
        Set the maximum number of new tokens to generate.

        Args:
            max_tokens: Maximum number of new tokens.
        """
        if max_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        self.max_new_tokens = max_tokens
        logger.debug(f"Max new tokens set to {max_tokens}")

    def set_stop_tokens(self, stop_tokens: List[str]) -> None:
        """
        Set the stop tokens for generation.

        Args:
            stop_tokens: List of stop tokens.
        """
        if not isinstance(stop_tokens, list):
            raise ValueError("stop_tokens must be a list")

        self.stop_tokens = stop_tokens
        logger.debug(f"Stop tokens set to {stop_tokens}")

    def set_sampler(self, sampler: Any) -> None:
        """
        Set the sampling strategy.

        Args:
            sampler: New sampling strategy.
        """
        if not hasattr(sampler, "sample"):
            raise ValueError("sampler must have a sample method")

        self.sampler = sampler
        logger.debug("Sampler updated")

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the chatbot.

        Returns:
            Dictionary containing chatbot information.
        """
        return {
            "generator": self.generator.get_model_info(),
            "context_manager": self.context_manager.get_context_summary(),
            "sampler": str(type(self.sampler).__name__),
            "system_prompt": self.system_prompt,
            "max_new_tokens": self.max_new_tokens,
            "stop_tokens": self.stop_tokens,
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the chatbot.

        Returns:
            Dictionary containing chatbot state.
        """
        return {
            "system_prompt": self.system_prompt,
            "max_new_tokens": self.max_new_tokens,
            "stop_tokens": self.stop_tokens,
            "context": self.context_manager.get_context_as_dict(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load chatbot state from a dictionary.

        Args:
            state: Dictionary containing chatbot state.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")

        # Validate required keys
        required_keys = ["system_prompt", "max_new_tokens", "stop_tokens"]
        if not all(key in state for key in required_keys):
            raise ValueError("state must contain all required keys")

        # Load state
        self.system_prompt = state["system_prompt"]

        if (
            not isinstance(state["max_new_tokens"], int)
            or state["max_new_tokens"] <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        self.max_new_tokens = state["max_new_tokens"]

        if not isinstance(state["stop_tokens"], list):
            raise ValueError("stop_tokens must be a list")
        self.stop_tokens = state["stop_tokens"]

        # Load context if provided
        if "context" in state:
            self.context_manager.load_context_from_dict(state["context"])

        logger.debug("ChatBot state loaded")

    def save_context(self, filepath: str) -> None:
        """
        Save the current context to a file.

        Args:
            filepath: Path to save the context.
        """
        self.context_manager.save_context(filepath)

    def load_context(self, filepath: str) -> None:
        """
        Load context from a file.

        Args:
            filepath: Path to load the context from.
        """
        self.context_manager.load_context(filepath)

    def get_message_count(self) -> int:
        """
        Get the number of messages in the context.

        Returns:
            Number of messages.
        """
        return self.context_manager.get_message_count()

    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get all messages with a specific role.

        Args:
            role: Role to filter by.

        Returns:
            List of messages with the specified role.
        """
        return self.context_manager.get_messages_by_role(role)

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message in the context.

        Returns:
            Last message dictionary or None if context is empty.
        """
        return self.context_manager.get_last_message()

    def get_last_message_by_role(self, role: str) -> Optional[Dict[str, Any]]:
        """
        Get the last message with a specific role.

        Args:
            role: Role to filter by.

        Returns:
            Last message with the specified role or None.
        """
        return self.context_manager.get_last_message_by_role(role)

    def remove_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Remove and return the last message from the context.

        Returns:
            Removed message dictionary or None if context is empty.
        """
        return self.context_manager.remove_last_message()

    def insert_message(
        self,
        content: str,
        role: str = "user",
        position: int = -1,
    ) -> None:
        """
        Insert a message at a specific position.

        Args:
            content: Message content.
            role: Message role.
            position: Position to insert at (-1 for end).
        """
        self.context_manager.insert_message(content, role, position)

    def replace_message(
        self,
        content: str,
        role: str = "user",
        position: int = -1,
    ) -> Optional[Dict[str, Any]]:
        """
        Replace a message at a specific position.

        Args:
            content: New message content.
            role: New message role.
            position: Position to replace (-1 for last).

        Returns:
            Replaced message dictionary or None if position is invalid.
        """
        return self.context_manager.replace_message(content, role, position)

    def benchmark_generation(
        self,
        prompt: str,
        num_runs: int = 5,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark generation performance.

        Args:
            prompt: Prompt to use for benchmarking.
            num_runs: Number of runs to average.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Dictionary containing benchmark results.
        """
        max_tokens = max_new_tokens or self.max_new_tokens

        times = []
        token_counts = []

        for _ in range(num_runs):
            start_time = time.time()
            response = self.generator.generate(
                prompt=prompt,
                sampler=self.sampler,
                max_new_tokens=max_tokens,
                stop_tokens=self.stop_tokens,
            )
            end_time = time.time()

            times.append(end_time - start_time)
            token_counts.append(len(response.split()))

        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0

        return {
            "avg_time": avg_time,
            "avg_tokens": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "num_runs": num_runs,
            "prompt_length": len(prompt),
        }
