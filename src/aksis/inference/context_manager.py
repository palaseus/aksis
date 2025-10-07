"""Context management for conversation history."""

import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages conversation context and history."""

    def __init__(
        self,
        tokenizer: Any,
        max_context_length: int = 512,
    ):
        """
        Initialize context manager.

        Args:
            tokenizer: Tokenizer for encoding/decoding text.
            max_context_length: Maximum number of tokens to keep in context.
        """
        if not hasattr(tokenizer, "encode") or not hasattr(
            tokenizer, "decode"
        ):
            raise ValueError("tokenizer must have encode and decode methods")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")

        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.context: List[Dict[str, Any]] = []
        self.context_length = 0

        logger.info(
            f"ContextManager initialized with max_length: {max_context_length}"
        )

    def add_message(
        self,
        content: str,
        role: str = "user",
    ) -> None:
        """
        Add a message to the context.

        Args:
            content: Message content.
            role: Message role ('user', 'assistant', or 'system').
        """
        if content is None:
            raise ValueError("content cannot be None")

        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        if role not in ["user", "assistant", "system"]:
            raise ValueError("role must be 'user', 'assistant', or 'system'")

        # Tokenize the content
        tokens = self.tokenizer.encode(content, add_special_tokens=True)

        # Create message entry
        message = {
            "role": role,
            "content": content,
            "tokens": tokens,
            "token_count": len(tokens),
        }

        # Add to context
        self.context.append(message)
        self.context_length += len(tokens)

        # Truncate if necessary
        self._truncate_context()

        logger.debug(f"Added {role} message with {len(tokens)} tokens")

    def get_context_tokens(self) -> List[int]:
        """
        Get all context tokens as a list.

        Returns:
            List of token IDs representing the full context.
        """
        if not self.context:
            return []

        tokens = []
        for i, message in enumerate(self.context):
            tokens.extend(message["tokens"])
            # Add separator token (EOS) between messages, but not after last
            if i < len(self.context) - 1 and hasattr(
                self.tokenizer, "eos_token_id"
            ):
                tokens.append(self.tokenizer.eos_token_id)

        return tokens

    def get_context_length(self) -> int:
        """
        Get the current context length in tokens.

        Returns:
            Number of tokens in the context.
        """
        return self.context_length

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.

        Returns:
            Dictionary containing context statistics.
        """
        return {
            "num_messages": len(self.context),
            "total_tokens": self.context_length,
            "max_context_length": self.max_context_length,
            "context_usage": self.context_length / self.max_context_length,
        }

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
        if not self.context:
            return "No context available."

        if "{role}" not in format_string or "{content}" not in format_string:
            raise ValueError("format_string must contain {role} and {content}")

        formatted_messages = []
        for message in self.context:
            formatted_message = format_string.format(
                role=message["role"], content=message["content"]
            )
            formatted_messages.append(formatted_message)

        return "".join(formatted_messages)

    def get_context_as_dict(self) -> List[Dict[str, Any]]:
        """
        Get the context as a list of dictionaries.

        Returns:
            List of message dictionaries.
        """
        return self.context.copy()

    def load_context_from_dict(
        self, context_data: List[Dict[str, Any]]
    ) -> None:
        """
        Load context from a list of dictionaries.

        Args:
            context_data: List of message dictionaries.
        """
        if not isinstance(context_data, list):
            raise ValueError("context_data must be a list")

        # Clear current context
        self.clear_context()

        # Load new context
        for message_data in context_data:
            if not isinstance(message_data, dict):
                raise ValueError("Each message must be a dictionary")

            required_keys = ["role", "content"]
            if not all(key in message_data for key in required_keys):
                raise ValueError(
                    "Each message must have 'role' and 'content' keys"
                )

            role = message_data["role"]
            content = message_data["content"]

            if role not in ["user", "assistant", "system"]:
                raise ValueError(
                    "role must be 'user', 'assistant', or 'system'"
                )

            if not content or not content.strip():
                raise ValueError("content cannot be empty")

            # Add message (this will tokenize and add to context)
            self.add_message(content, role)

        logger.info(f"Loaded context with {len(self.context)} messages")

    def clear_context(self) -> None:
        """Clear all context."""
        self.context.clear()
        self.context_length = 0
        logger.debug("Context cleared")

    def truncate_context(self, max_messages: int) -> None:
        """
        Truncate context to keep only the most recent messages.

        Args:
            max_messages: Maximum number of messages to keep.
        """
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")

        if len(self.context) <= max_messages:
            return

        # Remove oldest messages (FIFO)
        messages_to_remove = len(self.context) - max_messages
        removed_messages = self.context[:messages_to_remove]

        # Update context length
        for message in removed_messages:
            self.context_length -= message["token_count"]

        # Remove messages
        self.context = self.context[messages_to_remove:]

        logger.debug(f"Truncated context to {max_messages} messages")

    def _truncate_context(self) -> None:
        """Truncate context if it exceeds max_context_length."""
        if self.context_length <= self.max_context_length:
            return

        # Remove oldest messages until we're under the limit
        while self.context_length > self.max_context_length and self.context:
            oldest_message = self.context.pop(0)
            self.context_length -= oldest_message["token_count"]

        logger.debug(f"Truncated context to {self.context_length} tokens")

    def save_context(self, filepath: str) -> None:
        """
        Save context to a file.

        Args:
            filepath: Path to save the context.
        """
        context_data = {
            "context": self.get_context_as_dict(),
            "max_context_length": self.max_context_length,
            "context_length": self.context_length,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Context saved to {filepath}")

    def load_context(self, filepath: str) -> None:
        """
        Load context from a file.

        Args:
            filepath: Path to load the context from.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            context_data = json.load(f)

        if "context" not in context_data:
            raise ValueError("Invalid context file format")

        # Load context messages
        self.load_context_from_dict(context_data["context"])

        # Update max context length if specified
        if "max_context_length" in context_data:
            self.max_context_length = context_data["max_context_length"]

        logger.info(f"Context loaded from {filepath}")

    def get_message_count(self) -> int:
        """
        Get the number of messages in the context.

        Returns:
            Number of messages.
        """
        return len(self.context)

    def get_messages_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Get all messages with a specific role.

        Args:
            role: Role to filter by.

        Returns:
            List of messages with the specified role.
        """
        if role not in ["user", "assistant", "system"]:
            raise ValueError("role must be 'user', 'assistant', or 'system'")

        return [msg for msg in self.context if msg["role"] == role]

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message in the context.

        Returns:
            Last message dictionary or None if context is empty.
        """
        if not self.context:
            return None

        return self.context[-1]

    def get_last_message_by_role(self, role: str) -> Optional[Dict[str, Any]]:
        """
        Get the last message with a specific role.

        Args:
            role: Role to filter by.

        Returns:
            Last message with the specified role or None.
        """
        if role not in ["user", "assistant", "system"]:
            raise ValueError("role must be 'user', 'assistant', or 'system'")

        for message in reversed(self.context):
            if message["role"] == role:
                return message

        return None

    def remove_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Remove and return the last message from the context.

        Returns:
            Removed message dictionary or None if context is empty.
        """
        if not self.context:
            return None

        last_message = self.context.pop()
        self.context_length -= last_message["token_count"]

        logger.debug(f"Removed last {last_message['role']} message")
        return last_message

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
        if content is None:
            raise ValueError("content cannot be None")

        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        if role not in ["user", "assistant", "system"]:
            raise ValueError("role must be 'user', 'assistant', or 'system'")

        # Tokenize the content
        tokens = self.tokenizer.encode(content, add_special_tokens=True)

        # Create message entry
        message = {
            "role": role,
            "content": content,
            "tokens": tokens,
            "token_count": len(tokens),
        }

        # Insert at position
        if position == -1:
            self.context.append(message)
        else:
            self.context.insert(position, message)

        self.context_length += len(tokens)

        # Truncate if necessary
        self._truncate_context()

        logger.debug(f"Inserted {role} message at position {position}")

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
        if content is None:
            raise ValueError("content cannot be None")

        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        if role not in ["user", "assistant", "system"]:
            raise ValueError("role must be 'user', 'assistant', or 'system'")

        if not self.context:
            return None

        # Get position
        if position == -1:
            position = len(self.context) - 1

        if position < 0 or position >= len(self.context):
            return None

        # Get old message
        old_message = self.context[position]

        # Tokenize new content
        tokens = self.tokenizer.encode(content, add_special_tokens=True)

        # Create new message
        new_message = {
            "role": role,
            "content": content,
            "tokens": tokens,
            "token_count": len(tokens),
        }

        # Replace message
        self.context[position] = new_message

        # Update context length
        self.context_length -= old_message["token_count"]
        self.context_length += len(tokens)

        # Truncate if necessary
        self._truncate_context()

        logger.debug(f"Replaced message at position {position}")
        return old_message
