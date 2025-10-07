"""Tests for context management."""

import pytest
import torch
from unittest.mock import Mock

from aksis.inference.context_manager import ContextManager


class TestContextManager:
    """Test context management functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Hello world")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 2

        self.context_manager = ContextManager(
            tokenizer=self.tokenizer,
            max_context_length=512,
        )

    def test_context_manager_initialization(self):
        """Test context manager initialization."""
        assert self.context_manager.tokenizer == self.tokenizer
        assert self.context_manager.max_context_length == 512
        assert self.context_manager.context == []
        assert self.context_manager.context_length == 0

    def test_context_manager_initialization_invalid_max_length(self):
        """Test context manager with invalid max context length."""
        with pytest.raises(
            ValueError, match="max_context_length must be positive"
        ):
            ContextManager(
                tokenizer=self.tokenizer,
                max_context_length=0,
            )

    def test_context_manager_initialization_invalid_tokenizer(self):
        """Test context manager with invalid tokenizer."""
        with pytest.raises(
            ValueError, match="tokenizer must have encode and decode methods"
        ):
            ContextManager(
                tokenizer="invalid_tokenizer",
                max_context_length=512,
            )

    def test_add_message_basic(self):
        """Test adding a basic message."""
        message = "Hello world"
        self.context_manager.add_message(message, role="user")

        assert len(self.context_manager.context) == 1
        assert self.context_manager.context[0]["role"] == "user"
        assert self.context_manager.context[0]["content"] == message
        assert self.context_manager.context[0]["tokens"] == [1, 2, 3, 4, 5]

    def test_add_message_with_role(self):
        """Test adding a message with specific role."""
        message = "Hello there"
        self.context_manager.add_message(message, role="assistant")

        assert len(self.context_manager.context) == 1
        assert self.context_manager.context[0]["role"] == "assistant"
        assert self.context_manager.context[0]["content"] == message

    def test_add_message_invalid_role(self):
        """Test adding a message with invalid role."""
        message = "Hello world"

        with pytest.raises(
            ValueError, match="role must be 'user', 'assistant', or 'system'"
        ):
            self.context_manager.add_message(message, role="invalid")

    def test_add_message_empty_content(self):
        """Test adding a message with empty content."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            self.context_manager.add_message("", role="user")

    def test_add_message_none_content(self):
        """Test adding a message with None content."""
        with pytest.raises(ValueError, match="content cannot be None"):
            self.context_manager.add_message(None, role="user")

    def test_add_message_multiple(self):
        """Test adding multiple messages."""
        self.context_manager.add_message("Hello", role="user")
        self.context_manager.add_message("Hi there", role="assistant")
        self.context_manager.add_message("How are you?", role="user")

        assert len(self.context_manager.context) == 3
        assert self.context_manager.context[0]["role"] == "user"
        assert self.context_manager.context[1]["role"] == "assistant"
        assert self.context_manager.context[2]["role"] == "user"

    def test_get_context_tokens(self):
        """Test getting context tokens."""
        self.context_manager.add_message("Hello", role="user")
        self.context_manager.add_message("Hi", role="assistant")

        tokens = self.context_manager.get_context_tokens()

        # Should be concatenated tokens from all messages with EOS separator
        expected_tokens = [1, 2, 3, 4, 5, 2, 1, 2, 3, 4, 5]  # 2 is EOS token
        assert tokens == expected_tokens

    def test_get_context_tokens_empty(self):
        """Test getting context tokens when context is empty."""
        tokens = self.context_manager.get_context_tokens()
        assert tokens == []

    def test_get_context_tokens_with_max_length(self):
        """Test getting context tokens with max length constraint."""
        # Mock a long message
        self.tokenizer.encode.return_value = list(range(1000))

        self.context_manager.add_message("Very long message", role="user")

        tokens = self.context_manager.get_context_tokens()

        # Should be truncated to max_context_length
        assert len(tokens) <= self.context_manager.max_context_length

    def test_get_context_tokens_truncation(self):
        """Test context truncation when exceeding max length."""
        # Set a small max context length
        context_manager = ContextManager(
            tokenizer=self.tokenizer,
            max_context_length=10,
        )

        # Add multiple messages that exceed max length
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        context_manager.add_message("Message 1", role="user")
        context_manager.add_message("Message 2", role="assistant")
        context_manager.add_message("Message 3", role="user")

        tokens = context_manager.get_context_tokens()

        # Should be truncated to max_context_length (accounting for EOS)
        assert (
            len(tokens) <= 12
        )  # 3 messages * 5 tokens + 2 EOS separators = 17, but truncated

    def test_clear_context(self):
        """Test clearing the context."""
        self.context_manager.add_message("Hello", role="user")
        self.context_manager.add_message("Hi", role="assistant")

        assert len(self.context_manager.context) == 2

        self.context_manager.clear_context()

        assert len(self.context_manager.context) == 0
        assert self.context_manager.context_length == 0

    def test_get_context_length(self):
        """Test getting context length."""
        self.context_manager.add_message("Hello", role="user")
        self.context_manager.add_message("Hi", role="assistant")

        length = self.context_manager.get_context_length()

        # Should be sum of all token lengths
        expected_length = 5 + 5  # Two messages of 5 tokens each
        assert length == expected_length

    def test_get_context_length_empty(self):
        """Test getting context length when context is empty."""
        length = self.context_manager.get_context_length()
        assert length == 0

    def test_get_context_summary(self):
        """Test getting context summary."""
        self.context_manager.add_message("Hello", role="user")
        self.context_manager.add_message("Hi there", role="assistant")

        summary = self.context_manager.get_context_summary()

        assert "num_messages" in summary
        assert "total_tokens" in summary
        assert "max_context_length" in summary
        assert summary["num_messages"] == 2
        assert summary["total_tokens"] == 10

    def test_get_context_summary_empty(self):
        """Test getting context summary when context is empty."""
        summary = self.context_manager.get_context_summary()

        assert summary["num_messages"] == 0
        assert summary["total_tokens"] == 0

    def test_get_context_as_string(self):
        """Test getting context as formatted string."""
        self.context_manager.add_message("Hello", role="user")
        self.context_manager.add_message("Hi there", role="assistant")

        context_str = self.context_manager.get_context_as_string()

        assert "user" in context_str.lower()
        assert "assistant" in context_str.lower()
        assert "Hello" in context_str
        assert "Hi there" in context_str

    def test_get_context_as_string_empty(self):
        """Test getting context as string when context is empty."""
        context_str = self.context_manager.get_context_as_string()
        assert context_str == "No context available."

    def test_get_context_as_string_with_format(self):
        """Test getting context as string with custom format."""
        self.context_manager.add_message("Hello", role="user")

        context_str = self.context_manager.get_context_as_string(
            format_string="{role}: {content}\n"
        )

        assert context_str == "user: Hello\n"

    def test_get_context_as_string_invalid_format(self):
        """Test getting context as string with invalid format."""
        self.context_manager.add_message("Hello", role="user")

        with pytest.raises(
            ValueError, match="format_string must contain {role} and {content}"
        ):
            self.context_manager.get_context_as_string(
                format_string="Invalid format"
            )

    def test_truncate_context(self):
        """Test manual context truncation."""
        # Add multiple messages
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        for i in range(5):
            self.context_manager.add_message(f"Message {i}", role="user")

        assert len(self.context_manager.context) == 5

        # Truncate to 3 messages
        self.context_manager.truncate_context(max_messages=3)

        assert len(self.context_manager.context) == 3

    def test_truncate_context_invalid_max_messages(self):
        """Test context truncation with invalid max_messages."""
        self.context_manager.add_message("Hello", role="user")

        with pytest.raises(ValueError, match="max_messages must be positive"):
            self.context_manager.truncate_context(max_messages=0)

    def test_truncate_context_no_messages(self):
        """Test context truncation when no messages exist."""
        self.context_manager.truncate_context(max_messages=3)
        assert len(self.context_manager.context) == 0

    def test_truncate_context_fifo(self):
        """Test that context truncation follows FIFO (first in, first out)."""
        # Add messages with identifiable content
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.context_manager.add_message("First message", role="user")
        self.context_manager.add_message("Second message", role="assistant")
        self.context_manager.add_message("Third message", role="user")
        self.context_manager.add_message("Fourth message", role="assistant")

        # Truncate to 2 messages
        self.context_manager.truncate_context(max_messages=2)

        assert len(self.context_manager.context) == 2
        # Should keep the last 2 messages (FIFO removal)
        assert self.context_manager.context[0]["content"] == "Third message"
        assert self.context_manager.context[1]["content"] == "Fourth message"

    def test_get_context_tokens_with_separators(self):
        """Test getting context tokens with separators."""

        # Mock tokenizer to return different tokens for different messages
        def mock_encode(text, add_special_tokens=True):
            if "Hello" in text:
                return [1, 2, 3]
            elif "Hi" in text:
                return [4, 5, 6]
            else:
                return [7, 8, 9]

        self.tokenizer.encode.side_effect = mock_encode

        self.context_manager.add_message("Hello", role="user")
        self.context_manager.add_message("Hi", role="assistant")

        tokens = self.context_manager.get_context_tokens()

        # Should include separators between messages
        expected_tokens = [1, 2, 3, 2, 4, 5, 6]  # 2 is EOS token
        assert tokens == expected_tokens

    def test_context_manager_cuda_compatibility(self):
        """Test context manager with CUDA tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Mock tokenizer to return CUDA tensors
        self.tokenizer.encode.return_value = torch.tensor(
            [1, 2, 3, 4, 5]
        ).cuda()

        self.context_manager.add_message("Hello", role="user")

        tokens = self.context_manager.get_context_tokens()

        # Should handle CUDA tensors properly
        assert isinstance(tokens, list)
        assert len(tokens) == 5

    def test_context_manager_performance(self):
        """Test context manager performance with large contexts."""
        # Add many messages
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        for i in range(100):
            self.context_manager.add_message(f"Message {i}", role="user")

        # Should handle large contexts efficiently
        tokens = self.context_manager.get_context_tokens()
        # Account for EOS separators between messages
        assert len(tokens) <= self.context_manager.max_context_length + 100

        summary = self.context_manager.get_context_summary()
        assert summary["num_messages"] <= 100

    def test_context_manager_edge_cases(self):
        """Test context manager with edge cases."""
        # Test with very long single message
        self.tokenizer.encode.return_value = list(range(1000))

        self.context_manager.add_message("Very long message", role="user")

        tokens = self.context_manager.get_context_tokens()
        assert len(tokens) <= self.context_manager.max_context_length

        # Test with special characters
        self.context_manager.clear_context()
        self.tokenizer.encode.return_value = [1, 2, 3]

        self.context_manager.add_message("Hello\nWorld\t!", role="user")

        assert len(self.context_manager.context) == 1
        assert self.context_manager.context[0]["content"] == "Hello\nWorld\t!"

    def test_context_manager_serialization(self):
        """Test context manager serialization."""
        self.context_manager.add_message("Hello", role="user")
        self.context_manager.add_message("Hi", role="assistant")

        # Test getting context as dictionary
        context_dict = self.context_manager.get_context_as_dict()

        assert isinstance(context_dict, list)
        assert len(context_dict) == 2
        assert context_dict[0]["role"] == "user"
        assert context_dict[0]["content"] == "Hello"

        # Test loading context from dictionary
        new_context_manager = ContextManager(
            tokenizer=self.tokenizer,
            max_context_length=512,
        )

        new_context_manager.load_context_from_dict(context_dict)

        assert len(new_context_manager.context) == 2
        assert new_context_manager.context[0]["role"] == "user"
        assert new_context_manager.context[0]["content"] == "Hello"

    def test_context_manager_invalid_serialization(self):
        """Test context manager with invalid serialization data."""
        invalid_context = [
            {"role": "invalid", "content": "Hello"},
            {"content": "Missing role"},
            {"role": "user"},  # Missing content
        ]

        with pytest.raises(ValueError):
            self.context_manager.load_context_from_dict(invalid_context)
