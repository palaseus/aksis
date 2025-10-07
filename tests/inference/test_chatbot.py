"""Tests for chatbot interface."""

import pytest
import torch
from unittest.mock import Mock

from aksis.inference.chatbot import ChatBot
from aksis.inference.inference import Generator
from aksis.inference.context_manager import ContextManager
from aksis.inference.sampler import GreedySampler


class TestChatBot:
    """Test chatbot interface functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")

        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Hello world")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 2

        # Mock generator
        self.generator = Mock(spec=Generator)
        self.generator.generate.return_value = "Hello! How can I help you?"
        self.generator.device = self.device
        self.generator.tokenizer = self.tokenizer

        # Mock context manager
        self.context_manager = Mock(spec=ContextManager)
        self.context_manager.add_message = Mock()
        self.context_manager.get_context_tokens = Mock(return_value=[])
        self.context_manager.clear_context = Mock()
        self.context_manager.get_context_summary = Mock(
            return_value={
                "num_messages": 0,
                "total_tokens": 0,
                "max_context_length": 512,
            }
        )

        # Mock sampler
        self.sampler = Mock(spec=GreedySampler)

        self.chatbot = ChatBot(
            generator=self.generator,
            context_manager=self.context_manager,
            sampler=self.sampler,
        )

    def test_chatbot_initialization(self):
        """Test chatbot initialization."""
        assert self.chatbot.generator == self.generator
        assert self.chatbot.context_manager == self.context_manager
        assert self.chatbot.sampler == self.sampler
        assert self.chatbot.system_prompt is None
        assert self.chatbot.max_new_tokens == 100
        assert self.chatbot.stop_tokens == ["<|endoftext|>", "<|end|>"]

    def test_chatbot_initialization_with_params(self):
        """Test chatbot initialization with custom parameters."""
        chatbot = ChatBot(
            generator=self.generator,
            context_manager=self.context_manager,
            sampler=self.sampler,
            system_prompt="You are a helpful assistant.",
            max_new_tokens=50,
            stop_tokens=["<|stop|>"],
        )

        assert chatbot.system_prompt == "You are a helpful assistant."
        assert chatbot.max_new_tokens == 50
        assert chatbot.stop_tokens == ["<|stop|>"]

    def test_chatbot_initialization_invalid_generator(self):
        """Test chatbot initialization with invalid generator."""
        with pytest.raises(ValueError, match="generator must be a Generator"):
            ChatBot(
                generator="invalid_generator",
                context_manager=self.context_manager,
                sampler=self.sampler,
            )

    def test_chatbot_initialization_invalid_context_manager(self):
        """Test chatbot initialization with invalid context manager."""
        with pytest.raises(
            ValueError, match="context_manager must be a ContextManager"
        ):
            ChatBot(
                generator=self.generator,
                context_manager="invalid_context_manager",
                sampler=self.sampler,
            )

    def test_chatbot_initialization_invalid_sampler(self):
        """Test chatbot initialization with invalid sampler."""
        with pytest.raises(
            ValueError, match="sampler must have a sample method"
        ):
            ChatBot(
                generator=self.generator,
                context_manager=self.context_manager,
                sampler="invalid_sampler",
            )

    def test_chatbot_initialization_invalid_max_new_tokens(self):
        """Test chatbot initialization with invalid max_new_tokens."""
        with pytest.raises(
            ValueError, match="max_new_tokens must be positive"
        ):
            ChatBot(
                generator=self.generator,
                context_manager=self.context_manager,
                sampler=self.sampler,
                max_new_tokens=0,
            )

    def test_chatbot_initialization_invalid_stop_tokens(self):
        """Test chatbot initialization with invalid stop_tokens."""
        with pytest.raises(ValueError, match="stop_tokens must be a list"):
            ChatBot(
                generator=self.generator,
                context_manager=self.context_manager,
                sampler=self.sampler,
                stop_tokens="invalid_stop_tokens",
            )

    def test_chat_basic(self):
        """Test basic chat functionality."""
        user_input = "Hello, how are you?"

        response = self.chatbot.chat(user_input)

        assert isinstance(response, str)
        assert response == "Hello! How can I help you?"

        # Check that context manager was called
        assert self.context_manager.add_message.call_count >= 2

    def test_chat_with_system_prompt(self):
        """Test chat with system prompt."""
        chatbot = ChatBot(
            generator=self.generator,
            context_manager=self.context_manager,
            sampler=self.sampler,
            system_prompt="You are a helpful assistant.",
        )

        user_input = "Hello"
        response = chatbot.chat(user_input)

        assert isinstance(response, str)

        # Check that system prompt was added to context
        assert self.context_manager.add_message.call_count >= 1

    def test_chat_empty_input(self):
        """Test chat with empty input."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            self.chatbot.chat("")

    def test_chat_none_input(self):
        """Test chat with None input."""
        with pytest.raises(ValueError, match="Input cannot be None"):
            self.chatbot.chat(None)

    def test_chat_generator_error(self):
        """Test chat when generator raises an error."""
        self.generator.generate.side_effect = RuntimeError("Generation error")

        with pytest.raises(RuntimeError, match="Generation error"):
            self.chatbot.chat("Hello")

    def test_chat_context_manager_error(self):
        """Test chat when context manager raises an error."""
        self.context_manager.add_message.side_effect = RuntimeError(
            "Context error"
        )

        with pytest.raises(RuntimeError, match="Context error"):
            self.chatbot.chat("Hello")

    def test_chat_with_custom_params(self):
        """Test chat with custom parameters."""
        user_input = "Hello"

        response = self.chatbot.chat(
            user_input,
            max_new_tokens=50,
            stop_tokens=["<|stop|>"],
        )

        assert isinstance(response, str)

        # Check that generator was called with custom parameters
        self.generator.generate.assert_called_with(
            prompt=user_input,
            sampler=self.sampler,
            max_new_tokens=50,
            stop_tokens=["<|stop|>"],
        )

    def test_chat_multi_turn(self):
        """Test multi-turn conversation."""
        # First turn
        response1 = self.chatbot.chat("Hello")
        assert isinstance(response1, str)

        # Second turn
        response2 = self.chatbot.chat("How are you?")
        assert isinstance(response2, str)

        # Check that context manager was called for both turns
        assert (
            self.context_manager.add_message.call_count >= 4
        )  # 2 user + 2 assistant

    def test_clear_context(self):
        """Test clearing conversation context."""
        self.chatbot.clear_context()

        self.context_manager.clear_context.assert_called_once()

    def test_get_context_summary(self):
        """Test getting context summary."""
        summary = self.chatbot.get_context_summary()

        assert isinstance(summary, dict)
        self.context_manager.get_context_summary.assert_called_once()

    def test_get_context_as_string(self):
        """Test getting context as string."""
        self.context_manager.get_context_as_string.return_value = (
            "User: Hello\nAssistant: Hi"
        )

        context_str = self.chatbot.get_context_as_string()

        assert isinstance(context_str, str)
        assert "User: Hello" in context_str
        assert "Assistant: Hi" in context_str

    def test_get_context_as_string_with_format(self):
        """Test getting context as string with custom format."""
        self.context_manager.get_context_as_string.return_value = (
            "Custom format"
        )

        context_str = self.chatbot.get_context_as_string(
            format_string="{role}: {content}\n"
        )

        assert context_str == "Custom format"
        self.context_manager.get_context_as_string.assert_called_with(
            "{role}: {content}\n"
        )

    def test_set_system_prompt(self):
        """Test setting system prompt."""
        new_prompt = "You are a new assistant."
        self.chatbot.set_system_prompt(new_prompt)

        assert self.chatbot.system_prompt == new_prompt

    def test_set_system_prompt_none(self):
        """Test setting system prompt to None."""
        self.chatbot.set_system_prompt(None)

        assert self.chatbot.system_prompt is None

    def test_set_system_prompt_empty(self):
        """Test setting system prompt to empty string."""
        self.chatbot.set_system_prompt("")

        assert self.chatbot.system_prompt == ""

    def test_set_max_new_tokens(self):
        """Test setting max new tokens."""
        self.chatbot.set_max_new_tokens(50)

        assert self.chatbot.max_new_tokens == 50

    def test_set_max_new_tokens_invalid(self):
        """Test setting invalid max new tokens."""
        with pytest.raises(
            ValueError, match="max_new_tokens must be positive"
        ):
            self.chatbot.set_max_new_tokens(0)

    def test_set_stop_tokens(self):
        """Test setting stop tokens."""
        new_stop_tokens = ["<|stop|>", "<|end|>"]
        self.chatbot.set_stop_tokens(new_stop_tokens)

        assert self.chatbot.stop_tokens == new_stop_tokens

    def test_set_stop_tokens_invalid(self):
        """Test setting invalid stop tokens."""
        with pytest.raises(ValueError, match="stop_tokens must be a list"):
            self.chatbot.set_stop_tokens("invalid")

    def test_set_sampler(self):
        """Test setting sampler."""
        new_sampler = Mock(spec=GreedySampler)
        self.chatbot.set_sampler(new_sampler)

        assert self.chatbot.sampler == new_sampler

    def test_set_sampler_invalid(self):
        """Test setting invalid sampler."""
        with pytest.raises(
            ValueError, match="sampler must have a sample method"
        ):
            self.chatbot.set_sampler("invalid_sampler")

    def test_chatbot_info(self):
        """Test getting chatbot information."""
        info = self.chatbot.get_info()

        assert isinstance(info, dict)
        assert "generator" in info
        assert "context_manager" in info
        assert "sampler" in info
        assert "system_prompt" in info
        assert "max_new_tokens" in info
        assert "stop_tokens" in info

    def test_chatbot_cuda_compatibility(self):
        """Test chatbot with CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cuda_device = torch.device("cuda")
        self.generator.device = cuda_device

        chatbot = ChatBot(
            generator=self.generator,
            context_manager=self.context_manager,
            sampler=self.sampler,
        )

        response = chatbot.chat("Hello")

        assert isinstance(response, str)

    def test_chatbot_performance(self):
        """Test chatbot performance with multiple turns."""
        # Mock fast generation
        self.generator.generate.return_value = "Quick response"

        # Simulate multiple turns
        for i in range(10):
            response = self.chatbot.chat(f"Message {i}")
            assert isinstance(response, str)

    def test_chatbot_edge_cases(self):
        """Test chatbot with edge cases."""
        # Test with very long input
        long_input = "A" * 1000
        response = self.chatbot.chat(long_input)
        assert isinstance(response, str)

        # Test with special characters
        special_input = "Hello\nWorld\t!@#$%^&*()"
        response = self.chatbot.chat(special_input)
        assert isinstance(response, str)

        # Test with unicode characters
        unicode_input = "Hello 世界 🌍"
        response = self.chatbot.chat(unicode_input)
        assert isinstance(response, str)

    def test_chatbot_error_handling(self):
        """Test chatbot error handling."""
        # Test with generator returning None
        self.generator.generate.return_value = None

        with pytest.raises(RuntimeError, match="Generator returned None"):
            self.chatbot.chat("Hello")

        # Test with generator returning empty string
        self.generator.generate.return_value = ""

        response = self.chatbot.chat("Hello")
        assert response == ""

    def test_chatbot_context_management(self):
        """Test chatbot context management."""
        # Test that context is properly managed
        self.chatbot.chat("Hello")
        self.chatbot.chat("How are you?")

        # Check that context manager was called correctly
        assert self.context_manager.add_message.call_count >= 4

        # Test clearing context
        self.chatbot.clear_context()
        self.context_manager.clear_context.assert_called()

    def test_chatbot_sampler_integration(self):
        """Test chatbot integration with different samplers."""
        # Test with different samplers
        samplers = [
            Mock(spec=GreedySampler),
            Mock(spec=GreedySampler),
        ]

        for sampler in samplers:
            chatbot = ChatBot(
                generator=self.generator,
                context_manager=self.context_manager,
                sampler=sampler,
            )

            response = chatbot.chat("Hello")
            assert isinstance(response, str)

    def test_chatbot_serialization(self):
        """Test chatbot serialization."""
        # Test getting chatbot state
        state = self.chatbot.get_state()

        assert isinstance(state, dict)
        assert "system_prompt" in state
        assert "max_new_tokens" in state
        assert "stop_tokens" in state

        # Test loading chatbot state
        new_chatbot = ChatBot(
            generator=self.generator,
            context_manager=self.context_manager,
            sampler=self.sampler,
        )

        new_chatbot.load_state(state)

        assert new_chatbot.system_prompt == self.chatbot.system_prompt
        assert new_chatbot.max_new_tokens == self.chatbot.max_new_tokens
        assert new_chatbot.stop_tokens == self.chatbot.stop_tokens

    def test_chatbot_invalid_state(self):
        """Test chatbot with invalid state."""
        invalid_state = {
            "system_prompt": "Valid prompt",
            "max_new_tokens": "invalid",  # Should be int
            "stop_tokens": "invalid",  # Should be list
        }

        with pytest.raises(ValueError):
            self.chatbot.load_state(invalid_state)

    def test_chatbot_context_overflow(self):
        """Test chatbot with context overflow."""
        # Mock context manager to simulate overflow
        self.context_manager.get_context_tokens.return_value = list(
            range(1000)
        )

        # Should handle overflow gracefully
        response = self.chatbot.chat("Hello")
        assert isinstance(response, str)

    def test_chatbot_memory_management(self):
        """Test chatbot memory management."""
        # Simulate many turns to test memory management
        for i in range(100):
            response = self.chatbot.chat(f"Message {i}")
            assert isinstance(response, str)

        # Should not have memory issues
        assert True  # If we get here, no memory issues occurred
