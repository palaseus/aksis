"""Tests for chatbot dataset handling."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import json

from aksis.eval.dataset import ChatbotDataset


class TestChatBotDataset:
    """Test chatbot dataset functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Hello world")
        self.tokenizer.vocab_size_with_special = 1000
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 2
        
        # Sample chatbot data
        self.chatbot_data = [
            {
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "What's the weather like?"},
                    {"role": "assistant", "content": "I don't have access to weather data."},
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "Tell me a joke"},
                    {"role": "assistant", "content": "Why did the chicken cross the road?"},
                ]
            },
        ]
        
        self.dataset = ChatbotDataset(
            conversations=self.chatbot_data,
            tokenizer=self.tokenizer,
            max_length=512,
        )

    def test_dataset_initialization(self) -> None:
        """Test dataset initialization."""
        assert self.dataset.conversations == self.chatbot_data
        assert self.dataset.tokenizer == self.tokenizer
        assert self.dataset.max_length == 512

    def test_dataset_initialization_invalid_data(self) -> None:
        """Test dataset initialization with invalid data."""
        with pytest.raises(ValueError, match="Data must be a list"):
            ChatBotDataset(
                data="invalid_data",
                tokenizer=self.tokenizer,
                max_length=512,
            )

    def test_dataset_initialization_empty_data(self) -> None:
        """Test dataset initialization with empty data."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            ChatBotDataset(
                data=[],
                tokenizer=self.tokenizer,
                max_length=512,
            )

    def test_dataset_initialization_invalid_tokenizer(self) -> None:
        """Test dataset initialization with invalid tokenizer."""
        with pytest.raises(
            ValueError, match="tokenizer must have encode and decode methods"
        ):
            ChatBotDataset(
                data=self.chatbot_data,
                tokenizer="invalid_tokenizer",
                max_length=512,
            )

    def test_dataset_initialization_invalid_max_length(self) -> None:
        """Test dataset initialization with invalid max_length."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            ChatBotDataset(
                data=self.chatbot_data,
                tokenizer=self.tokenizer,
                max_length=0,
            )

    def test_dataset_length(self) -> None:
        """Test dataset length."""
        assert len(self.dataset) == 3

    def test_dataset_getitem(self) -> None:
        """Test dataset item access."""
        item = self.dataset[0]
        
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "target_ids" in item
        assert "attention_mask" in item
        
        # Check tensor shapes
        assert item["input_ids"].shape[0] <= 512
        assert item["target_ids"].shape[0] <= 512
        assert item["attention_mask"].shape[0] <= 512

    def test_dataset_getitem_invalid_index(self) -> None:
        """Test dataset item access with invalid index."""
        with pytest.raises(IndexError):
            self.dataset[10]

    def test_dataset_getitem_negative_index(self) -> None:
        """Test dataset item access with negative index."""
        item = self.dataset[-1]
        assert isinstance(item, dict)
        assert "input_ids" in item

    def test_dataset_tokenization(self) -> None:
        """Test dataset tokenization."""
        # Mock tokenizer to return different tokens for different inputs
        def mock_encode(text, add_special_tokens=True):
            if "Hello" in text:
                return [1, 2, 3]
            elif "weather" in text:
                return [4, 5, 6]
            else:
                return [7, 8, 9]
        
        self.tokenizer.encode.side_effect = mock_encode
        
        item = self.dataset[0]
        
        # Should have been tokenized
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["target_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)

    def test_dataset_padding(self) -> None:
        """Test dataset padding."""
        # Mock tokenizer to return short sequences
        self.tokenizer.encode.return_value = [1, 2, 3]
        
        item = self.dataset[0]
        
        # Should be padded to max_length
        assert item["input_ids"].shape[0] == 512
        assert item["target_ids"].shape[0] == 512
        assert item["attention_mask"].shape[0] == 512

    def test_dataset_truncation(self) -> None:
        """Test dataset truncation."""
        # Mock tokenizer to return long sequences
        long_sequence = list(range(1000))
        self.tokenizer.encode.return_value = long_sequence
        
        item = self.dataset[0]
        
        # Should be truncated to max_length
        assert item["input_ids"].shape[0] == 512
        assert item["target_ids"].shape[0] == 512
        assert item["attention_mask"].shape[0] == 512

    def test_dataset_attention_mask(self) -> None:
        """Test attention mask generation."""
        # Mock tokenizer to return short sequences
        self.tokenizer.encode.return_value = [1, 2, 3]
        
        item = self.dataset[0]
        
        # Attention mask should be 1 for real tokens, 0 for padding
        attention_mask = item["attention_mask"]
        assert attention_mask.sum().item() == 3  # Only 3 real tokens
        assert attention_mask[3:].sum().item() == 0  # Rest should be padding

    def test_dataset_conversation_formatting(self) -> None:
        """Test conversation formatting."""
        # Test that conversations are properly formatted
        item = self.dataset[0]
        
        # Should have been processed
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["target_ids"], torch.Tensor)

    def test_dataset_multiple_conversations(self) -> None:
        """Test dataset with multiple conversations."""
        multi_conv_data = [
            {
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ]
            }
        ]
        
        dataset = ChatBotDataset(
            data=multi_conv_data,
            tokenizer=self.tokenizer,
            max_length=512,
        )
        
        item = dataset[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "target_ids" in item

    def test_dataset_system_message(self) -> None:
        """Test dataset with system messages."""
        system_data = [
            {
                "conversations": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                ]
            }
        ]
        
        dataset = ChatBotDataset(
            data=system_data,
            tokenizer=self.tokenizer,
            max_length=512,
        )
        
        item = dataset[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "target_ids" in item

    def test_dataset_invalid_conversation_format(self) -> None:
        """Test dataset with invalid conversation format."""
        invalid_data = [
            {
                "conversations": [
                    {"invalid": "format"},
                ]
            }
        ]
        
        with pytest.raises(ValueError, match="Invalid conversation format"):
            ChatBotDataset(
                data=invalid_data,
                tokenizer=self.tokenizer,
                max_length=512,
            )

    def test_dataset_empty_conversation(self) -> None:
        """Test dataset with empty conversation."""
        empty_data = [
            {
                "conversations": []
            }
        ]
        
        with pytest.raises(ValueError, match="Conversation cannot be empty"):
            ChatBotDataset(
                data=empty_data,
                tokenizer=self.tokenizer,
                max_length=512,
            )

    def test_dataset_single_message_conversation(self) -> None:
        """Test dataset with single message conversation."""
        single_data = [
            {
                "conversations": [
                    {"role": "user", "content": "Hello"},
                ]
            }
        ]
        
        with pytest.raises(ValueError, match="Conversation must have at least 2 messages"):
            ChatBotDataset(
                data=single_data,
                tokenizer=self.tokenizer,
                max_length=512,
            )

    def test_dataset_invalid_role(self) -> None:
        """Test dataset with invalid role."""
        invalid_role_data = [
            {
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "invalid_role", "content": "Hi"},
                ]
            }
        ]
        
        with pytest.raises(ValueError, match="Invalid role: invalid_role"):
            ChatBotDataset(
                data=invalid_role_data,
                tokenizer=self.tokenizer,
                max_length=512,
            )

    def test_dataset_empty_message_content(self) -> None:
        """Test dataset with empty message content."""
        empty_content_data = [
            {
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": ""},
                ]
            }
        ]
        
        with pytest.raises(ValueError, match="Message content cannot be empty"):
            ChatBotDataset(
                data=empty_content_data,
                tokenizer=self.tokenizer,
                max_length=512,
            )

    def test_dataset_collate_fn(self) -> None:
        """Test dataset collate function."""
        # Create a batch of items
        batch = [self.dataset[0], self.dataset[1]]
        
        collated = self.dataset.collate_fn(batch)
        
        assert isinstance(collated, dict)
        assert "input_ids" in collated
        assert "target_ids" in collated
        assert "attention_mask" in collated
        
        # Check batch dimensions
        assert collated["input_ids"].shape[0] == 2  # Batch size
        assert collated["target_ids"].shape[0] == 2
        assert collated["attention_mask"].shape[0] == 2

    def test_dataset_collate_fn_empty_batch(self) -> None:
        """Test dataset collate function with empty batch."""
        with pytest.raises(ValueError, match="Batch cannot be empty"):
            self.dataset.collate_fn([])

    def test_dataset_collate_fn_different_lengths(self) -> None:
        """Test dataset collate function with different sequence lengths."""
        # Mock tokenizer to return different lengths
        def mock_encode(text, add_special_tokens=True):
            if "Hello" in text:
                return [1, 2, 3]
            else:
                return [4, 5, 6, 7, 8]
        
        self.tokenizer.encode.side_effect = mock_encode
        
        batch = [self.dataset[0], self.dataset[1]]
        collated = self.dataset.collate_fn(batch)
        
        # Should pad to the same length
        assert collated["input_ids"].shape[1] == 512
        assert collated["target_ids"].shape[1] == 512
        assert collated["attention_mask"].shape[1] == 512

    def test_dataset_from_file(self) -> None:
        """Test loading dataset from file."""
        # Create temporary file with chatbot data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.chatbot_data, f)
            temp_file = f.name
        
        try:
            dataset = ChatBotDataset.from_file(
                file_path=temp_file,
                tokenizer=self.tokenizer,
                max_length=512,
            )
            
            assert len(dataset) == 3
            assert dataset.data == self.chatbot_data
        finally:
            import os
            os.unlink(temp_file)

    def test_dataset_from_file_not_found(self) -> None:
        """Test loading dataset from non-existent file."""
        with pytest.raises(FileNotFoundError):
            ChatBotDataset.from_file(
                file_path="nonexistent.json",
                tokenizer=self.tokenizer,
                max_length=512,
            )

    def test_dataset_from_file_invalid_format(self) -> None:
        """Test loading dataset from file with invalid format."""
        # Create temporary file with invalid data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"invalid": "format"}, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Data must be a list"):
                ChatBotDataset.from_file(
                    file_path=temp_file,
                    tokenizer=self.tokenizer,
                    max_length=512,
                )
        finally:
            import os
            os.unlink(temp_file)

    def test_dataset_save_to_file(self) -> None:
        """Test saving dataset to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.dataset.save_to_file(temp_file)
            
            # Load and verify
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == self.chatbot_data
        finally:
            import os
            os.unlink(temp_file)

    def test_dataset_train_test_split(self) -> None:
        """Test train-test split."""
        train_dataset, test_dataset = self.dataset.train_test_split(test_size=0.33)
        
        assert len(train_dataset) == 2
        assert len(test_dataset) == 1
        assert len(train_dataset) + len(test_dataset) == len(self.dataset)

    def test_dataset_train_test_split_invalid_size(self) -> None:
        """Test train-test split with invalid test size."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            self.dataset.train_test_split(test_size=1.5)

    def test_dataset_train_test_split_zero_size(self) -> None:
        """Test train-test split with zero test size."""
        train_dataset, test_dataset = self.dataset.train_test_split(test_size=0.0)
        
        assert len(train_dataset) == 3
        assert len(test_dataset) == 0

    def test_dataset_train_test_split_full_size(self) -> None:
        """Test train-test split with full test size."""
        train_dataset, test_dataset = self.dataset.train_test_split(test_size=1.0)
        
        assert len(train_dataset) == 0
        assert len(test_dataset) == 3

    def test_dataset_train_val_test_split(self) -> None:
        """Test train-validation-test split."""
        train_dataset, val_dataset, test_dataset = self.dataset.train_val_test_split(
            val_size=0.33, test_size=0.33
        )
        
        assert len(train_dataset) == 1
        assert len(val_dataset) == 1
        assert len(test_dataset) == 1
        assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(self.dataset)

    def test_dataset_train_val_test_split_invalid_sizes(self) -> None:
        """Test train-validation-test split with invalid sizes."""
        with pytest.raises(ValueError, match="val_size and test_size must sum to less than 1"):
            self.dataset.train_val_test_split(val_size=0.5, test_size=0.6)

    def test_dataset_cuda_compatibility(self) -> None:
        """Test CUDA compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        cuda_dataset = ChatBotDataset(
            data=self.chatbot_data,
            tokenizer=self.tokenizer,
            max_length=512,
            device=torch.device("cuda"),
        )
        
        item = cuda_dataset[0]
        
        # Check that tensors are on CUDA
        assert item["input_ids"].device.type == "cuda"
        assert item["target_ids"].device.type == "cuda"
        assert item["attention_mask"].device.type == "cuda"

    def test_dataset_performance(self) -> None:
        """Test dataset performance."""
        import time
        
        start_time = time.time()
        
        # Access all items
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            assert isinstance(item, dict)
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0

    def test_dataset_memory_efficiency(self) -> None:
        """Test dataset memory efficiency."""
        # Create a larger dataset
        large_data = self.chatbot_data * 100  # 300 items
        
        large_dataset = ChatBotDataset(
            data=large_data,
            tokenizer=self.tokenizer,
            max_length=512,
        )
        
        # Should handle large datasets efficiently
        assert len(large_dataset) == 300
        
        # Access a few items
        for i in range(0, 300, 50):
            item = large_dataset[i]
            assert isinstance(item, dict)

    def test_dataset_reproducibility(self) -> None:
        """Test dataset reproducibility."""
        # Set random seed
        torch.manual_seed(42)
        
        # Create two datasets with same data
        dataset1 = ChatBotDataset(
            data=self.chatbot_data,
            tokenizer=self.tokenizer,
            max_length=512,
        )
        
        torch.manual_seed(42)
        
        dataset2 = ChatBotDataset(
            data=self.chatbot_data,
            tokenizer=self.tokenizer,
            max_length=512,
        )
        
        # Should produce same results
        item1 = dataset1[0]
        item2 = dataset2[0]
        
        assert torch.equal(item1["input_ids"], item2["input_ids"])
        assert torch.equal(item1["target_ids"], item2["target_ids"])
        assert torch.equal(item1["attention_mask"], item2["attention_mask"])
