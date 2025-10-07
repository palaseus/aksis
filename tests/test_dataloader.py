"""Tests for the DataLoader class."""

import torch

from aksis.data.tokenizer import Tokenizer
from aksis.data.dataloader import DataLoader, TextDataset


class TestTextDataset:
    """Test cases for the TextDataset class."""

    def test_dataset_initialization(self) -> None:
        """Test dataset initialization."""
        texts = ["Hello world!", "This is a test.", "Another sentence."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataset = TextDataset(texts, tokenizer)
        assert len(dataset) == len(texts)
        assert dataset.tokenizer == tokenizer

    def test_dataset_getitem(self) -> None:
        """Test dataset item retrieval."""
        texts = ["Hello world!", "This is a test."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataset = TextDataset(texts, tokenizer)

        # Test getting individual items
        item = dataset[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "attention_mask" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)

    def test_dataset_with_different_lengths(self) -> None:
        """Test dataset with texts of different lengths."""
        texts = ["Hi", "This is a much longer sentence with many words"]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataset = TextDataset(texts, tokenizer, max_length=10)

        for i in range(len(dataset)):
            item = dataset[i]
            # With truncation=True, sequences should be <= max_length
            # But we need to account for special tokens
            assert item["input_ids"].shape[0] <= 12  # max_length + 2 special tokens
            assert item["attention_mask"].shape[0] <= 12

    def test_dataset_padding(self) -> None:
        """Test dataset padding functionality."""
        texts = ["Hi", "This is a longer sentence"]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataset = TextDataset(texts, tokenizer, max_length=10, padding=True)

        # All items should have the same length
        lengths = [dataset[i]["input_ids"].shape[0] for i in range(len(dataset))]
        assert all(length == lengths[0] for length in lengths)

    def test_dataset_truncation(self) -> None:
        """Test dataset truncation functionality."""
        text = "This is a very long sentence that should be truncated"
        tokenizer = Tokenizer()
        tokenizer.build_vocab([text])

        dataset = TextDataset([text], tokenizer, max_length=5, truncation=True)

        item = dataset[0]
        assert item["input_ids"].shape[0] <= 5
        assert item["attention_mask"].shape[0] <= 5


class TestDataLoader:
    """Test cases for the DataLoader class."""

    def test_dataloader_initialization(self) -> None:
        """Test DataLoader initialization."""
        texts = ["Hello world!", "This is a test.", "Another sentence."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            shuffle=False,
        )

        assert dataloader.batch_size == 2
        assert not dataloader.shuffle
        assert dataloader.tokenizer == tokenizer

    def test_dataloader_iteration(self) -> None:
        """Test DataLoader iteration."""
        texts = ["Hello world!", "This is a test.", "Another sentence."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            shuffle=False,
        )

        batches = list(dataloader)
        assert len(batches) > 0

        for batch in batches:
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert batch["input_ids"].dim() == 2  # [batch_size, seq_len]
            assert batch["attention_mask"].dim() == 2

    def test_dataloader_batch_size(self) -> None:
        """Test DataLoader with different batch sizes."""
        texts = ["Hello world!", "This is a test.", "Another sentence.", "Fourth text."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        # Test batch size 2
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            shuffle=False,
        )

        batches = list(dataloader)
        for batch in batches:
            assert batch["input_ids"].shape[0] <= 2

    def test_dataloader_shuffle(self) -> None:
        """Test DataLoader shuffling."""
        texts = ["First", "Second", "Third", "Fourth"]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        # Test with shuffle=True
        dataloader_shuffled = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            shuffle=True,
        )

        # Test with shuffle=False
        dataloader_not_shuffled = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            shuffle=False,
        )

        # Get batches from both
        batches_shuffled = list(dataloader_shuffled)
        batches_not_shuffled = list(dataloader_not_shuffled)

        # Both should have the same number of batches
        assert len(batches_shuffled) == len(batches_not_shuffled)

    def test_dataloader_cuda_support(self) -> None:
        """Test DataLoader CUDA support."""
        texts = ["Hello world!", "This is a test."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        batch = next(iter(dataloader))
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # CUDA device might be cuda:0 instead of cuda
        assert batch["input_ids"].device.type == expected_device.type
        assert batch["attention_mask"].device.type == expected_device.type

    def test_dataloader_pin_memory(self) -> None:
        """Test DataLoader pin_memory functionality."""
        texts = ["Hello world!", "This is a test."]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            pin_memory=True,
        )

        batch = next(iter(dataloader))
        # Pin memory should be enabled if CUDA is available and using CPU
        if torch.cuda.is_available() and batch["input_ids"].device.type == "cpu":
            assert batch["input_ids"].is_pinned()

    def test_dataloader_max_length(self) -> None:
        """Test DataLoader with max_length parameter."""
        texts = ["This is a very long sentence that should be truncated"]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=1,
            max_length=5,
            truncation=True,
        )

        batch = next(iter(dataloader))
        assert batch["input_ids"].shape[1] <= 5
        assert batch["attention_mask"].shape[1] <= 5

    def test_dataloader_empty_texts(self) -> None:
        """Test DataLoader with empty texts list."""
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["dummy"])

        dataloader = DataLoader(
            texts=[],
            tokenizer=tokenizer,
            batch_size=2,
        )

        batches = list(dataloader)
        assert len(batches) == 0

    def test_dataloader_single_text(self) -> None:
        """Test DataLoader with single text."""
        texts = ["Single text"]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
        )

        batches = list(dataloader)
        assert len(batches) == 1
        batch = batches[0]
        assert batch["input_ids"].shape[0] == 1

    def test_dataloader_attention_mask(self) -> None:
        """Test DataLoader attention mask generation."""
        texts = ["Hello", "This is a longer sentence"]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=10,
            padding=True,
        )

        batch = next(iter(dataloader))
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]

        # Attention mask should have same shape as input_ids
        assert attention_mask.shape == input_ids.shape

        # Attention mask should be 1 for real tokens, 0 for padding
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                if input_ids[i, j] == tokenizer.pad_token_id:
                    assert attention_mask[i, j] == 0
                else:
                    assert attention_mask[i, j] == 1

    def test_dataloader_multiple_epochs(self) -> None:
        """Test DataLoader across multiple epochs."""
        texts = ["First", "Second", "Third"]
        tokenizer = Tokenizer()
        tokenizer.build_vocab(texts)

        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            shuffle=False,
        )

        # First epoch
        batches_epoch1 = list(dataloader)

        # Second epoch
        batches_epoch2 = list(dataloader)

        # Should be able to iterate multiple times
        assert len(batches_epoch1) > 0
        assert len(batches_epoch2) > 0
        assert len(batches_epoch1) == len(batches_epoch2)
