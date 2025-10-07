"""Tests for dataset integration."""

import torch
from unittest.mock import Mock, patch
from datasets import Dataset

from aksis.train.dataset import (
    load_wikitext2,
    load_shakespeare,
    create_dataloaders,
    create_aksis_dataloaders,
    collate_fn,
    get_dataset_info,
)
from aksis.data.tokenizer import Tokenizer


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = Mock(spec=Tokenizer)
        self.tokenizer.encode_batch = Mock(return_value=[[1, 2, 3, 4, 5]])
        self.tokenizer.build_vocab = Mock()

    @patch("aksis.train.dataset.load_dataset")
    def test_load_wikitext2(self, mock_load_dataset):
        """Test loading WikiText-2 dataset."""
        # Mock dataset
        mock_dataset = {
            "train": Dataset.from_dict(
                {"text": ["Sample text 1", "Sample text 2"]}
            ),
            "validation": Dataset.from_dict({"text": ["Val text 1"]}),
            "test": Dataset.from_dict({"text": ["Test text 1"]}),
        }
        mock_load_dataset.return_value = mock_dataset

        train_ds, val_ds, test_ds = load_wikitext2(self.tokenizer)

        assert len(train_ds) == 1  # After tokenization and filtering
        assert len(val_ds) == 1
        assert len(test_ds) == 1
        mock_load_dataset.assert_called_once_with(
            "wikitext", "wikitext-2-raw-v1"
        )

    @patch("aksis.train.dataset.load_dataset")
    def test_load_shakespeare(self, mock_load_dataset):
        """Test loading Shakespeare dataset."""
        # Mock dataset
        mock_dataset = {
            "train": Dataset.from_dict({"text": ["To be or not to be"]}),
            "validation": Dataset.from_dict(
                {"text": ["That is the question"]}
            ),
            "test": Dataset.from_dict({"text": ["Whether 'tis nobler"]}),
        }
        mock_load_dataset.return_value = mock_dataset

        train_ds, val_ds, test_ds = load_shakespeare(self.tokenizer)

        assert len(train_ds) == 1
        assert len(val_ds) == 1
        assert len(test_ds) == 1
        mock_load_dataset.assert_called_once_with("shakespeare")

    @patch("aksis.train.dataset.load_dataset")
    def test_load_wikitext2_with_custom_splits(self, mock_load_dataset):
        """Test loading WikiText-2 with custom split names."""
        mock_dataset = {
            "train": Dataset.from_dict({"text": ["Sample text"]}),
            "validation": Dataset.from_dict({"text": ["Val text"]}),
            "test": Dataset.from_dict({"text": ["Test text"]}),
        }
        mock_load_dataset.return_value = mock_dataset

        train_ds, val_ds, test_ds = load_wikitext2(
            self.tokenizer,
            train_split="train",
            val_split="validation",
            test_split="test",
        )

        assert len(train_ds) == 1
        assert len(val_ds) == 1
        assert len(test_ds) == 1

    @patch("aksis.train.dataset.load_dataset")
    def test_load_wikitext2_empty_texts(self, mock_load_dataset):
        """Test loading WikiText-2 with empty texts."""
        mock_dataset = {
            "train": Dataset.from_dict({"text": ["", "   ", "Valid text"]}),
            "validation": Dataset.from_dict({"text": ["Valid val text"]}),
            "test": Dataset.from_dict({"text": ["Valid test text"]}),
        }
        mock_load_dataset.return_value = mock_dataset

        train_ds, val_ds, test_ds = load_wikitext2(self.tokenizer)

        # Should filter out empty texts
        assert len(train_ds) == 1  # Only "Valid text"
        assert len(val_ds) == 1
        assert len(test_ds) == 1

    @patch("aksis.train.dataset.load_dataset")
    def test_load_wikitext2_max_length(self, mock_load_dataset):
        """Test loading WikiText-2 with max_length parameter."""
        mock_dataset = {
            "train": Dataset.from_dict({"text": ["Sample text"]}),
            "validation": Dataset.from_dict({"text": ["Val text"]}),
            "test": Dataset.from_dict({"text": ["Test text"]}),
        }
        mock_load_dataset.return_value = mock_dataset

        load_wikitext2(self.tokenizer, max_length=256)

        # Check that tokenizer was called with max_length
        self.tokenizer.encode_batch.assert_called()
        call_args = self.tokenizer.encode_batch.call_args
        assert call_args[1]["max_length"] == 256


class TestDataLoaderCreation:
    """Test DataLoader creation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.train_dataset = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "attention_mask": [[1, 1, 1], [1, 1, 1]],
            }
        )
        self.val_dataset = Dataset.from_dict(
            {"input_ids": [[7, 8, 9]], "attention_mask": [[1, 1, 1]]}
        )
        self.test_dataset = Dataset.from_dict(
            {"input_ids": [[10, 11, 12]], "attention_mask": [[1, 1, 1]]}
        )

    def test_create_dataloaders(self):
        """Test creating PyTorch DataLoaders."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=2,
        )

        assert len(train_loader) == 1  # 2 samples / 2 batch_size = 1 batch
        assert len(val_loader) == 1  # 1 sample / 2 batch_size = 1 batch
        assert len(test_loader) == 1  # 1 sample / 2 batch_size = 1 batch

        # Test that we can iterate through the loaders
        for batch in train_loader:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert batch["input_ids"].shape[0] == 2  # batch_size
            break

    def test_create_dataloaders_without_test(self):
        """Test creating DataLoaders without test dataset."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.train_dataset, self.val_dataset, batch_size=2
        )

        assert len(train_loader) == 1
        assert len(val_loader) == 1
        assert test_loader is None

    def test_create_dataloaders_different_batch_size(self):
        """Test creating DataLoaders with different batch sizes."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=1,
        )

        assert len(train_loader) == 2  # 2 samples / 1 batch_size = 2 batches
        assert len(val_loader) == 1  # 1 sample / 1 batch_size = 1 batch
        assert len(test_loader) == 1  # 1 sample / 1 batch_size = 1 batch

    @patch("aksis.train.dataset.AksisDataLoader")
    def test_create_aksis_dataloaders(self, mock_aksis_dataloader):
        """Test creating Aksis DataLoaders."""
        mock_aksis_dataloader.return_value = Mock()
        mock_aksis_dataloader.return_value.__len__ = Mock(return_value=1)

        train_loader, val_loader, test_loader = create_aksis_dataloaders(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=2,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check that AksisDataLoader was called with correct parameters
        assert mock_aksis_dataloader.call_count == 3

    def test_collate_fn(self):
        """Test collate function."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
            {
                "input_ids": torch.tensor([4, 5, 6]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
        ]

        result = collate_fn(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)

    def test_collate_fn_single_sample(self):
        """Test collate function with single sample."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            }
        ]

        result = collate_fn(batch)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (1, 3)
        assert result["attention_mask"].shape == (1, 3)


class TestDatasetInfo:
    """Test dataset information functionality."""

    def test_get_dataset_info_empty(self):
        """Test getting info from empty dataset."""
        dataset = Dataset.from_dict({})
        info = get_dataset_info(dataset)

        assert info["num_samples"] == 0
        assert info["avg_length"] == 0
        assert info["max_length"] == 0
        assert info["min_length"] == 0

    def test_get_dataset_info_with_input_ids(self):
        """Test getting info from dataset with input_ids."""
        dataset = Dataset.from_dict(
            {
                "input_ids": [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8],
                    [9, 10, 11, 12, 13, 14, 15],
                ]
            }
        )
        info = get_dataset_info(dataset)

        assert info["num_samples"] == 3
        assert info["avg_length"] == 5.0  # (5 + 3 + 7) / 3
        assert info["max_length"] == 7
        assert info["min_length"] == 3

    def test_get_dataset_info_with_text(self):
        """Test getting info from dataset with text."""
        dataset = Dataset.from_dict(
            {"text": ["short", "medium length text", "very long text here"]}
        )
        info = get_dataset_info(dataset)

        assert info["num_samples"] == 3
        assert info["avg_length"] == 14.0  # (5 + 17 + 20) / 3
        assert info["max_length"] == 19
        assert info["min_length"] == 5

    def test_get_dataset_info_no_relevant_columns(self):
        """Test getting info from dataset without relevant columns."""
        dataset = Dataset.from_dict({"other_column": ["a", "b", "c"]})
        info = get_dataset_info(dataset)

        assert info["num_samples"] == 3
        assert info["avg_length"] == 0
        assert info["max_length"] == 0
        assert info["min_length"] == 0


class TestDatasetIntegration:
    """Test end-to-end dataset integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = Mock(spec=Tokenizer)
        self.tokenizer.encode_batch = Mock(return_value=[[1, 2, 3, 4, 5]])
        self.tokenizer.build_vocab = Mock()

    @patch("aksis.train.dataset.load_dataset")
    def test_end_to_end_wikitext2_loading(self, mock_load_dataset):
        """Test end-to-end WikiText-2 loading and DataLoader creation."""
        # Mock dataset
        mock_dataset = {
            "train": Dataset.from_dict(
                {"text": ["Sample text 1", "Sample text 2"]}
            ),
            "validation": Dataset.from_dict({"text": ["Val text 1"]}),
            "test": Dataset.from_dict({"text": ["Test text 1"]}),
        }
        mock_load_dataset.return_value = mock_dataset

        # Load datasets
        train_ds, val_ds, test_ds = load_wikitext2(self.tokenizer)

        # Create DataLoaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=1
        )

        # Verify
        assert len(train_loader) == 1  # After tokenization and filtering
        assert len(val_loader) == 1
        assert len(test_loader) == 1

        # Test iteration
        for batch in train_loader:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            break

    @patch("aksis.train.dataset.load_dataset")
    def test_end_to_end_shakespeare_loading(self, mock_load_dataset):
        """Test end-to-end Shakespeare loading and DataLoader creation."""
        # Mock dataset
        mock_dataset = {
            "train": Dataset.from_dict({"text": ["To be or not to be"]}),
            "validation": Dataset.from_dict(
                {"text": ["That is the question"]}
            ),
            "test": Dataset.from_dict({"text": ["Whether 'tis nobler"]}),
        }
        mock_load_dataset.return_value = mock_dataset

        # Load datasets
        train_ds, val_ds, test_ds = load_shakespeare(self.tokenizer)

        # Create DataLoaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=1
        )

        # Verify
        assert len(train_loader) == 1
        assert len(val_loader) == 1
        assert len(test_loader) == 1

        # Test iteration
        for batch in train_loader:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            break
