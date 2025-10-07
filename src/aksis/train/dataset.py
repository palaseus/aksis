"""Dataset integration for training."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from aksis.data.dataloader import DataLoader as AksisDataLoader
from aksis.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def load_wikitext2(
    tokenizer: Tokenizer,
    max_length: int = 512,
    train_split: str = "train",
    val_split: str = "validation",
    test_split: str = "test",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load WikiText-2 dataset and tokenize it.

    Args:
        tokenizer: Tokenizer to use for tokenization.
        max_length: Maximum sequence length.
        train_split: Name of training split.
        val_split: Name of validation split.
        test_split: Name of test split.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    logger.info("Loading WikiText-2 dataset...")

    # Load the dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Get the splits
    train_dataset = dataset[train_split]
    val_dataset = dataset[val_split]
    test_dataset = dataset[test_split]

    logger.info(f"Loaded {len(train_dataset)} training samples")
    logger.info(f"Loaded {len(val_dataset)} validation samples")
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Build vocabulary from training data
    logger.info("Building vocabulary...")
    train_texts = [
        sample["text"] for sample in train_dataset if sample["text"].strip()
    ]
    tokenizer.build_vocab(train_texts)
    logger.info(
        f"Vocabulary built with {tokenizer.vocab_size_with_special} tokens"
    )

    # Tokenize the datasets
    def tokenize_function(examples):
        """Tokenize a batch of examples."""
        # Join all text in the batch
        texts = examples["text"]

        # Filter out empty texts
        texts = [text for text in texts if text.strip()]

        if not texts:
            return {"input_ids": [], "attention_mask": []}

        # Tokenize
        input_ids = tokenizer.encode_batch(
            texts, max_length=max_length, add_special_tokens=True
        )

        # Create attention masks (1 for real tokens, 0 for padding)
        attention_masks = []
        for ids in input_ids:
            mask = [1] * len(ids)
            attention_masks.append(mask)

        return {"input_ids": input_ids, "attention_mask": attention_masks}

    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data",
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data",
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test data",
    )

    # Filter out empty samples
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    test_dataset = test_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    logger.info(f"After tokenization: {len(train_dataset)} training samples")
    logger.info(f"After tokenization: {len(val_dataset)} validation samples")
    logger.info(f"After tokenization: {len(test_dataset)} test samples")

    return train_dataset, val_dataset, test_dataset


def load_shakespeare(
    tokenizer: Tokenizer,
    max_length: int = 512,
    train_split: str = "train",
    val_split: str = "validation",
    test_split: str = "test",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load Shakespeare dataset and tokenize it.

    Args:
        tokenizer: Tokenizer to use for tokenization.
        max_length: Maximum sequence length.
        train_split: Name of training split.
        val_split: Name of validation split.
        test_split: Name of test split.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    logger.info("Loading Shakespeare dataset...")

    # Load the dataset
    dataset = load_dataset("shakespeare")

    # Get the splits
    train_dataset = dataset[train_split]
    val_dataset = dataset[val_split]
    test_dataset = dataset[test_split]

    logger.info(f"Loaded {len(train_dataset)} training samples")
    logger.info(f"Loaded {len(val_dataset)} validation samples")
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Build vocabulary from training data
    logger.info("Building vocabulary...")
    train_texts = [
        sample["text"] for sample in train_dataset if sample["text"].strip()
    ]
    tokenizer.build_vocab(train_texts)
    logger.info(
        f"Vocabulary built with {tokenizer.vocab_size_with_special} tokens"
    )

    # Tokenize the datasets
    def tokenize_function(examples):
        """Tokenize a batch of examples."""
        # Get the text from the 'text' column
        texts = examples["text"]

        # Filter out empty texts
        texts = [text for text in texts if text.strip()]

        if not texts:
            return {"input_ids": [], "attention_mask": []}

        # Tokenize
        input_ids = tokenizer.encode_batch(
            texts, max_length=max_length, add_special_tokens=True
        )

        # Create attention masks (1 for real tokens, 0 for padding)
        attention_masks = []
        for ids in input_ids:
            mask = [1] * len(ids)
            attention_masks.append(mask)

        return {"input_ids": input_ids, "attention_mask": attention_masks}

    # Apply tokenization
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data",
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data",
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test data",
    )

    # Filter out empty samples
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    test_dataset = test_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    logger.info(f"After tokenization: {len(train_dataset)} training samples")
    logger.info(f"After tokenization: {len(val_dataset)} validation samples")
    logger.info(f"After tokenization: {len(test_dataset)} test samples")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders from datasets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset (optional).
        batch_size: Batch size.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    logger.info(f"Creating DataLoaders with batch_size={batch_size}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    logger.info(
        f"Created DataLoaders: train={len(train_loader)}, "
        f"val={len(val_loader)}, "
        f"test={len(test_loader) if test_loader else 'None'}"
    )

    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples.

    Returns:
        Batched data.
    """
    # Get the keys from the first sample
    keys = batch[0].keys()

    # Stack tensors for each key
    batched = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        # Convert lists to tensors if needed
        tensors = []
        for value in values:
            if isinstance(value, list):
                tensors.append(torch.tensor(value))
            else:
                tensors.append(value)

        # For variable-length sequences, pad to the same length
        if key in ["input_ids", "attention_mask"]:
            # Find the maximum length in the batch
            max_len = max(tensor.size(0) for tensor in tensors)

            # Pad all tensors to the same length
            padded_tensors = []
            for tensor in tensors:
                if tensor.size(0) < max_len:
                    # Pad with zeros (or appropriate padding token)
                    padding_size = max_len - tensor.size(0)
                    if key == "input_ids":
                        # Pad with 0 (assuming 0 is the padding token)
                        padding = torch.zeros(padding_size, dtype=tensor.dtype)
                    else:  # attention_mask
                        # Pad with 0 (no attention to padding tokens)
                        padding = torch.zeros(padding_size, dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, padding])
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)

            batched[key] = torch.stack(padded_tensors)
        else:
            batched[key] = torch.stack(tensors)

    return batched


def create_aksis_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[AksisDataLoader, AksisDataLoader, Optional[AksisDataLoader]]:
    """
    Create Aksis DataLoaders from datasets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset (optional).
        batch_size: Batch size.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    logger.info(f"Creating Aksis DataLoaders with batch_size={batch_size}")

    # Create DataLoaders
    train_loader = AksisDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = AksisDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = AksisDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    logger.info(
        f"Created Aksis DataLoaders: train={len(train_loader)}, "
        f"val={len(val_loader)}, "
        f"test={len(test_loader) if test_loader else 'None'}"
    )

    return train_loader, val_loader, test_loader


def get_dataset_info(dataset: Dataset) -> Dict[str, any]:
    """
    Get information about a dataset.

    Args:
        dataset: Dataset to analyze.

    Returns:
        Dataset information.
    """
    if len(dataset) == 0:
        return {
            "num_samples": 0,
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0,
        }

    # Get sequence lengths
    lengths = []
    for sample in dataset:
        if "input_ids" in sample:
            lengths.append(len(sample["input_ids"]))
        elif "text" in sample:
            lengths.append(len(sample["text"]))

    if not lengths:
        return {
            "num_samples": len(dataset),
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0,
        }

    return {
        "num_samples": len(dataset),
        "avg_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
    }
