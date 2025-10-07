"""DataLoader implementation for Aksis."""

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import List, Dict, Any, Optional, Union, Iterator
import logging

from aksis.data.tokenizer import Tokenizer
from aksis.utils.device import get_device

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    PyTorch Dataset for text data.
    
    This dataset handles tokenization, padding, and truncation of text sequences.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Tokenizer,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings.
            tokenizer: Tokenizer instance for encoding texts.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences to max_length.
            truncation: Whether to truncate sequences exceeding max_length.
            add_special_tokens: Whether to add BOS/EOS tokens.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens

    def __len__(self) -> int:
        """Return the number of texts in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve.
            
        Returns:
            Dictionary containing input_ids and attention_mask tensors.
        """
        text = self.texts[idx]
        
        # Encode the text
        input_ids = self.tokenizer.encode(
            text=text,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [
            1 if token_id != self.tokenizer.pad_token_id else 0
            for token_id in input_ids
        ]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DataLoader:
    """
    DataLoader for text data with CUDA support.
    
    This DataLoader wraps PyTorch's DataLoader with additional functionality
    for text processing and CUDA tensor management.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: Tokenizer,
        batch_size: int = 32,
        shuffle: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        pin_memory: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        Initialize the DataLoader.
        
        Args:
            texts: List of text strings to load.
            tokenizer: Tokenizer instance for encoding texts.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.
            max_length: Maximum sequence length.
            padding: Whether to pad sequences to max_length.
            truncation: Whether to truncate sequences exceeding max_length.
            add_special_tokens: Whether to add BOS/EOS tokens.
            device: Device to move tensors to (auto-detected if None).
            pin_memory: Whether to pin memory for faster GPU transfer.
            num_workers: Number of worker processes for data loading.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = get_device(device if isinstance(device, str) else None)
        # Only use pin_memory for CPU tensors
        self.pin_memory = pin_memory and self.device.type == "cpu"
        
        # Create dataset
        self.dataset = TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )
        
        # Create PyTorch DataLoader (only if dataset is not empty)
        if len(self.dataset) > 0:
            self.dataloader: Optional[TorchDataLoader] = TorchDataLoader(
                dataset=self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=self.pin_memory,
                drop_last=drop_last,
                collate_fn=self._collate_fn,
            )
        else:
            self.dataloader = None
        
        logger.info(
            f"DataLoader initialized with {len(texts)} texts, "
            f"batch_size={batch_size}, device={self.device}"
        )

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching.
        
        Args:
            batch: List of samples from the dataset.
            
        Returns:
            Batched tensors.
        """
        # Get max length in batch
        max_length = max(item["input_ids"].shape[0] for item in batch)
        
        # Pad sequences to max length
        padded_input_ids = []
        padded_attention_mask = []
        
        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            
            # Pad if necessary
            if input_ids.shape[0] < max_length:
                pad_length = max_length - input_ids.shape[0]
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ])
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
        
        # Stack tensors
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        
        # Move to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Return iterator over batches."""
        if self.dataloader is None:
            return iter([])
        return iter(self.dataloader)

    def __len__(self) -> int:
        """Return number of batches."""
        if self.dataloader is None:
            return 0
        else:
            return len(self.dataloader)

    def get_batch(self, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a specific batch by index.
        
        Args:
            batch_idx: Index of the batch to retrieve.
            
        Returns:
            Batch dictionary.
        """
        if batch_idx >= len(self):
            raise IndexError(f"Batch index {batch_idx} out of range")
        
        # Get the batch
        if self.dataloader is None:
            raise RuntimeError("DataLoader is empty")
        batch = list(self.dataloader)[batch_idx]
        return batch  # type: ignore

    def get_sample(self, sample_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample by index.
        
        Args:
            sample_idx: Index of the sample to retrieve.
            
        Returns:
            Sample dictionary.
        """
        if sample_idx >= len(self.dataset):
            raise IndexError(f"Sample index {sample_idx} out of range")
        
        sample = self.dataset[sample_idx]
        # Move to device
        sample = {k: v.to(self.device) for k, v in sample.items()}
        return sample

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics.
        """
        if not self.texts:
            return {"num_texts": 0}
        
        # Calculate sequence lengths
        lengths = []
        for text in self.texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            lengths.append(len(encoded))
        
        return {
            "num_texts": len(self.texts),
            "num_batches": len(self),
            "batch_size": self.batch_size,
            "avg_sequence_length": sum(lengths) / len(lengths),
            "min_sequence_length": min(lengths),
            "max_sequence_length": max(lengths),
            "vocab_size": self.tokenizer.vocab_size_with_special,
            "device": str(self.device),
        }
