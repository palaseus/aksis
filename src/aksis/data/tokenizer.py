"""Tokenizer implementation for Aksis."""

import json
import re
from typing import List, Dict, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    A simple tokenizer for text processing.
    
    This tokenizer implements basic word-level tokenization with support for
    special tokens, vocabulary building, and encoding/decoding operations.
    """

    def __init__(self, vocab_size: int = 50000) -> None:
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (excluding special tokens).
        """
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
        # Initialize with special tokens
        self.token_to_id[self.pad_token] = 0
        self.token_to_id[self.unk_token] = 1
        self.token_to_id[self.bos_token] = 2
        self.token_to_id[self.eos_token] = 3
        
        self.id_to_token[0] = self.pad_token
        self.id_to_token[1] = self.unk_token
        self.id_to_token[2] = self.bos_token
        self.id_to_token[3] = self.eos_token
        
        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        self._vocab_built = False

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization.
        
        Args:
            text: Input text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Add spaces around punctuation
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of tokens.
        """
        preprocessed = self._preprocess_text(text)
        return preprocessed.split()

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from.
        """
        logger.info(f"Building vocabulary from {len(texts)} texts")
        
        # Count token frequencies
        token_counts: Counter[str] = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            token_counts.update(tokens)
        
        logger.info(f"Found {len(token_counts)} unique tokens")
        
        # Add most frequent tokens (special tokens already initialized)
        most_common = token_counts.most_common(self.vocab_size)
        for token, _ in most_common:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
        
        self._vocab_built = True
        logger.info(f"Vocabulary built with {len(self.token_to_id)} tokens")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            max_length: Maximum sequence length.
            truncation: Whether to truncate if sequence exceeds max_length.
            padding: Whether to pad to max_length (requires max_length).
            
        Returns:
            List of token IDs.
        """
        if not self._vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = self._tokenize(text)
        token_ids = []
        
        # Add BOS token if requested
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        # Convert tokens to IDs
        for token in tokens:
            token_id = self.token_to_id.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        # Add EOS token if requested
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        # Truncate if necessary
        if max_length is not None and len(token_ids) > max_length:
            if truncation:
                token_ids = token_ids[:max_length]
            else:
                logger.warning(f"Sequence length {len(token_ids)} exceeds max_length {max_length}")
        
        # Pad if necessary
        if padding and max_length is not None:
            while len(token_ids) < max_length:
                token_ids.append(self.pad_token_id)
        
        return token_ids

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.
            max_length: Maximum sequence length.
            truncation: Whether to truncate if sequence exceeds max_length.
            padding: Whether to pad to max_length (requires max_length).
            
        Returns:
            List of encoded sequences.
        """
        return [
            self.encode(
                text=text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
            )
            for text in texts
        ]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.
            
        Returns:
            Decoded text string.
        """
        if not self._vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                tokens.append(token)
            else:
                logger.warning(f"Unknown token ID: {token_id}")
                if not skip_special_tokens:
                    tokens.append(self.unk_token)
        
        return " ".join(tokens)

    def decode_batch(
        self, 
        token_id_batches: List[List[int]], 
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.
        
        Args:
            token_id_batches: List of token ID sequences to decode.
            skip_special_tokens: Whether to skip special tokens in output.
            
        Returns:
            List of decoded text strings.
        """
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in token_id_batches
        ]

    def save_vocab(self, filepath: str) -> None:
        """
        Save vocabulary to a JSON file.
        
        Args:
            filepath: Path to save the vocabulary file.
        """
        if not self._vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
            },
            "special_token_ids": {
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
            },
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vocabulary saved to {filepath}")

    def load_vocab(self, filepath: str) -> None:
        """
        Load vocabulary from a JSON file.
        
        Args:
            filepath: Path to the vocabulary file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
        self.vocab_size = vocab_data["vocab_size"]
        
        # Load special tokens
        special_tokens = vocab_data["special_tokens"]
        self.pad_token = special_tokens["pad_token"]
        self.unk_token = special_tokens["unk_token"]
        self.bos_token = special_tokens["bos_token"]
        self.eos_token = special_tokens["eos_token"]
        
        # Load special token IDs
        special_token_ids = vocab_data["special_token_ids"]
        self.pad_token_id = special_token_ids["pad_token_id"]
        self.unk_token_id = special_token_ids["unk_token_id"]
        self.bos_token_id = special_token_ids["bos_token_id"]
        self.eos_token_id = special_token_ids["eos_token_id"]
        
        self._vocab_built = True
        logger.info(f"Vocabulary loaded from {filepath}")

    @property
    def vocab_size_with_special(self) -> int:
        """Get total vocabulary size including special tokens."""
        return len(self.token_to_id) if self._vocab_built else 0
