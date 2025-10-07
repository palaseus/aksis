"""Chatbot dataset handling for Aksis AI chatbot/LLM fine-tuning."""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import json
import random

from aksis.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class ChatbotDataset(Dataset):
    """Dataset class for chatbot conversation data."""

    def __init__(
        self,
        conversations: List[Dict[str, Any]],
        tokenizer: Tokenizer,
        max_length: int = 512,
        context_window: int = 3,
        include_system_prompts: bool = True,
    ) -> None:
        """
        Initialize the ChatbotDataset.

        Args:
            conversations: List of conversation dictionaries.
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
            context_window: Number of previous turns to include as context.
            include_system_prompts: Whether to include system prompts.
        """
        if not conversations:
            raise ValueError("Conversations list cannot be empty")
        if not hasattr(tokenizer, "encode") or not hasattr(
            tokenizer, "decode"
        ):
            raise ValueError("tokenizer must have encode and decode methods")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if context_window < 0:
            raise ValueError("context_window must be non-negative")

        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window
        self.include_system_prompts = include_system_prompts

        # Process conversations into training examples
        self.examples = self._process_conversations()

        logger.info(
            f"ChatbotDataset initialized with {len(self.examples)} examples"
        )

    def _process_conversations(self) -> List[Dict[str, Any]]:
        """Process conversations into training examples."""
        examples = []

        for conv_idx, conversation in enumerate(self.conversations):
            try:
                # Extract messages from conversation
                messages = self._extract_messages(conversation)
                if not messages:
                    continue

                # Create training examples from conversation
                conv_examples = self._create_examples_from_messages(
                    messages, conv_idx
                )
                examples.extend(conv_examples)

            except Exception as e:
                logger.warning(
                    f"Error processing conversation {conv_idx}: {e}"
                )
                continue

        return examples

    def _extract_messages(
        self, conversation: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract messages from a conversation."""
        messages = []

        # Handle different conversation formats
        if "messages" in conversation:
            # Format: {"messages": [{"role": "user", "content": "..."}, ...]}
            messages = conversation["messages"]
        elif "turns" in conversation:
            # Format: {"turns": [{"speaker": "user", "text": "..."}, ...]}
            for turn in conversation["turns"]:
                role = turn.get("speaker", "user")
                content = turn.get("text", "")
                messages.append({"role": role, "content": content})
        elif "dialogue" in conversation:
            # Format: {"dialogue": [{"speaker": "user", "text": "..."}, ...]}
            for turn in conversation["dialogue"]:
                role = turn.get("speaker", "user")
                content = turn.get("text", "")
                messages.append({"role": role, "content": content})
        else:
            # Try to infer format from keys
            for key, value in conversation.items():
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict) and "role" in value[0]:
                        messages = value
                        break
                    elif isinstance(value[0], dict) and "speaker" in value[0]:
                        for turn in value:
                            role = turn.get("speaker", "user")
                            content = turn.get("text", "")
                            messages.append({"role": role, "content": content})
                        break

        # Filter and validate messages
        valid_messages = []
        for msg in messages:
            if (
                isinstance(msg, dict)
                and "content" in msg
                and msg["content"].strip()
            ):
                role = msg.get("role", "user")
                if role in ["user", "assistant", "system"]:
                    valid_messages.append(
                        {"role": role, "content": msg["content"].strip()}
                    )

        return valid_messages

    def _create_examples_from_messages(
        self, messages: List[Dict[str, str]], conv_idx: int
    ) -> List[Dict[str, Any]]:
        """Create training examples from a list of messages."""
        examples = []

        # Group messages by speaker
        grouped_messages: List[Dict[str, str]] = []
        current_group: Dict[str, Any] = {"role": None, "content": []}

        for msg in messages:
            if msg["role"] == current_group["role"]:
                current_group["content"].append(msg["content"])
            else:
                if current_group["role"] is not None:
                    grouped_messages.append(
                        {
                            "role": current_group["role"],
                            "content": " ".join(current_group["content"]),
                        }
                    )
                current_group = {
                    "role": msg["role"],
                    "content": [msg["content"]],
                }

        # Add the last group
        if current_group["role"] is not None:
            grouped_messages.append(
                {
                    "role": current_group["role"],
                    "content": " ".join(current_group["content"]),
                }
            )

        # Create examples from grouped messages
        for i in range(len(grouped_messages)):
            if grouped_messages[i]["role"] == "assistant":
                # Create training example
                context_messages = grouped_messages[
                    max(0, i - self.context_window) : i
                ]
                target_message = grouped_messages[i]

                # Build context
                context = self._build_context(context_messages)
                target = target_message["content"]

                # Create example
                example = {
                    "input": context,
                    "target": target,
                    "conversation_id": conv_idx,
                    "message_index": i,
                }

                examples.append(example)

        return examples

    def _build_context(self, messages: List[Dict[str, str]]) -> str:
        """Build context string from messages."""
        context_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system" and self.include_system_prompts:
                context_parts.append(f"System: {content}")
            elif role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")

        return "\n".join(context_parts)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example by index."""
        if idx >= len(self.examples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size "
                f"{len(self.examples)}"
            )

        example = self.examples[idx]

        # Encode input and target
        input_text = example["input"]
        target_text = example["target"]

        # Tokenize
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        target_ids = self.tokenizer.encode(
            target_text, add_special_tokens=True
        )

        # Combine input and target for training
        combined_ids = input_ids + target_ids[1:]  # Skip BOS token from target

        # Truncate if too long
        if len(combined_ids) > self.max_length:
            combined_ids = combined_ids[: self.max_length]

        # Create attention mask
        attention_mask = [1] * len(combined_ids)

        # Pad to max_length
        padding_length = self.max_length - len(combined_ids)
        if padding_length > 0:
            combined_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)

        return {
            "input_ids": torch.tensor(combined_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "input_text": input_text,
            "target_text": target_text,
        }

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversations in the dataset."""
        stats = {
            "total_conversations": len(self.conversations),
            "total_examples": len(self.examples),
            "avg_examples_per_conversation": (
                len(self.examples) / len(self.conversations)
                if self.conversations
                else 0
            ),
            "role_distribution": {"user": 0, "assistant": 0, "system": 0},
            "avg_message_length": 0,
            "avg_context_length": 0,
        }

        total_message_length = 0
        total_context_length = 0
        message_count = 0

        for example in self.examples:
            # Count roles in context
            context = example["input"]
            role_dist = stats["role_distribution"]
            assert isinstance(role_dist, dict)
            if "User:" in context:
                role_dist["user"] += context.count("User:")
            if "Assistant:" in context:
                role_dist["assistant"] += context.count("Assistant:")
            if "System:" in context:
                role_dist["system"] += context.count("System:")

            # Calculate lengths
            total_message_length += len(example["target"])
            total_context_length += len(example["input"])
            message_count += 1

        if message_count > 0:
            stats["avg_message_length"] = total_message_length / message_count
            stats["avg_context_length"] = total_context_length / message_count

        return stats


class ChatbotDataLoader:
    """DataLoader for chatbot datasets with various dataset sources."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_length: int = 512,
        context_window: int = 3,
        include_system_prompts: bool = True,
    ) -> None:
        """
        Initialize the ChatbotDataLoader.

        Args:
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
            context_window: Number of previous turns to include as context.
            include_system_prompts: Whether to include system prompts.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window
        self.include_system_prompts = include_system_prompts

        logger.info("ChatbotDataLoader initialized")

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        num_samples: Optional[int] = None,
    ) -> ChatbotDataset:
        """
        Load a chatbot dataset.

        Args:
            dataset_name: Name of the dataset to load.
            split: Dataset split to load.
            num_samples: Number of samples to load (None for all).

        Returns:
            ChatbotDataset instance.
        """
        if dataset_name.lower() == "dailydialog":
            conversations = self._load_dailydialog(split, num_samples)
        elif dataset_name.lower() == "personachat":
            conversations = self._load_personachat(split, num_samples)
        elif dataset_name.lower() == "custom":
            raise ValueError("Use load_custom_dataset() for custom datasets")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return ChatbotDataset(
            conversations=conversations,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            context_window=self.context_window,
            include_system_prompts=self.include_system_prompts,
        )

    def load_custom_dataset(
        self,
        data_path: Union[str, Path],
        format: str = "json",
    ) -> ChatbotDataset:
        """
        Load a custom dataset from file.

        Args:
            data_path: Path to the dataset file.
            format: Format of the dataset file ("json" or "jsonl").

        Returns:
            ChatbotDataset instance.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        conversations = []

        if format.lower() == "json":
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    conversations = data
                elif isinstance(data, dict) and "conversations" in data:
                    conversations = data["conversations"]
                else:
                    raise ValueError("Invalid JSON format")

        elif format.lower() == "jsonl":
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        conversation = json.loads(line)
                        conversations.append(conversation)

        else:
            raise ValueError(f"Unsupported format: {format}")

        return ChatbotDataset(
            conversations=conversations,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            context_window=self.context_window,
            include_system_prompts=self.include_system_prompts,
        )

    def _load_dailydialog(
        self, split: str, num_samples: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Load DailyDialog dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library not available. "
                "Install with: pip install datasets"
            )

        logger.info(f"Loading DailyDialog {split} split...")

        # Load dataset
        dataset = load_dataset("daily_dialog", split=split)

        conversations = []
        for i, example in enumerate(dataset):
            if num_samples and i >= num_samples:
                break

            # Convert to our format
            conversation: Dict[str, List[Dict[str, str]]] = {"messages": []}

            # Add system prompt
            if self.include_system_prompts:
                conversation["messages"].append(
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant engaged in "
                            "daily conversation."
                        ),
                    }
                )

            # Add dialogue turns
            for turn in example["dialog"]:
                # Alternate between user and assistant
                role = (
                    "user"
                    if len(conversation["messages"]) % 2
                    == (1 if self.include_system_prompts else 0)
                    else "assistant"
                )
                conversation["messages"].append(
                    {"role": role, "content": turn}
                )

            conversations.append(conversation)

        logger.info(f"Loaded {len(conversations)} DailyDialog conversations")
        return conversations

    def _load_personachat(
        self, split: str, num_samples: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Load PersonaChat dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library not available. "
                "Install with: pip install datasets"
            )

        logger.info(f"Loading PersonaChat {split} split...")

        # Load dataset
        dataset = load_dataset("personachat", split=split)

        conversations = []
        for i, example in enumerate(dataset):
            if num_samples and i >= num_samples:
                break

            # Convert to our format
            conversation: Dict[str, List[Dict[str, str]]] = {"messages": []}

            # Add system prompt with persona
            if self.include_system_prompts and "persona" in example:
                persona_text = " ".join(example["persona"])
                conversation["messages"].append(
                    {
                        "role": "system",
                        "content": (
                            f"You are a helpful assistant with the "
                            f"following persona: {persona_text}"
                        ),
                    }
                )

            # Add dialogue turns
            for turn in example["utterances"]:
                for j, utterance in enumerate(turn):
                    role = "user" if j % 2 == 0 else "assistant"
                    conversation["messages"].append(
                        {"role": role, "content": utterance}
                    )

            conversations.append(conversation)

        logger.info(f"Loaded {len(conversations)} PersonaChat conversations")
        return conversations

    def create_dataloader(
        self,
        dataset: ChatbotDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create a DataLoader from a ChatbotDataset.

        Args:
            dataset: ChatbotDataset instance.
            batch_size: Batch size for the DataLoader.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes.
            pin_memory: Whether to pin memory for faster GPU transfer.

        Returns:
            DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Collate function for batching."""
        # Stack tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack(
            [item["attention_mask"] for item in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_texts": [item["input_text"] for item in batch],
            "target_texts": [item["target_text"] for item in batch],
        }

    def split_dataset(
        self,
        dataset: ChatbotDataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> Tuple[ChatbotDataset, ChatbotDataset, ChatbotDataset]:
        """
        Split a dataset into train, validation, and test sets.

        Args:
            dataset: ChatbotDataset to split.
            train_ratio: Ratio of data for training.
            val_ratio: Ratio of data for validation.
            test_ratio: Ratio of data for testing.
            random_seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        # Set random seed
        random.seed(random_seed)

        # Shuffle examples
        examples = dataset.examples.copy()
        random.shuffle(examples)

        # Calculate split indices
        total_examples = len(examples)
        train_end = int(total_examples * train_ratio)
        val_end = train_end + int(total_examples * val_ratio)

        # Split examples
        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]

        # Create new datasets
        train_dataset = ChatbotDataset(
            conversations=dataset.conversations,
            tokenizer=dataset.tokenizer,
            max_length=dataset.max_length,
            context_window=dataset.context_window,
            include_system_prompts=dataset.include_system_prompts,
        )
        train_dataset.examples = train_examples

        val_dataset = ChatbotDataset(
            conversations=dataset.conversations,
            tokenizer=dataset.tokenizer,
            max_length=dataset.max_length,
            context_window=dataset.context_window,
            include_system_prompts=dataset.include_system_prompts,
        )
        val_dataset.examples = val_examples

        test_dataset = ChatbotDataset(
            conversations=dataset.conversations,
            tokenizer=dataset.tokenizer,
            max_length=dataset.max_length,
            context_window=dataset.context_window,
            include_system_prompts=dataset.include_system_prompts,
        )
        test_dataset.examples = test_examples

        logger.info(
            f"Dataset split: {len(train_examples)} train, "
            f"{len(val_examples)} val, {len(test_examples)} test"
        )

        return train_dataset, val_dataset, test_dataset
