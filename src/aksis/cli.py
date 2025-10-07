"""Command-line interface for Aksis."""

import click
from typing import Optional

from aksis.utils.logging import setup_logging, get_logger
from aksis.utils.device import get_device

logger = get_logger(__name__)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level.",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to use for computations.",
)
def main(log_level: str, device: Optional[str]) -> None:
    """Aksis - AI Chatbot/LLM from Scratch."""
    setup_logging(level=log_level)

    if device:
        device_obj = get_device(device)
        logger.info(f"Using device: {device_obj}")

    logger.info("Aksis CLI initialized")


@main.command()
@click.option(
    "--text",
    default="Hello world! This is a test.",
    help="Text to tokenize.",
)
def tokenize(text: str) -> None:
    """Test tokenization functionality."""
    from aksis.data.tokenizer import Tokenizer

    logger.info(f"Tokenizing text: {text}")

    # Create tokenizer and build vocab
    tokenizer = Tokenizer()
    tokenizer.build_vocab([text])

    # Encode text
    encoded = tokenizer.encode(text, add_special_tokens=True)
    logger.info(f"Encoded: {encoded}")

    # Decode back
    decoded = tokenizer.decode(encoded)
    logger.info(f"Decoded: {decoded}")

    # Show vocabulary info
    logger.info(f"Vocabulary size: {tokenizer.vocab_size_with_special}")


@main.command()
@click.option(
    "--texts",
    default=["Hello world!", "This is a test.", "Another sentence."],
    multiple=True,
    help="Texts to process.",
)
@click.option(
    "--batch-size",
    default=2,
    help="Batch size for processing.",
)
def test_dataloader(texts: tuple, batch_size: int) -> None:
    """Test DataLoader functionality."""
    from aksis.data.tokenizer import Tokenizer
    from aksis.data.dataloader import DataLoader

    texts_list = list(texts)
    logger.info(f"Testing DataLoader with {len(texts_list)} texts")

    # Create tokenizer and build vocab
    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts_list)

    # Create DataLoader
    dataloader = DataLoader(
        texts=texts_list,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=20,
        padding=True,
        truncation=True,
    )

    # Show stats
    stats = dataloader.get_stats()
    logger.info(f"DataLoader stats: {stats}")

    # Process batches
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i}: input_ids shape={batch['input_ids'].shape}")
        logger.info(f"Batch {i}: attention_mask shape={batch['attention_mask'].shape}")


@main.command()
def info() -> None:
    """Show system information."""
    import torch

    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    device = get_device()
    logger.info(f"Default device: {device}")


if __name__ == "__main__":
    main()
