"""Command-line interface for Aksis."""

import click
import os
from typing import Optional

from aksis.utils.logging import setup_logging, get_logger
from aksis.utils.device import get_device
from aksis.model.transformer import TransformerDecoder
from aksis.data.tokenizer import Tokenizer
from aksis.data.dataloader import DataLoader

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
        logger.info(
            f"Batch {i}: attention_mask shape={batch['attention_mask'].shape}"
        )


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


@main.command()
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to use for computations.",
)
def test_model(device: Optional[str]) -> None:
    """Test the transformer model with a simple forward pass."""
    import torch

    logger.info("Testing transformer model...")

    # Get device
    device_obj = get_device(device)
    logger.info(f"Using device: {device_obj}")

    # Create sample data
    texts = [
        "Hello world, this is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    # Create tokenizer and build vocabulary
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.build_vocab(texts)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size_with_special}")

    # Create DataLoader
    dataloader = DataLoader(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=20,
        device=device_obj,
    )

    # Create model
    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size_with_special,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=1000,
        dropout=0.1,
    ).to(device_obj)

    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Test forward pass
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            logger.info(f"Input shape: {input_ids.shape}")
            logger.info(f"Attention mask shape: {attention_mask.shape}")

            # Forward pass
            output = model(input_ids, padding_mask=attention_mask)

            logger.info(f"Output shape: {output.shape}")
            logger.info(f"Output device: {output.device}")
            logger.info(f"Output dtype: {output.dtype}")

            # Check for NaN or Inf
            if torch.isnan(output).any():
                logger.error("Output contains NaN values!")
            elif torch.isinf(output).any():
                logger.error("Output contains Inf values!")
            else:
                logger.info("Output is numerically stable")

            break  # Only test first batch

    logger.info("Model test completed successfully!")


@main.command()
@click.option(
    "--dataset",
    default="wikitext2",
    type=click.Choice(["wikitext2", "shakespeare"]),
    help="Dataset to use for training.",
)
@click.option(
    "--epochs",
    default=1,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    default=32,
    help="Batch size for training.",
)
@click.option(
    "--learning-rate",
    default=5e-5,
    help="Learning rate for training.",
)
@click.option(
    "--max-length",
    default=512,
    help="Maximum sequence length.",
)
@click.option(
    "--checkpoint-dir",
    default="./checkpoints",
    help="Directory to save checkpoints.",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to use for training.",
)
@click.option(
    "--mixed-precision",
    is_flag=True,
    help="Use mixed precision training.",
)
@click.option(
    "--gradient-accumulation-steps",
    default=1,
    help="Number of gradient accumulation steps.",
)
@click.option(
    "--max-grad-norm",
    default=1.0,
    help="Maximum gradient norm for clipping.",
)
def train_model(
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    checkpoint_dir: str,
    device: Optional[str],
    mixed_precision: bool,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> None:
    """Train the transformer model."""
    import os
    from aksis.train.dataset import (
        load_wikitext2,
        load_shakespeare,
        create_dataloaders,
    )
    from aksis.train.trainer import Trainer
    from aksis.model.transformer import TransformerDecoder
    from aksis.data.tokenizer import Tokenizer

    logger.info("Starting model training...")

    # Get device
    device_obj = get_device(device)
    logger.info(f"Using device: {device_obj}")

    # Create tokenizer
    tokenizer = Tokenizer(vocab_size=10000)
    logger.info("Tokenizer created")

    # Load dataset
    if dataset == "wikitext2":
        train_dataset, val_dataset, test_dataset = load_wikitext2(
            tokenizer, max_length=max_length
        )
    elif dataset == "shakespeare":
        train_dataset, val_dataset, test_dataset = load_shakespeare(
            tokenizer, max_length=max_length
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info(f"Dataset loaded: {dataset}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    logger.info("DataLoaders created")

    # Create model
    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size_with_special,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=max_length,
        dropout=0.1,
    ).to(device_obj)

    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        lr=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        use_mixed_precision=mixed_precision,
    )

    logger.info("Trainer created")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train model
    logger.info(f"Starting training for {epochs} epochs...")
    trainer.train()

    logger.info("Training completed successfully!")


@main.command()
@click.option(
    "--checkpoint-path",
    required=True,
    help="Path to the checkpoint file.",
)
@click.option(
    "--dataset",
    default="wikitext2",
    type=click.Choice(["wikitext2", "shakespeare"]),
    help="Dataset to use for evaluation.",
)
@click.option(
    "--batch-size",
    default=32,
    help="Batch size for evaluation.",
)
@click.option(
    "--max-length",
    default=512,
    help="Maximum sequence length.",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to use for evaluation.",
)
def eval_model(
    checkpoint_path: str,
    dataset: str,
    batch_size: int,
    max_length: int,
    device: Optional[str],
) -> None:
    """Evaluate the trained model."""
    from aksis.train.dataset import (
        load_wikitext2,
        load_shakespeare,
        create_dataloaders,
    )
    from aksis.train.trainer import Trainer
    from aksis.model.transformer import TransformerDecoder
    from aksis.data.tokenizer import Tokenizer

    logger.info("Starting model evaluation...")

    # Get device
    device_obj = get_device(device)
    logger.info(f"Using device: {device_obj}")

    # Create tokenizer
    tokenizer = Tokenizer(vocab_size=10000)
    logger.info("Tokenizer created")

    # Load dataset
    if dataset == "wikitext2":
        train_dataset, val_dataset, test_dataset = load_wikitext2(
            tokenizer, max_length=max_length
        )
    elif dataset == "shakespeare":
        train_dataset, val_dataset, test_dataset = load_shakespeare(
            tokenizer, max_length=max_length
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info(f"Dataset loaded: {dataset}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    logger.info("DataLoaders created")

    # Create model
    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size_with_special,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=max_length,
        dropout=0.1,
    ).to(device_obj)

    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="./checkpoints",
    )

    logger.info("Trainer created")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    logger.info(
        f"Checkpoint loaded: epoch {checkpoint['epoch']}, loss {checkpoint['loss']}"
    )

    # Evaluate model
    logger.info("Starting evaluation...")
    val_loss, val_perplexity = trainer.evaluate()

    logger.info(
        f"Evaluation completed - Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}"
    )


@main.command()
@click.option(
    "--checkpoint-dir",
    default="./checkpoints",
    help="Directory containing checkpoints.",
)
def list_checkpoints(checkpoint_dir: str) -> None:
    """List available checkpoints."""
    import os
    from aksis.train.checkpoint import CheckpointManager

    logger.info(f"Listing checkpoints in {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # List checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()

    if not checkpoints:
        logger.info("No checkpoints found")
        return

    logger.info("Available checkpoints:")
    for checkpoint in checkpoints:
        logger.info(f"  - {checkpoint}")


@main.command()
@click.option(
    "--checkpoint-path",
    required=True,
    help="Path to the checkpoint file.",
)
@click.option(
    "--output-path",
    required=True,
    help="Path to save the model.",
)
def save_model(checkpoint_path: str, output_path: str) -> None:
    """Save model from checkpoint."""
    import torch
    from aksis.model.transformer import TransformerDecoder

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create model
    model = TransformerDecoder(
        vocab_size=checkpoint["model_state_dict"]["embedding.weight"].shape[0],
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=1000,
        dropout=0.1,
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Save model
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")


@main.command()
@click.option(
    "--checkpoint-path",
    required=True,
    help="Path to the checkpoint file.",
)
@click.option(
    "--prompt",
    required=True,
    help="Text prompt for generation.",
)
@click.option(
    "--max-new-tokens",
    default=100,
    help="Maximum number of new tokens to generate.",
)
@click.option(
    "--sampler",
    default="greedy",
    type=click.Choice(["greedy", "top-k", "top-p", "temperature"]),
    help="Sampling strategy to use.",
)
@click.option(
    "--temperature",
    default=0.7,
    help="Temperature for temperature sampling.",
)
@click.option(
    "--top-k",
    default=50,
    help="Top-k parameter for top-k sampling.",
)
@click.option(
    "--top-p",
    default=0.95,
    help="Top-p parameter for top-p sampling.",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to use for inference.",
)
@click.option(
    "--mixed-precision",
    is_flag=True,
    help="Use mixed precision inference.",
)
def generate_text(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    sampler: str,
    temperature: float,
    top_k: int,
    top_p: float,
    device: Optional[str],
    mixed_precision: bool,
) -> None:
    """Generate text from a prompt using a trained model."""
    from aksis.inference.inference import Generator
    from aksis.inference.sampler import (
        GreedySampler,
        TopKSampler,
        TopPSampler,
        TemperatureSampler,
    )
    from aksis.data.tokenizer import Tokenizer
    from aksis.utils.device import get_device

    logger.info(f"Generating text from prompt: {prompt[:50]}...")

    # Get device
    device_obj = get_device(device)

    # Load tokenizer
    tokenizer = Tokenizer()

    # Load generator from checkpoint
    generator = Generator.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        device=device_obj,
        use_mixed_precision=mixed_precision,
    )

    # Create sampler
    if sampler == "greedy":
        sampler_obj = GreedySampler()
    elif sampler == "top-k":
        sampler_obj = TopKSampler(k=top_k)
    elif sampler == "top-p":
        sampler_obj = TopPSampler(p=top_p)
    elif sampler == "temperature":
        sampler_obj = TemperatureSampler(temperature=temperature)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Generate text
    logger.info("Generating text...")
    generated_text = generator.generate(
        prompt=prompt,
        sampler=sampler_obj,
        max_new_tokens=max_new_tokens,
    )

    # Display results
    click.echo(f"\nPrompt: {prompt}")
    click.echo(f"Generated: {generated_text}")
    click.echo(f"\nSampler: {sampler}")
    click.echo(f"Max new tokens: {max_new_tokens}")
    click.echo(f"Device: {device_obj}")


@main.command()
@click.option(
    "--checkpoint-path",
    required=True,
    help="Path to the checkpoint file.",
)
@click.option(
    "--system-prompt",
    default=None,
    help="System prompt to set chatbot behavior.",
)
@click.option(
    "--max-new-tokens",
    default=100,
    help="Maximum number of new tokens to generate.",
)
@click.option(
    "--sampler",
    default="greedy",
    type=click.Choice(["greedy", "top-k", "top-p", "temperature"]),
    help="Sampling strategy to use.",
)
@click.option(
    "--temperature",
    default=0.7,
    help="Temperature for temperature sampling.",
)
@click.option(
    "--top-k",
    default=50,
    help="Top-k parameter for top-k sampling.",
)
@click.option(
    "--top-p",
    default=0.95,
    help="Top-p parameter for top-p sampling.",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to use for inference.",
)
@click.option(
    "--mixed-precision",
    is_flag=True,
    help="Use mixed precision inference.",
)
@click.option(
    "--context-file",
    default=None,
    help="File to save/load conversation context.",
)
def chat_with_model(
    checkpoint_path: str,
    system_prompt: Optional[str],
    max_new_tokens: int,
    sampler: str,
    temperature: float,
    top_k: int,
    top_p: float,
    device: Optional[str],
    mixed_precision: bool,
    context_file: Optional[str],
) -> None:
    """Start an interactive chat session with the model."""
    from aksis.inference.inference import Generator
    from aksis.inference.sampler import (
        GreedySampler,
        TopKSampler,
        TopPSampler,
        TemperatureSampler,
    )
    from aksis.inference.context_manager import ContextManager
    from aksis.inference.chatbot import ChatBot
    from aksis.data.tokenizer import Tokenizer
    from aksis.utils.device import get_device

    logger.info("Starting interactive chat session...")

    # Get device
    device_obj = get_device(device)

    # Load tokenizer
    tokenizer = Tokenizer()

    # Load generator from checkpoint
    generator = Generator.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        device=device_obj,
        use_mixed_precision=mixed_precision,
    )

    # Create sampler
    if sampler == "greedy":
        sampler_obj = GreedySampler()
    elif sampler == "top-k":
        sampler_obj = TopKSampler(k=top_k)
    elif sampler == "top-p":
        sampler_obj = TopPSampler(p=top_p)
    elif sampler == "temperature":
        sampler_obj = TemperatureSampler(temperature=temperature)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Create context manager
    context_manager = ContextManager(
        tokenizer=tokenizer,
        max_context_length=512,
    )

    # Create chatbot
    chatbot = ChatBot(
        generator=generator,
        context_manager=context_manager,
        sampler=sampler_obj,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )

    # Load context if file exists
    if context_file and os.path.exists(context_file):
        try:
            chatbot.load_context(context_file)
            logger.info(f"Loaded context from {context_file}")
        except Exception as e:
            logger.warning(f"Failed to load context: {e}")

    # Display welcome message
    click.echo("\n🤖 Aksis ChatBot - Interactive AI Assistant")
    click.echo("=" * 50)
    if system_prompt:
        click.echo(f"System: {system_prompt}")
    click.echo(f"Sampler: {sampler}")
    click.echo(f"Device: {device_obj}")
    click.echo("Type 'quit', 'exit', or 'bye' to end the session")
    click.echo("Type 'clear' to clear the conversation context")
    click.echo("Type 'info' to show chatbot information")
    click.echo("=" * 50)

    # Chat loop
    try:
        while True:
            # Get user input
            user_input = click.prompt("\nYou", type=str)

            # Handle special commands
            if user_input.lower() in ["quit", "exit", "bye"]:
                click.echo("Goodbye! 👋")
                break
            elif user_input.lower() == "clear":
                chatbot.clear_context()
                click.echo("Context cleared! 🧹")
                continue
            elif user_input.lower() == "info":
                info = chatbot.get_info()
                click.echo("\nChatBot Info:")
                click.echo(
                    f"  Messages: {info['context_manager']['num_messages']}"
                )
                click.echo(
                    f"  Tokens: {info['context_manager']['total_tokens']}"
                )
                click.echo(f"  Sampler: {info['sampler']}")
                click.echo(f"  Max tokens: {info['max_new_tokens']}")
                continue

            # Generate response
            try:
                response = chatbot.chat(user_input)
                click.echo(f"\n🤖 {response}")
            except Exception as e:
                click.echo(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")

    except KeyboardInterrupt:
        click.echo("\n\nGoodbye! 👋")

    # Save context if file specified
    if context_file:
        try:
            chatbot.save_context(context_file)
            logger.info(f"Saved context to {context_file}")
        except Exception as e:
            logger.warning(f"Failed to save context: {e}")


@main.command()
@click.option(
    "--checkpoint-path",
    required=True,
    help="Path to the checkpoint file.",
)
@click.option(
    "--prompt",
    default="The quick brown fox",
    help="Prompt to use for benchmarking.",
)
@click.option(
    "--max-new-tokens",
    default=50,
    help="Maximum number of new tokens to generate.",
)
@click.option(
    "--num-runs",
    default=5,
    help="Number of runs to average.",
)
@click.option(
    "--sampler",
    default="greedy",
    type=click.Choice(["greedy", "top-k", "top-p", "temperature"]),
    help="Sampling strategy to use.",
)
@click.option(
    "--temperature",
    default=0.7,
    help="Temperature for temperature sampling.",
)
@click.option(
    "--top-k",
    default=50,
    help="Top-k parameter for top-k sampling.",
)
@click.option(
    "--top-p",
    default=0.95,
    help="Top-p parameter for top-p sampling.",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to use for inference.",
)
@click.option(
    "--mixed-precision",
    is_flag=True,
    help="Use mixed precision inference.",
)
def benchmark_inference(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    num_runs: int,
    sampler: str,
    temperature: float,
    top_k: int,
    top_p: float,
    device: Optional[str],
    mixed_precision: bool,
) -> None:
    """Benchmark inference performance."""
    from aksis.inference.inference import Generator
    from aksis.inference.sampler import (
        GreedySampler,
        TopKSampler,
        TopPSampler,
        TemperatureSampler,
    )
    from aksis.inference.context_manager import ContextManager
    from aksis.inference.chatbot import ChatBot
    from aksis.data.tokenizer import Tokenizer
    from aksis.utils.device import get_device

    logger.info("Starting inference benchmark...")

    # Get device
    device_obj = get_device(device)

    # Load tokenizer
    tokenizer = Tokenizer()

    # Load generator from checkpoint
    generator = Generator.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        device=device_obj,
        use_mixed_precision=mixed_precision,
    )

    # Create sampler
    if sampler == "greedy":
        sampler_obj = GreedySampler()
    elif sampler == "top-k":
        sampler_obj = TopKSampler(k=top_k)
    elif sampler == "top-p":
        sampler_obj = TopPSampler(p=top_p)
    elif sampler == "temperature":
        sampler_obj = TemperatureSampler(temperature=temperature)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Create context manager and chatbot
    context_manager = ContextManager(tokenizer=tokenizer)
    chatbot = ChatBot(
        generator=generator,
        context_manager=context_manager,
        sampler=sampler_obj,
    )

    # Run benchmark
    logger.info(f"Running {num_runs} benchmark runs...")
    results = chatbot.benchmark_generation(
        prompt=prompt,
        num_runs=num_runs,
        max_new_tokens=max_new_tokens,
    )

    # Display results
    click.echo("\n🚀 Inference Benchmark Results")
    click.echo("=" * 40)
    click.echo(f"Prompt: {prompt}")
    click.echo(f"Sampler: {sampler}")
    click.echo(f"Device: {device_obj}")
    click.echo(f"Mixed precision: {mixed_precision}")
    click.echo(f"Runs: {num_runs}")
    click.echo("-" * 40)
    click.echo(f"Average time: {results['avg_time']:.3f}s")
    click.echo(f"Average tokens: {results['avg_tokens']:.1f}")
    click.echo(f"Tokens per second: {results['tokens_per_second']:.2f}")
    click.echo(f"Prompt length: {results['prompt_length']} chars")
    click.echo("=" * 40)


@main.command()
@click.option(
    "--checkpoint",
    required=True,
    help="Path to model checkpoint to evaluate.",
)
@click.option(
    "--dataset",
    default="wikitext-2",
    help="Dataset to evaluate on.",
)
@click.option(
    "--split",
    default="test",
    help="Dataset split to use.",
)
@click.option(
    "--output",
    default="evaluation_results.json",
    help="Output file for evaluation results.",
)
@click.option(
    "--batch-size",
    default=16,
    help="Batch size for evaluation.",
)
@click.option(
    "--max-length",
    default=512,
    help="Maximum sequence length.",
)
@click.option(
    "--mixed-precision/--no-mixed-precision",
    default=True,
    help="Use mixed precision evaluation.",
)
def eval_model_phase5(
    checkpoint: str,
    dataset: str,
    split: str,
    output: str,
    batch_size: int,
    max_length: int,
    mixed_precision: bool,
) -> None:
    """Evaluate a trained model on a dataset."""
    from aksis.eval.evaluator import Evaluator
    from aksis.data.dataloader import DataLoader
    from aksis.model.transformer import TransformerDecoder
    from aksis.data.tokenizer import Tokenizer
    import torch

    logger.info(f"Evaluating model: {checkpoint}")
    logger.info(f"Dataset: {dataset} ({split} split)")

    # Load model and tokenizer
    device = get_device()
    checkpoint_data = torch.load(checkpoint, map_location=device)

    # Create model
    model = TransformerDecoder(
        vocab_size=checkpoint_data.get("vocab_size", 10000),
        d_model=checkpoint_data.get("d_model", 512),
        num_heads=checkpoint_data.get("num_heads", 8),
        num_layers=checkpoint_data.get("num_layers", 6),
        d_ff=checkpoint_data.get("d_ff", 2048),
        max_length=checkpoint_data.get("max_length", 512),
        dropout=checkpoint_data.get("dropout", 0.1),
    )

    # Load model weights
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.to(device)

    # Create tokenizer
    tokenizer = Tokenizer()

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_mixed_precision=mixed_precision,
    )

    # Load dataset
    dataloader = DataLoader(
        dataset_name=dataset,
        split=split,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False,
    )

    # Convert dataloader to evaluation format
    eval_data = []
    for batch in dataloader:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            for i in range(input_ids.size(0)):
                # Decode input and target
                input_text = tokenizer.decode(input_ids[i].tolist())
                # For language modeling, target is same as input
                target_text = input_text
                eval_data.append({"input": input_text, "target": target_text})
        else:
            for i in range(batch.size(0)):
                input_text = tokenizer.decode(batch[i].tolist())
                eval_data.append({"input": input_text, "target": input_text})

    # Evaluate
    logger.info(f"Evaluating on {len(eval_data)} samples...")
    results = evaluator.evaluate_dataset(eval_data)

    # Save results
    evaluator.save_results(results, output)

    # Display results
    click.echo("\n📊 Evaluation Results")
    click.echo("=" * 40)
    click.echo(f"Dataset: {dataset} ({split})")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Samples: {len(eval_data)}")
    click.echo("-" * 40)
    for metric, value in results.items():
        if isinstance(value, float):
            click.echo(f"{metric}: {value:.4f}")
        else:
            click.echo(f"{metric}: {value}")
    click.echo(f"Results saved to: {output}")
    click.echo("=" * 40)


@main.command()
@click.option(
    "--checkpoint",
    required=True,
    help="Path to pre-trained model checkpoint.",
)
@click.option(
    "--dataset",
    default="dailydialog",
    help="Chatbot dataset to fine-tune on.",
)
@click.option(
    "--output-dir",
    default="fine_tuned_models",
    help="Directory to save fine-tuned models.",
)
@click.option(
    "--learning-rate",
    default=1e-4,
    help="Learning rate for fine-tuning.",
)
@click.option(
    "--batch-size",
    default=16,
    help="Batch size for fine-tuning.",
)
@click.option(
    "--epochs",
    default=3,
    help="Number of epochs to fine-tune.",
)
@click.option(
    "--max-length",
    default=512,
    help="Maximum sequence length.",
)
@click.option(
    "--mixed-precision/--no-mixed-precision",
    default=True,
    help="Use mixed precision training.",
)
@click.option(
    "--early-stopping-patience",
    default=3,
    help="Early stopping patience.",
)
def fine_tune_model(
    checkpoint: str,
    dataset: str,
    output_dir: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    max_length: int,
    mixed_precision: bool,
    early_stopping_patience: int,
) -> None:
    """Fine-tune a pre-trained model on a chatbot dataset."""
    from aksis.eval.fine_tuner import FineTuner
    from aksis.eval.dataset import ChatbotDataLoader
    from aksis.model.transformer import TransformerDecoder
    from aksis.data.tokenizer import Tokenizer
    import torch
    import os

    logger.info(f"Fine-tuning model: {checkpoint}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    device = get_device()
    checkpoint_data = torch.load(checkpoint, map_location=device)

    # Create model
    model = TransformerDecoder(
        vocab_size=checkpoint_data.get("vocab_size", 10000),
        d_model=checkpoint_data.get("d_model", 512),
        num_heads=checkpoint_data.get("num_heads", 8),
        num_layers=checkpoint_data.get("num_layers", 6),
        d_ff=checkpoint_data.get("d_ff", 2048),
        max_length=checkpoint_data.get("max_length", 512),
        dropout=checkpoint_data.get("dropout", 0.1),
    )

    # Load model weights
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)

    model.to(device)

    # Create tokenizer
    tokenizer = Tokenizer()

    # Create fine-tuner
    fine_tuner = FineTuner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        use_mixed_precision=mixed_precision,
    )

    # Load chatbot dataset
    data_loader = ChatbotDataLoader(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    try:
        # Load dataset
        chatbot_dataset = data_loader.load_dataset(
            dataset_name=dataset,
            split="train",
            num_samples=100,  # Limit for demo
        )

        # Split dataset
        train_dataset, val_dataset, test_dataset = data_loader.split_dataset(
            chatbot_dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        # Create data loaders
        train_loader = data_loader.create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = data_loader.create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # Fine-tune
        logger.info("Starting fine-tuning...")
        results = fine_tuner.fine_tune(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            output_dir=output_dir,
            save_best_only=True,
        )

        # Display results
        click.echo("\n🎯 Fine-tuning Results")
        click.echo("=" * 40)
        click.echo(f"Dataset: {dataset}")
        click.echo(f"Checkpoint: {checkpoint}")
        click.echo(f"Epochs: {epochs}")
        click.echo(f"Learning rate: {learning_rate}")
        click.echo(f"Batch size: {batch_size}")
        click.echo("-" * 40)
        click.echo(f"Total epochs: {results['total_epochs']}")
        click.echo(f"Best validation loss: {results['best_val_loss']:.4f}")
        click.echo(f"Output directory: {output_dir}")
        click.echo("=" * 40)

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        click.echo(f"❌ Fine-tuning failed: {e}")
        raise


@main.command()
@click.option(
    "--results-file",
    required=True,
    help="Path to evaluation results file.",
)
@click.option(
    "--output-dir",
    default="plots",
    help="Directory to save plots.",
)
@click.option(
    "--format",
    default="png",
    type=click.Choice(["png", "pdf", "svg"]),
    help="Plot format.",
)
def plot_metrics(
    results_file: str,
    output_dir: str,
    format: str,
) -> None:
    """Generate plots from evaluation results."""
    from aksis.eval.visualizer import Visualizer
    import json
    import os

    logger.info(f"Generating plots from: {results_file}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)

    # Create visualizer
    visualizer = Visualizer(output_dir=output_dir)

    # Generate plots
    if "perplexity" in results or "bleu" in results or "rouge" in results:
        # Evaluation metrics plot
        plot_path = visualizer.plot_evaluation_metrics(
            metrics=results,
            title="Model Evaluation Metrics",
            save_path=os.path.join(output_dir, f"evaluation_metrics.{format}"),
        )
        click.echo(f"📊 Evaluation metrics plot saved: {plot_path}")

    if "history" in results:
        # Training history plot
        plot_path = visualizer.plot_loss_perplexity(
            history=results["history"],
            title="Training History",
            save_path=os.path.join(output_dir, f"training_history.{format}"),
        )
        click.echo(f"📈 Training history plot saved: {plot_path}")

    click.echo(f"✅ Plots generated in: {output_dir}")


if __name__ == "__main__":
    main()
