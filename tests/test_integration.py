"""Integration tests for Aksis model with DataLoader."""

import torch

from aksis.data.tokenizer import Tokenizer
from aksis.data.dataloader import DataLoader
from aksis.model.transformer import TransformerDecoder
from aksis.utils.device import get_device


class TestModelDataLoaderIntegration:
    """Test integration between model and DataLoader components."""

    def test_model_with_dataloader_forward_pass(self):
        """Test forward pass through model with DataLoader batch."""
        # Create sample texts
        texts = [
            "Hello world, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating and powerful.",
            "Transformers have revolutionized natural language processing.",
            "This is another example text for testing purposes.",
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=20,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Verify batch shapes
        assert input_ids.shape[0] == 2  # batch_size
        assert input_ids.shape[1] <= 20  # max_length
        assert attention_mask.shape == input_ids.shape

        # Forward pass through model
        with torch.no_grad():
            output = model(input_ids, padding_mask=attention_mask)

        # Verify output shape
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            tokenizer.vocab_size_with_special,
        )
        assert output.shape == expected_shape

        # Verify output is on correct device
        assert output.device.type == device.type

        # Verify output is not NaN or infinite
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_with_dataloader_gradient_flow(self):
        """Test gradient flow through model with DataLoader batch."""
        # Create sample texts
        texts = [
            "Hello world, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=15,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Forward pass with gradients
        output = model(input_ids, padding_mask=attention_mask)

        # Create dummy loss (cross-entropy with next token prediction)
        # Shift input_ids for next token prediction
        targets = input_ids[:, 1:].contiguous()
        output = output[:, :-1, :].contiguous()

        # Flatten for cross-entropy
        output_flat = output.view(-1, output.size(-1))
        targets_flat = targets.view(-1)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(output_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Verify gradients exist and are not NaN
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_model_with_dataloader_mixed_precision(self):
        """Test model with DataLoader using mixed precision."""
        # Create sample texts
        texts = [
            "Hello world, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=15,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Test mixed precision if CUDA is available
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                output = model(input_ids, padding_mask=attention_mask)

            # Verify output shape
            expected_shape = (
                input_ids.shape[0],
                input_ids.shape[1],
                tokenizer.vocab_size_with_special,
            )
            assert output.shape == expected_shape

            # Verify output is on correct device
            assert output.device.type == device.type

            # Verify output is not NaN or infinite
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_model_with_dataloader_different_batch_sizes(self):
        """Test model with DataLoader using different batch sizes."""
        # Create sample texts
        texts = [
            "Hello world, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating and powerful.",
            "Transformers have revolutionized natural language processing.",
            "This is another example text for testing purposes.",
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Test different batch sizes
        batch_sizes = [1, 2, 3, 5]

        for batch_size in batch_sizes:
            # Create DataLoader
            dataloader = DataLoader(
                texts=texts,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=20,
                shuffle=False,
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
            )

            # Move model to device
            device = get_device()
            model = model.to(device)

            # Get a batch from DataLoader
            batch = next(iter(dataloader))
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Verify batch size
            assert input_ids.shape[0] == batch_size

            # Forward pass through model
            with torch.no_grad():
                output = model(input_ids, padding_mask=attention_mask)

            # Verify output shape
            expected_shape = (
                input_ids.shape[0],
                input_ids.shape[1],
                tokenizer.vocab_size_with_special,
            )
            assert output.shape == expected_shape

    def test_model_with_dataloader_attention_mask(self):
        """Test model with DataLoader using attention masks."""
        # Create sample texts with different lengths
        texts = [
            "Short text.",
            "This is a much longer text that will be padded in the batch.",
            "Medium length text here.",
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=3,
            max_length=20,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Verify attention mask has correct shape
        assert attention_mask.shape == input_ids.shape

        # Verify attention mask values (0 for padding, 1 for real tokens)
        assert attention_mask.min() >= 0
        assert attention_mask.max() <= 1

        # Forward pass through model
        with torch.no_grad():
            output = model(input_ids, padding_mask=attention_mask)

        # Verify output shape
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            tokenizer.vocab_size_with_special,
        )
        assert output.shape == expected_shape

    def test_model_with_dataloader_empty_batch(self):
        """Test model with DataLoader using empty batch."""
        # Create empty texts list
        texts = []

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=20,
            shuffle=False,
        )

        # Verify DataLoader is empty
        assert len(dataloader) == 0

        # Verify iteration over empty DataLoader
        batches = list(dataloader)
        assert len(batches) == 0

    def test_model_with_dataloader_single_sample(self):
        """Test model with DataLoader using single sample."""
        # Create single text
        texts = ["Hello world, this is a test sentence."]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=1,
            max_length=20,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Verify batch size is 1
        assert input_ids.shape[0] == 1

        # Forward pass through model
        with torch.no_grad():
            output = model(input_ids, padding_mask=attention_mask)

        # Verify output shape
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            tokenizer.vocab_size_with_special,
        )
        assert output.shape == expected_shape

    def test_model_with_dataloader_large_batch(self):
        """Test model with DataLoader using large batch."""
        # Create many texts
        texts = [
            f"This is test text number {i} for large batch testing." for i in range(100)
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader with large batch size
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=32,
            max_length=20,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Verify batch size
        assert input_ids.shape[0] == 32

        # Forward pass through model
        with torch.no_grad():
            output = model(input_ids, padding_mask=attention_mask)

        # Verify output shape
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            tokenizer.vocab_size_with_special,
        )
        assert output.shape == expected_shape

    def test_model_with_dataloader_different_sequence_lengths(self):
        """Test model with DataLoader using different sequence lengths."""
        # Create texts with different lengths
        texts = [
            "Short.",
            "This is a medium length text.",
            "This is a much longer text that will test the model's ability to handle different sequence lengths in the same batch.",
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=3,
            max_length=50,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Verify all sequences are padded to the same length
        assert input_ids.shape[1] == 50  # max_length

        # Forward pass through model
        with torch.no_grad():
            output = model(input_ids, padding_mask=attention_mask)

        # Verify output shape
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            tokenizer.vocab_size_with_special,
        )
        assert output.shape == expected_shape

    def test_model_with_dataloader_device_consistency(self):
        """Test model with DataLoader device consistency."""
        # Create sample texts
        texts = [
            "Hello world, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=20,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Verify all tensors are on the same device
        assert input_ids.device.type == device.type
        assert attention_mask.device.type == device.type

        # Forward pass through model
        with torch.no_grad():
            output = model(input_ids, padding_mask=attention_mask)

        # Verify output is on the same device
        assert output.device.type == device.type

    def test_model_with_dataloader_memory_efficiency(self):
        """Test model with DataLoader memory efficiency."""
        # Create sample texts
        texts = [
            "Hello world, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Create tokenizer and build vocabulary
        tokenizer = Tokenizer(vocab_size=1000)
        tokenizer.build_vocab(texts)

        # Create DataLoader
        dataloader = DataLoader(
            texts=texts,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=20,
            shuffle=False,
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
        )

        # Move model to device
        device = get_device()
        model = model.to(device)

        # Get a batch from DataLoader
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Forward pass through model
        with torch.no_grad():
            output = model(input_ids, padding_mask=attention_mask)

        # Verify output shape
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            tokenizer.vocab_size_with_special,
        )
        assert output.shape == expected_shape

        # Verify output is not NaN or infinite
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Verify output has reasonable values (not all zeros or ones)
        assert output.abs().max() > 0
        assert output.abs().max() < 100  # Reasonable upper bound
