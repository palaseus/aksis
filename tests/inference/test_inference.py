"""Tests for inference generation logic."""

import pytest
import torch
from unittest.mock import Mock, patch

from aksis.inference.inference import Generator
from aksis.model.transformer import TransformerDecoder
from aksis.data.tokenizer import Tokenizer


class TestGenerator:
    """Test inference generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.vocab_size = 1000
        self.max_length = 512

        # Mock model
        self.model = Mock(spec=TransformerDecoder)
        self.model.eval = Mock()
        self.model.to = Mock(return_value=self.model)
        self.model.parameters = Mock(return_value=[])

        # Mock tokenizer
        self.tokenizer = Mock(spec=Tokenizer)
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Hello world")
        self.tokenizer.vocab_size_with_special = self.vocab_size
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 2

        # Mock checkpoint
        self.checkpoint = {
            "model_state_dict": {
                "embedding.weight": torch.randn(self.vocab_size, 512)
            },
            "epoch": 1,
            "loss": 0.5,
        }

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        assert generator.model == self.model
        assert generator.tokenizer == self.tokenizer
        assert generator.device == self.device
        assert generator.max_length == 512
        assert generator.pad_token_id == 0
        assert generator.eos_token_id == 2

    def test_generator_initialization_with_params(self):
        """Test generator initialization with custom parameters."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=256,
        )

        assert generator.max_length == 256

    def test_generator_initialization_invalid_max_length(self):
        """Test generator initialization with invalid max_length."""
        with pytest.raises(ValueError, match="max_length must be positive"):
            Generator(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_length=0,
            )

    def test_generator_initialization_invalid_model(self):
        """Test generator initialization with invalid model."""
        with pytest.raises(
            ValueError, match="model must be a TransformerDecoder"
        ):
            Generator(
                model="invalid_model",
                tokenizer=self.tokenizer,
                device=self.device,
            )

    def test_generator_initialization_invalid_tokenizer(self):
        """Test generator initialization with invalid tokenizer."""
        with pytest.raises(
            ValueError, match="tokenizer must have encode and decode methods"
        ):
            Generator(
                model=self.model,
                tokenizer="invalid_tokenizer",
                device=self.device,
            )

    @patch("aksis.inference.inference.torch.load")
    @patch("aksis.inference.inference.Path.exists")
    def test_load_from_checkpoint(self, mock_exists, mock_torch_load):
        """Test loading generator from checkpoint."""
        mock_exists.return_value = True
        mock_torch_load.return_value = self.checkpoint

        generator = Generator.load_from_checkpoint(
            checkpoint_path="test.pt",
            tokenizer=self.tokenizer,
            device=self.device,
        )

        assert isinstance(generator, Generator)
        # Check that torch.load was called with the correct arguments
        mock_torch_load.assert_called_once()
        call_args = mock_torch_load.call_args
        assert str(call_args[0][0]) == "test.pt"
        assert call_args[1]["map_location"] == "cpu"

    @patch("aksis.inference.inference.torch.load")
    @patch("aksis.inference.inference.Path.exists")
    def test_load_from_checkpoint_with_device(
        self, mock_exists, mock_torch_load
    ):
        """Test loading generator from checkpoint with specific device."""
        mock_exists.return_value = True
        mock_torch_load.return_value = self.checkpoint

        generator = Generator.load_from_checkpoint(
            checkpoint_path="test.pt",
            tokenizer=self.tokenizer,
            device=torch.device("cuda"),
        )

        assert generator.device == torch.device("cuda")

    @patch("aksis.inference.inference.torch.load")
    def test_load_from_checkpoint_file_not_found(self, mock_torch_load):
        """Test loading generator from non-existent checkpoint."""
        mock_torch_load.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            Generator.load_from_checkpoint(
                checkpoint_path="nonexistent.pt",
                tokenizer=self.tokenizer,
                device=self.device,
            )

    def test_prepare_inputs(self):
        """Test input preparation."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        prompt = "Hello world"
        input_ids = generator._prepare_inputs(prompt)

        self.tokenizer.encode.assert_called_once_with(
            prompt, add_special_tokens=True
        )
        assert isinstance(input_ids, torch.Tensor)

    def test_prepare_inputs_empty_prompt(self):
        """Test input preparation with empty prompt."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        self.tokenizer.encode.return_value = []

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            generator._prepare_inputs("")

    def test_prepare_inputs_too_long(self):
        """Test input preparation with prompt too long."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=10,
        )

        # Mock a long prompt
        self.tokenizer.encode.return_value = list(range(20))

        with pytest.raises(ValueError, match="Prompt too long"):
            generator._prepare_inputs("Very long prompt")

    def test_generate_basic(self):
        """Test basic text generation."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.vocab_size)
        self.model.return_value = mock_logits

        # Mock sampler
        mock_sampler = Mock()
        mock_sampler.sample.return_value = torch.tensor([[6], [7], [8]])

        result = generator.generate(
            prompt="Hello",
            sampler=mock_sampler,
            max_new_tokens=3,
        )

        assert isinstance(result, str)
        self.tokenizer.decode.assert_called()

    def test_generate_with_stop_tokens(self):
        """Test generation with stop tokens."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.vocab_size)
        self.model.return_value = mock_logits

        # Mock sampler that returns stop token
        mock_sampler = Mock()
        mock_sampler.sample.return_value = torch.tensor([[6], [2]])  # 2 is EOS

        result = generator.generate(
            prompt="Hello",
            sampler=mock_sampler,
            max_new_tokens=10,
            stop_tokens=[2],
        )

        assert isinstance(result, str)

    def test_generate_max_length_exceeded(self):
        """Test generation when max length is exceeded."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=10,
        )

        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.vocab_size)
        self.model.return_value = mock_logits

        # Mock sampler
        mock_sampler = Mock()
        mock_sampler.sample.return_value = torch.tensor(
            [[6], [7], [8], [9], [10], [11]]
        )

        result = generator.generate(
            prompt="Hello",
            sampler=mock_sampler,
            max_new_tokens=20,
        )

        assert isinstance(result, str)

    def test_generate_invalid_sampler(self):
        """Test generation with invalid sampler."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        with pytest.raises(
            ValueError, match="sampler must have a sample method"
        ):
            generator.generate(
                prompt="Hello",
                sampler="invalid_sampler",
                max_new_tokens=5,
            )

    def test_generate_invalid_max_new_tokens(self):
        """Test generation with invalid max_new_tokens."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        mock_sampler = Mock()

        with pytest.raises(
            ValueError, match="max_new_tokens must be positive"
        ):
            generator.generate(
                prompt="Hello",
                sampler=mock_sampler,
                max_new_tokens=0,
            )

    def test_generate_model_error(self):
        """Test generation when model raises an error."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # Mock model to raise an error
        self.model.side_effect = RuntimeError("Model error")

        mock_sampler = Mock()

        with pytest.raises(RuntimeError, match="Model error"):
            generator.generate(
                prompt="Hello",
                sampler=mock_sampler,
                max_new_tokens=5,
            )

    def test_generate_nan_logits(self):
        """Test generation with NaN logits."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # Mock model to return NaN logits
        mock_logits = torch.full((1, 5, self.vocab_size), float("nan"))
        self.model.return_value = mock_logits

        mock_sampler = Mock()

        with pytest.raises(RuntimeError, match="NaN detected in model output"):
            generator.generate(
                prompt="Hello",
                sampler=mock_sampler,
                max_new_tokens=5,
            )

    def test_generate_inf_logits(self):
        """Test generation with infinite logits."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # Mock model to return infinite logits
        mock_logits = torch.full((1, 5, self.vocab_size), float("inf"))
        self.model.return_value = mock_logits

        mock_sampler = Mock()

        with pytest.raises(
            RuntimeError, match="Infinite values detected in model output"
        ):
            generator.generate(
                prompt="Hello",
                sampler=mock_sampler,
                max_new_tokens=5,
            )

    def test_generate_cuda_device(self):
        """Test generation on CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cuda_device = torch.device("cuda")
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=cuda_device,
        )

        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.vocab_size).to(cuda_device)
        self.model.return_value = mock_logits

        # Mock sampler
        mock_sampler = Mock()
        mock_sampler.sample.return_value = torch.tensor([[6], [7], [8]]).to(
            cuda_device
        )

        result = generator.generate(
            prompt="Hello",
            sampler=mock_sampler,
            max_new_tokens=3,
        )

        assert isinstance(result, str)

    def test_generate_mixed_precision(self):
        """Test generation with mixed precision."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_mixed_precision=True,
        )

        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.vocab_size)
        self.model.return_value = mock_logits

        # Mock sampler
        mock_sampler = Mock()
        mock_sampler.sample.return_value = torch.tensor([[6], [7], [8]])

        result = generator.generate(
            prompt="Hello",
            sampler=mock_sampler,
            max_new_tokens=3,
        )

        assert isinstance(result, str)

    def test_generate_batch(self):
        """Test batch generation."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # Mock model forward pass
        mock_logits = torch.randn(2, 5, self.vocab_size)
        self.model.return_value = mock_logits

        # Mock sampler
        mock_sampler = Mock()
        mock_sampler.sample.return_value = torch.tensor(
            [[6], [7], [8], [9], [10], [11]]
        )

        prompts = ["Hello", "Hi there"]
        results = generator.generate_batch(
            prompts=prompts,
            sampler=mock_sampler,
            max_new_tokens=3,
        )

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(result, str) for result in results)

    def test_generate_batch_empty_prompts(self):
        """Test batch generation with empty prompts list."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        mock_sampler = Mock()

        with pytest.raises(ValueError, match="prompts cannot be empty"):
            generator.generate_batch(
                prompts=[],
                sampler=mock_sampler,
                max_new_tokens=5,
            )

    def test_get_model_info(self):
        """Test getting model information."""
        generator = Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # Mock model parameters
        mock_params = [torch.randn(100, 100), torch.randn(50, 50)]
        self.model.parameters.return_value = mock_params

        info = generator.get_model_info()

        assert "total_parameters" in info
        assert "device" in info
        assert "max_length" in info
        assert "use_mixed_precision" in info
        assert info["device"] == str(self.device)
        assert info["max_length"] == 512
