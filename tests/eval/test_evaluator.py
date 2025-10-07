"""Tests for evaluation metrics computation."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from aksis.eval.evaluator import Evaluator
from aksis.model.transformer import TransformerDecoder
from aksis.data.tokenizer import Tokenizer


class TestEvaluator:
    """Test evaluation metrics computation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
        # Mock tokenizer
        self.tokenizer = Mock(spec=Tokenizer)
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Hello world")
        self.tokenizer.vocab_size_with_special = 1000
        self.tokenizer.pad_token_id = 0
        
        # Mock model
        self.model = Mock(spec=TransformerDecoder)
        self.model.eval = Mock()
        self.model.to = Mock(return_value=self.model)
        self.model.parameters = Mock(return_value=[])
        
        # Sample test data
        self.test_data = [
            {"input": "Hello", "target": "Hello world"},
            {"input": "How are", "target": "How are you?"},
            {"input": "The weather", "target": "The weather is nice"},
        ]
        
        self.evaluator = Evaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def test_evaluator_initialization(self) -> None:
        """Test evaluator initialization."""
        assert self.evaluator.model == self.model
        assert self.evaluator.tokenizer == self.tokenizer
        assert self.evaluator.device == self.device

    def test_evaluator_initialization_invalid_model(self) -> None:
        """Test evaluator initialization with invalid model."""
        with pytest.raises(ValueError, match="model must be a TransformerDecoder"):
            Evaluator(
                model="invalid_model",
                tokenizer=self.tokenizer,
                device=self.device,
            )

    def test_evaluator_initialization_invalid_tokenizer(self) -> None:
        """Test evaluator initialization with invalid tokenizer."""
        with pytest.raises(
            ValueError, match="tokenizer must have encode and decode methods"
        ):
            Evaluator(
                model=self.model,
                tokenizer="invalid_tokenizer",
                device=self.device,
            )

    def test_evaluator_initialization_invalid_device(self) -> None:
        """Test evaluator initialization with invalid device."""
        with pytest.raises(ValueError, match="device must be a torch.device"):
            Evaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                device="invalid_device",
            )

    def test_compute_perplexity(self) -> None:
        """Test perplexity computation."""
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock target tokens
        target_tokens = [1, 2, 3, 4, 5]
        self.tokenizer.encode.return_value = target_tokens
        
        perplexity = self.evaluator.compute_perplexity("Hello world")
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
        self.model.assert_called_once()

    def test_compute_perplexity_batch(self) -> None:
        """Test batch perplexity computation."""
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock target tokens
        target_tokens = [1, 2, 3, 4, 5]
        self.tokenizer.encode.return_value = target_tokens
        
        texts = ["Hello world", "How are you?"]
        perplexity = self.evaluator.compute_perplexity_batch(texts)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_compute_bleu_score(self) -> None:
        """Test BLEU score computation."""
        # Mock generated and reference texts
        generated = "Hello world"
        reference = "Hello world"
        
        # Mock tokenizer
        self.tokenizer.decode.return_value = generated
        
        bleu_score = self.evaluator.compute_bleu_score(generated, [reference])
        
        assert isinstance(bleu_score, float)
        assert 0 <= bleu_score <= 1

    def test_compute_bleu_score_multiple_references(self) -> None:
        """Test BLEU score with multiple references."""
        generated = "Hello world"
        references = ["Hello world", "Hi world", "Hello there"]
        
        self.tokenizer.decode.return_value = generated
        
        bleu_score = self.evaluator.compute_bleu_score(generated, references)
        
        assert isinstance(bleu_score, float)
        assert 0 <= bleu_score <= 1

    def test_compute_bleu_score_different_n_grams(self) -> None:
        """Test BLEU score with different n-gram orders."""
        generated = "Hello world"
        reference = "Hello world"
        
        self.tokenizer.decode.return_value = generated
        
        for n in [1, 2, 3, 4]:
            bleu_score = self.evaluator.compute_bleu_score(
                generated, [reference], n_gram=n
            )
            assert isinstance(bleu_score, float)
            assert 0 <= bleu_score <= 1

    def test_compute_rouge_score(self) -> None:
        """Test ROUGE score computation."""
        generated = "Hello world"
        reference = "Hello world"
        
        self.tokenizer.decode.return_value = generated
        
        rouge_scores = self.evaluator.compute_rouge_score(generated, [reference])
        
        assert isinstance(rouge_scores, dict)
        assert "rouge-1" in rouge_scores
        assert "rouge-2" in rouge_scores
        assert "rouge-l" in rouge_scores
        
        for score in rouge_scores.values():
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_compute_rouge_score_multiple_references(self) -> None:
        """Test ROUGE score with multiple references."""
        generated = "Hello world"
        references = ["Hello world", "Hi world", "Hello there"]
        
        self.tokenizer.decode.return_value = generated
        
        rouge_scores = self.evaluator.compute_rouge_score(generated, references)
        
        assert isinstance(rouge_scores, dict)
        for score in rouge_scores.values():
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_evaluate_dataset(self) -> None:
        """Test evaluation on a dataset."""
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock tokenizer
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Hello world"
        
        results = self.evaluator.evaluate_dataset(self.test_data)
        
        assert isinstance(results, dict)
        assert "perplexity" in results
        assert "bleu-1" in results
        assert "bleu-2" in results
        assert "bleu-3" in results
        assert "bleu-4" in results
        assert "rouge-1" in results
        assert "rouge-2" in results
        assert "rouge-l" in results
        
        for metric, value in results.items():
            assert isinstance(value, float)
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_evaluate_dataset_empty(self) -> None:
        """Test evaluation on empty dataset."""
        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            self.evaluator.evaluate_dataset([])

    def test_evaluate_dataset_invalid_format(self) -> None:
        """Test evaluation on dataset with invalid format."""
        invalid_data = [{"invalid": "data"}]
        
        with pytest.raises(ValueError, match="Dataset items must have 'input' and 'target' keys"):
            self.evaluator.evaluate_dataset(invalid_data)

    def test_evaluate_checkpoint(self) -> None:
        """Test evaluation on a checkpoint."""
        # Mock checkpoint loading
        with patch("aksis.eval.evaluator.torch.load") as mock_load:
            with patch("pathlib.Path.exists", return_value=True):
                mock_checkpoint = {
                    "model_state_dict": {"embedding.weight": torch.randn(1000, 512)},
                    "epoch": 1,
                    "loss": 0.5,
                }
                mock_load.return_value = mock_checkpoint
                
                # Mock model forward pass
                mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
                self.model.return_value = mock_logits
                
                # Mock tokenizer
                self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
                self.tokenizer.decode.return_value = "Hello world"
                
                results = self.evaluator.evaluate_checkpoint(
                    "test_checkpoint.pt", self.test_data
                )
                
                assert isinstance(results, dict)
                assert "checkpoint_path" in results
                assert "perplexity" in results
                assert "bleu-1" in results

    def test_evaluate_checkpoint_file_not_found(self) -> None:
        """Test evaluation on non-existent checkpoint."""
        with pytest.raises(FileNotFoundError):
            self.evaluator.evaluate_checkpoint("nonexistent.pt", self.test_data)

    def test_save_results(self) -> None:
        """Test saving evaluation results."""
        results = {
            "perplexity": 2.5,
            "bleu-1": 0.8,
            "bleu-2": 0.7,
            "rouge-1": 0.85,
        }
        
        with patch("builtins.open", create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            with patch("json.dump") as mock_json_dump:
                self.evaluator.save_results(results, "test_results.json")
                
                # Check that open was called with the correct arguments (Path object)
                call_args = mock_open.call_args
                assert str(call_args[0][0]) == "test_results.json"
                assert call_args[0][1] == "w"
                assert call_args[1]["encoding"] == "utf-8"
                mock_json_dump.assert_called_once()

    def test_save_results_csv(self) -> None:
        """Test saving evaluation results as CSV."""
        results = {
            "perplexity": 2.5,
            "bleu-1": 0.8,
            "bleu-2": 0.7,
            "rouge-1": 0.85,
        }
        
        with patch("builtins.open", Mock()) as mock_open:
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                self.evaluator.save_results(results, "test_results.csv")
                
                mock_to_csv.assert_called_once()

    def test_compute_perplexity_nan_handling(self) -> None:
        """Test perplexity computation with NaN handling."""
        # Mock model to return NaN logits
        mock_logits = torch.full(
            (1, 5, self.tokenizer.vocab_size_with_special), float("nan")
        )
        self.model.return_value = mock_logits
        
        with pytest.raises(RuntimeError, match="NaN detected in model output"):
            self.evaluator.compute_perplexity("Hello world")

    def test_compute_perplexity_inf_handling(self) -> None:
        """Test perplexity computation with Inf handling."""
        # Mock model to return Inf logits
        mock_logits = torch.full(
            (1, 5, self.tokenizer.vocab_size_with_special), float("inf")
        )
        self.model.return_value = mock_logits
        
        with pytest.raises(RuntimeError, match="Infinite values detected in model output"):
            self.evaluator.compute_perplexity("Hello world")

    def test_cuda_compatibility(self) -> None:
        """Test CUDA compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        cuda_device = torch.device("cuda")
        evaluator = Evaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=cuda_device,
        )
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special).to(
            cuda_device
        )
        self.model.return_value = mock_logits
        
        # Mock tokenizer
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Hello world"
        
        perplexity = evaluator.compute_perplexity("Hello world")
        
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_evaluation_performance(self) -> None:
        """Test evaluation performance."""
        import time
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock tokenizer
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Hello world"
        
        start_time = time.time()
        results = self.evaluator.evaluate_dataset(self.test_data)
        end_time = time.time()
        
        # Should complete in reasonable time (< 5 seconds for small dataset)
        assert end_time - start_time < 5.0
        assert isinstance(results, dict)

    def test_edge_cases(self) -> None:
        """Test edge cases in evaluation."""
        # Test with very short text
        self.tokenizer.encode.return_value = [1, 2]
        self.tokenizer.decode.return_value = "Hi"
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 2, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        perplexity = self.evaluator.compute_perplexity("Hi")
        assert isinstance(perplexity, float)
        assert perplexity > 0
        
        # Test with empty reference
        bleu_score = self.evaluator.compute_bleu_score("Hello", [])
        assert bleu_score == 0.0
        
        # Test with identical generated and reference
        bleu_score = self.evaluator.compute_bleu_score("Hello", ["Hello"])
        assert bleu_score > 0.0  # Should be positive for identical strings

    def test_metric_consistency(self) -> None:
        """Test metric consistency across multiple runs."""
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock tokenizer
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Hello world"
        
        # Run evaluation multiple times
        results1 = self.evaluator.evaluate_dataset(self.test_data)
        results2 = self.evaluator.evaluate_dataset(self.test_data)
        
        # Results should be consistent (same model, same data)
        for metric in results1:
            assert abs(results1[metric] - results2[metric]) < 1e-6

    def test_mixed_precision_evaluation(self) -> None:
        """Test evaluation with mixed precision."""
        evaluator = Evaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            use_mixed_precision=True,
        )
        
        # Mock model forward pass
        mock_logits = torch.randn(1, 5, self.tokenizer.vocab_size_with_special)
        self.model.return_value = mock_logits
        
        # Mock tokenizer
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Hello world"
        
        perplexity = evaluator.compute_perplexity("Hello world")
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
