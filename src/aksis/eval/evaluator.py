"""Evaluation metrics computation for Aksis AI chatbot/LLM."""

import logging
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path
import pandas as pd

from aksis.model.transformer import TransformerDecoder
from aksis.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates model performance using various metrics."""

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: Tokenizer,
        device: torch.device,
        use_mixed_precision: bool = False,
    ) -> None:
        """
        Initialize the Evaluator.

        Args:
            model: The trained PyTorch model to evaluate.
            tokenizer: The tokenizer used for encoding/decoding.
            device: The device (CPU or CUDA) to run evaluation on.
            use_mixed_precision: Whether to use mixed precision evaluation.
        """
        if not isinstance(model, TransformerDecoder):
            raise ValueError("model must be a TransformerDecoder")
        if not hasattr(tokenizer, "encode") or not hasattr(
            tokenizer, "decode"
        ):
            raise ValueError("tokenizer must have encode and decode methods")
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device object")

        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.use_mixed_precision = use_mixed_precision

        logger.info(f"Evaluator initialized on device: {self.device}")

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity for a single text.

        Args:
            text: The text to compute perplexity for.

        Returns:
            The perplexity score.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Tokenize text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if not tokens:
            raise ValueError("Text resulted in empty tokenization")

        # Convert to tensor
        input_ids = torch.tensor(
            [tokens], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_mixed_precision
            ):
                # Forward pass
                logits = self.model(input_ids)

                # Check for NaN or infinite values
                if torch.isnan(logits).any():
                    raise RuntimeError("NaN detected in model output")
                if torch.isinf(logits).any():
                    raise RuntimeError(
                        "Infinite values detected in model output"
                    )

                # Compute loss (cross-entropy)
                # Shift logits and targets for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                # Flatten for loss computation
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                )

                # Convert to perplexity
                perplexity = torch.exp(loss).item()

        return perplexity

    def compute_perplexity_batch(self, texts: List[str]) -> float:
        """
        Compute average perplexity for a batch of texts.

        Args:
            texts: List of texts to compute perplexity for.

        Returns:
            The average perplexity score.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        total_perplexity = 0.0
        valid_texts = 0

        for text in texts:
            try:
                perplexity = self.compute_perplexity(text)
                total_perplexity += perplexity
                valid_texts += 1
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Skipping invalid text: {e}")
                continue

        if valid_texts == 0:
            raise ValueError("No valid texts found")

        return total_perplexity / valid_texts

    def compute_bleu_score(
        self,
        generated: str,
        references: List[str],
        n_gram: int = 4,
    ) -> float:
        """
        Compute BLEU score for generated text against references.

        Args:
            generated: The generated text.
            references: List of reference texts.
            n_gram: The n-gram order for BLEU computation.

        Returns:
            The BLEU score.
        """
        if not generated or not generated.strip():
            return 0.0
        if not references:
            return 0.0

        try:
            from nltk.translate.bleu_score import (
                sentence_bleu,
                SmoothingFunction,
            )
        except ImportError:
            logger.warning(
                "NLTK not available, using simple BLEU approximation"
            )
            return self._simple_bleu_score(generated, references, n_gram)

        # Tokenize texts
        generated_tokens = generated.lower().split()
        reference_tokens = [ref.lower().split() for ref in references]

        # Use smoothing function to handle edge cases
        smoothing = SmoothingFunction().method1

        try:
            bleu_score = sentence_bleu(
                reference_tokens,
                generated_tokens,
                weights=tuple([1.0 / n_gram] * n_gram),
                smoothing_function=smoothing,
            )
        except ZeroDivisionError:
            bleu_score = 0.0

        return float(bleu_score)

    def _simple_bleu_score(
        self, generated: str, references: List[str], n_gram: int
    ) -> float:
        """Simple BLEU score approximation without NLTK."""
        generated_tokens = generated.lower().split()
        reference_tokens = [ref.lower().split() for ref in references]

        if not generated_tokens:
            return 0.0

        # Compute precision for each n-gram
        precisions = []
        for n in range(1, n_gram + 1):
            if len(generated_tokens) < n:
                precisions.append(0.0)
                continue

            # Get n-grams from generated text
            generated_ngrams = [
                tuple(generated_tokens[i : i + n])
                for i in range(len(generated_tokens) - n + 1)
            ]

            # Get n-grams from all references
            reference_ngrams = []
            for ref_tokens in reference_tokens:
                reference_ngrams.extend(
                    [
                        tuple(ref_tokens[i : i + n])
                        for i in range(len(ref_tokens) - n + 1)
                    ]
                )

            # Count matches
            matches = 0
            for ngram in generated_ngrams:
                if ngram in reference_ngrams:
                    matches += 1

            precision = (
                matches / len(generated_ngrams) if generated_ngrams else 0.0
            )
            precisions.append(precision)

        # Compute geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu_score = np.exp(np.mean(np.log(precisions)))
        else:
            bleu_score = 0.0

        # Apply brevity penalty
        generated_length = len(generated_tokens)
        reference_lengths = [len(ref.split()) for ref in references]
        closest_ref_length = min(
            reference_lengths, key=lambda x: abs(x - generated_length)
        )

        if generated_length < closest_ref_length:
            brevity_penalty = np.exp(1 - closest_ref_length / generated_length)
        else:
            brevity_penalty = 1.0

        return float(brevity_penalty * bleu_score)

    def compute_rouge_score(
        self, generated: str, references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores for generated text against references.

        Args:
            generated: The generated text.
            references: List of reference texts.

        Returns:
            Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
        """
        if not generated or not generated.strip():
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        if not references:
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

        try:
            from rouge_score import rouge_scorer
        except ImportError:
            logger.warning(
                "rouge-score not available, using simple ROUGE approximation"
            )
            return self._simple_rouge_score(generated, references)

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        # Compute ROUGE scores against all references
        rouge_scores = {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

        for reference in references:
            scores = scorer.score(reference, generated)
            rouge_scores["rouge-1"] += scores["rouge1"].fmeasure
            rouge_scores["rouge-2"] += scores["rouge2"].fmeasure
            rouge_scores["rouge-l"] += scores["rougeL"].fmeasure

        # Average across references
        num_refs = len(references)
        for key in rouge_scores:
            rouge_scores[key] /= num_refs

        return rouge_scores

    def _simple_rouge_score(
        self, generated: str, references: List[str]
    ) -> Dict[str, float]:
        """Simple ROUGE score approximation without rouge-score library."""
        generated_tokens = generated.lower().split()
        reference_tokens = [ref.lower().split() for ref in references]

        if not generated_tokens:
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

        # ROUGE-1 (unigram overlap)
        generated_unigrams = set(generated_tokens)
        rouge_1_scores = []
        for ref_tokens in reference_tokens:
            ref_unigrams = set(ref_tokens)
            overlap = len(generated_unigrams & ref_unigrams)
            precision = (
                overlap / len(generated_unigrams)
                if generated_unigrams
                else 0.0
            )
            recall = overlap / len(ref_unigrams) if ref_unigrams else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            rouge_1_scores.append(f1)

        # ROUGE-2 (bigram overlap)
        generated_bigrams = set(
            [
                tuple(generated_tokens[i : i + 2])
                for i in range(len(generated_tokens) - 1)
            ]
        )
        rouge_2_scores = []
        for ref_tokens in reference_tokens:
            ref_bigrams = set(
                [
                    tuple(ref_tokens[i : i + 2])
                    for i in range(len(ref_tokens) - 1)
                ]
            )
            overlap = len(generated_bigrams & ref_bigrams)
            precision = (
                overlap / len(generated_bigrams) if generated_bigrams else 0.0
            )
            recall = overlap / len(ref_bigrams) if ref_bigrams else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            rouge_2_scores.append(f1)

        # ROUGE-L (longest common subsequence)
        rouge_l_scores = []
        for ref_tokens in reference_tokens:
            lcs_length = self._lcs_length(generated_tokens, ref_tokens)
            precision = (
                lcs_length / len(generated_tokens) if generated_tokens else 0.0
            )
            recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            rouge_l_scores.append(f1)

        return {
            "rouge-1": float(np.mean(rouge_1_scores)),
            "rouge-2": float(np.mean(rouge_2_scores)),
            "rouge-l": float(np.mean(rouge_l_scores)),
        }

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def evaluate_dataset(
        self, dataset: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: List of dictionaries with 'input' and 'target' keys.

        Returns:
            Dictionary containing evaluation metrics.
        """
        if not dataset:
            raise ValueError("Dataset cannot be empty")

        # Validate dataset format
        for item in dataset:
            if (
                not isinstance(item, dict)
                or "input" not in item
                or "target" not in item
            ):
                raise ValueError(
                    "Dataset items must have 'input' and 'target' keys"
                )

        logger.info(f"Evaluating on {len(dataset)} samples")

        # Initialize metrics
        total_perplexity = 0.0
        total_bleu_1 = 0.0
        total_bleu_2 = 0.0
        total_bleu_3 = 0.0
        total_bleu_4 = 0.0
        total_rouge_1 = 0.0
        total_rouge_2 = 0.0
        total_rouge_l = 0.0
        valid_samples = 0

        for i, item in enumerate(dataset):
            try:
                target_text = item["target"]

                # Compute perplexity
                perplexity = self.compute_perplexity(target_text)
                total_perplexity += perplexity

                # Compute BLEU scores
                bleu_1 = self.compute_bleu_score(
                    target_text, [target_text], n_gram=1
                )
                bleu_2 = self.compute_bleu_score(
                    target_text, [target_text], n_gram=2
                )
                bleu_3 = self.compute_bleu_score(
                    target_text, [target_text], n_gram=3
                )
                bleu_4 = self.compute_bleu_score(
                    target_text, [target_text], n_gram=4
                )

                total_bleu_1 += bleu_1
                total_bleu_2 += bleu_2
                total_bleu_3 += bleu_3
                total_bleu_4 += bleu_4

                # Compute ROUGE scores
                rouge_scores = self.compute_rouge_score(
                    target_text, [target_text]
                )
                total_rouge_1 += rouge_scores["rouge-1"]
                total_rouge_2 += rouge_scores["rouge-2"]
                total_rouge_l += rouge_scores["rouge-l"]

                valid_samples += 1

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} samples")

            except Exception as e:
                logger.warning(f"Skipping sample {i}: {e}")
                continue

        if valid_samples == 0:
            raise ValueError("No valid samples found")

        # Compute averages
        results = {
            "perplexity": total_perplexity / valid_samples,
            "bleu-1": total_bleu_1 / valid_samples,
            "bleu-2": total_bleu_2 / valid_samples,
            "bleu-3": total_bleu_3 / valid_samples,
            "bleu-4": total_bleu_4 / valid_samples,
            "rouge-1": total_rouge_1 / valid_samples,
            "rouge-2": total_rouge_2 / valid_samples,
            "rouge-l": total_rouge_l / valid_samples,
        }

        logger.info(f"Evaluation completed on {valid_samples} samples")
        return results

    def evaluate_checkpoint(
        self, checkpoint_path: Union[str, Path], dataset: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Evaluate a model checkpoint on a dataset.

        Args:
            checkpoint_path: Path to the model checkpoint.
            dataset: List of dictionaries with 'input' and 'target' keys.

        Returns:
            Dictionary containing evaluation results and metadata.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract model state dict
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint

        # Load model weights
        self.model.load_state_dict(model_state_dict, strict=False)

        # Evaluate
        results = self.evaluate_dataset(dataset)

        # Add metadata
        results["checkpoint_path"] = str(checkpoint_path)  # type: ignore
        if "epoch" in checkpoint:
            results["epoch"] = checkpoint["epoch"]
        if "loss" in checkpoint:
            results["training_loss"] = checkpoint["loss"]

        return results

    def save_results(
        self, results: Dict[str, Any], output_path: Union[str, Path]
    ) -> None:
        """
        Save evaluation results to file.

        Args:
            results: Dictionary containing evaluation results.
            output_path: Path to save the results.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".csv":
            # Save as CSV
            df = pd.DataFrame([results])
            df.to_csv(output_path, index=False)
        else:
            # Save as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")
