# Aksis - AI Chatbot/LLM from Scratch

A complete transformer-based language model built from scratch using Python and PyTorch, designed to function as a chatbot with CUDA acceleration support.

## 🚀 Features

- **From Scratch Implementation**: Core transformer architecture built manually for educational purposes
- **CUDA Acceleration**: Automatic GPU detection and utilization with graceful CPU fallback
- **Test-Driven Development**: Comprehensive test suite with pytest
- **Code Quality**: Strict linting with Black, Flake8, and MyPy
- **Modular Architecture**: Clean, scalable codebase structure
- **Chatbot Interface**: Interactive CLI for real-time conversations
- **Evaluation & Fine-Tuning**: Comprehensive evaluation metrics (BLEU, ROUGE, Perplexity) and fine-tuning on chatbot datasets
- **Visualization Tools**: Plot training curves, evaluation metrics, and hyperparameter search results

## 📁 Project Structure

```
aksis/
├── src/                    # Source code
│   ├── aksis/             # Main package
│   │   ├── data/          # Data handling modules
│   │   ├── model/         # Transformer architecture
│   │   ├── train/         # Training loops and utilities
│   │   ├── inference/     # Inference and chat interface
│   │   ├── eval/          # Evaluation and fine-tuning
│   │   └── utils/         # Utility functions
│   └── cli.py             # Command-line interface
├── tests/                 # Test suite
├── data/                  # Datasets and data files
├── checkpoints/           # Model checkpoints
├── results/               # Evaluation results and plots
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
└── README.md             # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (optional, but recommended)
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd aksis
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # For CPU-only installation
   pip install -r requirements.txt

   # For CUDA support (recommended)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

5. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## 🧪 Testing

Run the complete test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=src --cov-report=html
```

## 🔧 Development

### Code Quality

The project enforces strict code quality standards:

- **Formatting**: Black (line length: 79)
- **Linting**: Flake8 (79 character line limit)
- **Type Checking**: MyPy (strict mode)
- **Testing**: Pytest (>80% coverage target for new modules)

Run all quality checks:

```bash
black src tests
flake8 src tests
mypy src --ignore-missing-imports
pytest --cov=src --cov-report=html
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit:

```bash
pre-commit run --all-files
```

## 🚀 Usage

### Training a Model

```bash
# Train on default dataset (WikiText2)
python -m aksis.cli train-model --dataset wikitext2 --epochs 10

# Train with custom parameters
python -m aksis.cli train-model --dataset shakespeare --batch-size 32 --learning-rate 5e-5 --mixed-precision
```

### Interactive Chat

```bash
# Start interactive chat session
python -m aksis.cli chat-with-model --checkpoint-path ./checkpoints/epoch_10.pt

# Chat with specific sampler and parameters
python -m aksis.cli chat-with-model \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --sampler temperature \
    --temperature 0.8 \
    --max-new-tokens 150 \
    --system-prompt "You are a helpful coding assistant."

# Chat with context saving
python -m aksis.cli chat-with-model \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --context-file ./chat_history.json
```

### Text Generation

```bash
# Generate text from a prompt
python -m aksis.cli generate-text \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --prompt "Once upon a time" \
    --max-new-tokens 100 \
    --sampler top-p \
    --top-p 0.9

# Generate with different sampling strategies
python -m aksis.cli generate-text \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --prompt "The meaning of life is" \
    --sampler greedy  # or top-k, top-p, temperature
```

### Benchmark Inference

```bash
# Benchmark inference performance
python -m aksis.cli benchmark-inference \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --prompt "The quick brown fox" \
    --max-new-tokens 50 \
    --num-runs 10 \
    --mixed-precision

# Benchmark with CUDA
python -m aksis.cli benchmark-inference \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --device cuda \
    --mixed-precision
```

### Data Processing

```bash
# Process and tokenize dataset
python -m aksis.cli load-data --dataset wikitext2 --batch-size 4 --max-length 256

# Tokenize custom text
python -m aksis.cli tokenize --text "Hello, world! This is Aksis."
```

### Model Evaluation

```bash
# Evaluate model on test dataset
python -m aksis.cli eval-model \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --dataset wikitext2 \
    --metrics bleu rouge perplexity \
    --output results/evaluation.json

# Evaluate with custom test data
python -m aksis.cli eval-model \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --test-file ./data/custom_test.json \
    --batch-size 16
```

### Fine-Tuning

```bash
# Fine-tune model on chatbot dataset
python -m aksis.cli fine-tune-model \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --dataset daily_dialog \
    --epochs 5 \
    --learning-rate 1e-4 \
    --batch-size 16 \
    --early-stopping

# Fine-tune with hyperparameter search
python -m aksis.cli fine-tune-model \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --dataset persona_chat \
    --hyperparameter-search \
    --learning-rates 1e-5 5e-5 1e-4 \
    --batch-sizes 8 16 32 \
    --max-trials 9

# Fine-tune on custom chatbot dataset
python -m aksis.cli fine-tune-model \
    --checkpoint-path ./checkpoints/epoch_10.pt \
    --train-file ./data/train_conversations.json \
    --val-file ./data/val_conversations.json \
    --epochs 10 \
    --mixed-precision
```

### Visualization

```bash
# Plot training metrics
python -m aksis.cli plot-metrics \
    --history-file results/training_history.json \
    --output plots/training_curves.png

# Plot evaluation metrics
python -m aksis.cli plot-metrics \
    --results-file results/evaluation.json \
    --output plots/evaluation_metrics.png

# Create summary report
python -m aksis.cli plot-metrics \
    --history-file results/training_history.json \
    --results-file results/evaluation.json \
    --output plots/summary_report.png
```

## 🤖 Inference & Sampling

Aksis supports multiple sampling strategies for text generation:

### Sampling Methods

- **Greedy Sampling**: Always selects the most probable token
- **Beam Search**: Explores multiple hypotheses (width=4 by default)
- **Top-K Sampling**: Samples from the top-k most probable tokens (k=50 by default)
- **Top-P (Nucleus) Sampling**: Samples from the smallest set with cumulative probability > p (p=0.95 by default)
- **Temperature Sampling**: Controls randomness via temperature scaling (temperature=0.7 by default)

### Context Management

The chatbot maintains conversation history with:

- **Maximum Context Length**: 512 tokens by default
- **FIFO Truncation**: Automatically removes oldest messages when context exceeds limit
- **Context Persistence**: Save and load conversation history to JSON
- **Role-Based Formatting**: Supports user, assistant, and system messages

### Performance Optimization

- **Mixed Precision Inference**: Automatic mixed precision (AMP) with CUDA
- **Device Detection**: Automatic GPU detection with CPU fallback
- **Efficient Generation**: Token-by-token generation with early stopping
- **KV Caching**: (planned) Cache key-value states for faster sequential decoding

## 📈 Evaluation & Fine-Tuning

### Evaluation Metrics

Aksis provides comprehensive model evaluation with:

- **BLEU Scores**: N-gram overlap (BLEU-1, BLEU-2, BLEU-3, BLEU-4) for text quality
- **ROUGE Scores**: Recall-oriented metrics (ROUGE-1, ROUGE-2, ROUGE-L) for summarization
- **Perplexity**: Model confidence and language modeling capability
- **Reference-Based**: Compare generated text against reference texts
- **Batch Processing**: Efficient evaluation on large test sets

### Fine-Tuning Features

- **Domain Adaptation**: Fine-tune on specific tasks or domains
- **Chatbot Datasets**: Support for DailyDialog, PersonaChat, and custom datasets
- **Hyperparameter Search**: Automated search for optimal hyperparameters
- **Early Stopping**: Prevent overfitting with validation-based stopping
- **Mixed Precision**: Fast training with automatic mixed precision
- **Checkpointing**: Save best models based on validation metrics

### Visualization Tools

- **Training Curves**: Plot loss and perplexity over epochs
- **Evaluation Metrics**: Visualize BLEU, ROUGE scores across checkpoints
- **Hyperparameter Search**: Compare results across different configurations
- **Summary Reports**: Comprehensive visualizations combining multiple metrics

## 📊 Model Architecture

The model follows the decoder-only transformer architecture:

- **Decoder-Only**: GPT-style causal language modeling
- **Multi-Head Attention**: 8 attention heads by default
- **Positional Encoding**: Learned positional embeddings
- **Layer Normalization**: Pre-norm architecture
- **Feed-Forward**: 4x hidden dimension expansion (2048 by default)

### Default Configuration

- **Layers**: 6 decoder layers
- **Hidden Dimension (d_model)**: 512
- **Attention Heads**: 8
- **Feed-Forward Dimension (d_ff)**: 2048
- **Vocabulary Size**: Dynamic (based on tokenizer)
- **Maximum Sequence Length**: 512 tokens (configurable)

## 📈 Training

### Supported Datasets

- **Shakespeare**: Complete works of Shakespeare
- **WikiText**: Wikipedia text corpus
- **Custom**: User-provided text files

### Training Features

- **Mixed Precision**: Automatic mixed precision (AMP) for CUDA
- **Gradient Accumulation**: Support for large effective batch sizes
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Checkpointing**: Automatic model saving and resuming
- **Logging**: TensorBoard and Weights & Biases integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Write tests for your changes
4. Implement your feature
5. Ensure all tests pass: `pytest`
6. Run quality checks: `black`, `flake8`, `mypy`
7. Commit your changes: `git commit -m "Add feature"`
8. Push to the branch: `git push origin feature-name`
9. Submit a pull request

### Development Guidelines

- **Test-Driven Development**: Write tests before implementation
- **Clean Code**: Follow PEP 8 and project conventions
- **Type Hints**: Use type annotations for all functions
- **Documentation**: Add docstrings for all public functions
- **Error Handling**: Implement robust error handling and logging

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for inspiration and dataset utilities
- The transformer paper authors for the foundational architecture

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers)

---

**Note**: This is an educational project built from scratch to understand transformer architectures and LLM training. For production use, consider using established libraries like Hugging Face Transformers.
