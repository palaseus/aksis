# Aksis - AI Chatbot/LLM from Scratch

A complete transformer-based language model built from scratch using Python and PyTorch, designed to function as a chatbot with CUDA acceleration support.

## 🚀 Features

- **From Scratch Implementation**: Core transformer architecture built manually for educational purposes
- **CUDA Acceleration**: Automatic GPU detection and utilization with graceful CPU fallback
- **Test-Driven Development**: Comprehensive test suite with pytest
- **Code Quality**: Strict linting with Black, Flake8, and MyPy
- **Modular Architecture**: Clean, scalable codebase structure
- **Chatbot Interface**: Interactive CLI for real-time conversations
- **Fine-tuning Capabilities**: Support for domain-specific training

## 📁 Project Structure

```
aksis/
├── src/                    # Source code
│   ├── aksis/             # Main package
│   │   ├── data/          # Data handling modules
│   │   ├── models/        # Model architectures
│   │   ├── training/      # Training loops and utilities
│   │   ├── inference/     # Inference and chat interface
│   │   └── utils/         # Utility functions
│   └── cli.py             # Command-line interface
├── tests/                 # Test suite
├── data/                  # Datasets and data files
├── models/                # Model checkpoints and logs
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
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

- **Formatting**: Black (line length: 88)
- **Linting**: Flake8
- **Type Checking**: MyPy
- **Testing**: Pytest

Run all quality checks:

```bash
black src tests
flake8 src tests
mypy src
pytest
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit:

```bash
pre-commit run --all-files
```

## 🚀 Usage

### Training a Model

```bash
# Train on default dataset (Shakespeare)
aksis train --dataset shakespeare --epochs 10

# Train with custom parameters
aksis train --dataset custom --batch-size 32 --learning-rate 1e-4
```

### Interactive Chat

```bash
# Start chat interface
aksis chat --model-path models/checkpoint.pth

# Chat with specific context length
aksis chat --model-path models/checkpoint.pth --context-length 512
```

### Data Processing

```bash
# Process and tokenize dataset
aksis process-data --input data/raw/text.txt --output data/processed/
```

## 📊 Model Architecture

The model follows the standard transformer architecture:

- **Encoder-Decoder**: Based on "Attention Is All You Need"
- **Multi-Head Attention**: 8 attention heads by default
- **Positional Encoding**: Learned positional embeddings
- **Layer Normalization**: Pre-norm architecture
- **Feed-Forward**: 4x hidden dimension expansion

### Default Configuration

- **Layers**: 6 encoder + 6 decoder layers
- **Hidden Dimension**: 512
- **Attention Heads**: 8
- **Vocabulary Size**: 50,000 (configurable)
- **Context Length**: 1024 tokens

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
