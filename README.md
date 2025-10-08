# Aksis: Transformer Language Model Research

A research implementation of transformer-based language models built from scratch for educational purposes. This project explores the fundamentals of modern language model architectures, training methodologies, and inference techniques.

## 🎯 Research Objectives

- **Architecture Understanding**: Implement transformer components from first principles
- **Training Dynamics**: Study language model training on various datasets
- **Inference Methods**: Compare different sampling strategies and their effects
- **Educational Value**: Provide clear, well-documented code for learning

## 🏗️ Architecture

**Decoder-Only Transformer** (GPT-style):
- 6 layers, 512 hidden dimensions, 8 attention heads
- 10,004 vocabulary size, 512 max sequence length
- Learned positional embeddings, pre-norm architecture

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd aksis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
# Train on WikiText-2
python -m aksis.cli train-model --dataset wikitext2 --epochs 10

# Train with mixed precision
python -m aksis.cli train-model --mixed-precision --batch-size 32
```

### Inference
```bash
# Interactive chat
python -m aksis.cli chat-with-model --checkpoint-path checkpoints/best.pt

# Text generation
python -m aksis.cli generate-text \
    --checkpoint-path checkpoints/best.pt \
    --prompt "The future of AI is" \
    --sampler temperature \
    --temperature 0.8
```

### Web Interface
```bash
# Start API server
python deploy/start_api_server.py

# Open web interface
xdg-open deploy/web_interface.html
```

## 🔬 Research Features

### Sampling Strategies
- **Greedy**: Deterministic, highest probability token
- **Temperature**: Controlled randomness via scaling
- **Top-K**: Sample from k most probable tokens
- **Top-P**: Nucleus sampling with cumulative probability threshold

### Evaluation Metrics
- **Perplexity**: Language modeling quality
- **BLEU**: N-gram overlap with references
- **ROUGE**: Recall-oriented evaluation
- **Generation Quality**: Human evaluation frameworks

### Training Experiments
- **Dataset Studies**: WikiText-2, Shakespeare, custom corpora
- **Architecture Variants**: Layer depth, attention heads, hidden dimensions
- **Optimization**: Learning rate schedules, mixed precision, gradient accumulation
- **Regularization**: Dropout, weight decay, early stopping

## 📊 Project Structure

```
aksis/
├── src/aksis/           # Core implementation
│   ├── model/          # Transformer architecture
│   ├── train/          # Training loops and utilities
│   ├── inference/      # Generation and sampling
│   ├── data/           # Tokenization and datasets
│   └── eval/           # Evaluation metrics
├── tests/              # Comprehensive test suite
├── training/           # Training scripts and logs
├── deploy/             # API server and web interface
└── docs/               # Research documentation
```

## 🧪 Experimental Setup

### Hardware Requirements
- **GPU**: CUDA-compatible (recommended for training)
- **RAM**: 8GB+ for training, 4GB+ for inference
- **Storage**: 10GB+ for datasets and checkpoints

### Software Dependencies
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional)
- Standard ML libraries (numpy, datasets, etc.)

## 📈 Research Applications

### Language Modeling
- Study perplexity reduction over training
- Compare different tokenization strategies
- Analyze attention patterns and learned representations

### Generation Quality
- Evaluate sampling strategy effects on coherence
- Study temperature scaling and its impact
- Compare greedy vs. stochastic generation

### Training Dynamics
- Monitor loss curves and convergence
- Study gradient flow and optimization
- Analyze model capacity vs. dataset size

## 🔍 Code Quality

- **Type Safety**: Full MyPy type checking
- **Testing**: 80%+ test coverage with pytest
- **Formatting**: Black code formatting, Flake8 linting
- **Documentation**: Comprehensive docstrings and examples

## 📚 Educational Resources

### Key Concepts Covered
- Attention mechanisms and multi-head attention
- Positional encoding and sequence modeling
- Language model training and optimization
- Text generation and sampling strategies
- Model evaluation and metrics

### Learning Path
1. **Architecture**: Study `src/aksis/model/` for transformer components
2. **Training**: Explore `src/aksis/train/` for training dynamics
3. **Inference**: Examine `src/aksis/inference/` for generation methods
4. **Experiments**: Run training and evaluation scripts

## 🤝 Contributing

This is a research project. Contributions should focus on:
- Educational improvements and documentation
- Research experiments and findings
- Code clarity and learning value
- Novel architectural or methodological insights

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Transformer paper authors for the foundational architecture
- Educational resources that inspired this implementation

---

**Research Note**: This implementation prioritizes educational value and research insights over production optimization. For production applications, consider established libraries like Hugging Face Transformers.
