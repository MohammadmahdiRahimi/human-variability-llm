# Getting Started

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-compatible GPU (recommended, but CPU works too)

## Installation

```bash
# Clone the repository
git clone https://github.com/mohammadmahdirahimi/human-variability-llm.git
cd human-variability-llm

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

Download the Provo Corpus:
- Visit: https://osf.io/sjefs/
- Download `Provo_Corpus.tsv`
- Place it in the project root directory

## Quick Usage

### Train a Model

```bash
python scripts/train.py --config configs/train_config.yaml
```

This will:
- Load the Provo Corpus
- Fine-tune GPT-2 with multi-label soft labels
- Save the model to `./gpt2-finetuned/`

### Evaluate Models

```bash
python scripts/evaluate.py --config configs/eval_config.yaml
```

This will:
- Compare original GPT-2 vs fine-tuned model
- Calculate TVD metrics
- Save results to `./results/tvd_results.pkl`
- Generate visualization plots

### Analyze Results

```bash
python scripts/analyze.py --results results/tvd_results.pkl --config configs/eval_config.yaml
```

This will:
- Extract features from test data
- Perform statistical analysis
- Generate correlation plots
- Create comprehensive analysis reports

## Configuration

Modify training parameters in `configs/train_config.yaml`:

```yaml
training:
  epochs: 12              # Number of training epochs
  learning_rate: 4.5e-5   # Learning rate
  batch_size: 1           # Batch size per device
```

## Using as a Library

```python
from src.models import MultiTokenSoftCETrainer
from src.evaluation import evaluate_models_with_sampling
from src.data import preprocess_provo_corpus

# Your code here
```

See `example_usage.py` for detailed examples.

## Next Steps

- Read the [README](README.md) for detailed documentation
- Check [CONTRIBUTING](CONTRIBUTING.md) to contribute
- See [QUICKREF](QUICKREF.md) for command reference

## Troubleshooting

**CUDA Out of Memory**: Reduce `batch_size` in config  
**Import Errors**: Ensure all dependencies are installed  
**Data Not Found**: Check that `Provo_Corpus.tsv` is in the root directory

## Support

For questions or issues, please open a GitHub issue.
