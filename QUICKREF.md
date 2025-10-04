# 🚀 Quick Reference

## Installation
```bash
git clone https://github.com/mohammadmahdirahimi/human-variability-llm.git
cd human-variability-llm
pip install -r requirements.txt
```

## Usage
```bash
# Train model
python scripts/train.py --config configs/train_config.yaml

# Evaluate models
python scripts/evaluate.py --config configs/eval_config.yaml

# Analyze results
python scripts/analyze.py --results results/tvd_results.pkl
```

## Use as Library
```python
from src.models import MultiTokenSoftCETrainer
from src.evaluation import evaluate_models_with_sampling
from src.analysis import plot_tvd_distributions

# Your code here...
```

## Configuration
Edit `configs/train_config.yaml`:
```yaml
training:
  epochs: 12
  learning_rate: 4.5e-5
  batch_size: 1
```

---
**Author:** Mohammadmahdi Rahimi | **License:** MIT
