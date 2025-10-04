"""
Evaluation script for comparing original and fine-tuned GPT-2 models.

Usage:
    python scripts/evaluate.py --config configs/eval_config.yaml
"""

import os
import argparse
import yaml
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import preprocess_provo_corpus, build_softlabel_dataset, split_dataset
from src.evaluation import evaluate_models_with_sampling
from src.analysis import plot_tvd_distributions, plot_tvd_density


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("GPT-2 Model Evaluation")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Load tokenizer
    print(f"\n📦 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(config['models']['original'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    print(f"\n📦 Loading original model: {config['models']['original']}")
    original_model = GPT2LMHeadModel.from_pretrained(config['models']['original'])
    
    print(f"📦 Loading fine-tuned model: {config['models']['finetuned']}")
    finetuned_model = GPT2LMHeadModel.from_pretrained(config['models']['finetuned'])
    
    # Load and preprocess data
    print(f"\n📊 Loading Provo Corpus from: {config['data']['provo_path']}")
    data = preprocess_provo_corpus(config['data']['provo_path'])
    
    # Build soft-label dataset
    print("\n🏗️  Building soft-label dataset...")
    soft_examples = build_softlabel_dataset(data, tokenizer)
    
    # Get test data
    _, _, test_data = split_dataset(
        soft_examples,
        train_text_ids=(1, 47),
        val_text_ids=(47, 50),
        test_text_ids=(50, 56)
    )
    print(f"\n📂 Test set size: {len(test_data)}")
    
    # Evaluate
    print("\n🔍 Evaluating models...")
    print("=" * 80)
    
    results = evaluate_models_with_sampling(
        original_model=original_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        test_data=test_data,
        n_samples=config['evaluation'].get('n_samples', 40),
        include_oracle=config['evaluation'].get('include_oracle', True),
        device=str(device)
    )
    
    # Save results
    output_dir = config['evaluation'].get('output_dir', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'tvd_results.pkl')
    print(f"\n💾 Saving results to: {results_path}")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Generate visualizations
    if config['evaluation'].get('generate_plots', True):
        print("\n📊 Generating visualizations...")
        
        plot_tvd_distributions(
            results,
            save_path=os.path.join(output_dir, 'tvd_histogram.png')
        )
        
        plot_tvd_density(
            results,
            save_path=os.path.join(output_dir, 'tvd_density.png')
        )
    
    print("\n✅ Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 models")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eval_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    main(args)
