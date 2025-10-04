"""
Example usage of the GPT-2 Human Variability package.
"""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments
from datasets import Dataset, DatasetDict

from src.data import (
    preprocess_provo_corpus,
    build_softlabel_dataset,
    split_dataset,
    SoftCECollator
)
from src.models import MultiTokenSoftCETrainer
from src.evaluation import evaluate_models_with_sampling


def train_example():
    """Example: Train a model with multi-label soft labels."""
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Load and preprocess data
    data = preprocess_provo_corpus("Provo_Corpus.tsv")
    soft_examples = build_softlabel_dataset(data, tokenizer)
    train_data, val_data, test_data = split_dataset(soft_examples)
    
    # Create datasets
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })
    
    # Setup training
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=1,
        num_train_epochs=3,
        learning_rate=4.5e-5,
        eval_strategy="epoch",
    )
    
    # Create trainer
    trainer = MultiTokenSoftCETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=SoftCECollator(tokenizer),
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    trainer.save_model("./my-finetuned-model")


def evaluate_example():
    """Example: Evaluate models and compare TVD."""
    # Load models
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    original_model = GPT2LMHeadModel.from_pretrained("gpt2")
    finetuned_model = GPT2LMHeadModel.from_pretrained("./my-finetuned-model")
    
    # Load test data
    data = preprocess_provo_corpus("Provo_Corpus.tsv")
    soft_examples = build_softlabel_dataset(data, tokenizer)
    _, _, test_data = split_dataset(soft_examples)
    
    # Evaluate
    results = evaluate_models_with_sampling(
        original_model=original_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        test_data=test_data,
        n_samples=40,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\nOriginal TVD: {results['mean_original_tvd']:.4f}")
    print(f"Fine-tuned TVD: {results['mean_finetuned_tvd']:.4f}")
    print(f"Improvement: {results['mean_original_tvd'] - results['mean_finetuned_tvd']:.4f}")


if __name__ == "__main__":
    print("GPT-2 Human Variability - Example Usage")
    print("=" * 80)
    print("\nSee scripts/ directory for command-line usage:")
    print("  - scripts/train.py")
    print("  - scripts/evaluate.py")
    print("  - scripts/analyze.py")

