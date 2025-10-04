"""
Training script for fine-tuning GPT-2 with multiple labels per context.

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""

import os
import argparse
import yaml
import torch
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments
from functools import partial

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import (
    preprocess_provo_corpus,
    build_softlabel_dataset,
    split_dataset,
    tokenize_candidates,
    SoftCECollator
)
from src.models import MultiTokenSoftCETrainer, freeze_bottom_layers


def tokenize_example(example, tokenizer):
    """Tokenize a single example."""
    ctx = tokenizer(example["context"], add_special_tokens=False)
    candidate_ids = [
        tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True)
        for word in example["next_words"]
    ]
    return {
        "input_ids": ctx["input_ids"],
        "attention_mask": ctx["attention_mask"],
        "candidate_ids": candidate_ids,
    }


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("GPT-2 Multi-Label Fine-tuning")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Load tokenizer and model
    print(f"\n📦 Loading model: {config['model']['name']}")
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(config['model']['name'])
    
    # Optional: freeze bottom layers
    if config['model'].get('freeze_layers', 0) > 0:
        freeze_layers = config['model']['freeze_layers']
        print(f"❄️  Freezing bottom {freeze_layers} layers")
        freeze_bottom_layers(model, freeze_layers=freeze_layers)
    
    model = model.to(device)
    
    # Load and preprocess data
    print(f"\n📊 Loading Provo Corpus from: {config['data']['provo_path']}")
    data = preprocess_provo_corpus(config['data']['provo_path'])
    print(f"   Loaded {len(data)} samples")
    
    # Build soft-label dataset
    print("\n🏗️  Building soft-label dataset...")
    soft_examples = build_softlabel_dataset(data, tokenizer)
    print(f"   Built {len(soft_examples)} examples")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(
        soft_examples,
        train_text_ids=(1, 47),
        val_text_ids=(47, 50),
        test_text_ids=(50, 56)
    )
    print(f"\n📂 Dataset splits:")
    print(f"   Train: {len(train_data)}")
    print(f"   Validation: {len(val_data)}")
    print(f"   Test: {len(test_data)}")
    
    # Create HuggingFace datasets
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })
    
    # Tokenize candidate words
    print("\n🔤 Tokenizing candidates...")
    tokenize_fn = partial(tokenize_candidates, tokenizer=tokenizer)
    dataset = dataset.map(tokenize_fn, remove_columns=[])
    
    # Tokenize examples
    tokenize_fn2 = partial(tokenize_example, tokenizer=tokenizer)
    dataset = dataset.map(tokenize_fn2)
    
    # Create collator
    collator = SoftCECollator(tokenizer)
    
    # Training arguments
    train_config = config['training']
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        overwrite_output_dir=True,
        per_device_train_batch_size=train_config['batch_size'],
        per_device_eval_batch_size=train_config['batch_size'],
        num_train_epochs=train_config['epochs'],
        learning_rate=train_config['learning_rate'],
        eval_strategy="epoch",
        save_strategy="epoch" if train_config.get('save_checkpoints', False) else "no",
        logging_dir=os.path.join(train_config['output_dir'], 'logs'),
        logging_steps=train_config.get('logging_steps', 50),
        save_total_limit=train_config.get('save_total_limit', 2),
        report_to="none",
        remove_unused_columns=False,
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 8),
        fp16=train_config.get('fp16', True) and torch.cuda.is_available(),
        warmup_ratio=train_config.get('warmup_ratio', 0.05),
        weight_decay=train_config.get('weight_decay', 0.01),
        load_best_model_at_end=False,
    )
    
    # Create trainer
    trainer = MultiTokenSoftCETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("\n🚀 Starting training...")
    print("=" * 80)
    trainer.train()
    
    # Save final model
    final_model_path = train_config.get('final_model_path', './gpt2-finetuned')
    print(f"\n💾 Saving final model to: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print("\n✅ Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT-2 with multiple labels")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    main(args)
