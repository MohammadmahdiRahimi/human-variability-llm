"""
Model sampling and evaluation utilities.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

from .metrics import (
    get_estimator, 
    get_common_support, 
    change_support, 
    get_tvd
)
from ..data import get_oracle_elements


def get_model_samples(
    model,
    tokenizer,
    context: str,
    n_samples: int = 40,
    add_tokens: int = 1,
    seed: int = 0,
    top_k: int = 0
) -> List[str]:
    """
    Generate next-word samples from a language model given a context.
    
    Args:
        model: Hugging Face language model
        tokenizer: Corresponding tokenizer
        context: Input context string
        n_samples: Number of samples to generate
        add_tokens: Number of tokens to generate (default 1 for next-word)
        seed: Random seed
        top_k: Top-k sampling parameter (0 = disabled)
        
    Returns:
        List of sampled next words
    """
    model.eval()
    input_ids = tokenizer(context, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)
    
    torch.manual_seed(seed)

    outputs = model.generate(
        input_ids=input_ids,
        do_sample=True,
        num_return_sequences=n_samples,
        max_new_tokens=add_tokens,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and extract only the generated word
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [o.replace(context, "").strip().split(" ")[0] for o in outputs]
    
    return outputs


def evaluate_models_with_sampling(
    original_model,
    finetuned_model,
    tokenizer,
    test_data: List[Dict],
    n_samples: int = 40,
    include_oracle: bool = True,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate two models by comparing their next-word distributions to human distributions.
    
    Args:
        original_model: Pre-trained baseline model
        finetuned_model: Fine-tuned model to compare
        tokenizer: Tokenizer for both models
        test_data: List of test examples with human predictions
        n_samples: Number of samples per context
        include_oracle: Whether to include human-human oracle comparison
        device: Device to run on
        
    Returns:
        Dictionary with TVD results and statistics
    """
    original_model.eval()
    finetuned_model.eval()
    
    original_model = original_model.to(device)
    finetuned_model = finetuned_model.to(device)

    original_tvds = []
    finetuned_tvds = []
    oracle_tvds = []

    print("Evaluating models using sampling-based TVD...")

    for sample in tqdm(test_data, desc="Evaluating"):
        context = sample['context']
        words = sample['next_words']
        probs = sample['probs']

        # Reconstruct human word list from probabilities
        words_list = []
        for word, prob in zip(words, probs):
            words_list.extend([word] * int(100 * prob))
        
        support_human, probs_human = get_estimator(words_list)

        if not support_human or not probs_human:
            continue

        # Oracle evaluation (human vs human)
        if include_oracle and len(words_list) >= 10:
            oracle1, oracle2 = get_oracle_elements(words_list.copy())
            if oracle1 and oracle2:
                s1, p1 = get_estimator(oracle1)
                s2, p2 = get_estimator(oracle2)
                union = get_common_support(s1, s2)
                _, p1 = change_support(s1, p1, union)
                _, p2 = change_support(s2, p2, union)
                oracle_tvds.append(get_tvd(p1, p2))

        # Model samples
        model1_samples = get_model_samples(
            original_model, tokenizer, context, n_samples=n_samples
        )
        model2_samples = get_model_samples(
            finetuned_model, tokenizer, context, n_samples=n_samples
        )

        s1, p1 = get_estimator(model1_samples)
        s2, p2 = get_estimator(model2_samples)

        if not s1 or not p1 or not s2 or not p2:
            continue

        # TVD - Original model vs Human
        union1 = get_common_support(support_human, s1)
        _, ph1 = change_support(support_human, probs_human, union1)
        _, pm1 = change_support(s1, p1, union1)
        original_tvds.append(get_tvd(ph1, pm1))

        # TVD - Fine-tuned model vs Human
        union2 = get_common_support(support_human, s2)
        _, ph2 = change_support(support_human, probs_human, union2)
        _, pm2 = change_support(s2, p2, union2)
        finetuned_tvds.append(get_tvd(ph2, pm2))

    # Summary statistics
    print(f"\nCollected TVDs - Original: {len(original_tvds)}, "
          f"Finetuned: {len(finetuned_tvds)}, Oracle: {len(oracle_tvds)}")

    mean_orig = np.mean(original_tvds)
    std_orig = np.std(original_tvds)
    mean_fine = np.mean(finetuned_tvds)
    std_fine = np.std(finetuned_tvds)
    
    mean_oracle = np.mean(oracle_tvds) if oracle_tvds else None
    std_oracle = np.std(oracle_tvds) if oracle_tvds else None

    print("\nEvaluation Summary:")
    print(f"Original GPT-2     - Mean TVD: {mean_orig:.4f} ± {std_orig:.4f}")
    print(f"Fine-tuned GPT-2   - Mean TVD: {mean_fine:.4f} ± {std_fine:.4f}")
    print(f"Δ TVD Improvement  = {mean_orig - mean_fine:.4f}")
    
    if oracle_tvds:
        print(f"Oracle (Human-Human) - Mean TVD: {mean_oracle:.4f} ± {std_oracle:.4f}")
        print(f"Distance to Oracle: Original = {mean_orig - mean_oracle:.4f}, "
              f"Fine-tuned = {mean_fine - mean_oracle:.4f}")

    return {
        "original_tvds": original_tvds,
        "finetuned_tvds": finetuned_tvds,
        "mean_original_tvd": mean_orig,
        "mean_finetuned_tvd": mean_fine,
        "std_original_tvd": std_orig,
        "std_finetuned_tvd": std_fine,
        "oracle_tvds": oracle_tvds if oracle_tvds else None,
        "mean_oracle_tvd": mean_oracle,
        "std_oracle_tvd": std_oracle
    }
