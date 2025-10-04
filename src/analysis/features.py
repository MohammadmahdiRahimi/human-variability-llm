"""
Feature extraction for analyzing TVD improvements.
"""

import numpy as np
import nltk
from collections import Counter
from typing import Dict, List
from scipy.stats import skew, kurtosis


# Ensure NLTK data is available
try:
    from nltk import pos_tag
except LookupError:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    from nltk import pos_tag


def context_length(sample: Dict) -> int:
    """Get the length of the context."""
    context = sample.get("context", "")
    if isinstance(context, list):
        return len(context)
    return len(context.split())


def entropy(probs: List[float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    p = np.array(probs)
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def avg_token_length(sample: Dict, tokenizer) -> float:
    """
    Compute average token length of next-word candidates.
    
    Args:
        sample: Dataset sample
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Average number of tokens per candidate word
    """
    return np.mean([
        len(tokenizer.encode(w, add_special_tokens=False)) 
        for w in sample["next_words"]
    ])


def majority_pos_tag(sample: Dict) -> str:
    """
    Get the most common POS tag among next-word candidates.
    
    Args:
        sample: Dataset sample with next_words
        
    Returns:
        Most common POS tag (or 'UNK' if none)
    """
    tags = pos_tag(sample["next_words"])
    tag_counts = Counter(tag for _, tag in tags)
    return tag_counts.most_common(1)[0][0] if tag_counts else "UNK"


def last_word_pos(sample: Dict) -> str:
    """
    Get the POS tag of the last word in the context.
    
    Args:
        sample: Dataset sample with context
        
    Returns:
        POS tag of last context word (or 'UNK')
    """
    context = sample.get("context", "")

    # Handle both string and list contexts
    if isinstance(context, str):
        words = context.strip().split()
    elif isinstance(context, list):
        words = context
    else:
        return "UNK"

    if not words:
        return "UNK"

    return pos_tag([words[-1]])[0][1]


def human_skewness(probs: List[float]) -> float:
    """Compute skewness of human probability distribution."""
    return float(skew(probs))


def human_kurtosis(probs: List[float]) -> float:
    """Compute kurtosis of human probability distribution."""
    return float(kurtosis(probs, fisher=False))


def extract_all_features(extended_test_data: List[Dict], tokenizer) -> Dict[str, List]:
    """
    Extract all analysis features from test data.
    
    Args:
        extended_test_data: List of test samples with TVD improvements
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Dictionary mapping feature names to lists of values
    """
    features = {
        "ctx_len": [],
        "entropy": [],
        "avg_tok_len": [],
        "tvd_improv": [],
        "next_pos": [],
        "last_pos": [],
        "human_skew": [],
        "human_kurtosis": [],
        "word_position": []
    }

    for sample in extended_test_data:
        probs = sample["probs"]
        
        features["ctx_len"].append(context_length(sample))
        features["entropy"].append(entropy(probs))
        features["avg_tok_len"].append(avg_token_length(sample, tokenizer))
        features["tvd_improv"].append(sample["tvd_improvement"])
        features["next_pos"].append(majority_pos_tag(sample))
        features["last_pos"].append(last_word_pos(sample))
        features["human_skew"].append(human_skewness(probs))
        features["human_kurtosis"].append(human_kurtosis(probs))
        features["word_position"].append(
            sample["original_positioning"]["word_num"]
        )

    return features
