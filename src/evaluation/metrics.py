"""
Total Variation Distance (TVD) and related distribution metrics.
"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Set


def get_estimator(elements: List[str]) -> Tuple[List[str], List[float]]:
    """
    Get the Maximum Likelihood Estimate (MLE) for a discrete distribution.
    
    Probability of a word equals its relative frequency in the sample.
    
    Args:
        elements: List of sampled elements
        
    Returns:
        Tuple of (support, probabilities)
    """
    c = Counter(elements)
    support = list(c.keys())
    counts = list(c.values())
    probs = [count / sum(counts) for count in counts]

    return (support, probs)


def get_common_support(support1: List[str], support2: List[str]) -> Set[str]:
    """
    Get union of supports from two distributions.
    
    Args:
        support1: Support (domain) of first distribution
        support2: Support (domain) of second distribution
        
    Returns:
        Set of all elements appearing in at least one distribution
    """
    return set(support1).union(set(support2))


def change_support(
    old_support: List[str], 
    old_probs: List[float], 
    new_support: Set[str]
) -> Tuple[List[str], List[float]]:
    """
    Expand a distribution's support by adding missing elements with probability 0.
    
    Args:
        old_support: Original support
        old_probs: Original probabilities
        new_support: New expanded support
        
    Returns:
        Tuple of (new_support_list, new_probs)
    """
    new_probs = []
    for item in new_support:
        if item in old_support:
            ind = old_support.index(item)
            new_probs.append(old_probs[ind])
        else:
            new_probs.append(0)
    return list(new_support), new_probs


def get_tvd(probs1: List[float], probs2: List[float]) -> float:
    """
    Compute Total Variation Distance (TVD) between two probability distributions.
    
    TVD = 0.5 * Σ |p1(x) - p2(x)|
    
    Args:
        probs1: Probability distribution 1 (must sum to 1)
        probs2: Probability distribution 2 (must sum to 1)
        
    Returns:
        TVD value in [0, 1]
    """
    tvd = np.sum(np.abs(np.array(probs1) - np.array(probs2))) / 2
    return tvd


def compute_entropy(probs: List[float]) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    H(p) = -Σ p(x) log p(x)
    
    Args:
        probs: Probability distribution
        
    Returns:
        Entropy value
    """
    p = np.array(probs)
    p = p[p > 0]  # Filter out zero probabilities
    return -(p * np.log(p)).sum()
