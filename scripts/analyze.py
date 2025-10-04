"""
Analysis script for understanding TVD improvements.

Usage:
    python scripts/analyze.py --results results/tvd_results.pkl --config configs/eval_config.yaml
"""

import os
import argparse
import yaml
import pickle
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import preprocess_provo_corpus, build_softlabel_dataset, split_dataset
from src.analysis import (
    extract_all_features,
    plot_feature_correlation,
    plot_pos_analysis,
    plot_word_position_analysis,
    plot_correlation_heatmap,
    paired_ttest,
    regression_analysis,
    compute_quartile_statistics
)


def main(args):
    print("=" * 80)
    print("TVD Results Analysis")
    print("=" * 80)
    
    # Load results
    print(f"\n📂 Loading results from: {args.results}")
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data for extended analysis
    print(f"\n📊 Loading Provo Corpus from: {config['data']['provo_path']}")
    tokenizer = GPT2Tokenizer.from_pretrained(config['models']['original'])
    tokenizer.pad_token = tokenizer.eos_token
    
    data = preprocess_provo_corpus(config['data']['provo_path'])
    soft_examples = build_softlabel_dataset(data, tokenizer)
    _, _, test_data = split_dataset(
        soft_examples,
        train_text_ids=(1, 47),
        val_text_ids=(47, 50),
        test_text_ids=(50, 56)
    )
    
    # Build extended test data with TVD improvements
    print("\n🔧 Building extended test data...")
    test_ids = set([s["text_id"] for s in test_data])
    test_data_with_meta = [
        entry for entry in data.values() 
        if entry["original_positioning"]["text_id"] in test_ids
    ]
    
    tvd_orig = results["original_tvds"]
    tvd_fine = results["finetuned_tvds"]
    
    extended_test_data = []
    for i, sample in enumerate(test_data_with_meta):
        if i >= len(tvd_orig):
            break
        
        new_sample = dict(sample)
        human_preds = new_sample.get("human_next_word_pred", [])
        
        filtered = [(w, c) for w, c in human_preds 
                   if isinstance(w, str) and w.strip() and c > 0]
        total = sum(freq for _, freq in filtered)
        
        if total == 0:
            continue
        
        probs = [freq / total for _, freq in filtered]
        words = [w for w, _ in filtered]
        
        new_sample["next_words"] = words
        new_sample["probs"] = probs
        new_sample["tvd_original"] = tvd_orig[i]
        new_sample["tvd_finetuned"] = tvd_fine[i]
        new_sample["tvd_improvement"] = tvd_orig[i] - tvd_fine[i]
        extended_test_data.append(new_sample)
    
    print(f"   Extended test data: {len(extended_test_data)} samples")
    
    # Output directory
    output_dir = args.output or os.path.dirname(args.results)
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== Statistical Significance =====
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)
    
    ttest_results = paired_ttest(
        np.array(results["original_tvds"]),
        np.array(results["finetuned_tvds"])
    )
    
    # Save statistics
    with open(os.path.join(output_dir, 'ttest_results.pkl'), 'wb') as f:
        pickle.dump(ttest_results, f)
    
    # ===== Quartile Statistics =====
    print("\n" + "=" * 80)
    print("QUARTILE STATISTICS")
    print("=" * 80)
    
    quartile_stats = compute_quartile_statistics(results)
    print("\n" + quartile_stats.to_string(index=False))
    quartile_stats.to_csv(os.path.join(output_dir, 'quartile_stats.csv'), index=False)
    
    # ===== Feature Extraction =====
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION")
    print("=" * 80)
    
    features = extract_all_features(extended_test_data, tokenizer)
    print(f"\n✅ Extracted features for {len(features['tvd_improv'])} samples")
    
    # ===== Feature Correlation Analysis =====
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    
    plot_feature_correlation(features, save_path=os.path.join(output_dir, 'correlation'))
    
    # ===== POS Tag Analysis =====
    print("\n" + "=" * 80)
    print("POS TAG ANALYSIS")
    print("=" * 80)
    
    plot_pos_analysis(features, save_path=os.path.join(output_dir, 'pos_analysis.png'))
    
    # ===== Word Position Analysis =====
    print("\n" + "=" * 80)
    print("WORD POSITION ANALYSIS")
    print("=" * 80)
    
    plot_word_position_analysis(
        features,
        save_path=os.path.join(output_dir, 'word_position_analysis.png')
    )
    
    # ===== Correlation Heatmap =====
    print("\n📊 Generating correlation heatmap...")
    plot_correlation_heatmap(
        features,
        save_path=os.path.join(output_dir, 'correlation_heatmap.png')
    )
    
    # ===== Regression Analysis =====
    print("\n" + "=" * 80)
    print("REGRESSION ANALYSIS")
    print("=" * 80)
    
    model, df_clean = regression_analysis(features)
    
    # Save regression results
    with open(os.path.join(output_dir, 'regression_summary.txt'), 'w') as f:
        f.write(model.summary().as_text())
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TVD results")
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to TVD results pickle file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eval_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for analysis results'
    )
    args = parser.parse_args()
    
    main(args)
