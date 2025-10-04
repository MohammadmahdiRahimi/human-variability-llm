"""Analysis utilities for TVD results."""

from .features import (
    context_length,
    entropy,
    avg_token_length,
    majority_pos_tag,
    last_word_pos,
    human_skewness,
    human_kurtosis,
    extract_all_features
)

from .visualizations import (
    plot_tvd_distributions,
    plot_tvd_density,
    plot_feature_correlation,
    plot_pos_analysis,
    plot_word_position_analysis,
    plot_correlation_heatmap
)

from .statistics import (
    paired_ttest,
    regression_analysis,
    compute_quartile_statistics
)

__all__ = [
    # Features
    'context_length',
    'entropy',
    'avg_token_length',
    'majority_pos_tag',
    'last_word_pos',
    'human_skewness',
    'human_kurtosis',
    'extract_all_features',
    # Visualizations
    'plot_tvd_distributions',
    'plot_tvd_density',
    'plot_feature_correlation',
    'plot_pos_analysis',
    'plot_word_position_analysis',
    'plot_correlation_heatmap',
    # Statistics
    'paired_ttest',
    'regression_analysis',
    'compute_quartile_statistics'
]
