"""
Deep Albedo — Metrics & Statistical Analysis

Statistical analysis of predicted skin parameter distributions.
Used by latent_space_validation.py.
"""

import numpy as np


def analyze_parameter_distribution(all_params, param_names, ranges, bio_ranges):
    """
    Analyse the distribution of predicted skin parameters.

    Checks:
      - Are values within expected LUT ranges?
      - Is there boundary clustering (model stuck at min/max)?
      - What fraction falls within biological literature ranges?
      - How many outliers exist?

    Args:
        all_params:  numpy array (N, 5)
        param_names: list of 5 parameter names
        ranges:      dict  name → (min, max)  — expected LUT range
        bio_ranges:  dict  name → (min, max)  — biological plausibility range
                     (may omit parameters; falls back to expected range)

    Returns:
        dict of per-parameter statistics
    """
    stats = {}
    for i, name in enumerate(param_names):
        v = all_params[:, i]
        exp_min, exp_max = ranges[name]
        span = exp_max - exp_min

        at_lower = (v < exp_min + 0.05 * span).sum() / len(v)
        at_upper = (v > exp_max - 0.05 * span).sum() / len(v)

        bio_min, bio_max = bio_ranges.get(name, (exp_min, exp_max))
        within_bio = ((v >= bio_min) & (v <= bio_max)).sum()
        outliers   = ((v < exp_min) | (v > exp_max)).sum()

        stats[name] = {
            'mean':   float(np.mean(v)),
            'std':    float(np.std(v)),
            'min':    float(np.min(v)),
            'max':    float(np.max(v)),
            'median': float(np.median(v)),
            'q25':    float(np.percentile(v, 25)),
            'q75':    float(np.percentile(v, 75)),
            'within_expected_range': bool(v.min() >= exp_min and v.max() <= exp_max),
            'boundary_clustering': {
                'lower_boundary_pct': float(at_lower * 100),
                'upper_boundary_pct': float(at_upper * 100),
                'is_problem':         bool(at_lower > 0.1 or at_upper > 0.1),
            },
            'biological_plausibility_pct': float(within_bio / len(v) * 100),
            'outlier_count': int(outliers),
            'outlier_pct':   float(outliers / len(v) * 100),
            'coefficient_of_variation': (
                float(np.std(v) / np.mean(v)) if np.mean(v) > 0 else None
            ),
        }
    return stats


def check_parameter_correlations(all_params, param_names):
    """
    Compute the correlation matrix and flag physically unusual correlations.

    Args:
        all_params:  numpy array (N, 5)
        param_names: list of 5 parameter names

    Returns:
        corr_matrix — (5, 5) float64 numpy array
        issues      — list of dicts describing problematic correlations
    """
    corr   = np.corrcoef(all_params.T)
    issues = []

    cm_i = param_names.index('Cm')
    ch_i = param_names.index('Ch')
    if corr[cm_i, ch_i] > 0.7:
        issues.append({
            'params':      ('Cm', 'Ch'),
            'correlation': float(corr[cm_i, ch_i]),
            'issue':       'High melanin usually obscures the blood signal',
        })

    return corr, issues
