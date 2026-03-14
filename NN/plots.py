"""
Deep Albedo — Validation Plots

Histogram and heatmap visualisations for latent-space validation.
Used by latent_space_validation.py.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_distribution_plots(all_params, param_names, ranges, output_dir):
    """
    Save a 2×3 grid of histograms — one per parameter — with expected-range
    overlays and mean lines.

    Args:
        all_params:  (N, 5) numpy array
        param_names: list of 5 names
        ranges:      dict  name → (min, max)
        output_dir:  directory to write parameter_distributions.png
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        v = all_params[:, i]
        exp_min, exp_max = ranges[name]

        ax.hist(v, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(exp_min, color='green', linestyle='--', linewidth=2,
                   label=f'Range [{exp_min}, {exp_max}]')
        ax.axvline(exp_max, color='green', linestyle='--', linewidth=2)
        ax.axvline(np.mean(v), color='red', linestyle='-', linewidth=2,
                   label=f'Mean {np.mean(v):.4f}')
        ax.set_title(f'{name} Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].remove()
    plt.tight_layout()
    out = f'{output_dir}/parameter_distributions.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Distribution plots → {out}")


def create_correlation_heatmap(corr_matrix, param_names, output_dir):
    """
    Save a correlation heatmap for the 5 skin parameters.

    Args:
        corr_matrix: (5, 5) numpy array
        param_names: list of 5 names
        output_dir:  directory to write correlation_matrix.png
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names)
    ax.set_yticklabels(param_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha="center", va="center", fontsize=10)

    ax.set_title("Parameter Correlation Matrix", fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    out = f'{output_dir}/correlation_matrix.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Correlation heatmap → {out}")
