"""
================================================================================
DELTA E VALIDATION - PERCEPTUAL ACCURACY TESTING
================================================================================

This module implements Delta E (CIEDE2000) testing to measure perceptual
color accuracy of reconstructed skin images.

Delta E measures how different two colors appear to the HUMAN EYE, not just
their mathematical RGB distance. It's the gold standard for color accuracy
in professional industries (printing, cosmetics, dermatology, CGI).

Delta E Scale:
- ΔE < 1.0:  Imperceptible (perfect match)
- ΔE 1-2:    Perceptible only upon close inspection (excellent)
- ΔE 2-10:   Perceptible at a glance (acceptable)
- ΔE > 10:   Colors appear different (poor)

Usage:
    from delta_e_validation import calculate_delta_e, analyze_delta_e_results
    
    results = calculate_delta_e(original_image, reconstructed_image)
    report = analyze_delta_e_results(results)

================================================================================
"""

import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# DELTA E CALCULATION
# ============================================================================

def rgb_to_lab(rgb_image):
    """
    Convert RGB image to LAB color space.
    
    LAB is perceptually uniform - equal distances in LAB space correspond
    to equal perceptual differences.
    
    Args:
        rgb_image: numpy array (H, W, 3) with values in [0, 1]
    
    Returns:
        lab_image: numpy array (H, W, 3) in LAB color space
    """
    # Ensure input is in [0, 1] range
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Convert RGB to LAB
    lab_image = color.rgb2lab(rgb_image)
    
    return lab_image


def calculate_delta_e_ciede2000(lab1, lab2):
    """
    Calculate CIEDE2000 Delta E between two LAB colors.
    
    CIEDE2000 is the most accurate Delta E formula, accounting for
    non-uniformities in LAB space and human color perception.
    
    Args:
        lab1: numpy array (..., 3) in LAB color space
        lab2: numpy array (..., 3) in LAB color space
    
    Returns:
        delta_e: numpy array of Delta E values
    """
    # Use scikit-image's built-in CIEDE2000 implementation
    delta_e = color.deltaE_ciede2000(lab1, lab2)
    
    return delta_e


def calculate_delta_e(original_rgb, reconstructed_rgb):
    """
    Calculate per-pixel Delta E between original and reconstructed images.
    
    Args:
        original_rgb: numpy array (H, W, 3) with values in [0, 1]
        reconstructed_rgb: numpy array (H, W, 3) with values in [0, 1]
    
    Returns:
        results: dict containing:
            - delta_e_map: (H, W) array of per-pixel Delta E values
            - mean_delta_e: average Delta E across image
            - max_delta_e: maximum Delta E
            - median_delta_e: median Delta E
            - percentile_95: 95th percentile Delta E
            - percent_imperceptible: % of pixels with ΔE < 1.0
            - percent_excellent: % of pixels with ΔE < 2.0
            - percent_acceptable: % of pixels with ΔE < 10.0
    """
    # Convert to LAB color space
    lab_original = rgb_to_lab(original_rgb)
    lab_reconstructed = rgb_to_lab(reconstructed_rgb)
    
    # Calculate Delta E for each pixel
    delta_e_map = calculate_delta_e_ciede2000(lab_original, lab_reconstructed)
    
    # Calculate statistics
    results = {
        'delta_e_map': delta_e_map,
        'mean_delta_e': float(np.mean(delta_e_map)),
        'max_delta_e': float(np.max(delta_e_map)),
        'median_delta_e': float(np.median(delta_e_map)),
        'std_delta_e': float(np.std(delta_e_map)),
        'percentile_95': float(np.percentile(delta_e_map, 95)),
        'percentile_99': float(np.percentile(delta_e_map, 99)),
        'percent_imperceptible': float(np.sum(delta_e_map < 1.0) / delta_e_map.size * 100),
        'percent_excellent': float(np.sum(delta_e_map < 2.0) / delta_e_map.size * 100),
        'percent_acceptable': float(np.sum(delta_e_map < 10.0) / delta_e_map.size * 100),
        'percent_poor': float(np.sum(delta_e_map >= 10.0) / delta_e_map.size * 100),
    }
    
    return results


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def calculate_delta_e_batch(original_images, reconstructed_images):
    """
    Calculate Delta E for multiple images.
    
    Args:
        original_images: list of numpy arrays (H, W, 3)
        reconstructed_images: list of numpy arrays (H, W, 3)
    
    Returns:
        batch_results: list of results dicts (one per image)
    """
    batch_results = []
    
    for original, reconstructed in zip(original_images, reconstructed_images):
        result = calculate_delta_e(original, reconstructed)
        batch_results.append(result)
    
    return batch_results


# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

def analyze_delta_e_results(batch_results):
    """
    Analyze Delta E results across multiple images.
    
    Args:
        batch_results: list of results dicts from calculate_delta_e
    
    Returns:
        summary: dict with aggregate statistics
    """
    # Extract metrics from all images
    mean_delta_es = [r['mean_delta_e'] for r in batch_results]
    max_delta_es = [r['max_delta_e'] for r in batch_results]
    median_delta_es = [r['median_delta_e'] for r in batch_results]
    p95_delta_es = [r['percentile_95'] for r in batch_results]
    
    imperceptible = [r['percent_imperceptible'] for r in batch_results]
    excellent = [r['percent_excellent'] for r in batch_results]
    acceptable = [r['percent_acceptable'] for r in batch_results]
    poor = [r['percent_poor'] for r in batch_results]
    
    # Aggregate statistics
    summary = {
        'num_images': len(batch_results),
        'overall_mean_delta_e': float(np.mean(mean_delta_es)),
        'overall_max_delta_e': float(np.max(max_delta_es)),
        'overall_median_delta_e': float(np.median(median_delta_es)),
        'mean_p95_delta_e': float(np.mean(p95_delta_es)),
        'avg_percent_imperceptible': float(np.mean(imperceptible)),
        'avg_percent_excellent': float(np.mean(excellent)),
        'avg_percent_acceptable': float(np.mean(acceptable)),
        'avg_percent_poor': float(np.mean(poor)),
        'best_image_idx': int(np.argmin(mean_delta_es)),
        'worst_image_idx': int(np.argmax(mean_delta_es)),
        'best_mean_delta_e': float(np.min(mean_delta_es)),
        'worst_mean_delta_e': float(np.max(mean_delta_es)),
    }
    
    return summary


def assess_quality(mean_delta_e, p95_delta_e, percent_excellent):
    """
    Assess overall quality based on Delta E metrics.
    
    Args:
        mean_delta_e: average Delta E
        p95_delta_e: 95th percentile Delta E
        percent_excellent: % of pixels with ΔE < 2.0
    
    Returns:
        quality: str ('Excellent', 'Good', 'Fair', 'Poor')
        color: str (for visualization)
    """
    if mean_delta_e < 2.0 and p95_delta_e < 3.0 and percent_excellent > 90:
        return 'Excellent', 'success'
    elif mean_delta_e < 3.5 and p95_delta_e < 6.0 and percent_excellent > 75:
        return 'Good', 'info'
    elif mean_delta_e < 6.0 and p95_delta_e < 10.0 and percent_excellent > 50:
        return 'Fair', 'warning'
    else:
        return 'Poor', 'error'


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_delta_e_visualization(original_rgb, reconstructed_rgb, delta_e_result):
    """
    Create visualization comparing original, reconstructed, and Delta E map.
    
    Args:
        original_rgb: numpy array (H, W, 3)
        reconstructed_rgb: numpy array (H, W, 3)
        delta_e_result: dict from calculate_delta_e
    
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Reconstructed
    axes[1].imshow(reconstructed_rgb)
    axes[1].set_title('Reconstructed', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Delta E map
    delta_e_map = delta_e_result['delta_e_map']
    im = axes[2].imshow(delta_e_map, cmap='viridis', vmin=0, vmax=10)
    axes[2].set_title(f"Delta E Map (Mean: {delta_e_result['mean_delta_e']:.2f})", 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Delta E', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    
    return fig


def create_delta_e_histogram(delta_e_result):
    """
    Create histogram of Delta E values with thresholds.
    
    Args:
        delta_e_result: dict from calculate_delta_e
    
    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delta_e_map = delta_e_result['delta_e_map'].flatten()
    
    # Histogram
    n, bins, patches = ax.hist(delta_e_map, bins=100, alpha=0.7, 
                                color='steelblue', edgecolor='black')
    
    # Threshold lines
    ax.axvline(1.0, color='green', linestyle='--', linewidth=2, 
               label='Imperceptible (ΔE = 1.0)')
    ax.axvline(2.0, color='orange', linestyle='--', linewidth=2, 
               label='Excellent (ΔE = 2.0)')
    ax.axvline(10.0, color='red', linestyle='--', linewidth=2, 
               label='Acceptable (ΔE = 10.0)')
    
    # Mean line
    ax.axvline(delta_e_result['mean_delta_e'], color='blue', linestyle='-', 
               linewidth=2, label=f"Mean: {delta_e_result['mean_delta_e']:.2f}")
    
    ax.set_xlabel('Delta E', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Delta E Values', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# REPORTING
# ============================================================================

def generate_delta_e_report(summary, output_path=None):
    """
    Generate text report of Delta E analysis.
    
    Args:
        summary: dict from analyze_delta_e_results
        output_path: optional path to save report
    
    Returns:
        report: str
    """
    report = []
    report.append("=" * 80)
    report.append("DELTA E (CIEDE2000) PERCEPTUAL ACCURACY REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"Images Analyzed: {summary['num_images']}")
    report.append("")
    
    report.append("-" * 80)
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Mean Delta E:           {summary['overall_mean_delta_e']:.3f}")
    report.append(f"Median Delta E:         {summary['overall_median_delta_e']:.3f}")
    report.append(f"Max Delta E:            {summary['overall_max_delta_e']:.3f}")
    report.append(f"95th Percentile:        {summary['mean_p95_delta_e']:.3f}")
    report.append("")
    
    report.append("-" * 80)
    report.append("QUALITY DISTRIBUTION")
    report.append("-" * 80)
    report.append(f"Imperceptible (ΔE<1):   {summary['avg_percent_imperceptible']:.1f}%")
    report.append(f"Excellent (ΔE<2):       {summary['avg_percent_excellent']:.1f}%")
    report.append(f"Acceptable (ΔE<10):     {summary['avg_percent_acceptable']:.1f}%")
    report.append(f"Poor (ΔE≥10):           {summary['avg_percent_poor']:.1f}%")
    report.append("")
    
    # Quality assessment
    quality, _ = assess_quality(
        summary['overall_mean_delta_e'],
        summary['mean_p95_delta_e'],
        summary['avg_percent_excellent']
    )
    
    report.append("-" * 80)
    report.append("OVERALL ASSESSMENT")
    report.append("-" * 80)
    report.append(f"Quality Rating: {quality}")
    report.append("")
    
    if quality == 'Excellent':
        report.append("✓ Professional-grade color reproduction")
        report.append("✓ Visually imperceptible differences")
        report.append("✓ Suitable for demanding applications")
    elif quality == 'Good':
        report.append("✓ High-quality color reproduction")
        report.append("⚠ Minor perceptible differences")
        report.append("✓ Suitable for most applications")
    elif quality == 'Fair':
        report.append("⚠ Acceptable color reproduction")
        report.append("⚠ Noticeable color differences")
        report.append("⚠ May need improvement for critical applications")
    else:
        report.append("✗ Poor color reproduction")
        report.append("✗ Significant color differences")
        report.append("✗ Requires model improvements")
    
    report.append("")
    report.append("-" * 80)
    report.append("BEST/WORST IMAGES")
    report.append("-" * 80)
    report.append(f"Best Image (Index {summary['best_image_idx']}):  "
                  f"Mean ΔE = {summary['best_mean_delta_e']:.3f}")
    report.append(f"Worst Image (Index {summary['worst_image_idx']}): "
                  f"Mean ΔE = {summary['worst_mean_delta_e']:.3f}")
    report.append("")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Delta E Validation Module")
    print("=" * 50)
    print("\nThis module provides Delta E (CIEDE2000) analysis.")
    print("\nUsage:")
    print("  from delta_e_validation import calculate_delta_e")
    print("  results = calculate_delta_e(original, reconstructed)")
    print("\nSee documentation for more details.")