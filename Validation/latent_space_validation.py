"""
================================================================================
LATENT SPACE RANGE VALIDATION SCRIPT
================================================================================

PURPOSE:
This script validates whether your trained autoencoder produces biologically
plausible parameters when given real-world skin images. It checks if the 
encoder's predictions fall within expected ranges and shows meaningful patterns.

WHAT IT DOES:
1. Loads random images from your dataset
2. Automatically detects and extracts skin regions
3. Runs the encoder to predict parameters (Cm, Ch, Bm, Bh, T)
4. Analyzes the distribution of predicted values
5. Identifies potential problems (boundary clustering, outliers, biases)
6. Generates a detailed statistical report

WHAT WE'RE LOOKING FOR:
✓ Parameters within expected biological ranges
✓ Proper distribution (not clustering at boundaries)
✓ Different skin tones produce different Cm values
✓ No impossible parameter combinations
✗ Values stuck at min/max boundaries (bad)
✗ All images produce similar parameters (bad)
✗ Many NaN or negative values (very bad)

================================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import random



# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for validation"""
    
    # Paths
    IMAGE_FOLDER = "path/to/your/images"  # CHANGE THIS
    CHECKPOINT_PATH = "checkpoints/your_run/best.pt"  # CHANGE THIS
    OUTPUT_DIR = "validation_results"
    
    # Analysis parameters
    NUM_IMAGES = 30  # Number of random images to analyze
    PIXELS_PER_IMAGE = 100  # Number of skin pixels to sample per image
    
    # Expected parameter ranges (from your Monte Carlo simulation)
    PARAM_RANGES = {
        'Cm': (0.001, 0.5),    # Melanin concentration
        'Ch': (0.001, 0.32),   # Blood concentration
        'Bm': (0.0, 1.0),      # Melanin blend
        'Bh': (0.6, 0.98),     # Blood oxygenation
        'T': (0.01, 0.25)      # Epidermis thickness
    }
    
    # Biological plausibility thresholds
    # These are literature values for normal skin
    BIOLOGICAL_RANGES = {
        'Cm': (0.013, 0.43),   # 1.3% - 43% melanin (typical)
        'Ch': (0.02, 0.07),    # 2% - 7% blood (typical, lips can be 30%)
        'Bh': (0.75, 0.98),    # 75% - 98% oxygenation (normal)
        'T': (0.05, 0.15)      # 0.5mm - 1.5mm epidermis (typical)
    }
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seed for reproducibility
    RANDOM_SEED = 42


# ============================================================================
# SKIN DETECTION
# ============================================================================

def detect_skin_pixels(image_array):
    """
    Automatic skin detection using RGB color space thresholds.
    
    WHY: We need to extract only skin pixels, excluding hair, background,
    clothing, and specular highlights.
    
    METHOD: Uses empirical RGB thresholds that work across different skin tones.
    Based on research in skin detection for face recognition.
    
    Args:
        image_array: numpy array of shape (H, W, 3) with RGB values 0-255
    
    Returns:
        mask: boolean array where True = skin pixel
    """
    # Convert to float for calculations
    img = image_array.astype(float)
    
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    # Rule 1: RGB thresholds (eliminates very dark and very bright pixels)
    rule1 = (R > 95) & (G > 40) & (B > 20)
    
    # Rule 2: Color ratios (skin has specific R-G-B relationships)
    rule2 = (np.maximum(R, np.maximum(G, B)) - np.minimum(R, np.minimum(G, B)) > 15)
    
    # Rule 3: RGB relationships (R > G > B is common for skin)
    rule3 = (np.abs(R - G) > 15) & (R > G) & (R > B)
    
    # Combine rules
    skin_mask = rule1 & rule2 & rule3
    
    return skin_mask


def sample_skin_pixels(image_path, num_pixels=100):
    """
    Load image and sample skin pixels.
    
    WHY: We need to extract representative skin RGB values from each image.
    
    Args:
        image_path: path to image file
        num_pixels: number of skin pixels to randomly sample
    
    Returns:
        pixels: numpy array of shape (num_pixels, 3) with RGB values 0-1
        skin_ratio: percentage of image that is skin (for quality check)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Detect skin
    skin_mask = detect_skin_pixels(img_array)
    
    # Calculate skin ratio (quality metric)
    skin_ratio = skin_mask.sum() / skin_mask.size
    
    # If too little skin detected, return None
    if skin_ratio < 0.05:  # Less than 5% skin
        return None, skin_ratio
    
    # Get skin pixel coordinates
    skin_coords = np.argwhere(skin_mask)
    
    # If we have fewer pixels than requested, use all
    if len(skin_coords) < num_pixels:
        sampled_coords = skin_coords
    else:
        # Randomly sample pixels
        indices = np.random.choice(len(skin_coords), num_pixels, replace=False)
        sampled_coords = skin_coords[indices]
    
    # Extract RGB values
    pixels = img_array[sampled_coords[:, 0], sampled_coords[:, 1]]
    
    # Normalize to 0-1 (your model expects this)
    pixels = pixels.astype(np.float32) / 255.0
    
    return pixels, skin_ratio

# ============================================================================
# MODEL ARCHITECTURE (Define locally to avoid import errors)
# ============================================================================


class Encoder(nn.Module):
    """Encoder network: RGB (3) → Latent Parameters (5)"""
    def __init__(self, in_dim=3, hidden_dim=256, num_layers=8, out_dim=5):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.mlp(x)
        return self.out(x)

class Decoder(nn.Module):
    """Decoder: Parameters (5) → RGB (3)"""
    def __init__(self, in_dim=5, hidden_dim=256, num_layers=8, out_dim=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.mlp(x)
        return self.out(x)
# ============================================================================
# MODEL LOADING
# ============================================================================

def load_encoder(checkpoint_path, device, num_neurons=256, num_layers=8):
    """
    Load the trained encoder model.
    
    Args:
        checkpoint_path: path to .pt checkpoint file
        device: torch device (cuda or cpu)
        num_neurons: hidden dimension (default 256)
        num_layers: number of layers (default 8)
    
    Returns:
        encoder: loaded encoder model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create encoder with specified architecture
    encoder = Encoder(
        in_dim=3, 
        hidden_dim=num_neurons, 
        num_layers=num_layers, 
        out_dim=5
    ).to(device)
    
    # Load weights (handle AutoEncoder wrapper if needed)
    try:
        # Try loading directly
        encoder.load_state_dict(state_dict)
        print(f"✓ Loaded encoder directly")
    except:
        # If checkpoint contains full AutoEncoder, extract encoder part
        encoder_state = {k.replace('encoder.', ''): v 
                        for k, v in state_dict.items() 
                        if k.startswith('encoder.')}
        
        if encoder_state:
            encoder.load_state_dict(encoder_state)
            print(f"✓ Extracted encoder from AutoEncoder checkpoint")
        else:
            raise ValueError("Could not load encoder from checkpoint")
    
    encoder.eval()
    print(f"✓ Encoder loaded successfully from {checkpoint_path}")
    print(f"✓ Architecture: {num_layers} layers, {num_neurons} neurons per layer")
    print(f"✓ Using device: {device}")
    
    return encoder

# ============================================================================
# PARAMETER PREDICTION
# ============================================================================

def predict_parameters(encoder, pixels, device):
    """
    Run encoder on skin pixels to predict parameters.
    
    WHY: This is the core inference step - converting RGB to latent parameters.
    
    Args:
        encoder: trained encoder model
        pixels: numpy array of shape (N, 3) with RGB values 0-1
        device: torch device
    
    Returns:
        params: numpy array of shape (N, 5) with predicted [Cm, Ch, Bm, Bh, T]
    """
    # Convert to tensor
    x = torch.from_numpy(pixels).float().to(device)
    
    # Predict
    with torch.no_grad():
        pred = encoder(x)
    
    # Convert back to numpy
    params = pred.cpu().numpy()
    
    return params


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_parameter_distribution(all_params, param_names, ranges, bio_ranges):
    """
    Analyze the distribution of predicted parameters.
    
    WHY: This is the core validation - checking if predictions are plausible.
    
    WHAT WE CHECK:
    1. Are values within expected ranges?
    2. Are values clustering at boundaries? (bad sign)
    3. What's the distribution shape?
    4. Are there outliers?
    5. Do values match biological expectations?
    
    Args:
        all_params: numpy array of shape (N, 5) with all predictions
        param_names: list of parameter names
        ranges: dict of expected ranges
        bio_ranges: dict of biological ranges
    
    Returns:
        stats: dictionary with detailed statistics
    """
    stats = {}
    
    for i, name in enumerate(param_names):
        param_values = all_params[:, i]
        
        # Basic statistics
        param_stats = {
            'mean': float(np.mean(param_values)),
            'std': float(np.std(param_values)),
            'min': float(np.min(param_values)),
            'max': float(np.max(param_values)),
            'median': float(np.median(param_values)),
            'q25': float(np.percentile(param_values, 25)),
            'q75': float(np.percentile(param_values, 75)),
        }
        
        # Range checks
        expected_min, expected_max = ranges[name]
        param_stats['within_expected_range'] = (
            param_stats['min'] >= expected_min and 
            param_stats['max'] <= expected_max
        )
        
        # Boundary clustering check
        # If >10% of values are within 5% of boundaries, flag it
        lower_threshold = expected_min + 0.05 * (expected_max - expected_min)
        upper_threshold = expected_max - 0.05 * (expected_max - expected_min)
        
        at_lower = np.sum(param_values < lower_threshold) / len(param_values)
        at_upper = np.sum(param_values > upper_threshold) / len(param_values)
        
        param_stats['boundary_clustering'] = {
            'lower_boundary_pct': float(at_lower * 100),
            'upper_boundary_pct': float(at_upper * 100),
            'is_problem': (at_lower > 0.1 or at_upper > 0.1)
        }
        
        # Biological plausibility check
        bio_min, bio_max = bio_ranges[name]
        within_bio = np.sum((param_values >= bio_min) & (param_values <= bio_max))
        param_stats['biological_plausibility_pct'] = float(within_bio / len(param_values) * 100)
        
        # Outlier detection (values beyond expected ranges)
        outliers = np.sum((param_values < expected_min) | (param_values > expected_max))
        param_stats['outlier_count'] = int(outliers)
        param_stats['outlier_pct'] = float(outliers / len(param_values) * 100)
        
        # Distribution shape (coefficient of variation)
        if param_stats['mean'] > 0:
            param_stats['coefficient_of_variation'] = param_stats['std'] / param_stats['mean']
        else:
            param_stats['coefficient_of_variation'] = None
        
        stats[name] = param_stats
    
    return stats


def check_parameter_correlations(all_params, param_names):
    """
    Check correlations between parameters.
    
    WHY: Some parameter combinations are physically impossible or unlikely.
    For example, very high melanin (dark skin) rarely has very high blood showing.
    
    Args:
        all_params: numpy array of shape (N, 5)
        param_names: list of parameter names
    
    Returns:
        correlation_matrix: numpy array of correlations
        problematic_correlations: list of unusual correlations
    """
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(all_params.T)
    
    # Check for problematic correlations
    problematic = []
    
    # High Cm (melanin) with high Ch (blood) is unusual
    cm_idx = param_names.index('Cm')
    ch_idx = param_names.index('Ch')
    if corr_matrix[cm_idx, ch_idx] > 0.7:
        problematic.append({
            'params': ('Cm', 'Ch'),
            'correlation': float(corr_matrix[cm_idx, ch_idx]),
            'issue': 'High melanin usually obscures blood'
        })
    
    return corr_matrix, problematic


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_distribution_plots(all_params, param_names, ranges, output_dir):
    """
    Create histogram plots for each parameter.
    
    WHY: Visual inspection is crucial for understanding distributions.
    
    Creates:
    - 5 histograms (one per parameter)
    - Expected range overlays
    - Statistical annotations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        param_values = all_params[:, i]
        
        # Histogram
        ax.hist(param_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Expected range
        expected_min, expected_max = ranges[name]
        ax.axvline(expected_min, color='green', linestyle='--', linewidth=2, 
                   label=f'Expected range: [{expected_min}, {expected_max}]')
        ax.axvline(expected_max, color='green', linestyle='--', linewidth=2)
        
        # Mean
        mean_val = np.mean(param_values)
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_val:.4f}')
        
        # Labels
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{name} Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Distribution plots saved to {output_dir}/parameter_distributions.png")


def create_correlation_heatmap(corr_matrix, param_names, output_dir):
    """Create correlation heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Ticks
    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names)
    ax.set_yticklabels(param_names)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title("Parameter Correlation Matrix", fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Correlation heatmap saved to {output_dir}/correlation_matrix.png")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(stats, corr_matrix, problematic_corrs, 
                    image_stats, output_dir):
    """
    Generate detailed text report.
    
    WHY: Provides human-readable summary of all findings.
    """
    report_path = f'{output_dir}/validation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LATENT SPACE RANGE VALIDATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images analyzed: {image_stats['total_images']}\n")
        f.write(f"Total skin pixels analyzed: {image_stats['total_pixels']}\n")
        f.write(f"Average skin detection ratio: {image_stats['avg_skin_ratio']:.2%}\n")
        f.write("\n")
        
        # Overall assessment
        f.write("="*80 + "\n")
        f.write("OVERALL ASSESSMENT\n")
        f.write("="*80 + "\n")
        
        all_within_range = all(s['within_expected_range'] for s in stats.values())
        boundary_problems = [name for name, s in stats.items() 
                            if s['boundary_clustering']['is_problem']]
        high_outliers = [name for name, s in stats.items() if s['outlier_pct'] > 5]
        
        if all_within_range and not boundary_problems and not high_outliers:
            f.write("✓ PASS: All parameters within expected ranges\n")
            f.write("✓ PASS: No significant boundary clustering\n")
            f.write("✓ PASS: Low outlier rate\n")
            f.write("\nCONCLUSION: Your model produces biologically plausible parameters!\n")
        else:
            f.write("⚠ ISSUES DETECTED:\n")
            if not all_within_range:
                f.write("  ✗ Some parameters outside expected ranges\n")
            if boundary_problems:
                f.write(f"  ✗ Boundary clustering detected in: {', '.join(boundary_problems)}\n")
            if high_outliers:
                f.write(f"  ✗ High outlier rate in: {', '.join(high_outliers)}\n")
            f.write("\nCONCLUSION: Your model needs investigation. See details below.\n")
        
        f.write("\n")
        
        # Parameter-by-parameter analysis
        f.write("="*80 + "\n")
        f.write("PARAMETER-BY-PARAMETER ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        param_names = ['Cm', 'Ch', 'Bm', 'Bh', 'T']
        param_descriptions = {
            'Cm': 'Melanin Concentration (0.1%-50%)',
            'Ch': 'Blood Concentration (0.1%-32%)',
            'Bm': 'Melanin Blend (0-1)',
            'Bh': 'Blood Oxygenation (60%-98%)',
            'T': 'Epidermis Thickness (0.1-2.5mm)'
        }
        
        for name in param_names:
            s = stats[name]
            f.write(f"{name} - {param_descriptions[name]}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Mean:              {s['mean']:.6f}\n")
            f.write(f"  Std Dev:           {s['std']:.6f}\n")
            f.write(f"  Min:               {s['min']:.6f}\n")
            f.write(f"  Max:               {s['max']:.6f}\n")
            f.write(f"  Median:            {s['median']:.6f}\n")
            f.write(f"  25th Percentile:   {s['q25']:.6f}\n")
            f.write(f"  75th Percentile:   {s['q75']:.6f}\n")
            f.write(f"\n")
            f.write(f"  Within Expected Range:     {'✓ YES' if s['within_expected_range'] else '✗ NO'}\n")
            f.write(f"  Biological Plausibility:   {s['biological_plausibility_pct']:.1f}%\n")
            f.write(f"  Outlier Percentage:        {s['outlier_pct']:.2f}%\n")
            f.write(f"\n")
            f.write(f"  Boundary Clustering:\n")
            f.write(f"    Lower boundary: {s['boundary_clustering']['lower_boundary_pct']:.1f}%\n")
            f.write(f"    Upper boundary: {s['boundary_clustering']['upper_boundary_pct']:.1f}%\n")
            f.write(f"    Problem? {'✗ YES - Too much clustering!' if s['boundary_clustering']['is_problem'] else '✓ NO'}\n")
            f.write(f"\n")
            
            # Interpretation
            f.write(f"  INTERPRETATION:\n")
            if s['boundary_clustering']['is_problem']:
                f.write(f"    ⚠ Values clustering at boundaries suggests your ranges may be\n")
                f.write(f"      too restrictive or the model hasn't learned this parameter well.\n")
            if s['outlier_pct'] > 5:
                f.write(f"    ⚠ High outlier rate suggests training data doesn't match real images.\n")
            if s['biological_plausibility_pct'] < 80:
                f.write(f"    ⚠ Low biological plausibility - predictions don't match literature values.\n")
            if (s['within_expected_range'] and not s['boundary_clustering']['is_problem'] 
                and s['outlier_pct'] < 5):
                f.write(f"    ✓ This parameter looks good!\n")
            f.write("\n\n")
        
        # Correlation analysis
        f.write("="*80 + "\n")
        f.write("PARAMETER CORRELATION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Correlation Matrix:\n")
        f.write("     " + "  ".join(f"{name:>6}" for name in param_names) + "\n")
        for i, name1 in enumerate(param_names):
            f.write(f"{name1:>4} ")
            for j in range(len(param_names)):
                f.write(f"{corr_matrix[i,j]:>6.2f} ")
            f.write("\n")
        f.write("\n")
        
        if problematic_corrs:
            f.write("⚠ PROBLEMATIC CORRELATIONS:\n")
            for corr in problematic_corrs:
                f.write(f"  {corr['params'][0]} vs {corr['params'][1]}: "
                       f"{corr['correlation']:.2f} - {corr['issue']}\n")
        else:
            f.write("✓ No problematic correlations detected\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        if boundary_problems:
            f.write("1. BOUNDARY CLUSTERING DETECTED:\n")
            f.write(f"   Parameters affected: {', '.join(boundary_problems)}\n")
            f.write("   Recommended actions:\n")
            f.write("   - Expand parameter ranges in Monte Carlo simulation\n")
            f.write("   - Retrain with expanded ranges\n")
            f.write("   - Check if model architecture has sufficient capacity\n\n")
        
        if high_outliers:
            f.write("2. HIGH OUTLIER RATE:\n")
            f.write(f"   Parameters affected: {', '.join(high_outliers)}\n")
            f.write("   Recommended actions:\n")
            f.write("   - Your synthetic data may not cover real-world diversity\n")
            f.write("   - Consider expanding parameter ranges\n")
            f.write("   - Add more diverse samples to training data\n\n")
        
        low_bio = [name for name, s in stats.items() 
                  if s['biological_plausibility_pct'] < 80]
        if low_bio:
            f.write("3. LOW BIOLOGICAL PLAUSIBILITY:\n")
            f.write(f"   Parameters affected: {', '.join(low_bio)}\n")
            f.write("   Recommended actions:\n")
            f.write("   - Review Monte Carlo simulation parameters\n")
            f.write("   - Compare against Leeds dataset spectral measurements\n")
            f.write("   - Verify your parameter ranges match literature\n\n")
        
        if not boundary_problems and not high_outliers and all(
            s['biological_plausibility_pct'] >= 80 for s in stats.values()):
            f.write("✓ Your model performs well! Ready for next validation step:\n")
            f.write("  - Proceed to Delta E (perceptual accuracy) testing\n")
            f.write("  - Test biophysical heuristics (lips, hair, etc.)\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Detailed report saved to {report_path}")
    return report_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main validation pipeline"""
    
    print("\n" + "="*80)
    print("LATENT SPACE RANGE VALIDATION")
    print("="*80 + "\n")
    
    # Set random seed
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {Config.OUTPUT_DIR}\n")
    
    # Step 1: Load encoder
    print("Step 1: Loading encoder model...")
    encoder = load_encoder(Config.CHECKPOINT_PATH, Config.DEVICE)
    print()
    
    # Step 2: Get image files
    print("Step 2: Scanning image directory...")
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(Path(Config.IMAGE_FOLDER).glob(ext))
    
    print(f"✓ Found {len(image_files)} images")
    
    # Randomly select images
    if len(image_files) > Config.NUM_IMAGES:
        selected_images = random.sample(image_files, Config.NUM_IMAGES)
    else:
        selected_images = image_files
    
    print(f"✓ Selected {len(selected_images)} images for analysis\n")
    
    # Step 3: Process images
    # Step 3: Process images
    print("Step 3: Processing images and extracting skin pixels...")
    all_params = []
    valid_images = 0
    total_pixels = 0
    skin_ratios = []

    for i, img_path in enumerate(selected_images):
        # ADD THESE DETAILED PRINTS:
        print(f"\n{'='*60}")
        print(f"Processing image {i+1}/{len(selected_images)}: {img_path.name}")
        
        # Check image size BEFORE loading
        import time
        from PIL import Image
        
        start_total = time.time()
        
        # Print file size
        file_size_mb = img_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        
        # Print image dimensions
        temp_img = Image.open(img_path)
        print(f"  Image dimensions: {temp_img.size[0]} x {temp_img.size[1]}")
        temp_img.close()
        
        # Extract skin pixels
        print(f"  Starting skin detection...")
        start_skin = time.time()
        pixels, skin_ratio = sample_skin_pixels(img_path, Config.PIXELS_PER_IMAGE)
        skin_time = time.time() - start_skin
        print(f"  Skin detection took: {skin_time:.2f} seconds")
        
        if pixels is None:
            print("  ✗ Insufficient skin detected, skipping")
            continue
        
        print(f"  ✓ Found {skin_ratio:.1%} skin pixels")
        print(f"  Sampled {len(pixels)} pixels")
        
        # Predict parameters
        print(f"  Running encoder...")
        start_pred = time.time()
        params = predict_parameters(encoder, pixels, Config.DEVICE)
        pred_time = time.time() - start_pred
        print(f"  Prediction took: {pred_time:.2f} seconds")
        
        total_time = time.time() - start_total
        print(f"  TOTAL TIME FOR THIS IMAGE: {total_time:.2f} seconds")
        print(f"{'='*60}")
        
        all_params.append(params)
        valid_images += 1
        total_pixels += len(pixels)
        skin_ratios.append(skin_ratio)
    
    # Step 4: Statistical analysis
    print("Step 4: Analyzing parameter distributions...")
    param_names = ['Cm', 'Ch', 'Bm', 'Bh', 'T']
    
    stats = analyze_parameter_distribution(
        all_params, 
        param_names,
        Config.PARAM_RANGES,
        Config.BIOLOGICAL_RANGES
    )
    
    corr_matrix, problematic_corrs = check_parameter_correlations(
        all_params, 
        param_names
    )
    
    print("✓ Statistical analysis complete\n")
    
    # Step 5: Visualization
    print("Step 5: Creating visualizations...")
    create_distribution_plots(
        all_params, 
        param_names, 
        Config.PARAM_RANGES,
        Config.OUTPUT_DIR
    )
    
    create_correlation_heatmap(
        corr_matrix, 
        param_names,
        Config.OUTPUT_DIR
    )
    print()
    
    # Step 6: Generate report
    print("Step 6: Generating detailed report...")
    image_stats = {
        'total_images': valid_images,
        'total_pixels': total_pixels,
        'avg_skin_ratio': np.mean(skin_ratios)
    }
    
    report_path = generate_report(
        stats, 
        corr_matrix, 
        problematic_corrs,
        image_stats,
        Config.OUTPUT_DIR
    )
    
    # Save statistics as JSON for programmatic access
    json_path = f'{Config.OUTPUT_DIR}/validation_statistics.json'
    with open(json_path, 'w') as f:
        json.dump({
            'statistics': stats,
            'image_stats': image_stats,
            'correlation_matrix': corr_matrix.tolist(),
            'problematic_correlations': problematic_corrs
        }, f, indent=4)
    print(f"✓ JSON statistics saved to {json_path}\n")
    
    # Final summary
    print("="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {Config.OUTPUT_DIR}/")
    print(f"  - validation_report.txt (detailed text report)")
    print(f"  - validation_statistics.json (structured data)")
    print(f"  - parameter_distributions.png (histograms)")
    print(f"  - correlation_matrix.png (correlation heatmap)")
    print(f"\nNext step: Read the validation report to assess your model's performance")
    print("="*80 + "\n")

