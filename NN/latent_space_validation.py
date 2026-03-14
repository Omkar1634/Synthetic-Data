"""
Deep Albedo — Latent Space Range Validation

Validates whether the trained encoder produces biologically plausible skin
parameters on real face images.

What it checks:
  ✓ Values fall within expected biological ranges
  ✓ No boundary clustering (model isn't just outputting min/max)
  ✓ Meaningful variation across skin tones
  ✗ Values stuck at boundaries
  ✗ All images produce the same parameters
  ✗ Outliers / negative values

Usage:
    Edit Config paths below, then:
    python latent_space_validation.py
"""

import os
import sys
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model        import Encoder
from image_utils  import sample_skin_pixels
from metrics      import analyze_parameter_distribution, check_parameter_correlations
from plots        import create_distribution_plots, create_correlation_heatmap
from io_utils     import generate_validation_report


# ── Configuration ─────────────────────────────────────────────────────────────

class Config:
    # Paths — edit before running
    IMAGE_FOLDER    = "path/to/your/images"           # CHANGE THIS
    CHECKPOINT_PATH = "checkpoints/your_run/best.pt"  # CHANGE THIS
    OUTPUT_DIR      = "validation_results"

    # Sampling
    NUM_IMAGES       = 30    # random images to analyse
    PIXELS_PER_IMAGE = 100   # skin pixels sampled per image

    # Expected ranges from the v6 Monte Carlo LUT (Cm, Ch, Bm, Bh, T)
    PARAM_RANGES = {
        'Cm': (0.05,  0.50),
        'Ch': (0.02,  0.20),
        'Bm': (0.0,   1.0),
        'Bh': (0.60,  0.98),
        'T':  (0.005, 0.020),
    }

    # Literature-based biological plausibility ranges
    BIOLOGICAL_RANGES = {
        'Cm': (0.013, 0.43),
        'Ch': (0.02,  0.07),
        'Bh': (0.75,  0.98),
        'T':  (0.005, 0.015),
    }

    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42


# ── Model loading ─────────────────────────────────────────────────────────────

def load_encoder(checkpoint_path, device):
    """Load encoder from checkpoint, auto-detecting architecture."""
    ckpt       = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    enc_state  = {k.replace("encoder.", ""): v
                  for k, v in state_dict.items() if k.startswith("encoder.")}
    hidden_dim = enc_state['mlp.0.weight'].shape[0]
    num_layers = sum(1 for k in enc_state
                     if k.startswith('mlp.') and k.endswith('.weight'))

    encoder = Encoder(in_dim=3, hidden_dim=hidden_dim,
                      num_layers=num_layers, out_dim=5).to(device)
    encoder.load_state_dict(enc_state)
    encoder.eval()
    return encoder


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_parameters(encoder, pixels, device):
    """Run encoder on skin pixels. Returns (N, 5) numpy array."""
    with torch.no_grad():
        pred = encoder(torch.from_numpy(pixels).float().to(device))
    return pred.cpu().numpy()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"Output: {Config.OUTPUT_DIR}\n")

    encoder = load_encoder(Config.CHECKPOINT_PATH, Config.DEVICE)

    image_files = [p for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG')
                   for p in Path(Config.IMAGE_FOLDER).glob(ext)]
    print(f"Found {len(image_files)} images")

    if len(image_files) > Config.NUM_IMAGES:
        image_files = random.sample(image_files, Config.NUM_IMAGES)

    all_params_list, skin_ratios = [], []

    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {img_path.name}  "
              f"({img_path.stat().st_size / 1e6:.1f} MB)")

        t0 = time.time()
        pixels, skin_ratio = sample_skin_pixels(img_path, Config.PIXELS_PER_IMAGE)

        if pixels is None:
            print("  ✗ Insufficient skin, skipping")
            continue

        params = predict_parameters(encoder, pixels, Config.DEVICE)
        print(f"  skin={skin_ratio:.1%}  pixels={len(pixels)}  "
              f"t={time.time() - t0:.2f}s")

        all_params_list.append(params)
        skin_ratios.append(skin_ratio)

    if not all_params_list:
        print("No valid images processed.")
        return

    all_params  = np.concatenate(all_params_list, axis=0)
    param_names = ['Cm', 'Ch', 'Bm', 'Bh', 'T']

    stats        = analyze_parameter_distribution(
        all_params, param_names, Config.PARAM_RANGES, Config.BIOLOGICAL_RANGES)
    corr, issues = check_parameter_correlations(all_params, param_names)

    create_distribution_plots(all_params, param_names,
                               Config.PARAM_RANGES, Config.OUTPUT_DIR)
    create_correlation_heatmap(corr, param_names, Config.OUTPUT_DIR)

    image_stats = {
        'total_images':   len(all_params_list),
        'total_pixels':   len(all_params),
        'avg_skin_ratio': float(np.mean(skin_ratios)),
    }

    generate_validation_report(stats, corr, issues, image_stats, Config.OUTPUT_DIR)

    json_path = f'{Config.OUTPUT_DIR}/validation_statistics.json'
    with open(json_path, 'w') as f:
        json.dump({
            'statistics':               stats,
            'image_stats':              image_stats,
            'correlation_matrix':       corr.tolist(),
            'problematic_correlations': issues,
        }, f, indent=4)
    print(f"✓ JSON stats → {json_path}")


if __name__ == '__main__':
    main()
