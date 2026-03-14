"""
Deep Albedo — Image Utilities

Image I/O, preprocessing, and skin-pixel detection.
Used by both inference.py and latent_space_validation.py.
"""

import numpy as np
import cv2
import rawpy
from pathlib import Path
from PIL import Image


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_image_any(image_path):
    """
    Read a standard image (JPG/PNG/TIFF) or Canon CR3 RAW file.
    Returns an RGB uint8 array [0, 255].
    """
    path = Path(image_path)
    if path.suffix.lower() == ".cr3":
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
        rgb = np.power(np.clip(rgb.astype(np.float32) / 65535.0, 0, 1), 1 / 2.2)
        return (rgb * 255).astype(np.uint8)

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def crop_center_ratio(image, ratio=0.8):
    """Centre-crop image to `ratio` of its smaller dimension."""
    h, w = image.shape[:2]
    size = int(min(h, w) * ratio)
    cy, cx = h // 2, w // 2
    half = size // 2
    return image[cy - half:cy + half, cx - half:cx + half]


# ── Skin detection ────────────────────────────────────────────────────────────

def detect_skin_pixels(image_array):
    """
    Return a boolean mask (H, W) where True = likely skin pixel.

    Uses empirical RGB thresholds that generalise across skin tones.
    image_array: uint8 (H, W, 3) RGB.
    """
    img = image_array.astype(float)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    rule1 = (R > 95) & (G > 40) & (B > 20)
    rule2 = (np.maximum(R, np.maximum(G, B)) - np.minimum(R, np.minimum(G, B))) > 15
    rule3 = (np.abs(R - G) > 15) & (R > G) & (R > B)

    return rule1 & rule2 & rule3


def sample_skin_pixels(image_path, num_pixels=100):
    """
    Load an image and randomly sample `num_pixels` skin pixels.

    Returns:
        pixels    — (N, 3) float32 in [0, 1], or None if < 5 % skin
        skin_ratio — fraction of total pixels classified as skin
    """
    img_array  = np.array(Image.open(image_path).convert('RGB'))
    mask       = detect_skin_pixels(img_array)
    skin_ratio = mask.sum() / mask.size

    if skin_ratio < 0.05:
        return None, skin_ratio

    coords = np.argwhere(mask)
    if len(coords) > num_pixels:
        coords = coords[np.random.choice(len(coords), num_pixels, replace=False)]

    pixels = img_array[coords[:, 0], coords[:, 1]].astype(np.float32) / 255.0
    return pixels, skin_ratio
