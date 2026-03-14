"""
Deep Albedo — Inference

Encodes images to skin parameters (Cm, Ch, Bm, Bh, T) and decodes them
back to RGB using the trained autoencoder.

Usage:
    python inference.py --input path/to/image.jpg
    python inference.py --input path/to/folder/
    python inference.py --input path/to/folder/ --output results/
    python inference.py --input image.jpg --checkpoint path/to/model.pt
    python inference.py --input image.jpg --cpu
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model       import Encoder, Decoder, AutoEncoder
from image_utils import read_image_any, crop_center_ratio
from io_utils    import save_results, save_summary_report

import cv2


# ── Core encode / decode ──────────────────────────────────────────────────────

def encode(image, encoder, device):
    """
    Encode an RGB image to skin parameter maps.

    Args:
        image:   (H, W, 3) float32 normalised to [0, 1]
        encoder: Encoder in eval mode
        device:  torch.device

    Returns:
        parameter_maps: (H*W, 5) float32
        elapsed:        seconds
        dimensions:     (H, W)
    """
    if len(image.shape) == 2:
        H = W = int(math.sqrt(image.shape[0]))
    else:
        H, W = image.shape[0], image.shape[1]

    pixels = np.asarray(image, dtype="float32").reshape(H * W, 3)

    import time
    start = time.time()
    encoder.eval()
    with torch.no_grad():
        pred = encoder(torch.from_numpy(pixels).to(device))
    elapsed = time.time() - start

    return pred.cpu().numpy().reshape(H * W, 5), elapsed, (H, W)


def decode(encoded, decoder, device):
    """
    Decode skin parameter maps back to RGB.

    Args:
        encoded: (H*W, 5) or (H, W, 5) float32
        decoder: Decoder in eval mode
        device:  torch.device

    Returns:
        recovered:  (H, W, 3) float32
        elapsed:    seconds
        dimensions: (H, W)
    """
    if len(encoded.shape) == 2:
        H = W = int(math.sqrt(encoded.shape[0]))
    else:
        H, W = encoded.shape[0], encoded.shape[1]

    flat = np.asarray(encoded, dtype="float32").reshape(H * W, 5)

    import time
    start = time.time()
    decoder.eval()
    with torch.no_grad():
        recovered = decoder(torch.from_numpy(flat).to(device))
    elapsed = time.time() - start

    return recovered.cpu().numpy().reshape(H, W, 3), elapsed, (H, W)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_single_image(image_path, encoder, decoder, device,
                         output_dir=None, target_size=(256, 256),
                         save=True, use_crop=False):
    """
    Read → (crop) → resize → encode → decode → (save).

    Returns a dict with keys:
        image_path, original, recovered, parameter_maps,
        dimensions, encode_time, decode_time, total_time
    """
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import preprocess

    image_rgb = read_image_any(image_path)

    if use_crop:
        try:
            image_rgb = preprocess.crop_face(image_rgb)[0]
        except Exception as e:
            print(f"Warning: crop failed ({e}), using full image.")

    image_rgb = cv2.resize(image_rgb, target_size).astype("float32") / 255.0

    parameter_maps, enc_time, dims = encode(image_rgb, encoder, device)
    recovered, dec_time, _         = decode(parameter_maps, decoder, device)

    results = {
        'image_path':     str(image_path),
        'original':       image_rgb,
        'recovered':      recovered,
        'parameter_maps': parameter_maps,
        'dimensions':     dims,
        'encode_time':    enc_time,
        'decode_time':    dec_time,
        'total_time':     enc_time + dec_time,
    }

    if output_dir and save:
        save_results(image_path, image_rgb, recovered, parameter_maps,
                     dims, enc_time, dec_time, output_dir)

    return results


def process_folder(input_folder, encoder, decoder, device,
                   output_dir=None, target_size=(256, 256)):
    """Process every image in a folder. Returns list of result dicts."""
    input_path = Path(input_folder)
    supported  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.cr3'}

    if output_dir:
        output_dir = Path(output_dir) / input_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults will be saved to: {output_dir}/")

    image_files = sorted(p for p in input_path.iterdir()
                         if p.is_file() and p.suffix.lower() in supported)

    if not image_files:
        print(f"No images found in {input_folder}")
        return []

    print(f"Found {len(image_files)} images")
    results, failed = [], []

    for img_path in tqdm(image_files, desc="Processing"):
        try:
            results.append(
                process_single_image(img_path, encoder, decoder, device,
                                     output_dir=output_dir, target_size=target_size)
            )
        except Exception as e:
            print(f"✗ {img_path.name}: {e}")
            failed.append({'file': img_path.name, 'error': str(e)})

    if output_dir and results:
        save_summary_report(results, failed, output_dir, input_folder)

    return results


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    """Load encoder + decoder from a checkpoint, auto-detecting architecture."""
    print(f"\nLoading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    def _load(prefix, cls):
        s      = {k.replace(prefix + '.', ''): v
                  for k, v in state_dict.items() if k.startswith(prefix + '.')}
        hidden = s['mlp.0.weight'].shape[0]
        layers = sum(1 for k in s if k.startswith('mlp.') and k.endswith('.weight'))
        m = cls(hidden_dim=hidden, num_layers=layers).to(device)
        m.load_state_dict(s)
        m.eval()
        return m, hidden, layers

    encoder, eh, el = _load('encoder', Encoder)
    decoder, dh, dl = _load('decoder', Decoder)

    print(f"✓ encoder {eh}×{el}  decoder {dh}×{dl}  "
          f"epoch {checkpoint.get('epoch', 'N/A')}  "
          f"loss {checkpoint.get('best_train_loss', 'N/A')}")
    return encoder, decoder


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Deep Albedo — Extract skin parameters from images')
    parser.add_argument('--input',      '-i', required=True,
                        help='Image file or folder')
    parser.add_argument('--output',     '-o', default='output',
                        help='Output directory (default: output/)')
    parser.add_argument('--checkpoint', '-c',
                        default='checkpoints/2026-03-14_14-26-36/best.pt',
                        help='Path to checkpoint .pt file')
    parser.add_argument('--size', '-s', type=int, nargs=2, default=[256, 256],
                        help='Target image size: width height')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()

    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    try:
        encoder, decoder = load_model(args.checkpoint, device)
    except FileNotFoundError:
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    input_path  = Path(args.input)
    target_size = tuple(args.size)

    if not input_path.exists():
        print(f"✗ Input path does not exist: {args.input}")
        return

    if input_path.is_file():
        r = process_single_image(input_path, encoder, decoder, device,
                                 output_dir=args.output, target_size=target_size)
        print(f"\n✓ encode {r['encode_time']:.4f}s  "
              f"decode {r['decode_time']:.4f}s  "
              f"total {r['total_time']:.4f}s")

    elif input_path.is_dir():
        results = process_folder(input_path, encoder, decoder, device,
                                 output_dir=args.output, target_size=target_size)
        if results:
            avg = np.mean([r['total_time'] for r in results])
            print(f"\n✓ {len(results)} images  avg {avg:.4f}s/image")
        else:
            print("✗ No images processed")
    else:
        print("✗ Input must be a file or directory")


if __name__ == '__main__':
    main()
