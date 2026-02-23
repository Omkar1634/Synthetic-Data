"""
Deep Albedo - Inference Script
Processes images to extract skin parameters (Cm, Ch, Bm, Bh, T)

Usage:
    python inference.py --input path/to/image.jpg
    python inference.py --input path/to/folder/
    python inference.py --input path/to/folder/ --output results/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import cv2
import rawpy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import math
from pathlib import Path
from tqdm import tqdm
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import preprocess
from utils import plotting


# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================

NUM_NEURONS = 75
NUM_LAYERS = 2

class Encoder(nn.Module):
    """Encoder: RGB (3D) -> Parameters (5D)"""
    def __init__(self, in_dim=3, hidden_dim=NUM_NEURONS, num_layers=NUM_LAYERS, out_dim=5):
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
    """Decoder: Parameters (5D) -> RGB (3D)"""
    def __init__(self, in_dim=5, hidden_dim=NUM_NEURONS, num_layers=NUM_LAYERS, out_dim=3):
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


class AutoEncoder(nn.Module):
    """Combined AutoEncoder model"""
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_in, decoder_in, end_to_end_in):
        enc_out = self.encoder(encoder_in)
        dec_out = self.decoder(decoder_in)
        end_out = self.decoder(self.encoder(end_to_end_in))
        return enc_out, dec_out, end_out


# =====================================================================
# INFERENCE FUNCTIONS
# =====================================================================

def encode(image, encoder, device):
    """
    Encode RGB image to parameter maps
    
    Args:
        image: numpy array (H, W, 3) normalized to [0, 1]
        encoder: PyTorch Encoder model
        device: torch device
    
    Returns:
        parameter_maps: numpy array (H*W, 5)
        elapsed: inference time in seconds
        dimensions: (WIDTH, HEIGHT)
    """
    # Handle 2D flattened input vs image input
    if len(image.shape) == 2:
        WIDTH = HEIGHT = int(math.sqrt(image.shape[0]))
        image = np.asarray(image).reshape(-1, 4).astype("float32")
    else:
        WIDTH = image.shape[0]
        HEIGHT = image.shape[1]
        image = np.asarray(image).astype("float32")

    # match your Keras reshape
    image = image.reshape(WIDTH * HEIGHT, 3)

    start = time.time()
    print(f"Image shape before encoder inference: {image.shape}")

    # Convert to torch tensor and run model inference
    x = torch.from_numpy(image).to(device)

    encoder.eval()
    with torch.no_grad():
        pred_maps = encoder(x)

    # Convert back to numpy
    pred_maps = pred_maps.detach().cpu().numpy()

    end = time.time()
    elapsed = end - start

    # reshape output to (WIDTH*HEIGHT, 5)
    pred_maps = pred_maps.reshape(WIDTH * HEIGHT, 5)

    return pred_maps, elapsed, (WIDTH, HEIGHT)



def decode(encoded, decoder, device):
    """
    Decode parameter maps to RGB image
    
    Args:
        encoded: numpy array (H*W, 5) or (H, W, 5)
        decoder: PyTorch Decoder model
        device: torch device
    
    Returns:
        recovered: numpy array (H, W, 3)
        elapsed: inference time in seconds
        dimensions: (WIDTH, HEIGHT)
    """
    if len(encoded.shape) == 2:
        WIDTH = HEIGHT = int(math.sqrt(encoded.shape[0]))
        encoded = np.asarray(encoded).reshape(-1, 5).astype("float32")
    else:
        WIDTH = encoded.shape[0]
        HEIGHT = encoded.shape[1]
        encoded = np.asarray(encoded).astype("float32")

    start = time.time()
    
    # Convert to torch tensor and run inference
    x = torch.from_numpy(encoded).to(device)
    
    decoder.eval()
    with torch.no_grad():
        recovered = decoder(x)
    
    recovered = recovered.detach().cpu().numpy()
    
    elapsed = time.time() - start
    
    # Reshape output to RGB image
    recovered = recovered.reshape(WIDTH, HEIGHT, 3)
    
    return recovered, elapsed, (WIDTH, HEIGHT)


def normalize_parameters(parameter_maps):
    """
    Normalize parameter maps to proper ranges
    
    Args:
        parameter_maps: numpy array (H*W, 5)
    
    Returns:
        normalized parameter_maps
    """
    Cm = parameter_maps[:, 0]
    Ch = parameter_maps[:, 1]
    Bm = parameter_maps[:, 2]
    Bh = parameter_maps[:, 3]
    T  = parameter_maps[:, 4]
    
    # Normalize to [0, 1]
    def norm01(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    
    Cm = norm01(Cm)
    Ch = norm01(Ch)
    Bm = norm01(Bm)
    Bh = norm01(Bh)
    T  = norm01(T)
    
    # Scale to proper ranges
    parameter_maps[:, 0] = Cm * 0.62 + 0.001
    parameter_maps[:, 1] = Ch * 0.31 + 0.001
    parameter_maps[:, 2] = Bm * 0.8  + 0.2
    parameter_maps[:, 3] = Bh * 0.3  + 0.6
    parameter_maps[:, 4] = T  * 0.2  + 0.05
    
    return parameter_maps


# =====================================================================
# RESULT SAVING FUNCTIONS
# =====================================================================

def save_results(image_path, original, recovered, parameter_maps, dimensions, 
                encode_time, decode_time, output_dir):
    """
    Save all results in an organized folder structure
    
    Creates:
        output_dir/
        ├── visualizations/     # Combined analysis plots
        ├── recovered/          # Reconstructed images
        ├── parameter_maps/     # Individual parameter maps
        │   ├── Cm/
        │   ├── Ch/
        │   ├── Bm/
        │   ├── Bh/
        │   └── T/
        └── data/              # Raw numpy arrays
    """
    output_path = Path(output_dir)
    image_name = Path(image_path).stem
    
    # Create folder structure
    viz_dir = output_path / "visualizations"
    recovered_dir = output_path / "recovered"
    param_dir = output_path / "parameter_maps"
    data_dir = output_path / "data"
    
    for directory in [viz_dir, recovered_dir, param_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each parameter
    param_names = ['Cm', 'Ch', 'Bm', 'Bh', 'T']
    for param_name in param_names:
        (param_dir / param_name).mkdir(parents=True, exist_ok=True)
    
    # 1. Save complete visualization
    plt.style.use("dark_background")
    plt.rcParams["axes.grid"] = False
    
    viz_path = viz_dir / f"{image_name}_analysis.png"
    plotting.PLOT_TEX_MAPS(
        recovered, 
        parameter_maps,
        title=f"Analysis: {image_name}",
        save=True,
        text_below=f"Encode: {encode_time:.4f}s | Decode: {decode_time:.4f}s | Total: {encode_time+decode_time:.4f}s"
    )
    
    # Move the generated plot to correct location
    temp_plot_name = f"tex_maps_Analysis: {image_name}.png"
    if Path(temp_plot_name).exists():
        import shutil
        shutil.move(temp_plot_name, str(viz_path))
    
    plt.close('all')
    
    # 2. Save recovered image
    recovered_rgb = (recovered * 255).astype(np.uint8)
    recovered_bgr = cv2.cvtColor(recovered_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(recovered_dir / f"{image_name}_recovered.png"), recovered_bgr)
    
    # 3. Save individual parameter maps
    pm_reshaped = parameter_maps.reshape(dimensions[0], dimensions[1], 5)
    
    for i, param_name in enumerate(param_names):
        param_map = pm_reshaped[:, :, i]
        
        # Save as grayscale image (normalized to 0-255)
        param_normalized = ((param_map - param_map.min()) / 
                           (param_map.max() - param_map.min() + 1e-8) * 255).astype(np.uint8)
        
        param_img_path = param_dir / param_name / f"{image_name}_{param_name}.png"
        cv2.imwrite(str(param_img_path), param_normalized)
        
        # Also save with colormap
        param_colored_path = param_dir / param_name / f"{image_name}_{param_name}_colored.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(param_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'{param_name} - {image_name}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(param_colored_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    # 4. Save raw data as numpy arrays
    np.save(data_dir / f"{image_name}_parameters.npy", parameter_maps)
    np.save(data_dir / f"{image_name}_original.npy", original)
    np.save(data_dir / f"{image_name}_recovered.npy", recovered)
    
    # 5. Save metadata as JSON
    import json
    metadata = {
        'image_name': image_name,
        'image_path': str(image_path),
        'dimensions': dimensions,
        'encode_time': float(encode_time),
        'decode_time': float(decode_time),
        'total_time': float(encode_time + decode_time),
        'parameters': {
            'Cm': {'min': float(pm_reshaped[:, :, 0].min()), 'max': float(pm_reshaped[:, :, 0].max()), 'mean': float(pm_reshaped[:, :, 0].mean())},
            'Ch': {'min': float(pm_reshaped[:, :, 1].min()), 'max': float(pm_reshaped[:, :, 1].max()), 'mean': float(pm_reshaped[:, :, 1].mean())},
            'Bm': {'min': float(pm_reshaped[:, :, 2].min()), 'max': float(pm_reshaped[:, :, 2].max()), 'mean': float(pm_reshaped[:, :, 2].mean())},
            'Bh': {'min': float(pm_reshaped[:, :, 3].min()), 'max': float(pm_reshaped[:, :, 3].max()), 'mean': float(pm_reshaped[:, :, 3].mean())},
            'T':  {'min': float(pm_reshaped[:, :, 4].min()), 'max': float(pm_reshaped[:, :, 4].max()), 'mean': float(pm_reshaped[:, :, 4].mean())}
        }
    }
    
    with open(data_dir / f"{image_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    pass
    
    # print(f"  ✓ Saved to: {output_dir}/")
    # print(f"    - Visualization: visualizations/{image_name}_analysis.png")
    # print(f"    - Recovered: recovered/{image_name}_recovered.png")
    # print(f"    - Parameters: parameter_maps/[Cm,Ch,Bm,Bh,T]/{image_name}_*.png")
    # print(f"    - Raw data: data/{image_name}_*.npy")
    # print(f"    - Metadata: data/{image_name}_metadata.json")


# =====================================================================
# IMAGE PROCESSING PIPELINE
# =====================================================================

def crop_center_ratio(image, ratio=0.8):
    """
    Center-crop an image by a given ratio.

    Args:
        image: RGB image (H, W, 3)
        ratio: fraction of min(H, W) to keep

    Returns:
        Cropped image
    """
    h, w, _ = image.shape
    size = int(min(h, w) * ratio)

    cy, cx = h // 2, w // 2
    half = size // 2

    return image[
        cy - half: cy + half,
        cx - half: cx + half
    ]


def read_image_any(image_path):
    """
    Read standard images + Canon CR3 RAW files
    Returns RGB image in uint8 [0,255]
    """
    suffix = image_path.suffix.lower()

    if suffix == ".cr3":
        with rawpy.imread(str(image_path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=16
            )

        # Normalize 16-bit RAW → 8-bit RGB
        rgb = rgb.astype(np.float32) / 65535.0
        rgb = np.clip(rgb, 0, 1)

        # Gamma correction (important for JPG-trained models)
        rgb = np.power(rgb, 1 / 2.2)

        return (rgb * 255).astype(np.uint8)

    # Standard images (JPG / PNG / TIFF)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def process_single_image(image_path, encoder, decoder, device, 
                        output_dir=None, target_size=(256, 256), 
                        show_plot=True, save_plot=True,
                        use_crop=False):

    """
    Process a single image through the pipeline
    
    Args:
        image_path: path to input image
        encoder: PyTorch Encoder model
        decoder: PyTorch Decoder model
        device: torch device
        output_dir: directory to save results (optional)
        target_size: (width, height) for resizing
        show_plot: whether to display the plot
        save_plot: whether to save the plot
    
    Returns:
        dict with processing results
    """
    # Read and preprocess image
    image_rgb = read_image_any(Path(image_path))

    
    # Optional crop
    if use_crop:
        try:
            image_rgb = preprocess.crop_face(image_rgb)[0]
        except Exception as e:
            print(f"Warning: Crop failed for {image_path}, using full image. Error: {e}")

    
    # Resize
    image_rgb = cv2.resize(image_rgb, target_size)
    
    # Normalize to [0, 1]
    image_rgb = np.asarray(image_rgb).astype("float32") / 255.0
    
    # Encode
    parameter_maps, enc_time, dimensions = encode(image_rgb, encoder, device)
    
    # Normalize parameters
    parameter_maps = normalize_parameters(parameter_maps)
    
    # Decode
    pm_reshaped = parameter_maps.copy().reshape(dimensions[0], dimensions[1], 5)
    recovered, decode_time, _ = decode(pm_reshaped.reshape(-1, 5), decoder, device)
    
    # Prepare results
    results = {
        'image_path': str(image_path),
        'original': image_rgb,
        'recovered': recovered,
        'parameter_maps': parameter_maps,
        'dimensions': dimensions,
        'encode_time': enc_time,
        'decode_time': decode_time,
        'total_time': enc_time + decode_time
    }
    
    # Save results to organized folder structure
    if output_dir and save_plot:
        save_results(
            image_path=image_path,
            original=image_rgb,
            recovered=recovered,
            parameter_maps=parameter_maps,
            dimensions=dimensions,
            encode_time=enc_time,
            decode_time=decode_time,
            output_dir=output_dir
        )
    
    # Show visualization
    # if show_plot:
    #     plt.style.use("dark_background")
    #     plt.rcParams["axes.grid"] = False
        
    #     plotting.PLOT_TEX_MAPS(
    #         recovered, 
    #         parameter_maps,
    #         title=f"Analysis: {Path(image_path).stem}",
    #         save=False,
    #         text_below=f"Encode: {enc_time:.4f}s | Decode: {decode_time:.4f}s | Total: {enc_time+decode_time:.4f}s"
    #     )
    #     plt.show()
    
    return results


def process_folder(input_folder, encoder, decoder, device, 
                   output_dir=None, target_size=(256, 256)):
    """
    Process all images in a folder
    
    Args:
        input_folder: path to folder containing images
        encoder: PyTorch Encoder model
        decoder: PyTorch Decoder model
        device: torch device
        output_dir: directory to save results
        target_size: (width, height) for resizing
    
    Returns:
        list of results for each image
    """
    input_path = Path(input_folder)
    
    # Create output subfolder with input folder name
    if output_dir:
        input_folder_name = input_path.name
        output_dir = Path(output_dir) / input_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults will be saved to: {output_dir}/")
    
    # Supported image extensions
    supported_exts = {
        '.jpg', '.jpeg', '.png', '.bmp',
        '.tiff', '.tif', '.cr3'
    }    
    # Find all images
    image_files = sorted(
    p for p in input_path.iterdir()
    if p.is_file() and p.suffix.lower() in supported_exts
    )
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    results = []
    failed_images = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            result = process_single_image(
                image_path, 
                encoder, 
                decoder, 
                device,
                output_dir=output_dir,
                target_size=target_size,
                show_plot=False,
                save_plot=True
            )
            results.append(result)
        except Exception as e:
            print(f"✗ {image_path.name}: Error - {str(e)}")
            failed_images.append({'file': image_path.name, 'error': str(e)})
    
    # Generate summary report
    if output_dir and results:
        generate_summary_report(results, failed_images, output_dir, input_folder)
    
    return results




def generate_summary_report(results, failed_images, output_dir, input_folder):
    """
    Generate a summary report for batch processing
    
    Args:
        results: list of processing results
        failed_images: list of failed image info
        output_dir: output directory
        input_folder: input folder path
    """
    import json
    from datetime import datetime
    
    output_path = Path(output_dir)
    
    # Calculate statistics
    total_images = len(results) + len(failed_images)
    successful = len(results)
    failed = len(failed_images)
    
    avg_encode_time = np.mean([r['encode_time'] for r in results]) if results else 0
    avg_decode_time = np.mean([r['decode_time'] for r in results]) if results else 0
    avg_total_time = np.mean([r['total_time'] for r in results]) if results else 0
    
    total_processing_time = sum([r['total_time'] for r in results])
    
    # Create summary report
    summary = {
        'processing_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_folder': str(input_folder),
            'output_folder': str(output_dir),
            'total_images': total_images,
            'successful': successful,
            'failed': failed,
            'success_rate': f"{(successful/total_images*100):.2f}%" if total_images > 0 else "0%"
        },
        'performance': {
            'avg_encode_time': f"{avg_encode_time:.4f}s",
            'avg_decode_time': f"{avg_decode_time:.4f}s",
            'avg_total_time': f"{avg_total_time:.4f}s",
            'total_processing_time': f"{total_processing_time:.2f}s"
        },
        'processed_images': [
            {
                'filename': Path(r['image_path']).name,
                'encode_time': f"{r['encode_time']:.4f}s",
                'decode_time': f"{r['decode_time']:.4f}s",
                'total_time': f"{r['total_time']:.4f}s"
            }
            for r in results
        ],
        'failed_images': failed_images
    }
    
    # Save JSON summary
    summary_path = output_path / 'processing_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:  
        json.dump(summary, f, indent=4)
    
    # Generate text summary
    text_summary_path = output_path / 'processing_summary.txt'
    with open(text_summary_path, 'w', encoding='utf-8') as f: 
        f.write("=" * 70 + "\n")
        f.write("DEEP ALBEDO - BATCH PROCESSING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Date: {summary['processing_info']['date']}\n")
        f.write(f"Input Folder: {summary['processing_info']['input_folder']}\n")
        f.write(f"Output Folder: {summary['processing_info']['output_folder']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PROCESSING STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Images: {summary['processing_info']['total_images']}\n")
        f.write(f"Successfully Processed: {summary['processing_info']['successful']}\n")
        f.write(f"Failed: {summary['processing_info']['failed']}\n")
        f.write(f"Success Rate: {summary['processing_info']['success_rate']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average Encode Time: {summary['performance']['avg_encode_time']}\n")
        f.write(f"Average Decode Time: {summary['performance']['avg_decode_time']}\n")
        f.write(f"Average Total Time: {summary['performance']['avg_total_time']}\n")
        f.write(f"Total Processing Time: {summary['performance']['total_processing_time']}\n\n")
        
        if failed_images:
            f.write("-" * 70 + "\n")
            f.write("FAILED IMAGES\n")
            f.write("-" * 70 + "\n")
            for fail in failed_images:
                f.write(f"  - {fail['file']}: {fail['error']}\n")
            f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("OUTPUT STRUCTURE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{output_dir}/\n")
        f.write("├── visualizations/         # Complete analysis plots\n")
        f.write("├── recovered/              # Reconstructed images\n")
        f.write("├── parameter_maps/         # Individual parameter maps\n")
        f.write("│   ├── Cm/                 # Melanin concentration\n")
        f.write("│   ├── Ch/                 # Hemoglobin concentration\n")
        f.write("│   ├── Bm/                 # Melanin baseline\n")
        f.write("│   ├── Bh/                 # Hemoglobin baseline\n")
        f.write("│   └── T/                  # Thickness\n")
        f.write("├── data/                   # Raw numpy arrays & metadata\n")
        f.write("├── processing_summary.json # Detailed JSON summary\n")
        f.write("└── processing_summary.txt  # This file\n\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"\n✓ Summary reports generated:")
    print(f"  - {summary_path}")
    print(f"  - {text_summary_path}")


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: path to checkpoint file
        device: torch device
    
    Returns:
        encoder, decoder models
    """
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Initialize model
    encoder_net = Encoder().to(device)
    decoder_net = Decoder().to(device)
    model = AutoEncoder(encoder_net, decoder_net).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract encoder and decoder
    encoder = model.encoder
    decoder = model.decoder
    
    encoder.eval()
    decoder.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Best train loss: {checkpoint.get('best_train_loss', 'N/A')}")
    
    return encoder, decoder


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Deep Albedo - Extract skin parameters from images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process single image:
    python inference.py --input image.jpg
    
  Process folder:
    python inference.py --input images_folder/
    
  Process with custom output:
    python inference.py --input images_folder/ --output results/
    
  Use custom checkpoint:
    python inference.py --input image.jpg --checkpoint path/to/model.pt
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input image file or folder containing images')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory for results (default: output/)')
    parser.add_argument('--checkpoint', '-c', type=str, 
                       default='checkpoints/2025-12-31_18-54-39/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--size', '-s', type=int, nargs=2, default=[256, 256],
                       help='Target image size (width height), default: 256 256')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU inference (even if GPU available)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
        print("Using CPU (forced)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    # Load model
    try:
        encoder, decoder = load_model(args.checkpoint, device)
    except FileNotFoundError:
        print(f"\n✗ Error: Checkpoint not found at {args.checkpoint}")
        print("Please check the path or train a model first.")
        return
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return
    
    # Process input
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"\n✗ Error: Input path does not exist: {args.input}")
        return
    
    target_size = tuple(args.size)
    
    if input_path.is_file():
        # Process single image
        print(f"\nProcessing single image: {args.input}")
        try:
            result = process_single_image(
                input_path,
                encoder,
                decoder,
                device,
                output_dir=args.output,
                target_size=target_size,
                show_plot=not args.no_show,
                save_plot=True
            )
            print(f"\n✓ Processing complete!")
            print(f"  - Encode time: {result['encode_time']:.4f}s")
            print(f"  - Decode time: {result['decode_time']:.4f}s")
            print(f"  - Total time: {result['total_time']:.4f}s")
            if args.output:
                print(f"  - Results saved to: {args.output}/")
        except Exception as e:
            print(f"\n✗ Error processing image: {e}")
    
    elif input_path.is_dir():
        # Process folder
        print(f"\nProcessing folder: {args.input}")
        results = process_folder(
            input_path,
            encoder,
            decoder,
            device,
            output_dir=args.output,
            target_size=target_size
        )
        
        if results:
            avg_time = np.mean([r['total_time'] for r in results])
            print(f"\n✓ Processing complete!")
            print(f"  - Images processed: {len(results)}")
            print(f"  - Average time per image: {avg_time:.4f}s")
            if args.output:
                print(f"  - Results saved to: {args.output}/")
        else:
            print("\n✗ No images were processed")
    
    else:
        print(f"\n✗ Error: Input must be a file or directory")


if __name__ == '__main__':
    main()