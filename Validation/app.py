"""
================================================================================
STREAMLIT WEB APP FOR LATENT SPACE VALIDATION
================================================================================

This app provides a web interface for validating your autoencoder.
It uses functions from latent_space_validation.py

FEATURES:
- Upload images or select folder

- Configure validation parameters
- Run validation with live progress
- View results in interactive dashboard
- Download reports and visualizations

HOW TO RUN:
    streamlit run streamlit_validation_app.py

================================================================================
"""

import streamlit as st
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import torch
import cv2
from plotly.subplots import make_subplots
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt

# Import functions from the validation script
from latent_space_validation import (
    load_encoder,
    sample_skin_pixels,
    predict_parameters,
    analyze_parameter_distribution,
    check_parameter_correlations,
    create_distribution_plots,
    create_correlation_heatmap
)
from delta_e import (
    calculate_delta_e,
    analyze_delta_e_results,
    assess_quality,
    create_delta_e_visualization,
    create_delta_e_histogram,
    generate_delta_e_report
)

# ============================================================================
# FULL IMAGE INFERENCE FUNCTIONS
# ============================================================================

def process_full_image(image_path, encoder, decoder, device, resize_to=512):
    """
    Process entire image (not just skin pixels) to generate parameter maps
    
    Args:
        image_path: path to image
        encoder: trained encoder model
        decoder: trained decoder  
        device: torch device
        resize_to: resize images larger than this (for speed)
    
    Returns:
        original_img: original RGB image
        recovered_img: reconstructed RGB image
        parameter_maps: (H, W, 5) array of parameters
        processing_time: total time in seconds
    """
    import time
    from PIL import Image, ImageOps
    
    start_time = time.time()
    
    # Load image and fix orientation
    img = Image.open(image_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    
    # Resize if too large (for speed)
    if max(img.size) > resize_to:
        img.thumbnail((resize_to, resize_to), Image.LANCZOS)
    
    # Convert to array
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    original_img = img_array.copy()
    
    H, W, C = img_array.shape
    
    # Flatten to (H*W, 3)
    pixels_flat = img_array.reshape(-1, 3)
    
    # Convert to tensor
    pixels_tensor = torch.from_numpy(pixels_flat).float().to(device)
    
    # Encode: RGB -> Parameters
    encoder.eval()
    with torch.no_grad():
        params_flat = encoder(pixels_tensor)
    
    # Decode: Parameters -> RGB
    decoder.eval()
    with torch.no_grad():
        recovered_flat = decoder(params_flat)
    
    # Convert back to numpy
    params_np = params_flat.cpu().numpy()
    recovered_np = recovered_flat.cpu().numpy()
    
    # Reshape to image dimensions
    parameter_maps = params_np.reshape(H, W, 5)
    recovered_img = recovered_np.reshape(H, W, 3)
    
    # Clip to valid range
    recovered_img = np.clip(recovered_img, 0, 1)
    
    processing_time = time.time() - start_time
    
    return original_img, recovered_img, parameter_maps, processing_time


def create_parameter_visualization(original_img, recovered_img, parameter_maps, image_name):
    """
    Simple vertical layout using st.image and plotly heatmaps
    No subplots - just stack them vertically
    """
    import plotly.graph_objects as go
    
    # We'll return a list of figures instead of one combined figure
    # Display them separately in the app
    
    results = {
        'original': original_img,
        'recovered': recovered_img,
        'param_maps': parameter_maps,
        'image_name': image_name
    }
    
    return results

def create_statistics_table(parameter_maps):
    """
    Create table of parameter statistics
    
    Returns:
        pandas DataFrame
    """
    param_names = ['Cm', 'Ch', 'Bm', 'Bh', 'T']
    stats_data = []
    
    for i, name in enumerate(param_names):
        param_map = parameter_maps[:, :, i]
        stats_data.append({
            'Parameter': name,
            'Min': f"{param_map.min():.4f}",
            'Max': f"{param_map.max():.4f}",
            'Mean': f"{param_map.mean():.4f}",
            'Std': f"{param_map.std():.4f}",
            'Median': f"{np.median(param_map):.4f}"
        })
    
    return pd.DataFrame(stats_data)


# ============================================================================
# ADD TO STREAMLIT APP - FULL IMAGE VISUALIZATION SECTION
# ============================================================================

def add_full_image_visualization_tab(st, results, image_folder, encoder, decoder, device):
    """
    Add a new tab for full image parameter map visualization
    
    Call this in your streamlit app after validation completes
    """
    
    st.markdown("---")
    st.header("🖼️ Full Image Parameter Maps")
    st.markdown("Process entire images to see parameter distributions across the whole image")
    
    # Get list of images
    from pathlib import Path
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(image_folder).glob(ext))
    
    if not image_files:
        st.warning("No images found in folder")
        return
    
    # Image selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_image = st.selectbox(
            "Select Image to Analyze",
            image_files,
            format_func=lambda x: x.name
        )
    
    with col2:
        resize_size = st.slider("Image Size", 256, 1024, 512, 64, 
                               help="Resize to this size for faster processing")
    
    # Process button
    if st.button("🔍 Analyze Full Image", type="primary", use_container_width=True):
        
        with st.spinner(f"Processing {selected_image.name}..."):
            try:
                # Process entire image
                original, recovered, param_maps, proc_time = process_full_image(
                    selected_image,
                    encoder,
                    decoder,
                    device,
                    resize_to=resize_size
                )
                
                # Store in session state
                st.session_state.full_image_result = {
                    'original': original,
                    'recovered': recovered,
                    'param_maps': param_maps,
                    'proc_time': proc_time,
                    'image_name': selected_image.name
                }
                
                st.success(f"✓ Processed in {proc_time:.2f}s")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
    
    # Display results if available
    if 'full_image_result' in st.session_state:
        result = st.session_state.full_image_result
        
        st.markdown("---")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Image", result['image_name'])
        with col2:
            h, w = result['param_maps'].shape[:2]
            st.metric("Size", f"{w}x{h}")
        with col3:
            st.metric("Processing Time", f"{result['proc_time']:.2f}s")
        with col4:
            error = np.abs(result['original'] - result['recovered']).mean()
            st.metric("Avg Error", f"{error:.4f}")
        
        # Main visualization
        st.subheader("Parameter Maps Visualization")
        
        fig = create_parameter_visualization(
            result['original'],
            result['recovered'],
            result['param_maps'],
            result['image_name']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.subheader("Parameter Statistics")
        stats_df = create_statistics_table(result['param_maps'])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Download options
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Save parameter maps as numpy
            npz_data = {
                'original': result['original'],
                'recovered': result['recovered'],
                'Cm': result['param_maps'][:, :, 0],
                'Ch': result['param_maps'][:, :, 1],
                'Bm': result['param_maps'][:, :, 2],
                'Bh': result['param_maps'][:, :, 3],
                'T': result['param_maps'][:, :, 4]
            }
            
            import io
            buf = io.BytesIO()
            np.savez_compressed(buf, **npz_data)
            buf.seek(0)
            
            st.download_button(
                "📥 Download Arrays (.npz)",
                data=buf,
                file_name=f"{Path(result['image_name']).stem}_params.npz",
                mime="application/octet-stream"
            )
        
        with col2:
            # Save statistics as CSV
            csv = stats_df.to_csv(index=False)
            st.download_button(
                "📥 Download Stats (CSV)",
                data=csv,
                file_name=f"{Path(result['image_name']).stem}_stats.csv",
                mime="text/csv"
            )
        
        with col3:
            # Save recovered image
            recovered_uint8 = (result['recovered'] * 255).astype(np.uint8)
            from PIL import Image
            img_buf = io.BytesIO()
            Image.fromarray(recovered_uint8).save(img_buf, format='PNG')
            img_buf.seek(0)
            
            st.download_button(
                "📥 Download Reconstructed",
                data=img_buf,
                file_name=f"{Path(result['image_name']).stem}_reconstructed.png",
                mime="image/png"
            )

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Autoencoder Validation Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'validation_complete' not in st.session_state:
    st.session_state.validation_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'encoder_loaded' not in st.session_state:
    st.session_state.encoder_loaded = False
if 'encoder' not in st.session_state:
    st.session_state.encoder = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_param_name(param):
    """Format parameter names for display"""
    names = {
        'Cm': 'Melanin Concentration',
        'Ch': 'Blood Concentration',
        'Bm': 'Melanin Blend',
        'Bh': 'Blood Oxygenation',
        'T': 'Epidermis Thickness'
    }
    return names.get(param, param)


def get_param_description(param):
    """Get parameter descriptions"""
    descriptions = {
        'Cm': 'Controls skin darkness (0.1% - 50%)',
        'Ch': 'Controls blood/redness (0.1% - 32%)',
        'Bm': 'Melanin distribution (0 - 1)',
        'Bh': 'Oxygen level in blood (60% - 98%)',
        'T': 'Skin layer thickness (0.1 - 2.5mm)'
    }
    return descriptions.get(param, '')


def assess_parameter(stats):
    """Assess if parameter is good or has issues"""
    issues = []
    
    if not stats['within_expected_range']:
        issues.append('Outside expected range')
    
    if stats['boundary_clustering']['is_problem']:
        issues.append('Boundary clustering detected')
    
    if stats['outlier_pct'] > 5:
        issues.append(f"High outlier rate ({stats['outlier_pct']:.1f}%)")
    
    if stats['biological_plausibility_pct'] < 80:
        issues.append(f"Low bio-plausibility ({stats['biological_plausibility_pct']:.1f}%)")
    
    if not issues:
        return "✅ Good", "success"
    elif len(issues) == 1:
        return f"⚠️ {issues[0]}", "warning"
    else:
        return f"❌ {len(issues)} issues", "error"


# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

st.sidebar.title("🔬 Validation Settings")
st.sidebar.markdown("---")

# Model Configuration
st.sidebar.subheader("1️⃣ Model Configuration")

# Base paths
BASE_CHECKPOINT_DIR = r"D:\Github\PhD Code\Synthetic Data\NN\checkpoints"
BASE_IMAGE_DIR = r"D:\Github\PhD Code\Synthetic Data\Testing_Data"

base_dir = Path(BASE_CHECKPOINT_DIR)

if base_dir.exists():

    checkpoint_files = []

    for run_folder in base_dir.iterdir():
        if run_folder.is_dir():

            # 1️⃣ Check directly inside
            direct_best = run_folder / "best.pt"
            if direct_best.exists():
                checkpoint_files.append(direct_best)

            # 2️⃣ Check one level down
            for subfolder in run_folder.iterdir():
                if subfolder.is_dir():
                    nested_best = subfolder / "best.pt"
                    if nested_best.exists():
                        checkpoint_files.append(nested_best)

    # Sort by modification time
    checkpoint_files.sort(
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if checkpoint_files:
        selected_checkpoint = st.sidebar.selectbox(
            "Select Checkpoint",
            checkpoint_files,
            format_func=lambda x: str(x.relative_to(base_dir))
        )

        checkpoint_path = str(selected_checkpoint)
        st.sidebar.success(f"Using: {selected_checkpoint.name}")

    else:
        st.sidebar.warning("No best.pt files found!")
        checkpoint_path = st.sidebar.text_input("Checkpoint Path", value="")

else:
    st.sidebar.warning(f"Checkpoint directory not found: {BASE_CHECKPOINT_DIR}")
    checkpoint_path = st.sidebar.text_input("Checkpoint Path", value="")
# Show selected path
st.sidebar.caption(f"📁 {checkpoint_path}")

# Model architecture (you need to know these from your training)
st.sidebar.markdown("**Model Architecture:**")
col1, col2 = st.sidebar.columns(2)
with col1:
    num_neurons = st.sidebar.number_input("Neurons", value=256, min_value=64, max_value=1024, step=64)
with col2:
    num_layers = st.sidebar.number_input("Layers", value=8, min_value=2, max_value=20, step=1)

device_option = st.sidebar.selectbox(
    "Device",
    ["cuda", "cpu"],
    help="Use GPU if available for faster processing"
)

# Load Encoder Button
if st.sidebar.button("Load Model", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading encoder and decoder..."):
            device = torch.device(device_option if torch.cuda.is_available() else "cpu")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Create encoder
            from latent_space_validation import Encoder, Decoder  # Import your model classes
            encoder = Encoder(in_dim=3, hidden_dim=num_neurons, num_layers=num_layers, out_dim=5).to(device)
            
            # Extract encoder weights
            encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            encoder.load_state_dict(encoder_state)
            encoder.eval()
            
            decoder = Decoder(in_dim=5, hidden_dim=num_neurons, num_layers=num_layers, out_dim=3).to(device)
            
            # Extract decoder weights
            decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
            decoder.load_state_dict(decoder_state)
            decoder.eval()
            
            # Save both to session state
            st.session_state.encoder = encoder
            st.session_state.decoder = decoder
            st.session_state.device = device
            st.session_state.encoder_loaded = True
            
            st.sidebar.success("✅ Encoder & Decoder loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        st.sidebar.exception(e)
st.sidebar.markdown("---")

# Image Configuration
st.sidebar.subheader("2️⃣ Image Configuration")

# Auto-detect image folders
if os.path.exists(BASE_IMAGE_DIR):
    image_folders = [f.name for f in Path(BASE_IMAGE_DIR).iterdir() if f.is_dir()]
    
    # Add option to browse or use detected folders
    use_detected = st.sidebar.checkbox("Use detected folders", value=True)
    
    if use_detected and image_folders:
        selected_image_folder = st.sidebar.selectbox(
            "Select Image Folder",
            [""] + image_folders,
            help="Choose from available image folders"
        )
        if selected_image_folder:
            image_folder = os.path.join(BASE_IMAGE_DIR, selected_image_folder)
        else:
            image_folder = BASE_IMAGE_DIR
    else:
        image_folder = st.sidebar.text_input(
            "Image Folder Path",
            value=BASE_IMAGE_DIR,
            help="Folder containing your skin images"
        )
else:
    image_folder = st.sidebar.text_input(
        "Image Folder Path",
        value="path/to/your/images",
        help="Folder containing your skin images"
    )

# Show image count
# Show image count
if os.path.exists(image_folder):
    # Case-insensitive image counting (FIXED!)
    image_files = []
    for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
        image_files.extend(list(Path(image_folder).glob(ext)))
    # Remove duplicates (same file counted twice)
    image_files = list(set(image_files))
    image_count = len(image_files)
    st.sidebar.caption(f"📁 {image_folder}")
    st.sidebar.caption(f"🖼️ {image_count} images found")
else:
    st.sidebar.error("❌ Folder not found")

num_images = st.sidebar.slider(
    "Number of Images to Analyze",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
    help="More images = more accurate validation but slower"
)

pixels_per_image = st.sidebar.slider(
    "Pixels per Image",
    min_value=50,
    max_value=500,
    value=100,
    step=50,
    help="Number of skin pixels to sample from each image"
)

st.sidebar.markdown("---")

# Parameter Ranges
st.sidebar.subheader("3️⃣ Expected Ranges")

with st.sidebar.expander("View/Edit Parameter Ranges"):
    st.caption("Expected ranges from Monte Carlo simulation")
    
    param_ranges = {
        'Cm': st.slider("Cm (Melanin)", 0.0, 1.0, (0.001, 0.5), 0.001, key='cm_range'),
        'Ch': st.slider("Ch (Blood)", 0.0, 1.0, (0.001, 0.32), 0.001, key='ch_range'),
        'Bm': st.slider("Bm (Blend)", 0.0, 1.0, (0.0, 1.0), 0.01, key='bm_range'),
        'Bh': st.slider("Bh (Oxygen)", 0.0, 1.0, (0.6, 0.98), 0.01, key='bh_range'),
        'T': st.slider("T (Thickness)", 0.0, 1.0, (0.01, 0.25), 0.01, key='t_range'),
    }

biological_ranges = {
    'Cm': (0.013, 0.43),
    'Ch': (0.02, 0.07),
    'Bm': (0.0, 1.0),
    'Bh': (0.75, 0.98),
    'T': (0.05, 0.15)
}


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

st.title("🔬 Autoencoder Latent Space Validation")
st.markdown("### Validate your skin autoencoder with real-world images")

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.encoder_loaded:
        st.success("✅ Encoder Loaded")
    else:
        st.warning("⏳ Encoder Not Loaded")

with col2:
    if os.path.exists(image_folder):
        st.success("✅ Image Folder Found")
    else:
        st.error("❌ Image Folder Not Found")

with col3:
    if st.session_state.validation_complete:
        st.success("✅ Validation Complete")
    else:
        st.info("⏳ Ready to Validate")

st.markdown("---")


# ============================================================================
# RUN VALIDATION
# ============================================================================

if st.button("🚀 Run Validation", type="primary", use_container_width=True, 
             disabled=not st.session_state.encoder_loaded):
    
    if not os.path.exists(image_folder):
        st.error("❌ Image folder does not exist. Please check the path.")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Get image files
            status_text.text("📁 Scanning image directory...")
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend(Path(image_folder).glob(ext))
            
            if len(image_files) == 0:
                st.error("❌ No images found in the folder!")
                st.stop()
            
            progress_bar.progress(10)
            
            # Randomly select images
            import random
            random.seed(42)
            if len(image_files) > num_images:
                selected_images = random.sample(image_files, num_images)
            else:
                selected_images = image_files
            
            st.info(f"📊 Selected {len(selected_images)} images for analysis")
            progress_bar.progress(20)
            
            # Step 2: Process images
            status_text.text("🖼️ Processing images and extracting skin pixels...")
            all_params = []
            valid_images = 0
            total_pixels = 0
            skin_ratios = []
            failed_images = []
            
            progress_increment = 60 / len(selected_images)
            
            # Create a placeholder for image processing status
            img_status = st.empty()
            
            for i, img_path in enumerate(selected_images):
                img_status.text(f"Processing image {i+1}/{len(selected_images)}: {img_path.name}")
                
                # Extract skin pixels
                pixels, skin_ratio = sample_skin_pixels(img_path, pixels_per_image)
                
                if pixels is None:
                    failed_images.append(img_path.name)
                    continue
                
                # Predict parameters
                params = predict_parameters(
                    st.session_state.encoder, 
                    pixels, 
                    st.session_state.device
                )
                
                all_params.append(params)
                valid_images += 1
                total_pixels += len(pixels)
                skin_ratios.append(skin_ratio)
                
                progress_bar.progress(20 + int((i + 1) * progress_increment))
            
            img_status.empty()
            
            if valid_images == 0:
                st.error("❌ No valid images processed. Check skin detection thresholds.")
                st.stop()
            
            progress_bar.progress(80)
            
            # Step 3: Analyze
            status_text.text("📊 Analyzing parameter distributions...")
            all_params = np.vstack(all_params)
            param_names = ['Cm', 'Ch', 'Bm', 'Bh', 'T']
            
            stats = analyze_parameter_distribution(
                all_params,
                param_names,
                param_ranges,
                biological_ranges
            )
            
            corr_matrix, problematic_corrs = check_parameter_correlations(
                all_params,
                param_names
            )
            
            progress_bar.progress(90)
            
            # Step 4: Store results
            status_text.text("💾 Saving results...")
            st.session_state.results = {
                'stats': stats,
                'all_params': all_params,
                'param_names': param_names,
                'corr_matrix': corr_matrix,
                'problematic_corrs': problematic_corrs,
                'image_stats': {
                    'total_images': valid_images,
                    'total_pixels': total_pixels,
                    'avg_skin_ratio': np.mean(skin_ratios),
                    'failed_images': failed_images
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Step 5: Process full images for parameter maps
            if 'full_image_results' not in st.session_state or not st.session_state.full_image_results:

                status_text.text("🖼️ Processing full images for parameter maps...")
                # Select up to 3 images for full processing
                num_full_images = len(selected_images)
                full_image_results = []
                
                for idx in range(num_full_images):
                    img_path = selected_images[idx]
                    
                    try:
                        original, recovered, param_maps, proc_time = process_full_image(
                            img_path,
                            st.session_state.encoder,
                            st.session_state.decoder,
                            st.session_state.device,
                            resize_to=512
                        )
                        
                        full_image_results.append({
                            'image_name': img_path.name,
                            'image_path': str(img_path),
                            'original': original,
                            'recovered': recovered,
                            'param_maps': param_maps,
                            'proc_time': proc_time
                        })
                        
                    except Exception as e:
                        print(f"Failed to process {img_path.name}: {e}")
                
                # Store full image results
                st.session_state.full_image_results = full_image_results
            
            st.session_state.validation_complete = True
            progress_bar.progress(100)
            status_text.text("✅ Validation complete!")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during validation: {str(e)}")
            st.exception(e)
    


# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if st.session_state.validation_complete and st.session_state.results:
    
    st.markdown("---")
    st.header("📊 Validation Results")
    
    results = st.session_state.results
    stats = results['stats']
    all_params = results['all_params']
    param_names = results['param_names']
    corr_matrix = results['corr_matrix']
    image_stats = results['image_stats']
    
    # Overall Summary
    st.subheader("📋 Overall Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Images Analyzed", image_stats['total_images'])
    with col2:
        st.metric("Total Pixels", f"{image_stats['total_pixels']:,}")
    with col3:
        st.metric("Avg Skin Ratio", f"{image_stats['avg_skin_ratio']:.1%}")
    with col4:
        failed = len(image_stats['failed_images'])
        st.metric("Failed Images", failed)
    
    
    
    st.markdown("---")
    
    # Parameter Cards
    st.subheader("🎯 Parameter Analysis")
    
    # Create tabs for each parameter
    tabs = st.tabs([f"{p} - {format_param_name(p)}" for p in param_names])
    
    for i, (tab, param) in enumerate(zip(tabs, param_names)):
        with tab:
            s = stats[param]
            status, status_type = assess_parameter(s)
            
            # Header with status
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {format_param_name(param)}")
                st.caption(get_param_description(param))
            with col2:
                if status_type == "success":
                    st.success(status)
                elif status_type == "warning":
                    st.warning(status)
                else:
                    st.error(status)
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{s['mean']:.4f}")
                st.metric("Std Dev", f"{s['std']:.4f}")
            with col2:
                st.metric("Min", f"{s['min']:.4f}")
                st.metric("Max", f"{s['max']:.4f}")
            with col3:
                st.metric("Median", f"{s['median']:.4f}")
                st.metric("Q25-Q75", f"{s['q25']:.3f} - {s['q75']:.3f}")
            with col4:
                st.metric("Bio-Plausibility", f"{s['biological_plausibility_pct']:.1f}%")
                st.metric("Outliers", f"{s['outlier_pct']:.1f}%")
            
            # Boundary clustering info
            st.markdown("**Boundary Clustering:**")
            col1, col2 = st.columns(2)
            with col1:
                lower_pct = s['boundary_clustering']['lower_boundary_pct']
                st.progress(lower_pct / 100)
                st.caption(f"Lower boundary: {lower_pct:.1f}%")
            with col2:
                upper_pct = s['boundary_clustering']['upper_boundary_pct']
                st.progress(upper_pct / 100)
                st.caption(f"Upper boundary: {upper_pct:.1f}%")
            
            if s['boundary_clustering']['is_problem']:
                st.error("⚠️ Excessive boundary clustering detected!")
            
            # Distribution plot
            st.markdown("**Distribution:**")
            param_values = all_params[:, i]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=param_values,
                nbinsx=50,
                name='Distribution',
                marker_color='steelblue'
            ))
            
            # Add expected range
            expected_min, expected_max = param_ranges[param]
            fig.add_vline(x=expected_min, line_dash="dash", line_color="green",
                         annotation_text="Expected Min")
            fig.add_vline(x=expected_max, line_dash="dash", line_color="green",
                         annotation_text="Expected Max")
            
            # Add mean
            fig.add_vline(x=s['mean'], line_color="red",
                         annotation_text=f"Mean: {s['mean']:.4f}")
            
            fig.update_layout(
                title=f"{param} Distribution",
                xaxis_title="Value",
                yaxis_title="Frequency",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation Matrix
    st.subheader("🔗 Parameter Correlations")
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=param_names,
        y=param_names,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Parameter Correlation Heatmap",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if results['problematic_corrs']:
        st.warning("⚠️ Problematic Correlations Detected:")
        for corr in results['problematic_corrs']:
            st.write(f"- **{corr['params'][0]} vs {corr['params'][1]}**: "
                    f"{corr['correlation']:.2f} - {corr['issue']}")
    else:
        st.success("✅ No problematic correlations detected")
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    boundary_problems = [name for name, s in stats.items() 
                        if s['boundary_clustering']['is_problem']]
    high_outliers = [name for name, s in stats.items() if s['outlier_pct'] > 5]
    low_bio = [name for name, s in stats.items() 
              if s['biological_plausibility_pct'] < 80]
    
    if not boundary_problems and not high_outliers and not low_bio:
        st.success("🎉 **Excellent!** Your model produces biologically plausible parameters!")
        st.info("**Next Steps:**\n"
                "- Proceed to Delta E (perceptual accuracy) testing\n"
                "- Test biophysical heuristics (lips, hair, elderly skin)")
    else:
        if boundary_problems:
            st.error(f"**⚠️ Boundary Clustering:** {', '.join(boundary_problems)}")
            st.write("**Actions:**")
            st.write("- Expand parameter ranges in Monte Carlo simulation")
            st.write("- Retrain with expanded ranges")
            st.write("- Check model architecture capacity")
        
        if high_outliers:
            st.warning(f"**⚠️ High Outlier Rate:** {', '.join(high_outliers)}")
            st.write("**Actions:**")
            st.write("- Add more diverse samples to training data")
            st.write("- Review if synthetic data covers real-world diversity")
        
        if low_bio:
            st.warning(f"**⚠️ Low Biological Plausibility:** {', '.join(low_bio)}")
            st.write("**Actions:**")
            st.write("- Review Monte Carlo simulation parameters")
            st.write("- Compare against Leeds dataset")
            st.write("- Verify parameter ranges match literature")
    
    st.markdown("---")

    if 'full_image_results' in st.session_state and st.session_state.full_image_results:
        st.header("Full Image Parameter Maps")
        st.markdown("Automatically processed sample images showing parameter distributions")
        
        
        # Auto-calculate Delta E (runs once)
        if 'delta_e_results' not in st.session_state:
            delta_e_results = []
            for result in st.session_state.full_image_results:
                de_result = calculate_delta_e(result['original'], result['recovered'])
                de_result['image_name'] = result['image_name']
                delta_e_results.append(de_result)
            st.session_state.delta_e_results = delta_e_results
            st.session_state.delta_e_summary = analyze_delta_e_results(delta_e_results)
        
        # Create tabs for each processed image
        if len(st.session_state.full_image_results) == 1:
            # Single image - no tabs needed
            result = st.session_state.full_image_results[0]
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Image", result['image_name'])
            with col2:
                h, w = result['param_maps'].shape[:2]
                st.metric("Size", f"{w}x{h}")
            with col3:
                st.metric("Processing", f"{result['proc_time']:.2f}s")
            with col4:
                error = np.abs(result['original'] - result['recovered']).mean()
                st.metric("Avg Error", f"{error:.4f}")
            
            # Visualization
            fig = create_parameter_visualization(
                result['original'],
                result['recovered'],
                result['param_maps'],
                result['image_name']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            with st.expander("📊 Parameter Statistics"):
                stats_df = create_statistics_table(result['param_maps'])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        else:
            # Multiple images - use tabs
            tab_names = [r['image_name'] for r in st.session_state.full_image_results]
            tabs = st.tabs(tab_names)
            
            for idx, (tab, result) in enumerate(zip(tabs, st.session_state.full_image_results)):
                with tab:
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Image", result['image_name'])
                    with col2:
                        h, w = result['param_maps'].shape[:2]
                        st.metric("Size", f"{w}x{h}")
                    with col3:
                        st.metric("Processing", f"{result['proc_time']:.2f}s")
                    with col4:
                        error = np.abs(result['original'] - result['recovered']).mean()
                        st.metric("Avg Error", f"{error:.4f}")
            
                    # Display results
                    with st.container():
                        st.subheader(f"Analysis: {result['image_name']}")
                        
                        # Row 1: Original and Reconstructed
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Image**")
                            st.image(result['original'], use_column_width=True)
                        with col2:
                            st.markdown("**Reconstructed**")
                            st.image(result['recovered'], use_column_width=True)

                        # Row 2: Cm and Ch
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Cm (Melanin)**")
                            fig = go.Figure(data=go.Heatmap(z=np.flipud(result['param_maps'][:, :, 0]), colorscale='Viridis'))
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                            fig.update_xaxes(showticklabels=False)
                            fig.update_yaxes(showticklabels=False)
                            st.plotly_chart(fig, use_container_width=True, key=f"tab{idx}_cm_{result['image_name']}")
                        with col2:
                            st.markdown("**Ch (Blood)**")
                            fig = go.Figure(data=go.Heatmap(z=np.flipud(result['param_maps'][:, :, 1]), colorscale='Viridis'))
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                            fig.update_xaxes(showticklabels=False)
                            fig.update_yaxes(showticklabels=False)
                            st.plotly_chart(fig, use_container_width=True, key=f"tab{idx}_ch_{result['image_name']}")
                        # Row 3: Error and Bm
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Reconstruction Error**")
                            error = np.abs(result['original'] - result['recovered']).mean(axis=2)
                            fig = go.Figure(data=go.Heatmap(z=np.flipud(error), colorscale='Viridis'))
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                            fig.update_xaxes(showticklabels=False)
                            fig.update_yaxes(showticklabels=False)
                            st.plotly_chart(fig, use_container_width=True, key=f"tab{idx}_error_{result['image_name']}")
                        with col2:
                            st.markdown("**Bm (Melanin Blend)**")
                            fig = go.Figure(data=go.Heatmap(z=np.flipud(result['param_maps'][:, :, 2]), colorscale='Viridis'))
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                            fig.update_xaxes(showticklabels=False)
                            fig.update_yaxes(showticklabels=False)
                            st.plotly_chart(fig, use_container_width=True, key=f"tab{idx}_bm_{result['image_name']}")
                        # Row 4: Bh and T
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Bh (Blood Oxygenation)**")
                            fig = go.Figure(data=go.Heatmap(z=np.flipud(result['param_maps'][:, :, 3]), colorscale='Viridis'))
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                            fig.update_xaxes(showticklabels=False)
                            fig.update_yaxes(showticklabels=False)
                            st.plotly_chart(fig, use_container_width=True, key=f"tab{idx}_bh_{result['image_name']}")
                        with col2:
                            st.markdown("**T (Thickness)**")
                            fig = go.Figure(data=go.Heatmap(z=np.flipud(result['param_maps'][:, :, 4]), colorscale='Viridis'))
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                            fig.update_xaxes(showticklabels=False)
                            fig.update_yaxes(showticklabels=False)
                            st.plotly_chart(fig, use_container_width=True, key=f"tab{idx}_t_{result['image_name']}")

                    
                    # Statistics
                    with st.expander("📊 Parameter Statistics"):
                        stats_df = create_statistics_table(result['param_maps'])
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                    st.header("🎨 Delta E Perceptual Accuracy")
                    st.markdown("Measure how different colors appear to the **human eye** (not just mathematical distance)")
                    
                    # Info box
                    with st.expander("ℹ️ What is Delta E?"):
                        st.markdown("""
                        **Delta E (CIEDE2000)** measures perceptual color difference:
                        - **ΔE < 1.0**: Imperceptible (perfect match) ✅
                        - **ΔE 1-2**: Perceptible only upon close inspection (excellent) ✅
                        - **ΔE 2-10**: Perceptible at a glance (acceptable) ⚠️
                        - **ΔE > 10**: Colors appear different (poor) ❌
                        
                        This is the **gold standard** for color accuracy in professional industries 
                        (printing, cosmetics, dermatology, CGI).
                        """)    
                    # Delta E 
                    if 'delta_e_results' in st.session_state:
                        st.markdown("---")
                        st.markdown("**🎨 Delta E**")
                        de = st.session_state.delta_e_results[idx]
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean ΔE", f"{de['mean_delta_e']:.2f}")
                        with col2:
                            st.metric("Excellent %", f"{de['percent_excellent']:.1f}%")
                        with col3:
                            quality, _ = assess_quality(de['mean_delta_e'], de['percentile_95'], de['percent_excellent'])
                            st.metric("Quality", quality)
                        with col4:
                            st.metric("95th %", f"{de['percentile_95']:.2f}")
                        with st.expander("🔍 View Delta E Heatmap"):
                            fig_de = create_delta_e_visualization(result['original'], result['recovered'], de)
                            st.pyplot(fig_de, use_container_width=True)
                            plt.close(fig_de)
                        # Histogram
                        with st.expander("📊 Delta E Distribution"):
                            fig_hist = create_delta_e_histogram(de)  # ← Changed this
                            st.pyplot(fig_hist, use_container_width=True)
                            plt.close(fig_hist)

                        # Detailed Statistics
                        with st.expander("📈 Detailed Statistics"):
                            stats_data = {
                                'Metric': ['Mean', 'Median', 'Std Dev', 'Max', '95th %ile', '99th %ile'],
                                'Value': [
                                    f"{de['mean_delta_e']:.3f}",
                                    f"{de['median_delta_e']:.3f}",
                                    f"{de['std_delta_e']:.3f}",
                                    f"{de['max_delta_e']:.3f}",
                                    f"{de['percentile_95']:.3f}",
                                    f"{de['percentile_99']:.3f}"
                                ]
                            }
                            st.table(stats_data)
                            
                            quality_data = {
                                'Quality Level': ['Imperceptible (ΔE<1)', 'Excellent (ΔE<2)', 
                                                'Acceptable (ΔE<10)', 'Poor (ΔE≥10)'],
                                'Percentage': [
                                    f"{de['percent_imperceptible']:.1f}%",
                                    f"{de['percent_excellent']:.1f}%",
                                    f"{de['percent_acceptable']:.1f}%",
                                    f"{de['percent_poor']:.1f}%"
                                ]
                            }
                            st.table(quality_data)
                        
# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Autoencoder Latent Space Validation Tool | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)



