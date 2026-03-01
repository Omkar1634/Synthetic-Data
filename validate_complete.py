"""
COMPLETE VALIDATION: Your Synthetic Skin Data vs Hyper-Skin Real Measurements
==============================================================================
This script performs full spectral comparison between your Monte Carlo
generated data and real hyperspectral measurements from Hyper-Skin dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from pathlib import Path
import glob
import h5py

# ===== CONFIGURATION - UPDATE THESE PATHS =====
YOUR_CSV_PATH = r"D:\Github\PhD Code\Synthetic Data\monte_carlo\lut_rgb_BaseLine_3.csv"
HYPERSKIN_VIS_FOLDER = r"D:\Hyper-Skin\Hyper-Skin(RGB, VIS)\train\VIS"  # UPDATE THIS!
OUTPUT_DIR = r"D:\Github\PhD Code\Synthetic Data\validation_results\lut_rgb_BaseLine_3"  # Where to save results
# ==============================================

def load_your_synthetic_data(csv_path):
    """Load your synthetic skin data with spectral reflectance"""
    
    print("="*80)
    print("STEP 1: LOADING YOUR SYNTHETIC DATA")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df):,} samples")
    print(f"   Columns: {len(df.columns)}")
    
    # Convert RGB from [0, 255] to [0, 1] if needed
    if 'sR' in df.columns and df['sR'].max() > 1:
        print("\n📊 Converting RGB from [0, 255] to [0, 1]...")
        for col in ['sR', 'sG', 'sB']:
            if col in df.columns:
                df[col] = df[col] / 255.0
    
    # Extract spectral columns
    spectral_cols = [col for col in df.columns if col.startswith('R_') and col.endswith('nm')]
    
    if len(spectral_cols) == 0:
        raise ValueError("❌ No spectral columns found! Need columns like R_380nm, R_385nm, etc.")
    
    # Extract wavelengths
    wavelengths = np.array([int(col.replace('R_', '').replace('nm', '')) 
                           for col in spectral_cols])
    
    # Get spectral data
    spectral_data = df[spectral_cols].values
    
    print(f"\n✅ Spectral data found:")
    print(f"   Wavelength range: {wavelengths[0]}-{wavelengths[-1]} nm")
    print(f"   Number of wavelengths: {len(wavelengths)}")
    print(f"   Spectral data shape: {spectral_data.shape}")
    
    # Get parameters
    param_cols = ['melanin_concentration(Cm)', 'blood_concentration(Ch)', 
                  'melanin_blend(Bm)', 'BloodOxy', 'epidermis_thickness(T)']
    
    params = df[param_cols] if all(col in df.columns for col in param_cols) else None
    
    return df, wavelengths, spectral_data, params


def load_hyperskin_sample(mat_file_path):
    """Load a single Hyper-Skin VIS sample (.mat file - MATLAB v7.3 format)"""
    
    try:
        # Try h5py first (for MATLAB v7.3 files)
        with h5py.File(mat_file_path, 'r') as f:
            # Debug: print available keys
            keys = list(f.keys())
            
            print(f"   Available keys in .mat file: {keys}")
            
            # Try common key names for Hyper-Skin dataset
            # Based on the repository structure, common keys are:
            # 'cube', 'hsi', 'data', 'hyperspectral', etc.
            
            hyperspectral_cube = None
            
            if 'cube' in f:
                hyperspectral_cube = f['cube'][:]
            elif 'hsi' in f:
                hyperspectral_cube = f['hsi'][:]
            elif 'data' in f:
                hyperspectral_cube = f['data'][:]
            elif 'hyperspectral' in f:
                hyperspectral_cube = f['hyperspectral'][:]
            else:
                # Take the first non-metadata key
                for key in keys:
                    if not key.startswith('#'):
                        hyperspectral_cube = f[key][:]
                        print(f"   Using key: '{key}'")
                        break
            
            if hyperspectral_cube is None:
                print(f"   ⚠️ Could not find hyperspectral data in keys: {keys}")
                return None, None, keys
            
            # Wavelengths for VIS: 400-700nm, 31 bands
            if 'wavelengths' in f:
                wavelengths = f['wavelengths'][:].flatten()
            elif 'wl' in f:
                wavelengths = f['wl'][:].flatten()
            elif 'lambda' in f:
                wavelengths = f['lambda'][:].flatten()
            else:
                # Default VIS wavelengths: 400-700nm with ~10nm spacing for 31 bands
                wavelengths = np.linspace(400, 700, 31)
                print(f"   Using default wavelengths: 400-700nm, 31 bands")
            
            return wavelengths, hyperspectral_cube, keys
        
    except Exception as e:
        # Fallback to scipy.io.loadmat for older MATLAB formats
        try:
            data = loadmat(mat_file_path)
            keys = [k for k in data.keys() if not k.startswith('__')]
            
            if 'cube' in data:
                hyperspectral_cube = data['cube']
            elif 'hsi' in data:
                hyperspectral_cube = data['hsi']
            else:
                largest_key = max(keys, key=lambda k: data[k].size if isinstance(data[k], np.ndarray) else 0)
                hyperspectral_cube = data[largest_key]
            
            wavelengths = data.get('wavelengths', np.linspace(400, 700, 31)).flatten()
            
            return wavelengths, hyperspectral_cube, keys
            
        except Exception as e2:
            print(f"   ⚠️ Error loading {mat_file_path}: {e2}")
            return None, None, None


def extract_skin_region(hyperspectral_cube, region='center'):
    """
    Extract skin pixels from hyperspectral cube
    
    Parameters:
    -----------
    hyperspectral_cube : numpy.array
        Shape: (height, width, n_wavelengths) or (n_wavelengths, height, width)
    region : str
        'center', 'forehead', 'cheek', or 'full'
    """
    
    print(f"   Input cube shape: {hyperspectral_cube.shape}")
    
    # Handle different cube orientations
    # Common formats: (n_wavelengths, height, width) or (height, width, n_wavelengths)
    # Hyper-Skin uses: (n_wavelengths, height, width)
    
    if hyperspectral_cube.shape[0] < 100:  # Likely n_wavelengths is first dimension
        # (n_wavelengths, height, width) → (height, width, n_wavelengths)
        cube = np.transpose(hyperspectral_cube, (1, 2, 0))
        print(f"   Transposed to: {cube.shape}")
    else:
        cube = hyperspectral_cube
    
    height, width, n_bands = cube.shape
    print(f"   Final shape (H, W, bands): {height} x {width} x {n_bands}")
    
    # Define regions (approximate - adjust based on face alignment)
    regions = {
        'center': (
            slice(height//3, 2*height//3),
            slice(width//3, 2*width//3)
        ),
        'forehead': (
            slice(height//4, height//2),
            slice(width//3, 2*width//3)
        ),
        'cheek': (
            slice(height//2, 3*height//4),
            slice(width//4, width//2)
        ),
        'full': (
            slice(None),
            slice(None)
        )
    }
    
    row_slice, col_slice = regions.get(region, regions['center'])
    
    # Extract region
    region_cube = cube[row_slice, col_slice, :]
    
    # Reshape to (n_pixels, n_wavelengths)
    n_pixels = region_cube.shape[0] * region_cube.shape[1]
    spectra = region_cube.reshape(n_pixels, n_bands)
    
    print(f"   Extracted spectra shape: {spectra.shape}")
    
    # Remove invalid spectra (negative, NaN, or zero)
    valid_mask = (
        np.all(spectra >= 0, axis=1) & 
        ~np.any(np.isnan(spectra), axis=1) &
        ~np.all(spectra == 0, axis=1) &
        (np.sum(spectra, axis=1) > 0.01)  # Remove very dark pixels
    )
    
    valid_spectra = spectra[valid_mask]
    
    print(f"   Valid spectra: {len(valid_spectra)} / {len(spectra)}")
    
    return valid_spectra


def align_wavelengths(wl_source, spec_source, wl_target):
    """
    Interpolate spectrum to align wavelengths
    
    Parameters:
    -----------
    wl_source : array
        Source wavelengths
    spec_source : array
        Source spectrum
    wl_target : array
        Target wavelengths
    """
    
    # Find overlapping range
    wl_min = max(wl_source.min(), wl_target.min())
    wl_max = min(wl_source.max(), wl_target.max())
    
    # Create target wavelengths within overlap
    wl_overlap = wl_target[(wl_target >= wl_min) & (wl_target <= wl_max)]
    
    # Interpolate
    interp_func = interp1d(wl_source, spec_source, kind='cubic', 
                          fill_value='extrapolate', bounds_error=False)
    spec_aligned = interp_func(wl_overlap)
    
    return wl_overlap, spec_aligned


def compare_spectral_curves(your_wl, your_spectra, hyperskin_wl, hyperskin_spectra, 
                           output_dir, n_samples=5):
    """
    Create comprehensive comparison plots
    """
    
    print("\n" + "="*80)
    print("STEP 3: CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find overlapping wavelength range
    wl_min = max(your_wl.min(), hyperskin_wl.min())
    wl_max = min(your_wl.max(), hyperskin_wl.max())
    
    print(f"\n📊 Wavelength alignment:")
    print(f"   Your data: {your_wl.min()}-{your_wl.max()} nm ({len(your_wl)} points)")
    print(f"   Hyper-Skin: {hyperskin_wl.min()}-{hyperskin_wl.max()} nm ({len(hyperskin_wl)} points)")
    print(f"   Overlap: {wl_min}-{wl_max} nm")
    
    # Create common wavelength grid (use Hyper-Skin wavelengths as reference)
    wl_common = hyperskin_wl[(hyperskin_wl >= wl_min) & (hyperskin_wl <= wl_max)]
    
    # Align your synthetic spectra
    your_spectra_aligned = []
    for spec in your_spectra[:n_samples]:
        _, spec_aligned = align_wavelengths(your_wl, spec, wl_common)
        your_spectra_aligned.append(spec_aligned)
    
    # Align Hyper-Skin spectra  
    hyperskin_spectra_aligned = []
    for spec in hyperskin_spectra[:min(n_samples, len(hyperskin_spectra))]:
        _, spec_aligned = align_wavelengths(hyperskin_wl, spec, wl_common)
        hyperskin_spectra_aligned.append(spec_aligned)
    
    # Compute statistics
    your_mean = np.mean(your_spectra_aligned, axis=0)
    your_std = np.std(your_spectra_aligned, axis=0)
    
    hyperskin_mean = np.mean(hyperskin_spectra_aligned, axis=0)
    hyperskin_std = np.std(hyperskin_spectra_aligned, axis=0)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((your_mean - hyperskin_mean)**2))
    mae = np.mean(np.abs(your_mean - hyperskin_mean))
    
    # ===== PLOT 1: Mean Comparison =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    ax.plot(wl_common, hyperskin_mean, 'b-', linewidth=2.5, 
           label='Hyper-Skin (Real)', alpha=0.8)
    ax.fill_between(wl_common, 
                    hyperskin_mean - hyperskin_std,
                    hyperskin_mean + hyperskin_std,
                    color='blue', alpha=0.2, label='±1 std (Real)')
    
    ax.plot(wl_common, your_mean, 'r--', linewidth=2.5,
           label='Your Synthetic', alpha=0.8)
    ax.fill_between(wl_common,
                    your_mean - your_std,
                    your_mean + your_std,
                    color='red', alpha=0.2, label='±1 std (Synthetic)')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=13, fontweight='bold')
    ax.set_title(f'Mean Spectral Comparison\nRMSE: {rmse:.4f}, MAE: {mae:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # ===== PLOT 2: Difference =====
    ax = axes[0, 1]
    difference = your_mean - hyperskin_mean
    ax.plot(wl_common, difference, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between(wl_common, 0, difference, 
                    where=(difference >= 0), color='green', alpha=0.3,
                    label='Synthetic > Real')
    ax.fill_between(wl_common, 0, difference,
                    where=(difference < 0), color='red', alpha=0.3,
                    label='Synthetic < Real')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Difference (Synthetic - Real)', fontsize=13, fontweight='bold')
    ax.set_title('Spectral Difference', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ===== PLOT 3: Individual Real Spectra =====
    ax = axes[1, 0]
    for spec in hyperskin_spectra_aligned:
        ax.plot(wl_common, spec, 'b-', alpha=0.3, linewidth=1)
    ax.plot(wl_common, hyperskin_mean, 'r-', linewidth=3, 
           label='Mean', alpha=0.9)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=13, fontweight='bold')
    ax.set_title(f'Hyper-Skin Real Spectra (n={len(hyperskin_spectra_aligned)})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ===== PLOT 4: Individual Synthetic Spectra =====
    ax = axes[1, 1]
    for spec in your_spectra_aligned:
        ax.plot(wl_common, spec, 'r-', alpha=0.3, linewidth=1)
    ax.plot(wl_common, your_mean, 'b-', linewidth=3,
           label='Mean', alpha=0.9)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=13, fontweight='bold')
    ax.set_title(f'Your Synthetic Spectra (n={len(your_spectra_aligned)})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / "spectral_comparison.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()
    
    return rmse, mae, wl_common, your_mean, hyperskin_mean


def generate_validation_report(rmse, mae, output_dir):
    """Generate text validation report"""
    
    report_path = Path(output_dir) / "validation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SPECTRAL VALIDATION REPORT\n")
        f.write("Your Synthetic Data vs Hyper-Skin Real Measurements\n")
        f.write("="*80 + "\n\n")
        
        f.write("QUANTITATIVE METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"RMSE (Root Mean Square Error): {rmse:.6f}\n")
        f.write(f"MAE (Mean Absolute Error):     {mae:.6f}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-"*80 + "\n")
        
        if rmse < 0.03:
            f.write("✅ EXCELLENT! Your synthetic data matches real skin very closely.\n")
            f.write("   This validates your Monte Carlo implementation.\n")
        elif rmse < 0.05:
            f.write("✅ VERY GOOD! Minor differences, but overall realistic.\n")
            f.write("   Your biophysical model is accurate.\n")
        elif rmse < 0.10:
            f.write("✅ GOOD! Some differences present, but acceptable.\n")
            f.write("   Consider checking absorption coefficients.\n")
        elif rmse < 0.20:
            f.write("⚠️  MODERATE. Noticeable differences.\n")
            f.write("   Review melanin/hemoglobin absorption formulas.\n")
        else:
            f.write("❌ SIGNIFICANT DIFFERENCES. Review implementation.\n")
            f.write("   Check: absorption equations, scattering, Monte Carlo logic.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NEXT STEPS:\n")
        f.write("="*80 + "\n")
        f.write("1. Review the spectral_comparison.png plot\n")
        f.write("2. Check if absorption peaks match (hemoglobin at 540nm, 570nm)\n")
        f.write("3. Verify melanin shows wavelength^-3.33 dependency\n")
        f.write("4. If RMSE > 0.10, debug your Monte Carlo implementation\n")
    
    print(f"\n✅ Saved: {report_path}")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        print("\n" + f.read())


def main():
    """Main validation workflow"""
    
    print("\n" + "🔬 COMPLETE SPECTRAL VALIDATION 🔬".center(80))
    print("Comparing Your Synthetic Data vs Hyper-Skin Real Measurements\n")
    
    # Step 1: Load your synthetic data
    df, your_wl, your_spectra, params = load_your_synthetic_data(YOUR_CSV_PATH)
    
    # Step 2: Load Hyper-Skin data
    print("\n" + "="*80)
    print("STEP 2: LOADING HYPER-SKIN REAL DATA")
    print("="*80)
    
    hyperskin_folder = Path(HYPERSKIN_VIS_FOLDER)
    mat_files = list(hyperskin_folder.glob("*.mat"))
    
    if len(mat_files) == 0:
        print(f"❌ No .mat files found in {hyperskin_folder}")
        print("   Please update HYPERSKIN_VIS_FOLDER path!")
        return
    
    print(f"✅ Found {len(mat_files)} Hyper-Skin samples")
    
    # Load first sample to get structure
    print(f"\n📂 Loading sample: {mat_files[0].name}")
    hyperskin_wl, hyperspectral_cube, keys = load_hyperskin_sample(mat_files[0])
    
    if hyperspectral_cube is None:
        print("❌ Failed to load Hyper-Skin data")
        print(f"   Available keys in .mat file: {keys}")
        return
    
    print(f"✅ Loaded hyperspectral cube")
    print(f"   Shape: {hyperspectral_cube.shape}")
    print(f"   Wavelengths: {len(hyperskin_wl)} bands")
    print(f"   Range: {hyperskin_wl[0]:.1f}-{hyperskin_wl[-1]:.1f} nm")
    
    # Extract skin region
    print(f"\n📊 Extracting skin pixels from center region...")
    hyperskin_spectra = extract_skin_region(hyperspectral_cube, region='center')
    print(f"✅ Extracted {len(hyperskin_spectra)} valid skin pixels")
    
    # Sample random synthetic spectra for comparison
    n_synthetic_samples = min(1000, len(your_spectra))
    indices = np.random.choice(len(your_spectra), n_synthetic_samples, replace=False)
    your_spectra_sample = your_spectra[indices]
    
    # Step 3: Compare
    rmse, mae, wl_common, your_mean, hyperskin_mean = compare_spectral_curves(
        your_wl, your_spectra_sample,
        hyperskin_wl, hyperskin_spectra,
        OUTPUT_DIR,
        n_samples=min(100, len(hyperskin_spectra))
    )
    
    # Step 4: Generate report
    generate_validation_report(rmse, mae, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nKey Findings:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    
    if rmse < 0.10:
        print("\n🎉 SUCCESS! Your synthetic data matches real skin measurements!")
    else:
        print("\n⚠️  Review the plots and consider adjusting your implementation.")


if __name__ == "__main__":
    main()