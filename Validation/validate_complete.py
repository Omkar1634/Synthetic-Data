"""
COMPLETE VALIDATION: Synthetic Skin Data vs Hyper-Skin Real Measurements
==============================================================================
Compares Monte Carlo-generated synthetic skin reflectance spectra against the
Hyper-Skin 2023 hyperspectral ground truth dataset.

------------------------------------------------------------------------------
GROUND TRUTH DATASET: Hyper-Skin 2023
------------------------------------------------------------------------------
  Reference : Ng et al., "Hyper-Skin: A Hyperspectral Dataset for
              Reconstructing Facial Skin-Spectra from RGB Images",
              NeurIPS Datasets and Benchmarks Track, 2023.
  GitHub    : https://github.com/hyperspectral-skin/Hyper-Skin-2023
  License   : MIT (access via signed EULA)

  Specs:
    - 51 subjects, 306 hyperspectral cubes (neutral + smile poses)
    - Spatial resolution : 1024 × 1024 pixels per cube (~1M spectra/image)
    - Original bands     : 448 spectral bands (VIS + NIR)
    - VIS resampled      : 31 bands, 400–700 nm  ← used here
    - NIR resampled      : 31 bands, 700–1000 nm
    - File format        : MATLAB .mat (HDF5/v7.3), key = 'cube'
    - Data split         : 264 train / 18 val / 24 test (subject-level split)

------------------------------------------------------------------------------
SIMULATION CHANGELOG — changes applied to simulation/main.cpp
------------------------------------------------------------------------------
  v1 — Baseline (RMSE: 0.3206, MAE: 0.3103)
    - Original Monte Carlo, Nphotons=1000
    - BUG: RFresnel(1.0, nt, -uz) — wrong argument order at tissue-air boundary;
           TIR never triggered, ~33% of oblique photons wrongly escaped,
           inflating reflectance by ~0.25–0.35
    - BUG: s1 = fabs(z / uz) — computed path *after* surface, not *to* surface;
           subsequent position assignments x=(s-s1)*ux etc. ignored accumulated
           lateral position, corrupting photon trajectory

  v2 — Bug fixes (RMSE: 0.0749, MAE: 0.0627)
    - FIX: RFresnel(nt, 1.0, -uz) — correct tissue→air Fresnel; TIR now fires
           for angles > 48.75° (critical angle for n=1.33)
    - FIX: Rewrote surface-hit block — recover z_old = z - s*uz, compute
           s1 = z_old/(-uz), track x_surf/y_surf explicitly, continue from
           surface position for the remaining path s_remaining = s - s1
    - FIX: Nphotons 1000 → 10000 to reduce per-wavelength shot noise

  v3 — Parameter floor (RMSE: 0.0626, MAE: 0.0490)
    - Cm minimum raised: 0.01 → 0.05 (removes unrealistically pale skin
      not present in Hyper-Skin; capped max synthetic reflectance ~0.65)

  v4 — Calibration (RMSE: 0.0690, MAE: 0.0525)
    - Dermis scattering: Us_dermis = 0.75 → 0.65 × Us_epidermis
      (reduces over-scattering in 600–700 nm, lifts red reflectance)
    - Ch minimum raised: 0.002 → 0.02 (ensures haemoglobin double-dip
      at 540/570 nm is visible in generated spectra)
    - Surface specular added: R_specular = ((n-1)/(n+1))^2 ≈ 0.020
      NOTE: later removed — adds uniform offset, overcorrects 450–560 nm

  v5 — Grid expansion + specular removed (RMSE: 0.0641, MAE: 0.0457)
    - Specular term removed (commented out) — was inflating green band
    - LUT grid expanded to 60,000 samples (20×20×5×10×3)
    - Bm range restored to 0.0–1.0 (was 0.5–1.0; exclusion biased mean darker)
    - T confirmed as cm: 0.005–0.020 cm = 50–200 μm (physiologically correct)
    - generateSequence exponent encoding: Cm=2, Ch=2, Bm=2 (power-law spacing)
    - New validation metrics (20 subjects): RMSE=0.0799, SAM=14.3°, ΔE=11.5

  v6 — Stratum corneum layer added  ← current
    - NEW: MonteCarlo upgraded from 2-layer to 3-layer model
      Stack order (top → bottom): SC → epidermis → dermis
    - SC optical properties (wavelength-dependent, fixed parameters, not free):
        sc_thickness = 0.0015 cm (15 μm)
        sc_mua       = alpha_base  (background absorption only, no chromophores)
        sc_mus       = 100 × (400/λ)^0.8 cm⁻¹  (~100 at 400 nm, ~62 at 700 nm)
        sc_g         = 0.70  (slightly more isotropic than epidermis g=0.62)
    - Expected improvement: +0.10–0.15 reflectance at 400–440 nm, closing the
      deficit that drove SAM=14.3°, ΔE=11.5, and PCA/t-SNE separation

------------------------------------------------------------------------------
KNOWN LIMITATIONS OF THE TWO-LAYER MODEL
------------------------------------------------------------------------------
  - 400–440 nm deficit (~−0.18): missing stratum corneum (SC) layer; the SC
    is a ~10–20 μm dead-cell layer with high scattering and no melanin that
    acts as a diffuse backscatterer, contributing ~0.10–0.15 reflectance
    at short wavelengths
  - Current model: epidermis (melanin + scattering) + dermis (Hb + scattering)
  - Planned: add SC as a third purely-scattering layer

------------------------------------------------------------------------------
VALIDATION METHODOLOGY NOTES (for publication)
------------------------------------------------------------------------------
  Current:  RMSE/MAE between mean synthetic and mean real spectrum (1 subject)
  Needed for conference publication:
    1. Spectral Angle Mapper (SAM) — shape-only metric, standard in HSI
    2. Per-wavelength distribution comparison (box plots / violin plots)
    3. Nearest-neighbour coverage: % real spectra within RMSE < threshold
    4. Wasserstein / MMD distance — proper distribution similarity test
    5. Delta E (CIELAB) — perceptual colour accuracy (already in delta_e.py)
    6. Multi-subject: loop over all mat_files, report mean ± std across subjects
    7. Reproducibility: set np.random.seed(42) before all random sampling
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
from pathlib import Path
import glob
import h5py

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ℹ️  scikit-learn not found — PCA/t-SNE plot will be skipped.  pip install scikit-learn")

# ===== CONFIGURATION - UPDATE THESE PATHS =====
YOUR_CSV_PATH = r"D:\Github\PhD Code\Synthetic Data\simulation\lut_rgb_60k20260310_093817.csv"
HYPERSKIN_VIS_FOLDER = r"D:\Hyper-Skin\Hyper-Skin(RGB, VIS)\train\VIS"  
OUTPUT_DIR = r"D:\Github\PhD Code\Synthetic Data\validation\lut_rgb_60k20260310_093817_new_metrics"  # Where to save results
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


# ─── METRIC FUNCTIONS ────────────────────────────────────────────────────────

def spectral_angle_mapper(a, b):
    """
    Spectral Angle Mapper (SAM) between two mean spectra — degrees.
    Shape-only metric (magnitude-independent). Standard in HSI literature.
    < 2° excellent,  2–5° good,  5–10° fair,  > 10° poor.
    """
    dot  = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))


def nearest_neighbour_coverage(synthetic_spectra, real_spectra,
                                thresholds=(0.03, 0.05, 0.10)):
    """
    For each real spectrum find the closest synthetic spectrum (min RMSE).
    Returns:
        coverage  – dict  { threshold : % of real spectra covered }
        nn_dists  – array of per-real nearest-neighbour RMSE values
    Subsamples real to max 500 to keep runtime manageable.
    """
    n_real   = min(500, len(real_spectra))
    idx      = np.random.choice(len(real_spectra), n_real, replace=False)
    real_sub = real_spectra[idx]

    nn_dists = np.array([
        np.sqrt(np.mean((synthetic_spectra - r) ** 2, axis=1)).min()
        for r in real_sub
    ])
    coverage = {t: float(np.mean(nn_dists <= t) * 100) for t in thresholds}
    return coverage, nn_dists


def wasserstein_per_wavelength(synthetic_aligned, real_aligned):
    """
    Per-wavelength Wasserstein-1 distance (Earth Mover's Distance).
    Compares full distributions at each band, not just means.
    Returns (per-wavelength array, scalar mean).
    """
    w_dists = np.array([
        wasserstein_distance(synthetic_aligned[:, i], real_aligned[:, i])
        for i in range(synthetic_aligned.shape[1])
    ])
    return w_dists, float(np.mean(w_dists))


# CIE 1931 2° CMF at 10 nm, 400–700 nm (31 bands) + D65 illuminant
_CIE_WL = np.arange(400, 701, 10, dtype=float)
_CIE_X  = np.array([0.01431,0.04351,0.13438,0.28390,0.34828,0.33620,0.29080,
                     0.19536,0.09564,0.03201,0.00495,0.00932,0.06327,0.16550,
                     0.29040,0.43345,0.59450,0.76210,0.91630,1.02630,1.06220,
                     1.00260,0.85440,0.64240,0.44790,0.28350,0.16490,0.08740,
                     0.04677,0.02270,0.01135])
_CIE_Y  = np.array([0.000396,0.001210,0.004000,0.011600,0.023000,0.038000,
                     0.060000,0.091000,0.139000,0.208000,0.323000,0.503000,
                     0.710000,0.862000,0.954000,0.994950,0.995000,0.952000,
                     0.870000,0.757000,0.631000,0.503000,0.381000,0.265000,
                     0.175000,0.107000,0.061000,0.032000,0.017000,0.008210,0.004102])
_CIE_Z  = np.array([0.06790,0.20740,0.64560,1.38560,1.74706,1.77211,1.66920,
                     1.28764,0.81295,0.46518,0.27200,0.15820,0.07820,0.04220,
                     0.02030,0.00870,0.00390,0.00210,0.00170,0.00110,0.00080,
                     0.00034,0.00019,0.00000,0.00000,0.00000,0.00000,0.00000,
                     0.00000,0.00000,0.00000])
_D65    = np.array([82.75,91.49,93.43,86.68,104.86,117.01,117.81,114.86,
                    115.92,108.81,105.35,95.79,112.40,125.86,125.60,116.34,
                    108.58,100.00,98.00,102.10,102.32,100.00,97.74,98.95,
                    95.24,98.88,95.70,97.23,100.40,102.00,100.00])


def _interp_cmf(wl_target):
    """Interpolate CIE CMF + D65 to an arbitrary wavelength grid."""
    def f(data):
        return interp1d(_CIE_WL, data, kind='linear',
                        fill_value=0.0, bounds_error=False)(wl_target)
    return f(_CIE_X), f(_CIE_Y), f(_CIE_Z), f(_D65)


def spectra_to_lab(spectra, wl_common):
    """
    Convert (N, bands) reflectance spectra → (N, 3) CIE L*a*b* under D65.
    Uses CIE 1931 2° observer, D65 white point.
    """
    x_bar, y_bar, z_bar, d65 = _interp_cmf(wl_common)
    dw = np.gradient(wl_common)
    k  = 1.0 / np.sum(y_bar * d65 * dw)

    X = k * (spectra @ (x_bar * d65 * dw))
    Y = k * (spectra @ (y_bar * d65 * dw))
    Z = k * (spectra @ (z_bar * d65 * dw))

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    def f_lab(t):
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3,
                        np.cbrt(np.maximum(t, 0)),
                        t / (3 * delta ** 2) + 4.0 / 29.0)

    fx, fy, fz = f_lab(X / Xn), f_lab(Y / Yn), f_lab(Z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def delta_e76(lab1, lab2):
    """CIE Delta E 1976 between two (N, 3) Lab arrays. Returns (N,) array."""
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=1))


# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────

def _plot_coverage_curve(nn_dists, output_path):
    """Coverage curve: % real spectra covered vs RMSE threshold."""
    thresholds = np.linspace(0.0, 0.20, 200)
    coverage   = [np.mean(nn_dists <= t) * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, coverage, 'b-', lw=2)
    for t, label in [(0.03, '0.03'), (0.05, '0.05'), (0.10, '0.10')]:
        pct = float(np.mean(nn_dists <= t) * 100)
        ax.axvline(t, color='gray', ls='--', lw=1, alpha=0.6)
        ax.annotate(f'{pct:.0f}% @ {label}',
                    xy=(t, pct), xytext=(t + 0.005, pct - 8),
                    fontsize=9, color='darkred')
    ax.set_xlabel('RMSE Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Real Spectra Covered', fontsize=12, fontweight='bold')
    ax.set_title('Nearest-Neighbour Coverage\n'
                 '(% of real skin spectra within threshold of a synthetic match)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 0.20); ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_distribution_boxplots(your_aligned, real_aligned, wl_common, output_path):
    """Side-by-side per-wavelength box plots for real vs synthetic."""
    step    = max(1, len(wl_common) // 15)
    wl_idx  = np.arange(0, len(wl_common), step)
    wl_ticks = wl_common[wl_idx].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, data, color, mcolor, title in [
        (axes[0], real_aligned,  'cornflowerblue', 'navy',    'Real Spectra'),
        (axes[1], your_aligned,  'salmon',         'darkred', 'Synthetic Spectra'),
    ]:
        ax.boxplot(
            [data[:, i] for i in wl_idx],
            positions=range(len(wl_idx)), widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color=mcolor, lw=2),
            flierprops=dict(marker='.', markersize=2, alpha=0.3),
            whiskerprops=dict(lw=1), capprops=dict(lw=1)
        )
        ax.set_xticks(range(len(wl_idx)))
        ax.set_xticklabels(wl_ticks, rotation=45, fontsize=9)
        ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reflectance',     fontsize=12, fontweight='bold')
        ax.set_title(f'{title} — Per-Wavelength Distribution',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Distribution Comparison: Real vs Synthetic\n'
                 '(box = IQR, whiskers = 1.5×IQR, line = median)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_wasserstein(w_dists, wl_common, w_mean, output_path):
    """Bar chart of per-wavelength Wasserstein distance."""
    dw  = wl_common[1] - wl_common[0] if len(wl_common) > 1 else 10
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(wl_common, w_dists, width=dw * 0.85, color='steelblue', alpha=0.7)
    ax.axhline(w_mean, color='red', ls='--', lw=1.5,
               label=f'Mean = {w_mean:.4f}')
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Wasserstein Distance', fontsize=12, fontweight='bold')
    ax.set_title('Per-Wavelength Wasserstein Distance\n'
                 '(distribution similarity — lower is better)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def _plot_pca_tsne(your_aligned, real_aligned, output_path):
    """PCA (and t-SNE if sklearn available) 2D projection of both distributions."""
    if not HAS_SKLEARN:
        print("   ⚠️  Skipping PCA/t-SNE — sklearn not installed.")
        return

    combined = np.vstack([real_aligned, your_aligned])
    labels   = np.array(['Real'] * len(real_aligned) + ['Synthetic'] * len(your_aligned))

    pca       = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)
    var_exp   = pca.explained_variance_ratio_ * 100

    n_cols = 2 if len(combined) <= 2000 else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    # PCA plot
    ax = axes[0]
    for lab, col, mk in [('Real', 'royalblue', 'o'), ('Synthetic', 'tomato', 'x')]:
        mask = labels == lab
        ax.scatter(combined_pca[mask, 0], combined_pca[mask, 1],
                   c=col, marker=mk, alpha=0.4, s=15, label=lab)
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var)', fontsize=11)
    ax.set_title('PCA Projection\n(good coverage = red surrounds blue)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # t-SNE plot (only if dataset is small enough)
    if n_cols == 2:
        tsne         = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
        combined_tsne = tsne.fit_transform(combined)
        ax = axes[1]
        for lab, col, mk in [('Real', 'royalblue', 'o'), ('Synthetic', 'tomato', 'x')]:
            mask = labels == lab
            ax.scatter(combined_tsne[mask, 0], combined_tsne[mask, 1],
                       c=col, marker=mk, alpha=0.4, s=15, label=lab)
        ax.set_xlabel('t-SNE dim 1', fontsize=11)
        ax.set_ylabel('t-SNE dim 2', fontsize=11)
        ax.set_title('t-SNE Projection\n(good coverage = red surrounds blue)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.suptitle('Spectral Space Coverage: Synthetic vs Real\n'
                 'Hyper-Skin 2023 (Ng et al., NeurIPS 2023)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────

def compare_spectral_curves(your_wl, your_spectra, hyperskin_wl, hyperskin_spectra,
                           output_dir, n_samples=100):
    """
    Comprehensive spectral comparison. Produces 4 publication-ready plots:
      spectral_comparison.png  – mean ± std, difference, individual spectra
      distribution_plot.png   – per-wavelength box plots (IQR coverage)
      wasserstein_plot.png    – per-wavelength Wasserstein distance
      pca_tsne_plot.png       – PCA (+ t-SNE) projection of spectral space
    Returns: rmse, mae, sam, w_mean, wl_common, your_aligned, real_aligned
    """

    print("\n" + "="*80)
    print("STEP 3: COMPUTING METRICS & CREATING VISUALISATIONS")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Wavelength alignment ──────────────────────────────────────────────
    wl_min = max(your_wl.min(), hyperskin_wl.min())
    wl_max = min(your_wl.max(), hyperskin_wl.max())
    print(f"\n📊 Wavelength alignment:")
    print(f"   Your data:   {your_wl.min()}-{your_wl.max()} nm ({len(your_wl)} pts)")
    print(f"   Hyper-Skin:  {hyperskin_wl.min()}-{hyperskin_wl.max()} nm ({len(hyperskin_wl)} pts)")
    print(f"   Overlap:     {wl_min}-{wl_max} nm")

    wl_common = hyperskin_wl[(hyperskin_wl >= wl_min) & (hyperskin_wl <= wl_max)]

    # ── Align ALL spectra (used for every metric) ─────────────────────────
    print(f"\n   Aligning {len(your_spectra)} synthetic spectra …")
    your_aligned = np.array([
        align_wavelengths(your_wl, s, wl_common)[1] for s in your_spectra
    ])

    n_real_use  = min(1000, len(hyperskin_spectra))
    idx_real    = np.random.choice(len(hyperskin_spectra), n_real_use, replace=False)
    real_aligned = np.array([
        align_wavelengths(hyperskin_wl, s, wl_common)[1]
        for s in hyperskin_spectra[idx_real]
    ])
    print(f"   Aligned {n_real_use} real spectra.")

    # ── Core statistics ───────────────────────────────────────────────────
    your_mean      = your_aligned.mean(axis=0)
    your_std       = your_aligned.std(axis=0)
    hyperskin_mean = real_aligned.mean(axis=0)
    hyperskin_std  = real_aligned.std(axis=0)

    # ── Scalar metrics ────────────────────────────────────────────────────
    rmse   = float(np.sqrt(np.mean((your_mean - hyperskin_mean) ** 2)))
    mae    = float(np.mean(np.abs(your_mean - hyperskin_mean)))
    sam    = spectral_angle_mapper(your_mean, hyperskin_mean)
    w_dists, w_mean = wasserstein_per_wavelength(your_aligned, real_aligned)

    print(f"\n📐 Metrics:")
    print(f"   RMSE (mean-vs-mean)     : {rmse:.4f}")
    print(f"   MAE  (mean-vs-mean)     : {mae:.4f}")
    print(f"   SAM  (spectral angle)   : {sam:.2f}°")
    print(f"   Wasserstein (mean/band) : {w_mean:.4f}")

    n_plot = min(n_samples, len(your_aligned), len(real_aligned))

    # ═══ FIGURE 1: Mean comparison (2×2) ══════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.plot(wl_common, hyperskin_mean, 'b-', lw=2.5, label='Hyper-Skin (Real)', alpha=0.8)
    ax.fill_between(wl_common, hyperskin_mean - hyperskin_std, hyperskin_mean + hyperskin_std,
                    color='blue', alpha=0.2, label='±1 std (Real)')
    ax.plot(wl_common, your_mean, 'r--', lw=2.5, label='Your Synthetic', alpha=0.8)
    ax.fill_between(wl_common, your_mean - your_std, your_mean + your_std,
                    color='red', alpha=0.2, label='±1 std (Synthetic)')
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=13, fontweight='bold')
    ax.set_title(f'Mean Spectral Comparison\n'
                 f'RMSE: {rmse:.4f}  MAE: {mae:.4f}  SAM: {sam:.1f}°',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    diff = your_mean - hyperskin_mean
    ax.plot(wl_common, diff, 'g-', lw=2)
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.fill_between(wl_common, 0, diff, where=(diff >= 0), color='green', alpha=0.3,
                    label='Synthetic > Real')
    ax.fill_between(wl_common, 0, diff, where=(diff < 0),  color='red',   alpha=0.3,
                    label='Synthetic < Real')
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Difference (Synthetic − Real)', fontsize=13, fontweight='bold')
    ax.set_title(f'Spectral Difference  |  Mean Wasserstein: {w_mean:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for s in real_aligned[:n_plot]:
        ax.plot(wl_common, s, 'b-', alpha=0.3, lw=1)
    ax.plot(wl_common, hyperskin_mean, 'r-', lw=3, label='Mean', alpha=0.9)
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=13, fontweight='bold')
    ax.set_title(f'Hyper-Skin Real Spectra (n={n_plot})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for s in your_aligned[:n_plot]:
        ax.plot(wl_common, s, 'r-', alpha=0.3, lw=1)
    ax.plot(wl_common, your_mean, 'b-', lw=3, label='Mean', alpha=0.9)
    ax.set_xlabel('Wavelength (nm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reflectance', fontsize=13, fontweight='bold')
    ax.set_title(f'Your Synthetic Spectra (n={n_plot})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p1 = output_path / "spectral_comparison.png"
    plt.savefig(p1, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {p1}")
    plt.close()

    # ═══ FIGURES 2–4: Distribution, Wasserstein, PCA/t-SNE ════════════════
    _plot_distribution_boxplots(your_aligned, real_aligned, wl_common,
                                output_path / "distribution_plot.png")
    _plot_wasserstein(w_dists, wl_common, w_mean,
                      output_path / "wasserstein_plot.png")
    _plot_pca_tsne(your_aligned, real_aligned,
                   output_path / "pca_tsne_plot.png")

    return rmse, mae, sam, w_mean, wl_common, your_aligned, real_aligned


def generate_validation_report(rmse, mae, sam, w_mean, coverage,
                               delta_e_mean, n_subjects, output_dir):
    """Generate comprehensive text validation report with all publication metrics."""

    report_path = Path(output_dir) / "validation_report.txt"

    def grade(val, thresholds, labels):
        for t, l in zip(thresholds, labels):
            if val < t:
                return l
        return labels[-1]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SPECTRAL VALIDATION REPORT\n")
        f.write("Synthetic Skin Data vs Hyper-Skin 2023 Real Measurements\n")
        f.write("Ng et al., NeurIPS Datasets and Benchmarks, 2023\n")
        f.write(f"Subjects loaded: {n_subjects}\n")
        f.write("=" * 80 + "\n\n")

        f.write("QUANTITATIVE METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"RMSE  (mean-vs-mean spectral)      : {rmse:.6f}\n")
        f.write(f"MAE   (mean-vs-mean spectral)      : {mae:.6f}\n")
        f.write(f"SAM   (spectral angle, degrees)    : {sam:.4f}°\n")
        f.write(f"Wasserstein dist (mean across bands): {w_mean:.6f}\n")
        if delta_e_mean is not None:
            f.write(f"Delta E 76 (mean, real vs synth)   : {delta_e_mean:.4f}\n")
        f.write("\nNearest-Neighbour Coverage:\n")
        for t, pct in coverage.items():
            f.write(f"   % real spectra within RMSE {t:.2f} : {pct:.1f}%\n")

        f.write("\nINTERPRETATION:\n")
        f.write("-" * 80 + "\n")

        rmse_grade = grade(rmse, [0.03, 0.05, 0.10, 0.20],
                           ["✅ EXCELLENT (<0.03)", "✅ VERY GOOD (<0.05)",
                            "✅ GOOD (<0.10)", "⚠️  MODERATE (<0.20)", "❌ POOR (>=0.20)"])
        sam_grade  = grade(sam,  [2.0, 5.0, 10.0],
                           ["✅ EXCELLENT (<2°)", "✅ GOOD (<5°)",
                            "⚠️  FAIR (<10°)", "❌ POOR (>=10°)"])
        w_grade    = grade(w_mean, [0.02, 0.05],
                           ["✅ EXCELLENT (<0.02)", "✅ GOOD (<0.05)", "⚠️  MODERATE (>=0.05)"])

        f.write(f"RMSE  : {rmse_grade}\n")
        f.write(f"SAM   : {sam_grade}\n")
        f.write(f"W1    : {w_grade}\n")

        if delta_e_mean is not None:
            de_grade = grade(delta_e_mean, [2.0, 5.0, 10.0],
                             ["✅ EXCELLENT ΔE<2 (imperceptible)",
                              "✅ GOOD ΔE<5 (minor difference)",
                              "⚠️  FAIR ΔE<10 (noticeable)",
                              "❌ POOR ΔE>=10 (significant)"])
            f.write(f"ΔE76  : {de_grade}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("OUTPUT FILES:\n")
        f.write("=" * 80 + "\n")
        f.write("  spectral_comparison.png — mean spectra, ±std, difference curve\n")
        f.write("  distribution_plot.png   — per-wavelength box plots (IQR coverage)\n")
        f.write("  wasserstein_plot.png    — per-wavelength distribution distance\n")
        f.write("  pca_tsne_plot.png       — PCA/t-SNE spectral space coverage\n")
        f.write("  coverage_plot.png       — % real spectra covered vs RMSE threshold\n")

    print(f"\n✅ Saved: {report_path}")
    with open(report_path, 'r', encoding='utf-8') as f:
        print("\n" + f.read())


def main():
    """Main validation workflow"""

    np.random.seed(42)   # Reproducibility — fixed seed for all random sampling

    print("\n" + "🔬 COMPLETE SPECTRAL VALIDATION 🔬".center(80))
    print("Comparing Your Synthetic Data vs Hyper-Skin Real Measurements\n")

    # ── Step 1: Load synthetic data ───────────────────────────────────────
    df, your_wl, your_spectra, params = load_your_synthetic_data(YOUR_CSV_PATH)

    # ── Step 2: Load ALL available Hyper-Skin subjects ────────────────────
    print("\n" + "="*80)
    print("STEP 2: LOADING HYPER-SKIN REAL DATA (ALL SUBJECTS)")
    print("="*80)

    hyperskin_folder = Path(HYPERSKIN_VIS_FOLDER)
    mat_files = sorted(hyperskin_folder.glob("*.mat"))

    if len(mat_files) == 0:
        print(f"❌ No .mat files found in {hyperskin_folder}")
        print("   Please update HYPERSKIN_VIS_FOLDER path!")
        return

    print(f"✅ Found {len(mat_files)} Hyper-Skin .mat files")

    # Load first file to get wavelength array
    hyperskin_wl, first_cube, keys = load_hyperskin_sample(mat_files[0])
    if first_cube is None:
        print(f"❌ Failed to load first sample — keys: {keys}"); return

    # Aggregate spectra from up to 20 subjects (500 pixels each) for speed
    all_hyperskin_spectra = []
    n_subjects_loaded = 0
    MAX_SUBJECTS = 20
    PX_PER_SUBJECT = 500

    for mat_file in mat_files[:MAX_SUBJECTS]:
        wl, cube, _ = load_hyperskin_sample(mat_file)
        if cube is None:
            continue
        spectra = extract_skin_region(cube, region='center')
        if len(spectra) == 0:
            continue
        n_take = min(PX_PER_SUBJECT, len(spectra))
        idx    = np.random.choice(len(spectra), n_take, replace=False)
        all_hyperskin_spectra.append(spectra[idx])
        n_subjects_loaded += 1

    if len(all_hyperskin_spectra) == 0:
        print("❌ No valid real spectra loaded."); return

    hyperskin_spectra = np.vstack(all_hyperskin_spectra)
    print(f"\n✅ Loaded {n_subjects_loaded} subjects → {len(hyperskin_spectra):,} real spectra")

    # ── Step 3: Random synthetic sample ───────────────────────────────────
    n_synth = min(1000, len(your_spectra))
    idx     = np.random.choice(len(your_spectra), n_synth, replace=False)
    your_spectra_sample = your_spectra[idx]

    # ── Step 4: Spectral comparison + all plots ────────────────────────────
    rmse, mae, sam, w_mean, wl_common, your_aligned, real_aligned = \
        compare_spectral_curves(
            your_wl, your_spectra_sample,
            hyperskin_wl, hyperskin_spectra,
            OUTPUT_DIR, n_samples=100
        )

    # ── Step 5: Nearest-neighbour coverage + coverage curve ───────────────
    print("\n📊 Computing nearest-neighbour coverage …")
    coverage, nn_dists = nearest_neighbour_coverage(
        your_aligned, real_aligned, thresholds=(0.03, 0.05, 0.10)
    )
    for t, pct in coverage.items():
        print(f"   Coverage RMSE < {t:.2f} : {pct:.1f}%")

    _plot_coverage_curve(nn_dists, Path(OUTPUT_DIR) / "coverage_plot.png")

    # ── Step 6: Delta E (mean synthetic mean vs real mean, in Lab space) ──
    print("\n🎨 Computing Delta E 76 (CIE Lab) …")
    try:
        syn_lab  = spectra_to_lab(your_aligned,  wl_common)
        real_lab = spectra_to_lab(real_aligned,  wl_common)
        # Compare mean-Lab vectors (representative summary for the report)
        de_mean_vec = delta_e76(syn_lab.mean(axis=0, keepdims=True),
                                real_lab.mean(axis=0, keepdims=True))
        delta_e_mean = float(de_mean_vec[0])
        # Per-sample Delta E distribution for printing
        n_pairs = min(len(syn_lab), len(real_lab))
        de_per_sample = delta_e76(syn_lab[:n_pairs], real_lab[:n_pairs])
        print(f"   ΔE76 (mean Lab vs mean Lab) : {delta_e_mean:.4f}")
        print(f"   ΔE76 per-sample median      : {np.median(de_per_sample):.4f}")
    except Exception as e:
        print(f"   ⚠️  Delta E computation failed: {e}")
        delta_e_mean = None

    # ── Step 7: Generate report ────────────────────────────────────────────
    generate_validation_report(rmse, mae, sam, w_mean, coverage,
                                delta_e_mean, n_subjects_loaded, OUTPUT_DIR)

    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nSummary:")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  SAM   : {sam:.2f}°")
    print(f"  W1    : {w_mean:.4f}")
    if delta_e_mean is not None:
        print(f"  ΔE76  : {delta_e_mean:.2f}")
    print(f"\nOutput files:")
    for fname in ["spectral_comparison.png", "distribution_plot.png",
                  "wasserstein_plot.png", "pca_tsne_plot.png",
                  "coverage_plot.png", "validation_report.txt"]:
        print(f"  {OUTPUT_DIR}/{fname}")


if __name__ == "__main__":
    main()