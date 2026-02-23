# Deep Albedo Configuration File

# Model Configuration
NUM_NEURONS = 75
NUM_LAYERS = 2

# Default Checkpoint
DEFAULT_CHECKPOINT = "checkpoints/2025-12-31_18-54-39/best.pt"

# Image Processing
DEFAULT_IMAGE_SIZE = (256, 256)  # (width, height)

# Parameter Normalization Ranges
# Format: (scale, offset) for each parameter
PARAM_RANGES = {
    'Cm': (0.62, 0.001),  # Melanin concentration
    'Ch': (0.31, 0.001),  # Hemoglobin concentration
    'Bm': (0.8, 0.2),     # Melanin baseline
    'Bh': (0.3, 0.6),     # Hemoglobin baseline
    'T': (0.2, 0.05)      # Thickness
}

# Output Configuration
DEFAULT_OUTPUT_DIR = "output"
SAVE_FORMAT = "png"
DPI = 100

# Visualization
PLOT_STYLE = "dark_background"
SHOW_GRID = False