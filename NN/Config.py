# Deep Albedo - Configuration
# Single source of truth for hyperparameters and file paths.
# Model architecture constants are also in model.py (PARAM_MINS / PARAM_MAXS).

# ── Model Architecture ────────────────────────────────────────────────────────
ENC_HIDDEN_DIM = 70     # encoder hidden layer width
ENC_NUM_LAYERS = 4      # encoder hidden layer count
DEC_HIDDEN_DIM = 256    # decoder hidden layer width
DEC_NUM_LAYERS = 4      # decoder hidden layer count

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 4096
NUM_EPOCHS   = 400
LR           = 1e-4
MIN_LR       = 1e-6
LOSS_WEIGHTS = (0.3, 0.1, 0.6)   # (parameter, albedo, end-to-end)
RANDOM_SEED  = 7

# ── Default Paths ─────────────────────────────────────────────────────────────
# Auto-discover the most-recently-modified best.pt under checkpoints/.
# Falls back to a literal string only if no checkpoint exists yet.
import os as _os, glob as _glob

def _latest_checkpoint(base="checkpoints"):
    """Return the best.pt from the most recently modified training run."""
    _here = _os.path.dirname(_os.path.abspath(__file__))
    pattern = _os.path.join(_here, base, "*", "best.pt")
    candidates = _glob.glob(pattern)
    if not candidates:
        return _os.path.join(base, "best.pt")   # placeholder if none exist
    return max(candidates, key=_os.path.getmtime)

DEFAULT_CHECKPOINT = _latest_checkpoint()
DEFAULT_LUT_PATH   = "../simulation/data/lut_rgb.csv"
DEFAULT_OUTPUT_DIR = "output"

# ── Image Processing ──────────────────────────────────────────────────────────
DEFAULT_IMAGE_SIZE = (256, 256)   # (width, height)
SAVE_FORMAT        = "png"
DPI                = 100

# ── Biological Parameter Bounds ───────────────────────────────────────────────
# Order: Cm (melanin), Ch (haemoglobin), Bm (mel. blend), Bh (blood oxy), T (thickness)
PARAM_NAMES = ['Cm',   'Ch',   'Bm',  'Bh',   'T'    ]
PARAM_MINS  = [0.05,   0.02,   0.0,   0.60,   0.005  ]
PARAM_MAXS  = [0.50,   0.20,   1.0,   0.98,   0.020  ]
