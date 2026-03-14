"""
Stage 3 & 4 Pipeline — Aliaga et al. (2022)
Aligned to your actual codebase (model.py / train.py / dataset.py)

Stage 3 : Algebraic parameter map editing  (8 manipulation cases from Figure 10)
Stage 4 : Decoder MLP reconstruction  →  edited RGB albedo

Parameter channel order  (matches dataset.py column order):
  Ch 0  Cm  — Melanin concentration          range [0.05,  0.50]
  Ch 1  Ch  — Blood  concentration           range [0.02,  0.20]
  Ch 2  Bm  — Melanin blend (type ratio)     range [0.00,  1.00]  (0=pheo, 1=eu)
  Ch 3  Bh  — Blood oxygenation              range [0.60,  0.98]
  Ch 4  T   — Epidermal thickness            range [0.005, 0.020]

Checkpoint format  (from train.py):
  torch.save({
      "model_state_dict":     model.state_dict(),   <- AutoEncoder (encoder + decoder)
      "optimizer_state_dict": ...,
      "epoch":                ...,
      "architecture":         ...,
  }, "checkpoints/<run>/best.pt")
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 0.  Re-use your model classes directly from model.py
#     Place this script in the same directory as model.py, or adjust sys.path
# ---------------------------------------------------------------------------

try:
    from model import Encoder, Decoder, AutoEncoder, PARAM_MINS, PARAM_MAXS
except ImportError as e:
    raise ImportError(
        "Cannot import from model.py. Make sure stage3_4_pipeline.py "
        "lives in the same directory as model.py."
    ) from e


# ---------------------------------------------------------------------------
# 1.  Parameter metadata  (must match dataset.py column order exactly)
# ---------------------------------------------------------------------------

PARAM_NAMES  = ["Cm", "Ch", "Bm", "Bh", "T"]
PARAM_RANGES = list(zip(PARAM_MINS, PARAM_MAXS))
# [(0.05, 0.50), (0.02, 0.20), (0.00, 1.00), (0.60, 0.98), (0.005, 0.020)]

NUM_PARAMS = 5
NUM_RGB    = 3


# ---------------------------------------------------------------------------
# 2.  Checkpoint loader
#     Reads "model_state_dict" from best.pt / last.pt and returns the
#     separate Encoder and Decoder submodules ready for inference.
# ---------------------------------------------------------------------------

def load_checkpoint(
    ckpt_path: str,
    device: torch.device,
    enc_hidden_dim: int = 70,
    enc_num_layers: int = 4,
    dec_hidden_dim: int = 256,
    dec_num_layers: int = 4,
):
    """
    Load encoder and decoder from a training checkpoint.

    The checkpoint is saved by train.py as:
        torch.save({"model_state_dict": model.state_dict(), ...}, path)
    where model is AutoEncoder(encoder, decoder).
    The state_dict contains keys prefixed with "encoder.*" and "decoder.*".

    Parameters
    ----------
    ckpt_path      : path to best.pt or last.pt
    device         : torch device to load onto
    enc_hidden_dim : must match value used during training  (default 70)
    enc_num_layers : must match value used during training  (default 4)
    dec_hidden_dim : must match value used during training  (default 256)
    dec_num_layers : must match value used during training  (default 4)

    Returns
    -------
    encoder, decoder  — both in eval() mode on device
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Locate the state dict — handle all formats train.py might produce
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("encoder.") for k in ckpt):
        state = ckpt                          # saved directly as state_dict
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()             # full model object
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {ckpt_path}.\n"
            f"Top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    # Read architecture dims from checkpoint if available (written by extract_architecture)
    if "architecture" in ckpt:
        arch           = ckpt["architecture"]
        enc_hidden_dim = arch["encoder"]["hidden_dim"]
        enc_num_layers = arch["encoder"]["num_layers"]
        dec_hidden_dim = arch["decoder"]["hidden_dim"]
        dec_num_layers = arch["decoder"]["num_layers"]
        print(f"  Architecture from checkpoint: "
              f"enc(hidden={enc_hidden_dim}, layers={enc_num_layers})  "
              f"dec(hidden={dec_hidden_dim}, layers={dec_num_layers})")

    # Build submodels
    encoder = Encoder(hidden_dim=enc_hidden_dim, num_layers=enc_num_layers).to(device)
    decoder = Decoder(hidden_dim=dec_hidden_dim, num_layers=dec_num_layers).to(device)

    # Split AutoEncoder state_dict into encoder / decoder halves
    enc_state = {k[len("encoder."):]: v
                 for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v
                 for k, v in state.items() if k.startswith("decoder.")}

    if not enc_state:
        raise KeyError(
            "No 'encoder.*' keys found in the checkpoint state_dict.\n"
            f"Key prefixes present: {sorted({k.split('.')[0] for k in state})}"
        )
    if not dec_state:
        raise KeyError(
            "No 'decoder.*' keys found in the checkpoint state_dict.\n"
            f"Key prefixes present: {sorted({k.split('.')[0] for k in state})}"
        )

    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    encoder.eval()
    decoder.eval()

    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("best_train_loss", ckpt.get("train_total_loss", "?"))
    print(f"  Loaded: {ckpt_path.name}  |  epoch={epoch}  |  loss={loss}")

    return encoder, decoder


# ---------------------------------------------------------------------------
# 3.  Stage 3 — Parameter Editing
# ---------------------------------------------------------------------------

@dataclass
class EditConfig:
    """
    One Figure-10 manipulation case.

    ch_scale : {channel_index: scale_factor}  multiplicative edits
    ch_set   : {channel_index: value}          absolute set edits
    spatial  : if True, ch_set is applied only inside vitiligo_mask
    """
    name        : str
    label       : str
    description : str
    ch_scale    : dict = field(default_factory=dict)
    ch_set      : dict = field(default_factory=dict)
    spatial     : bool = False


# All 8 cases from Figure 10 — Aliaga et al. (2022)
#
# Channel mapping:
#   0 = Cm  (melanin concentration)      range [0.05, 0.50]
#   1 = Ch  (blood concentration)        range [0.02, 0.20]
#   2 = Bm  (melanin blend / type)       range [0.00, 1.00]
#   3 = Bh  (blood oxygenation)          range [0.60, 0.98]
#   4 = T   (epidermal thickness)        range [0.005, 0.020]

EDIT_CONFIGS = [
    EditConfig(
        name        = "original",
        label       = "(a) Original",
        description = "Unmodified — baseline reference",
    ),
    EditConfig(
        name        = "deoxy",
        label       = "(b) Deoxygenated blood",
        description = "Bh -> 0.60 (min)  |  fully deoxygenated -> purplish tint",
        ch_set      = {3: 0.60},          # Bh floor = PARAM_MINS[3]
    ),
    EditConfig(
        name        = "oxy",
        label       = "(c) Oxygenated blood",
        description = "Bh -> 0.98 (max)  |  fully oxygenated -> saturated red",
        ch_set      = {3: 0.98},          # Bh ceiling = PARAM_MAXS[3]
    ),
    EditConfig(
        name        = "thin",
        label       = "(d) Epidermal thinning",
        description = "T -> 0.005 (T_MIN)  |  thinner -> more blood visible",
        ch_set      = {4: 0.005},         # T floor
    ),
    EditConfig(
        name        = "thick",
        label       = "(e) Epidermal thickening",
        description = "T -> 0.020 (T_MAX)  |  thicker epidermis -> paler skin",
        ch_set      = {4: 0.020},         # T ceiling
    ),
    EditConfig(
        name        = "tan",
        label       = "(f) Tanning",
        description = "Cm x 1.4  +  Bm -> 0.0  |  +40% melanin, full pheomelanin",
        ch_scale    = {0: 1.4},           # Cm +40%
        ch_set      = {2: 0.0},           # Bm -> 0 (full pheomelanin)
    ),
    EditConfig(
        name        = "flush",
        label       = "(g) Flushing",
        description = "Ch x 1.7  +  Bh -> 0.98  |  +70% blood, fully oxygenated",
        ch_scale    = {1: 1.7},           # Ch +70%
        ch_set      = {3: 0.98},          # Bh -> max
    ),
    EditConfig(
        name        = "vitiligo",
        label       = "(h) Vitiligo",
        description = "Cm -> 0.05 (min) in mask  |  spatially selective depigmentation",
        ch_set      = {0: 0.05},          # Cm -> PARAM_MINS[0]  (not zero — encoder min)
        spatial     = True,
    ),
]


def apply_edit(
    param_maps: np.ndarray,
    config: EditConfig,
    vitiligo_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Stage 3: apply one EditConfig to a (H, W, 5) parameter map.

    Parameters
    ----------
    param_maps    : (H, W, 5)  float32  from encoder inference
    config        : EditConfig describing the manipulation
    vitiligo_mask : (H, W)     uint8/bool  required when config.spatial=True

    Returns
    -------
    edited        : (H, W, 5)  float32  clamped to PARAM_MINS / PARAM_MAXS
    """
    edited = param_maps.copy()

    # 1. Multiplicative edits
    for ch, factor in config.ch_scale.items():
        edited[:, :, ch] = edited[:, :, ch] * factor

    # 2. Absolute set edits
    for ch, value in config.ch_set.items():
        if config.spatial:
            assert vitiligo_mask is not None, (
                f"apply_edit: vitiligo_mask required for spatial edit (case '{config.name}')"
            )
            edited[:, :, ch] = np.where(
                vitiligo_mask.astype(bool), value, edited[:, :, ch]
            )
        else:
            edited[:, :, ch] = value

    # 3. Clamp each channel to its physical range
    #    Uses the same bounds as Encoder.register_buffer so the decoder
    #    never receives out-of-distribution inputs.
    for ch, (lo, hi) in enumerate(PARAM_RANGES):
        edited[:, :, ch] = np.clip(edited[:, :, ch], lo, hi)

    return edited


def run_stage3(
    param_maps: np.ndarray,
    vitiligo_mask: Optional[np.ndarray] = None,
    cases: Optional[list] = None,
) -> dict:
    """
    Run all 8 (or a named subset) of Stage-3 edits.

    Parameters
    ----------
    param_maps    : (H, W, 5)  from encoder inference
    vitiligo_mask : (H, W)     binary mask for case 'vitiligo'
    cases         : list of case names to run; None -> all 8

    Returns
    -------
    dict  name -> edited (H, W, 5) array
    """
    configs = {c.name: c for c in EDIT_CONFIGS}
    if cases is not None:
        configs = {k: v for k, v in configs.items() if k in cases}

    results = {}
    for name, cfg in configs.items():
        results[name] = apply_edit(param_maps, cfg, vitiligo_mask)
        e = results[name]
        print(f"  [Stage 3] {cfg.label:35s} "
              f"Cm={e[:,:,0].mean():.4f}  "
              f"Ch={e[:,:,1].mean():.4f}  "
              f"Bh={e[:,:,3].mean():.4f}")

    return results


# ---------------------------------------------------------------------------
# 4.  Stage 4 — Decoder inference  (edited params -> RGB albedo)
# ---------------------------------------------------------------------------

def run_stage4(
    edited_maps: np.ndarray,
    decoder: Decoder,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Stage 4: run the decoder on one edited parameter map.

    Note: Decoder in model.py has no Sigmoid, so output is raw linear.
    We apply np.clip(0, 1) after to ensure valid albedo values.

    Parameters
    ----------
    edited_maps : (H, W, 5)  float32  from Stage 3
    decoder     : loaded Decoder in eval() mode
    device      : torch device
    batch_size  : pixels per forward pass  (4096 = paper batch size)

    Returns
    -------
    albedo_rgb  : (H, W, 3)  float32  in [0, 1]
    """
    H, W, _ = edited_maps.shape
    pixels   = edited_maps.reshape(-1, NUM_PARAMS).astype(np.float32)
    N        = pixels.shape[0]
    out_buf  = np.empty((N, NUM_RGB), dtype=np.float32)

    decoder.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            batch = torch.from_numpy(pixels[start:end]).to(device)
            out_buf[start:end] = decoder(batch).cpu().numpy()

    # Decoder has no Sigmoid -> clip to valid albedo range
    return np.clip(out_buf.reshape(H, W, NUM_RGB), 0.0, 1.0)


def run_stage4_all(
    stage3_results: dict,
    decoder: Decoder,
    device: torch.device,
    batch_size: int = 4096,
) -> dict:
    """Run Stage 4 for every edited map produced by Stage 3."""
    albedos = {}
    for name, edited_maps in stage3_results.items():
        print(f"  [Stage 4] Decoding '{name}' ...")
        albedos[name] = run_stage4(edited_maps, decoder, device, batch_size)
    return albedos


# ---------------------------------------------------------------------------
# 5.  Encoder inference helper  (Stage 2 -> Stage 3 bridge)
# ---------------------------------------------------------------------------

def encode_image(
    rgb_image: np.ndarray,
    encoder: Encoder,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Run the encoder on a full RGB face image to produce parameter maps.

    Parameters
    ----------
    rgb_image  : (H, W, 3)  float32  albedo in [0, 1]
    encoder    : loaded Encoder in eval() mode
    device     : torch device
    batch_size : pixels per forward pass

    Returns
    -------
    param_maps : (H, W, 5)  float32  already clamped by Encoder.forward()
    """
    H, W, _ = rgb_image.shape
    pixels   = rgb_image.reshape(-1, NUM_RGB).astype(np.float32)
    N        = pixels.shape[0]
    out_buf  = np.empty((N, NUM_PARAMS), dtype=np.float32)

    encoder.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            batch = torch.from_numpy(pixels[start:end]).to(device)
            out_buf[start:end] = encoder(batch).cpu().numpy()

    return out_buf.reshape(H, W, NUM_PARAMS)


# ---------------------------------------------------------------------------
# 6.  Visualisation
# ---------------------------------------------------------------------------

PARAM_CMAPS  = ["YlOrBr", "Reds",    "RdYlBu_r", "RdBu_r",      "viridis"]
PARAM_LABELS = ["Cm\nMelanin", "Ch\nBlood", "Bm\nMelanin blend",
                "Bh\nOxygenation", "T\nThickness"]


def visualise_param_maps(
    param_maps: np.ndarray,
    title: str = "Parameter Maps",
    save_path: Optional[Path] = None,
) -> None:
    """Plot all 5 biophysical parameter maps as heatmaps (Figure 8 style)."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (ax, cmap, label, (vmin, vmax)) in enumerate(
        zip(axes, PARAM_CMAPS, PARAM_LABELS, PARAM_RANGES)
    ):
        im = ax.imshow(param_maps[:, :, i], cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def visualise_figure10(
    albedos: dict,
    save_path: Optional[Path] = None,
) -> None:
    """Replicate Figure 10 layout: 8 reconstructed albedos in a row."""
    ordered   = [c.name for c in EDIT_CONFIGS if c.name in albedos]
    n         = len(ordered)
    label_map = {c.name: c.label       for c in EDIT_CONFIGS}
    desc_map  = {c.name: c.description for c in EDIT_CONFIGS}

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle("Figure 10 — Biophysical Skin Parameter Manipulation",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, name in zip(axes, ordered):
        ax.imshow(np.clip(albedos[name], 0, 1))
        ax.set_title(label_map[name], fontsize=10, pad=4)
        ax.set_xlabel(desc_map[name], fontsize=7, wrap=True)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def visualise_delta_maps(
    maps_t0: np.ndarray,
    maps_t1: np.ndarray,
    title: str = "Delta Parameter Maps  (T1 - T0)",
    save_path: Optional[Path] = None,
) -> None:
    """
    Diverging heatmaps of per-channel change between two timepoints.
    Used for longitudinal chemotherapy toxicity monitoring.
    """
    delta     = maps_t1 - maps_t0
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (ax, label) in enumerate(zip(axes, PARAM_LABELS)):
        d   = delta[:, :, i]
        lim = max(abs(float(np.percentile(d, 1))),
                  abs(float(np.percentile(d, 99))), 1e-6)
        im  = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim,
                        interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Delta {label}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 7.  CTCAE grade classifier  (research extension)
# ---------------------------------------------------------------------------

# Thresholds for mean delta -> CTCAE Grade 0-4
# Calibrate these against your clinical ground-truth labels.
# Ch range = [0.02, 0.20] so thresholds are in that unit.
CTCAE_THRESHOLDS = {
    "Ch": [0.005, 0.010, 0.030, 0.060],   # blood up -> erythema
    "Cm": [0.010, 0.025, 0.060, 0.120],   # melanin up -> hyperpigmentation
}


def classify_ctcae(
    maps_t0: np.ndarray,
    maps_t1: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Classify CTCAE skin toxicity grade from longitudinal parameter maps.

    Parameters
    ----------
    maps_t0  : (H, W, 5)  baseline
    maps_t1  : (H, W, 5)  during / post treatment
    roi_mask : (H, W)     optional ROI — restrict to facial skin region

    Returns
    -------
    dict: delta_Ch, delta_Cm, grade_Ch, grade_Cm, grade_overall
    """
    delta = maps_t1 - maps_t0

    if roi_mask is not None:
        m    = roi_mask.astype(bool)
        d_ch = float(delta[:, :, 1][m].mean())   # Ch
        d_cm = float(delta[:, :, 0][m].mean())   # Cm
    else:
        d_ch = float(delta[:, :, 1].mean())
        d_cm = float(delta[:, :, 0].mean())

    def _grade(val: float, thresholds: list) -> int:
        for g, thresh in enumerate(thresholds, start=1):
            if abs(val) < thresh:
                return g - 1
        return 4

    grade_ch = _grade(d_ch, CTCAE_THRESHOLDS["Ch"])
    grade_cm = _grade(d_cm, CTCAE_THRESHOLDS["Cm"])

    return {
        "delta_Ch":      d_ch,
        "delta_Cm":      d_cm,
        "grade_Ch":      grade_ch,    # erythema / blood
        "grade_Cm":      grade_cm,    # hyperpigmentation / melanin
        "grade_overall": max(grade_ch, grade_cm),
    }


# ---------------------------------------------------------------------------
# 8.  Full pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    ckpt_path: str,
    rgb_image: np.ndarray,
    device: torch.device,
    output_dir: Path = Path("outputs/stage3_4"),
    vitiligo_mask: Optional[np.ndarray] = None,
    cases: Optional[list] = None,
    rgb_image_t1: Optional[np.ndarray] = None,
    roi_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    End-to-end  Stage 2 -> Stage 3 -> Stage 4  from a single checkpoint.

    Parameters
    ----------
    ckpt_path     : path to best.pt or last.pt from your training run
    rgb_image     : (H, W, 3)  float32  albedo in [0, 1] — baseline / T0
    device        : torch device
    output_dir    : directory for saved figures
    vitiligo_mask : (H, W)     binary mask for case 'vitiligo'
    cases         : list of case names; None -> all 8
    rgb_image_t1  : (H, W, 3)  optional second timepoint for CTCAE analysis
    roi_mask      : (H, W)     optional ROI mask for CTCAE grading

    Returns
    -------
    dict: encoder, decoder, param_maps, stage3_results,
          stage4_results, ctcae (if t1 provided)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder and decoder from the single checkpoint
    print("\n" + "=" * 60)
    print("LOADING CHECKPOINT")
    print("=" * 60)
    encoder, decoder = load_checkpoint(ckpt_path, device)

    # Stage 2: encode baseline image -> parameter maps
    print("\n" + "=" * 60)
    print("STAGE 2  --  Encoder Inference")
    print("=" * 60)
    param_maps = encode_image(rgb_image, encoder, device)
    print(f"  param_maps shape : {param_maps.shape}")
    for i, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, PARAM_RANGES)):
        v = param_maps[:, :, i]
        print(f"  {name:4s}  min={v.min():.5f}  max={v.max():.5f}  "
              f"mean={v.mean():.5f}  range=[{lo}, {hi}]")

    visualise_param_maps(
        param_maps,
        title="Stage 2 Output -- Estimated Parameter Maps (T0)",
        save_path=output_dir / "param_maps_T0.png",
    )

    # Stage 3: edit parameter maps
    print("\n" + "=" * 60)
    print("STAGE 3  --  Parameter Editing")
    print("=" * 60)
    stage3_results = run_stage3(param_maps, vitiligo_mask, cases)

    # Stage 4: decode -> RGB albedos
    print("\n" + "=" * 60)
    print("STAGE 4  --  Decoder Reconstruction")
    print("=" * 60)
    stage4_results = run_stage4_all(stage3_results, decoder, device)

    visualise_figure10(
        stage4_results,
        save_path=output_dir / "figure10_reconstruction.png",
    )

    output = {
        "encoder":        encoder,
        "decoder":        decoder,
        "param_maps":     param_maps,
        "stage3_results": stage3_results,
        "stage4_results": stage4_results,
    }

    # Optional longitudinal CTCAE analysis
    if rgb_image_t1 is not None:
        print("\n" + "=" * 60)
        print("LONGITUDINAL  --  CTCAE Toxicity Grading")
        print("=" * 60)
        param_maps_t1 = encode_image(rgb_image_t1, encoder, device)

        visualise_param_maps(
            param_maps_t1,
            title="Stage 2 Output -- Estimated Parameter Maps (T1)",
            save_path=output_dir / "param_maps_T1.png",
        )
        visualise_delta_maps(
            param_maps, param_maps_t1,
            title="Delta Parameter Maps  (T1 - T0)  -- Chemo Toxicity",
            save_path=output_dir / "delta_maps_T0_T1.png",
        )

        ctcae = classify_ctcae(param_maps, param_maps_t1, roi_mask)
        output["ctcae"]         = ctcae
        output["param_maps_t1"] = param_maps_t1

        print(f"  dCh (blood)   = {ctcae['delta_Ch']:+.5f}"
              f"  ->  CTCAE Grade {ctcae['grade_Ch']}  (erythema)")
        print(f"  dCm (melanin) = {ctcae['delta_Cm']:+.5f}"
              f"  ->  CTCAE Grade {ctcae['grade_Cm']}  (hyperpigmentation)")
        print(f"  Overall Grade = {ctcae['grade_overall']}")

    return output


# ---------------------------------------------------------------------------
# 9.  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 3 & 4 -- biophysical skin parameter editing + reconstruction"
    )
    parser.add_argument("--ckpt",     required=True,
                        help="Path to best.pt or last.pt  "
                             "e.g. checkpoints/2026-03-14_.../best.pt")
    parser.add_argument("--image",    required=True,
                        help="Path to input albedo image (H x W x 3, PNG/JPG)")
    parser.add_argument("--image-t1",
                        help="Optional second timepoint image for CTCAE analysis")
    parser.add_argument("--cases",    nargs="+",
                        choices=[c.name for c in EDIT_CONFIGS],
                        help="Subset of cases to run (default: all 8)")
    parser.add_argument("--outdir",   default="outputs/stage3_4",
                        help="Directory for saved figures")
    parser.add_argument("--cpu",      action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    from PIL import Image

    def _load_rgb(path: str) -> np.ndarray:
        return np.array(
            Image.open(path).convert("RGB"), dtype=np.float32
        ) / 255.0

    DEVICE = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {DEVICE}")

    results = run_pipeline(
        ckpt_path    = args.ckpt,
        rgb_image    = _load_rgb(args.image),
        device       = DEVICE,
        output_dir   = Path(args.outdir),
        cases        = args.cases,
        rgb_image_t1 = _load_rgb(args.image_t1) if args.image_t1 else None,
    )

    print("\nDone. Figures saved to:", args.outdir)