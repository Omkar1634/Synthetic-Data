"""
Deep Albedo - Dataset

Loads the Monte Carlo LUT CSV and provides a PyTorch Dataset for training.

Usage:
    from dataset import AEDataset, load_lut

    x_train, x_test, y_train, y_test = load_lut("../simulation/data/lut_rgb.csv")
    train_ds = AEDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class AEDataset(Dataset):
    """
    PyTorch dataset for autoencoder training.

    Each item returns six tensors to support three simultaneous loss heads:
        enc_in   — RGB input for encoder         (shape 3)
        dec_in   — param input for decoder        (shape 5)
        end_in   — RGB input for end-to-end path  (shape 3)
        enc_true — ground-truth params            (shape 5)
        dec_true — ground-truth RGB               (shape 3)
        end_true — ground-truth RGB               (shape 3)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()  # RGB,    (N, 3), normalized [0,1]
        self.y = torch.from_numpy(y).float()  # params, (N, 5)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_i, y_i = self.x[idx], self.y[idx]
        return x_i, y_i, x_i, y_i, x_i, x_i


def load_lut(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load and preprocess the Monte Carlo LUT CSV.

    Steps:
      1. Select the 8 relevant columns and rename them.
      2. Drop the artefact first row, coerce to float, drop NaN.
      3. Deduplicate on rounded RGB (keep the row closest to parameter midpoints).
      4. Normalise RGB to [0, 1].
      5. Split into train/test sets.

    Args:
        csv_path:     Path to lut_rgb.csv (or variant).
        test_size:    Fraction held out for validation (default 0.2).
        random_state: RNG seed for reproducible splits.

    Returns:
        x_train, x_test — float32 arrays, shape (N, 3), RGB in [0, 1]
        y_train, y_test — float32 arrays, shape (N, 5), params in physical units
    """
    data = pd.read_csv(csv_path)

    df = data[['melanin_concentration(Cm)', 'blood_concentration(Ch)',
               'melanin_blend(Bm)', 'BloodOxy', 'epidermis_thickness(T)',
               'sR', 'sG', 'sB']].rename(columns={
        'melanin_concentration(Cm)': 'Cm',
        'blood_concentration(Ch)':  'Ch',
        'melanin_blend(Bm)':        'Bm',
        'BloodOxy':                 'Bh',
        'epidermis_thickness(T)':   'T',
    })

    df = df.iloc[1:].apply(pd.to_numeric, errors='coerce').dropna()

    # Deduplicate on rounded RGB: keep the row whose params are closest to midpoints
    upper = [0.50, 0.20, 1.0, 0.98, 0.020]
    lower = [0.05, 0.02, 0.0, 0.60, 0.005]
    mids  = [(u + l) / 2 for u, l in zip(upper, lower)]

    rounded = df[['sR', 'sG', 'sB']].round(0)
    is_dup  = rounded.duplicated(keep=False)

    if is_dup.any():
        dups = df[is_dup].copy()
        dups['_dist'] = sum(
            abs(dups[col] - mid)
            for col, mid in zip(['Cm', 'Ch', 'Bm', 'Bh', 'T'], mids)
        )
        best_dups = (dups.sort_values('_dist')
                         .drop_duplicates(subset=['sR', 'sG', 'sB'])
                         .drop(columns='_dist'))
        df = pd.concat([df[~is_dup], best_dups]).sort_index()

    x = df[['sR', 'sG', 'sB']].to_numpy(dtype='float32') / 255.0
    y = df[['Cm', 'Ch', 'Bm', 'Bh', 'T']].to_numpy(dtype='float32')

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Strip the artefact first row that survives the iloc[1:] + split
    return x_train[1:], x_test[1:], y_train[1:], y_test[1:]


def make_loaders(csv_path: str, batch_size: int = 4096,
                 test_size: float = 0.2, random_state: int = 42):
    """Convenience wrapper: load LUT → AEDataset → DataLoader pair."""
    x_train, x_test, y_train, y_test = load_lut(csv_path, test_size, random_state)
    train_loader = DataLoader(AEDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(AEDataset(x_test,  y_test),  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
