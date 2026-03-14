"""
Deep Albedo - Model Architecture
Single source of truth for Encoder, Decoder, AutoEncoder.
Import this everywhere instead of redefining the classes.
"""
import torch
import torch.nn as nn

# Biological parameter bounds  (Cm, Ch, Bm, Bh, T)
PARAM_MINS = [0.05, 0.02, 0.0,  0.60, 0.005]
PARAM_MAXS = [0.50, 0.20, 1.0,  0.98, 0.020]


class Encoder(nn.Module):
    """RGB (3) → skin parameters (5): Cm, Ch, Bm, Bh, T

    Output is hard-clamped to biological ranges via register_buffer,
    so the clamp values are saved with the checkpoint and always match.
    Default architecture matches checkpoints/2026-03-14_**/best.pt.
    """

    def __init__(self, in_dim=3, hidden_dim=70, num_layers=4, out_dim=5):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.register_buffer('param_mins', torch.tensor(PARAM_MINS))
        self.register_buffer('param_maxs', torch.tensor(PARAM_MAXS))

    def forward(self, x):
        params = self.out(self.mlp(x))
        return torch.stack([
            torch.clamp(params[:, i], self.param_mins[i], self.param_maxs[i])
            for i in range(5)
        ], dim=1)


class Decoder(nn.Module):
    """Skin parameters (5) → RGB (3)

    Default architecture matches checkpoints/2026-03-14_**/best.pt.
    """

    def __init__(self, in_dim=5, hidden_dim=256, num_layers=4, out_dim=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.out(self.mlp(x))


class AutoEncoder(nn.Module):
    """Combined autoencoder for training with three simultaneous loss heads.

    Takes three separate inputs so all three loss heads (encoder/decoder/
    end-to-end) can be computed in one forward pass:
        enc_out  = encoder(encoder_in)          → parameter loss
        dec_out  = decoder(decoder_in)          → albedo loss
        end_out  = decoder(encoder(end_in))     → end-to-end loss
    """

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_in, decoder_in, end_to_end_in):
        enc_out = self.encoder(encoder_in)
        dec_out = self.decoder(decoder_in)
        end_out = self.decoder(self.encoder(end_to_end_in))
        return enc_out, dec_out, end_out
