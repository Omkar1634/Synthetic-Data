import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np

import torch
import torch.nn as nn
import time
import math

# device setup (equivalent to tf.device('/device:GPU:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL ARCHITECTURE (must match training)
# -----------------------------
NUM_NEURONS = 75
NUM_LAYERS = 2

class Encoder(nn.Module):
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
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_in, decoder_in, end_to_end_in):
        enc_out = self.encoder(encoder_in)
        dec_out = self.decoder(decoder_in)
        end_out = self.decoder(self.encoder(end_to_end_in))
        return enc_out, dec_out, end_out


def encode(image, encoder):
    """
    PyTorch version of TensorFlow encode()
    Input: image (numpy array), encoder (torch.nn.Module)
    Output: pred_maps (numpy), elapsed (float), (WIDTH, HEIGHT)
    """
    print(f"Image shape at start of encoder method: {image.shape}")

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


def decode(encoded, decoder):
    """
    PyTorch version of TensorFlow decode()
    Input: encoded (numpy array), decoder (torch.nn.Module)
    Output: recovered (numpy), elapsed (float), (WIDTH, HEIGHT)
    """
    print(f"Image shape going into decoder: {encoded.shape}")

    start = time.time()

    # Handle 2D flattened input vs (W,H,C) style input
    if len(encoded.shape) == 2:
        WIDTH = HEIGHT = int(math.sqrt(encoded.shape[0]))
        encoded = np.asarray(encoded).reshape(-1, 5).astype("float32")
    else:
        WIDTH = encoded.shape[0]
        HEIGHT = encoded.shape[1]
        encoded = np.asarray(encoded).astype("float32")

    print(f"encoded shape going into decoder: {encoded.shape}")

    # Torch inference
    x = torch.from_numpy(encoded).to(device)

    decoder.eval()
    with torch.no_grad():
        recovered = decoder(x)

    recovered = recovered.detach().cpu().numpy()

    end = time.time()
    elapsed = end - start

    # reshape output to RGB image
    recovered = recovered.reshape(WIDTH, HEIGHT, 3)

    return recovered, elapsed, (WIDTH, HEIGHT)


# ------------------------------------------------------
# LOAD TRAINED MODELS
# ------------------------------------------------------
print("Loading models...")

# Initialize the full AutoEncoder model
encoder_net = Encoder().to(device)
decoder_net = Decoder().to(device)
model = AutoEncoder(encoder_net, decoder_net).to(device)

# Load trained weights from checkpoint
checkpoint_path = 'checkpoints/2025-12-31_18-54-39/best.pt'

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the full model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract encoder and decoder for easier use
    encoder = model.encoder
    decoder = model.decoder
    
    print(f"Models loaded successfully from {checkpoint_path}!")
    print(f"Checkpoint was from epoch: {checkpoint['epoch']}")
    print(f"Best train loss: {checkpoint.get('best_train_loss', 'N/A')}")
    
except FileNotFoundError:
    print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
    print("Please check the path and try again.")
    exit()
except Exception as e:
    print(f"ERROR loading checkpoint: {e}")
    exit()

encoder.eval()
decoder.eval()


import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib
import sys

sys.path.append('../')

from utils import preprocess
from utils import plotting

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
WIDTH = 256
HEIGHT = 256

# Reload modules
importlib.reload(preprocess)
importlib.reload(plotting)

# Matplotlib styling
plt.style.use("dark_background")
plt.rcParams["axes.grid"] = False


# ------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------
image_rgb = cv2.imread(r"D:\Github\Deep-Albedo\images\IMG_8.jpeg")
image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

# Crop face
image_rgb = preprocess.crop_face(image_rgb)[0]

# Resize
image_rgb = cv2.resize(image_rgb, (WIDTH, HEIGHT))

# Normalize to float32 in [0,1]
image_rgb = np.asarray(image_rgb).astype("float32") / 255.0

plt.imshow(image_rgb)
plt.title("original")
plt.axis("off")
plt.show()


# ------------------------------------------------------
# ENCODE
# ------------------------------------------------------
parameter_maps, elapsed, d1 = encode(image_rgb, encoder)

pm1 = parameter_maps.copy().reshape(d1[0], d1[1], 5)


# ------------------------------------------------------
# NORMALIZE PARAM MAPS
# ------------------------------------------------------
Cm = parameter_maps[:, 0]
Ch = parameter_maps[:, 1]
Bm = parameter_maps[:, 2]
Bh = parameter_maps[:, 3]
T  = parameter_maps[:, 4]

# normalize helper
def norm01(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

Cm = norm01(Cm)
Ch = norm01(Ch)
Bm = norm01(Bm)
Bh = norm01(Bh)
T  = norm01(T)

parameter_maps[:, 0] = Cm * 0.62 + 0.001
parameter_maps[:, 1] = Ch * 0.31 + 0.001
parameter_maps[:, 2] = Bm * 0.8  + 0.2
parameter_maps[:, 3] = Bh * 0.3  + 0.6
parameter_maps[:, 4] = T  * 0.2  + 0.05


print(f"CM {Cm.min()} {Cm.max()}")
print(f"CH {Ch.min()} {Ch.max()}")
print(f"BM {Bm.min()} {Bm.max()}")
print(f"BH {Bh.min()} {Bh.max()}")
print(f"T  {T.min()} {T.max()}")

WIDTH, HEIGHT = d1
print(f"encode time {elapsed}")


# ------------------------------------------------------
# DECODE
# ------------------------------------------------------
recovered, elapsed, d2 = decode(pm1.reshape(-1, 5), decoder)

WIDTH, HEIGHT = d2
recovered = recovered  # keep as is

print(f"decode time {elapsed}")


# ------------------------------------------------------
# PLOT
# ------------------------------------------------------
plotting.PLOT_TEX_MAPS(recovered, parameter_maps)