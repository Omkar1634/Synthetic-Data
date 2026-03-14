"""
Deep Albedo — PyTorch Training Script
Equivalent to train_pytorch_ae.ipynb, structured for command-line execution.

Usage:
    python NN/train.py
    python NN/train.py --lut simulation/data/lut_rgb.csv
    python NN/train.py --epochs 200 --cpu
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import Config
from model   import Encoder, Decoder, AutoEncoder
from losses  import parameter_loss, albedo_loss, end_to_end_loss, reduce_loss
from dataset import make_loaders


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = Config.RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_model(device: torch.device) -> AutoEncoder:
    enc = Encoder(hidden_dim=Config.ENC_HIDDEN_DIM, num_layers=Config.ENC_NUM_LAYERS).to(device)
    dec = Decoder(hidden_dim=Config.DEC_HIDDEN_DIM, num_layers=Config.DEC_NUM_LAYERS).to(device)
    return AutoEncoder(enc, dec).to(device)


def extract_architecture(model: AutoEncoder) -> dict:
    return {
        "encoder": {
            "in_dim":     model.encoder.mlp[0].in_features,
            "hidden_dim": model.encoder.mlp[0].out_features,
            "num_layers": len(model.encoder.mlp) // 2,
            "out_dim":    model.encoder.out.out_features,
        },
        "decoder": {
            "in_dim":     model.decoder.mlp[0].in_features,
            "hidden_dim": model.decoder.mlp[0].out_features,
            "num_layers": len(model.decoder.mlp) // 2,
            "out_dim":    model.decoder.out.out_features,
        },
    }


def compute_total_loss(enc_true, enc_pred, dec_true, dec_pred, end_true, end_pred,
                       weights=Config.LOSS_WEIGHTS):
    w1, w2, w3 = weights
    return (w1 * reduce_loss(parameter_loss(enc_true, enc_pred)) +
            w2 * reduce_loss(albedo_loss(dec_true, dec_pred))    +
            w3 * reduce_loss(end_to_end_loss(end_true, end_pred)))


# ── Training Visualisation ────────────────────────────────────────────────────
def show_training_progress(encoder, decoder, x_batch, y_batch, epoch, save_dir):
    """Save a 3x5 grid: input / true params / encoder output / reconstructed / error."""
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        lat_pred = encoder(x_batch[:3])
        rgb_out  = decoder(lat_pred)

    rgb_in   = (x_batch[:3].cpu().numpy() * 255).astype(int)
    lat_pred = lat_pred.cpu().numpy()
    lat_true = y_batch[:3].cpu().numpy()
    rgb_rec  = (rgb_out.cpu().numpy() * 255).astype(int)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(f'Epoch {epoch} — Training Progress', fontsize=14, fontweight='bold')

    for i in range(3):
        axes[i, 0].imshow([[rgb_in[i] / 255.0]])
        axes[i, 0].set_title(f'Input\n{rgb_in[i]}')
        axes[i, 0].axis('off')

        axes[i, 1].barh(Config.PARAM_NAMES, lat_true[i], color='green', alpha=0.6)
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].set_title('True Params')

        axes[i, 2].barh(Config.PARAM_NAMES, lat_pred[i], color='orange', alpha=0.6)
        axes[i, 2].set_xlim(0, 1)
        axes[i, 2].set_title('Encoder Output')

        axes[i, 3].imshow([[rgb_rec[i] / 255.0]])
        axes[i, 3].set_title(f'Output\n{rgb_rec[i]}')
        axes[i, 3].axis('off')

        err = np.abs(rgb_in[i] - rgb_rec[i])
        axes[i, 4].axis('off')
        axes[i, 4].text(0.1, 0.5, f'Error:\n{err}\n\nAvg: {err.mean():.1f}',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:04d}.png'), dpi=100, bbox_inches='tight')
    plt.close()

    return [{'sample': i,
             'rgb_input':  rgb_in[i].tolist(),
             'rgb_output': rgb_rec[i].tolist(),
             'mean_error': float(np.abs(rgb_in[i] - rgb_rec[i]).mean())}
            for i in range(3)]


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataloader, device, weights=Config.LOSS_WEIGHTS):
    model.eval()
    sums, n = [0.0] * 4, 0
    for enc_in, dec_in, end_in, enc_true, dec_true, end_true in dataloader:
        enc_in, dec_in, end_in = enc_in.to(device), dec_in.to(device), end_in.to(device)
        enc_true = enc_true.to(device)
        dec_true = dec_true.to(device)
        end_true = end_true.to(device)

        enc_pred, dec_pred, end_pred = model(enc_in, dec_in, end_in)
        l1 = reduce_loss(parameter_loss(enc_true, enc_pred))
        l2 = reduce_loss(albedo_loss(dec_true, dec_pred))
        l3 = reduce_loss(end_to_end_loss(end_true, end_pred))
        lt = weights[0]*l1 + weights[1]*l2 + weights[2]*l3

        sums[0] += lt.item(); sums[1] += l1.item()
        sums[2] += l2.item(); sums[3] += l3.item()
        n += 1

    if n == 0:
        return {"total": 0.0, "param": 0.0, "albedo": 0.0, "e2e": 0.0}
    return {"total": sums[0]/n, "param": sums[1]/n, "albedo": sums[2]/n, "e2e": sums[3]/n}


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, optimizer, scheduler, device,
          num_epochs=Config.NUM_EPOCHS, weights=Config.LOSS_WEIGHTS,
          base_ckpt_dir="checkpoints", checkpoint_period=200, print_period=5,
          save_json_each_epoch=True):

    run_dir = os.path.join(base_ckpt_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    viz_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    best_ckpt = os.path.join(run_dir, "best.pt")
    last_ckpt = os.path.join(run_dir, "last.pt")
    json_path = os.path.join(run_dir, "history.json")
    log_path  = os.path.join(viz_dir, "training_log.txt")

    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\nAUTOENCODER TRAINING LOG\n" + "=" * 80 + "\n")
        f.write(f"Start:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Device:     {device}\n"
                f"Epochs:     {num_epochs}\n"
                f"Batch size: {train_loader.batch_size}\n"
                f"Weights:    {weights}\n" + "=" * 80 + "\n\n")

    print(f"Device:  {device}\nRun dir: {run_dir}")

    best_train_loss = float("inf")
    history = {
        "config": {
            "num_epochs":   num_epochs,
            "loss_weights": weights,
            "optimizer":    optimizer.__class__.__name__,
            "scheduler":    scheduler.__class__.__name__,
            "device":       str(device),
            "architecture": extract_architecture(model),
        },
        "epochs": []
    }

    for epoch in range(num_epochs):
        model.train()
        sums, n_batches = [0.0] * 4, 0

        for enc_in, dec_in, end_in, enc_true, dec_true, end_true in train_loader:
            enc_in, dec_in, end_in = enc_in.to(device), dec_in.to(device), end_in.to(device)
            enc_true = enc_true.to(device)
            dec_true = dec_true.to(device)
            end_true = end_true.to(device)

            optimizer.zero_grad()
            enc_pred, dec_pred, end_pred = model(enc_in, dec_in, end_in)

            l1 = reduce_loss(parameter_loss(enc_true, enc_pred))
            l2 = reduce_loss(albedo_loss(dec_true, dec_pred))
            l3 = reduce_loss(end_to_end_loss(end_true, end_pred))
            lt = weights[0]*l1 + weights[1]*l2 + weights[2]*l3

            lt.backward()
            optimizer.step()

            sums[0] += lt.item(); sums[1] += l1.item()
            sums[2] += l2.item(); sums[3] += l3.item()
            n_batches += 1

        nb = max(n_batches, 1)
        tr = {"total": sums[0]/nb, "param": sums[1]/nb, "albedo": sums[2]/nb, "e2e": sums[3]/nb}
        vl = evaluate(model, val_loader, device, weights)

        scheduler.step(tr["total"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["epochs"].append({"epoch": epoch, "lr": current_lr, "train": tr, "val": vl})
        if save_json_each_epoch:
            with open(json_path, "w") as f:
                json.dump(history, f, indent=4)

        if epoch % print_period == 0:
            print(f"[{epoch:4d}/{num_epochs}] "
                  f"train={tr['total']:.6f} "
                  f"(p={tr['param']:.4f} a={tr['albedo']:.4f} e={tr['e2e']:.4f}) | "
                  f"val={vl['total']:.6f} | lr={current_lr:.2e}")

        # Visualise every epoch
        for val_batch in val_loader:
            break
        vis_metrics = show_training_progress(
            model.encoder, model.decoder,
            val_batch[2].to(device), val_batch[3].to(device),
            epoch, viz_dir,
        )

        with open(log_path, 'a') as f:
            f.write(f"\n{'='*80}\nEPOCH {epoch}\n{'='*80}\n"
                    f"Train: {tr['total']:.6f}  (p={tr['param']:.6f} a={tr['albedo']:.6f} e={tr['e2e']:.6f})\n"
                    f"Val:   {vl['total']:.6f}  (p={vl['param']:.6f} a={vl['albedo']:.6f} e={vl['e2e']:.6f})\n"
                    f"LR:    {current_lr:.2e}\n"
                    f"Viz:   " +
                    "  ".join(f"S{m['sample']}={m['mean_error']:.2f}" for m in vis_metrics) +
                    f"\nSaved: epoch_{epoch:04d}.png\n")

        # Save best checkpoint
        if tr["total"] < best_train_loss:
            best_train_loss = tr["total"]
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture":         extract_architecture(model),
                "best_train_loss":      best_train_loss,
                "history_path":         json_path,
            }, best_ckpt)

        # Periodic checkpoint
        if (epoch + 1) % checkpoint_period == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_total_loss":     tr["total"],
                "val_total_loss":       vl["total"],
                "history_path":         json_path,
            }, last_ckpt)

    with open(json_path, "w") as f:
        json.dump(history, f, indent=4)
    with open(log_path, 'a') as f:
        f.write(f"\n{'='*80}\nTRAINING COMPLETE\n{'='*80}\n"
                f"End:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Best train loss: {best_train_loss:.6f}\n")

    print(f"\nDone. Best checkpoint: {best_ckpt}")
    return history, run_dir


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Deep Albedo autoencoder")
    parser.add_argument("--lut",      default=Config.DEFAULT_LUT_PATH,
                        help="Path to lut_rgb.csv")
    parser.add_argument("--epochs",   type=int,   default=Config.NUM_EPOCHS)
    parser.add_argument("--batch",    type=int,   default=Config.BATCH_SIZE)
    parser.add_argument("--lr",       type=float, default=Config.LR)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--cpu",      action="store_true")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    train_loader, val_loader = make_loaders(args.lut, batch_size=args.batch)

    model     = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.01, patience=5,
        threshold=1e-4, cooldown=0, min_lr=Config.MIN_LR,
    )

    train(
        model, train_loader, val_loader, optimizer, scheduler, device,
        num_epochs=args.epochs,
        base_ckpt_dir=args.ckpt_dir,
    )


if __name__ == "__main__":
    main()
