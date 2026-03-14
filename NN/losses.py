"""
Deep Albedo - Loss Functions

Three losses trained simultaneously:
    parameter_loss  — encoder output vs ground-truth skin params  (L2)
    albedo_loss     — decoder output vs ground-truth RGB           (L1)
    end_to_end_loss — full pipeline RGB reconstruction             (L1)

Loss weights used during training: (0.3, 0.1, 0.6)
"""
import torch


def parameter_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 distance between predicted and true skin parameters."""
    return torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1) + 1e-12)


def albedo_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-sample L1 distance between predicted and true RGB (albedo head)."""
    return torch.sum(torch.abs(y_pred - y_true), dim=-1)


def end_to_end_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Per-sample L1 distance for full encoder→decoder RGB reconstruction."""
    return torch.sum(torch.abs(y_pred - y_true), dim=-1)


def reduce_loss(loss_per_sample: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    raise ValueError("reduction must be 'mean' or 'sum'")
