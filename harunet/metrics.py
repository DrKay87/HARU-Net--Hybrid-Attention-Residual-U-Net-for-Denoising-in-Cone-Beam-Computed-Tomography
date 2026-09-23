"""Image-quality metrics used by the HARU-Net experiments."""
import torch
import torch.nn.functional as F
import numpy as np


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range / np.sqrt(mse))


def batch_psnr(pred, target, data_range=1.0):
    return float(np.mean([
        psnr(pred[i], target[i], data_range=data_range)
        for i in range(pred.shape[0])
    ]))


def gmsd(img1, img2, c=0.0026):
    """Gradient Magnitude Similarity Deviation for tensors BxCxHxW."""
    if img1.ndim != 4 or img2.ndim != 4:
        raise ValueError("Expected BxCxHxW tensors")

    sobel_x = torch.tensor(
        [[1/3, 0, -1/3], [1/3, 0, -1/3], [1/3, 0, -1/3]],
        dtype=img1.dtype, device=img1.device
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    g1x = F.conv2d(img1, sobel_x, padding=1)
    g1y = F.conv2d(img1, sobel_y, padding=1)
    g2x = F.conv2d(img2, sobel_x, padding=1)
    g2y = F.conv2d(img2, sobel_y, padding=1)

    gm1 = torch.sqrt(g1x.square() + g1y.square() + 1e-12)
    gm2 = torch.sqrt(g2x.square() + g2y.square() + 1e-12)
    gms = (2 * gm1 * gm2 + c) / (gm1.square() + gm2.square() + c)
    return torch.std(gms.flatten(1), dim=1).mean().item()
