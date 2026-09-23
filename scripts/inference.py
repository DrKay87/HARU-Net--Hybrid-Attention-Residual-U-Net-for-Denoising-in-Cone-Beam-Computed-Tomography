"""Run HARU-Net inference on a NumPy 2D image or stack."""
import argparse
from pathlib import Path
import numpy as np
import torch
from harunet.model import HARU_net


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help=".npy image or stack")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="results/denoised.npy")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--hab-heads", type=int, default=8)
    p.add_argument("--hab-window", type=int, default=16)
    p.add_argument("--hab-depth", type=int, default=6)
    return p.parse_args()


def load_state(path, device):
    obj = torch.load(path, map_location=device)
    return obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HARU_net(
        dim=args.dim, hab_heads=args.hab_heads,
        hab_ws=args.hab_window, hab_depth=args.hab_depth
    ).to(device)
    model.load_state_dict(load_state(args.checkpoint, device))
    model.eval()

    arr = np.load(args.input).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    outputs = []
    with torch.no_grad():
        for sl in arr:
            x = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)
            outputs.append(model(x).squeeze().cpu().numpy())

    out = np.stack(outputs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
