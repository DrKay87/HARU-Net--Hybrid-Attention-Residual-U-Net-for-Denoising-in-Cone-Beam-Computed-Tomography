"""Evaluate a HARU-Net checkpoint on paired pickled test patches."""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from harunet.model import HARU_net
from harunet.data import load_pickle_array, PairedPatchDataset
from harunet.metrics import batch_psnr, gmsd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-inputs", required=True)
    p.add_argument("--test-targets", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairedPatchDataset(load_pickle_array(args.test_inputs),
                            load_pickle_array(args.test_targets))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model = HARU_net().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    psnrs, gmsds = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            psnrs.append(batch_psnr(pred, y))
            gmsds.append(gmsd(pred, y))
    print(f"PSNR: {np.mean(psnrs):.4f} dB")
    print(f"GMSD: {np.mean(gmsds):.4f}")


if __name__ == "__main__":
    main()
