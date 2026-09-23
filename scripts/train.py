"""Train HARU-Net on paired pickled CBCT patches."""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from harunet.model import HARU_net
from harunet.data import load_pickle_array, PairedPatchDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-inputs", required=True)
    p.add_argument("--train-targets", required=True)
    p.add_argument("--val-inputs")
    p.add_argument("--val-targets")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--hab-heads", type=int, default=8)
    p.add_argument("--hab-window", type=int, default=16)
    p.add_argument("--hab-depth", type=int, default=6)
    return p.parse_args()


def make_loader(x_path, y_path, batch_size, shuffle, workers):
    ds = PairedPatchDataset(load_pickle_array(x_path), load_pickle_array(y_path))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(
        args.train_inputs, args.train_targets, args.batch_size, True, args.num_workers
    )
    val_loader = None
    if args.val_inputs and args.val_targets:
        val_loader = make_loader(
            args.val_inputs, args.val_targets, args.batch_size, False, args.num_workers
        )

    model = HARU_net(
        dim=args.dim, hab_heads=args.hab_heads, hab_ws=args.hab_window,
        hab_depth=args.hab_depth
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    total += criterion(model(x), y).item() * x.size(0)
            val_loss = total / len(val_loader.dataset)

        score = val_loss if val_loss is not None else train_loss
        print(f"Epoch {epoch:03d}: train={train_loss:.6f}" +
              (f", val={val_loss:.6f}" if val_loss is not None else ""))

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                   ckpt_dir / "last.pt")
        if score < best:
            best = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
