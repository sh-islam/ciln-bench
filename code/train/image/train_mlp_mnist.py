"""MLP on MNIST — Srivastava 2014 dropout MLP, 784-1024-1024-2048-10.

Dropout: input p=0.2, hidden p=0.5.
Max-norm constraint c=2.0 (applied each step in common.apply_max_norm).
Optimizer: Adam lr=1e-3 batch=128 200 ep (Han 2018 Co-Teaching schedule for MNIST pool consistency).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms

from common import VoterConfig, build_loaders, train_voter
from models import DropoutMLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = VoterConfig(
        voter_name="mlp",
        dataset="mnist",
        epochs=200,
        batch_size=128,
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        scheduler="linear_decay_after",
        decay_start_epoch=80,
        max_norm=2.0,
        notes="Srivastava 2014 dropout-MLP arch (0.5 hidden / 0.2 input) + max-norm 2.0 + Han 2018 Adam schedule",
    )

    tx = transforms.ToTensor()
    train_loader, val_loader, test_loader = build_loaders(
        "mnist", train_transform=tx, eval_transform=tx,
        batch_size=cfg.batch_size, val_frac=cfg.val_frac, num_workers=cfg.num_workers,
    )

    model = DropoutMLP(num_classes=10, input_dim=784)
    train_voter(model, cfg, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
