"""LeNet-5 on MNIST — Han 2018 Co-Teaching recipe.

Adam lr=1e-3, batch 128, 200 ep, no aug, ToTensor only.
Linear lr decay starts at epoch 80.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms

from common import VoterConfig, build_loaders, train_voter
from models import LeNet5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = VoterConfig(
        voter_name="lenet5",
        dataset="mnist",
        epochs=200,
        batch_size=128,
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        scheduler="linear_decay_after",
        decay_start_epoch=80,
        label_smoothing=0.0,
        notes="Han 2018 Co-Teaching MNIST recipe; modern LeNet-5 (ReLU+CE+Adam)",
    )

    # ToTensor only (matches Han 2018 official code; paper text mentions zero-mean unit-var but code uses ToTensor only)
    tx = transforms.ToTensor()
    train_loader, val_loader, test_loader = build_loaders(
        "mnist", train_transform=tx, eval_transform=tx,
        batch_size=cfg.batch_size, val_frac=cfg.val_frac, num_workers=cfg.num_workers,
    )

    model = LeNet5(num_classes=10)
    train_voter(model, cfg, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
