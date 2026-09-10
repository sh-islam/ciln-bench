"""ResNet-20 on MNIST — Han 2018 (Co-Teaching) recipe applied (no canonical recipe for ResNet-on-MNIST).
Adam lr=1e-3, batch 128, 200 ep, no aug.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms

from common import VoterConfig, build_loaders, train_voter
from models import ResNet20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = VoterConfig(
        voter_name="resnet20",
        dataset="mnist",
        epochs=200,
        batch_size=128,
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        scheduler="linear_decay_after",
        decay_start_epoch=80,
        notes="Han 2018 Co-Teaching recipe applied (no canonical ResNet-MNIST recipe)",
    )

    tx = transforms.ToTensor()
    train_loader, val_loader, test_loader = build_loaders(
        "mnist", train_transform=tx, eval_transform=tx,
        batch_size=cfg.batch_size, val_frac=cfg.val_frac, num_workers=cfg.num_workers,
    )

    model = ResNet20(num_classes=10, in_channels=1)
    train_voter(model, cfg, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
