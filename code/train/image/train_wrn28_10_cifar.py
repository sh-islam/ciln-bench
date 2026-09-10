"""WRN-28-10 on CIFAR-10 — Zagoruyko & Komodakis 2016 canonical recipe.

SGD Nesterov momentum 0.9, weight_decay 5e-4, batch 128, 200 ep,
LR 0.1 step ×0.2 at ep 60/120/160, dropout 0.3,
augmentation: 4-px zero-pad + random 32x32 crop + horizontal flip + per-channel mean/std normalization.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms

from common import VoterConfig, build_loaders, train_voter
from models import WideResNet


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = VoterConfig(
        voter_name="wrn28_10",
        dataset="cifar10",
        epochs=200,
        batch_size=128,
        optimizer="sgd_nesterov",
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        scheduler="multistep",
        milestones=[60, 120, 160],
        gamma=0.2,
        notes="Zagoruyko 2016 §4 canonical recipe; dropout 0.3; train.lua defaults",
    )

    train_tx = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    eval_tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_loader, val_loader, test_loader = build_loaders(
        "cifar10", train_transform=train_tx, eval_transform=eval_tx,
        batch_size=cfg.batch_size, val_frac=cfg.val_frac, num_workers=cfg.num_workers,
    )

    model = WideResNet(depth=28, widen_factor=10, num_classes=10, in_channels=3, dropout=0.3)
    train_voter(model, cfg, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
