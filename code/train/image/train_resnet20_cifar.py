"""ResNet-20 on CIFAR-10 — He 2016 canonical recipe.

SGD momentum 0.9, weight_decay 1e-4, batch 128, 200 ep,
LR 0.1 step ×0.1 at ep 100/150,
augmentation: 4-px zero-pad + random 32x32 crop + horizontal flip + per-channel mean/std normalization.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms

from common import VoterConfig, build_loaders, train_voter
from models import ResNet20


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = VoterConfig(
        voter_name="resnet20",
        dataset="cifar10",
        epochs=200,
        batch_size=128,
        optimizer="sgd",
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        scheduler="multistep",
        milestones=[100, 150],
        gamma=0.1,
        notes="He 2016 §4.2 canonical recipe; community step at epochs 100/150",
    )

    train_tx = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # zero-pad by default in torchvision
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

    model = ResNet20(num_classes=10, in_channels=3)
    train_voter(model, cfg, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
