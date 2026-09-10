"""DeiT3-Small on CIFAR-10 — Touvron 2022 README_revenge.md fine-tune recipe.

AdamW lr=1e-5, wd=0.1, batch 128, 20 ep, cosine + 5 ep warmup,
drop_path 0.05, label_smoothing 0.1,
augmentation: RandAugment rand-m9-mstd0.5-inc1 + MixUp 0.8 + CutMix 1.0.
ImageNet-1K pretrained weights via timm.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch

from common import VoterConfig, build_loaders, train_voter
from _deit_helpers import build_deit3_model, build_deit3_transforms, build_mixup_fn, make_deit3_train_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = VoterConfig(
        voter_name="deit3_small",
        dataset="cifar10",
        epochs=20,
        batch_size=128,
        optimizer="adamw",
        lr=1e-5,
        weight_decay=0.1,
        scheduler="cosine",
        warmup_epochs=5,
        warmup_lr=1e-6,
        label_smoothing=0.1,
        num_workers=4,
        notes="Touvron 2022 README_revenge.md recipe (full MixUp+CutMix+RandAug+LS+DropPath)",
    )

    train_tx, eval_tx = build_deit3_transforms(is_mnist=False)
    train_loader, val_loader, test_loader = build_loaders(
        "cifar10", train_transform=train_tx, eval_transform=eval_tx,
        batch_size=cfg.batch_size, val_frac=cfg.val_frac, num_workers=cfg.num_workers,
    )

    model = build_deit3_model(num_classes=10, drop_path_rate=0.05)
    mixup_fn = build_mixup_fn(num_classes=10)
    train_step = make_deit3_train_step(mixup_fn, use_amp=cfg.use_amp)

    train_voter(model, cfg, train_loader, val_loader, test_loader, device, train_step_fn=train_step)


if __name__ == "__main__":
    main()
