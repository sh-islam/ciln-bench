"""DeiT3 fine-tune utilities: MixUp/CutMix step + transform builders.

Implements the official `facebookresearch/deit` README_revenge.md fine-tune recipe:
- timm Mixup (mixup=0.8, cutmix=1.0, label_smoothing=0.1)
- RandAugment rand-m9-mstd0.5-inc1
- AdamW lr=1e-5, wd=0.1, cosine + 5 ep warmup
- drop_path 0.05
"""
from __future__ import annotations
import timm
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.data.transforms_factory import create_transform
from timm.loss import SoftTargetCrossEntropy
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_deit3_model(num_classes: int = 10, drop_path_rate: float = 0.05):
    """DeiT3-Small backbone, ImageNet-1K pretrained, new classifier head."""
    model = timm.create_model(
        "deit3_small_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model


def build_deit3_transforms(is_mnist: bool = False):
    """Returns (train_transform, eval_transform).
    On MNIST we channel-replicate 1->3 and resize to 224.
    On CIFAR we resize 32->224.
    """
    if is_mnist:
        # 1ch 28x28 -> 3ch 224x224
        train_pre = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
        eval_pre = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    else:
        # 3ch 32x32 -> 3ch 224x224
        train_pre = [transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)]
        eval_pre = [transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)]

    # Augmentation is RandAugment rand-m9-mstd0.5-inc1 (DeiT3 README)
    train_aug = create_transform(
        input_size=224,
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.0,     # no random erasing
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )
    # the create_transform output expects PIL input; we just compose pre + aug
    train_transform = transforms.Compose(train_pre + [train_aug])

    eval_transform = transforms.Compose(eval_pre + [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def build_mixup_fn(num_classes: int = 10):
    """timm Mixup with the DeiT3 fine-tune values."""
    return Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=num_classes,
    )


def make_deit3_train_step(mixup_fn: Mixup, use_amp: bool = True):
    """Custom training step closure that applies mixup and uses SoftTargetCrossEntropy.
    Honors AMP if use_amp=True (FP16 forward + GradScaler).
    """
    soft_ce = SoftTargetCrossEntropy()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    def step(model, x, y, _ce_unused, optim_, _cfg_unused):
        x, y_soft = mixup_fn(x, y)
        optim_.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = soft_ce(logits, y_soft)
            scaler.scale(loss).backward()
            scaler.step(optim_)
            scaler.update()
        else:
            logits = model(x)
            loss = soft_ce(logits, y_soft)
            loss.backward()
            optim_.step()
        return loss.item()
    return step
