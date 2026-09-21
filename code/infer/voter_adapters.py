"""Voter adapters for build_noisy_benchmark.py.

Each adapter is three functions: `load_model(ckpt_path)`, `preprocess(images)`,
`postprocess(logits)`. Together they turn (uint8 images, weights file) into
(softmax probabilities). At the bottom of this file is a `REGISTRY` mapping
(dataset, voter_name) → that triple.

Currently shipped: 3 simple image adapters (resnet20/wrn28_10 on CIFAR-10,
lenet5 on MNIST). More to follow as they get verified end-to-end.

To plug in a custom voter: copy the closest adapter below into your own
module, rewrite the three functions for your model, then pass the module
path to build_noisy_benchmark.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Pull the model defs from pipeline/train_image/
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "train_image"))
from models import LeNet5, ResNet20, WideResNet, DropoutMLP  # noqa: E402
from _deit_helpers import build_deit3_model, build_deit3_transforms  # noqa: E402


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# Going through PIL.Image + transforms.ToTensor is required for bit-identity
# with the shipped softmaxes; pure numpy /255 drifts after the first conv.
_CIFAR_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
_MNIST_TFM = transforms.ToTensor()


def _load_state(ckpt_path):
    """Our checkpoints are dicts with 'model_state_dict'; bare state_dicts are
    also accepted so a user can drop in their own."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def _softmax_to_numpy(logits):
    return F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)


def _preprocess_via_pil(images_uint8, mode, tfm):
    """Route each image through PIL.Image so torchvision.transforms.ToTensor
    sees a PIL image (not a raw ndarray). Matches training-time preprocessing.

    images_uint8: (N, H, W) for mode='L', or (N, H, W, 3) for mode='RGB'.
    """
    tensors = []
    for arr in images_uint8:
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        tensors.append(tfm(Image.fromarray(arr, mode=mode)))
    return torch.stack(tensors)


# ============================================================
# CIFAR-10: ResNet-20
# ============================================================

def _resnet20_cifar_load(ckpt_path):
    model = ResNet20(num_classes=10, in_channels=3)
    model.load_state_dict(_load_state(ckpt_path))
    return model.eval()


def _cifar_preprocess(images_uint8):
    return _preprocess_via_pil(images_uint8, mode="RGB", tfm=_CIFAR_TFM)


# ============================================================
# CIFAR-10: WRN-28-10
# Same preprocessing as ResNet-20; only the model differs.
# ============================================================

def _wrn28_10_cifar_load(ckpt_path):
    model = WideResNet(num_classes=10, in_channels=3)
    model.load_state_dict(_load_state(ckpt_path))
    return model.eval()


# ============================================================
# MNIST: LeNet-5
# ============================================================

def _lenet5_mnist_load(ckpt_path):
    model = LeNet5(num_classes=10)
    model.load_state_dict(_load_state(ckpt_path))
    return model.eval()


def _mnist_preprocess(images_uint8):
    return _preprocess_via_pil(images_uint8, mode="L", tfm=_MNIST_TFM)


# ============================================================
# MNIST: ResNet-20
# Same preprocessing as LeNet-5; different architecture.
# ============================================================

def _resnet20_mnist_load(ckpt_path):
    model = ResNet20(num_classes=10, in_channels=1)
    model.load_state_dict(_load_state(ckpt_path))
    return model.eval()


# ============================================================
# MNIST: MLP (DropoutMLP)
# Same per-image preprocessing as LeNet-5, then flatten to (N, 784).
# ============================================================

def _mlp_mnist_load(ckpt_path):
    model = DropoutMLP(num_classes=10, input_dim=784)
    model.load_state_dict(_load_state(ckpt_path))
    return model.eval()


def _mlp_mnist_preprocess(images_uint8):
    t = _preprocess_via_pil(images_uint8, mode="L", tfm=_MNIST_TFM)
    return t.flatten(1)


# ============================================================
# CIFAR-10 / MNIST: DeiT3-Small
# Upscale to 224x224 with bicubic, ImageNet-normalize.
# MNIST also needs grayscale -> 3 channels.
# ============================================================

# eval transforms built once. timm's create_transform internals depend on
# torch state, so cache the result.
_DEIT3_CIFAR_TFM = build_deit3_transforms(is_mnist=False)[1]
_DEIT3_MNIST_TFM = build_deit3_transforms(is_mnist=True)[1]


def _deit3_load(ckpt_path):
    model = build_deit3_model(num_classes=10, drop_path_rate=0.0)
    model.load_state_dict(_load_state(ckpt_path))
    return model.eval()


def _deit3_cifar_preprocess(images_uint8):
    return _preprocess_via_pil(images_uint8, mode="RGB", tfm=_DEIT3_CIFAR_TFM)


def _deit3_mnist_preprocess(images_uint8):
    # DeiT MNIST transform handles grayscale->RGB itself (Grayscale(num_output_channels=3)).
    return _preprocess_via_pil(images_uint8, mode="L", tfm=_DEIT3_MNIST_TFM)


# ============================================================
# CIFAR-10: CLIP ViT-B/32 (zero-shot)
# CLIP doesn't have a normal load/preprocess/postprocess split: the model
# carries its own text features and image preprocessing. The adapter wraps
# CLIPVoter so the three-function contract still works.
# CLIP has no local checkpoint; load() ignores its argument.
# ============================================================

from clip_voter import CLIPVoter  # noqa: E402


class _CLIPModelWrapper:
    """Thin wrapper so CLIP fits the (model)(tensor)->logits contract."""
    def __init__(self, voter): self.voter = voter
    def __call__(self, images_uint8):
        # images_uint8 here is the raw numpy (B, H, W, 3) we got from preprocess.
        return torch.from_numpy(self.voter.predict(images_uint8, batch_size=len(images_uint8)))
    def eval(self): return self
    def to(self, device): return self


def _clip_load(ckpt_path):
    # ckpt_path is unused — CLIP loads from OpenAI weights via open_clip.
    return _CLIPModelWrapper(CLIPVoter.load())


def _clip_preprocess(images_uint8):
    # Return the array as-is; CLIPVoter.predict() does its own resize + normalize.
    # For MNIST (which we don't use here), the caller would need to broadcast to 3-channel.
    return images_uint8


def _clip_postprocess(probs_tensor):
    return probs_tensor.numpy().astype(np.float32)


# ============================================================
# Registry
# ============================================================
# Each value is (load_model, preprocess, postprocess).

# Each entry: (load_model, preprocess, postprocess, batch_size). The batch
# size matters for bit-identity: cuDNN picks different conv/attn algorithms
# at different batch sizes. Match the values used at release time.
REGISTRY = {
    ("cifar10", "resnet20"):    (_resnet20_cifar_load, _cifar_preprocess,        _softmax_to_numpy, 256),
    ("cifar10", "wrn28_10"):    (_wrn28_10_cifar_load, _cifar_preprocess,        _softmax_to_numpy, 256),
    ("cifar10", "deit3_small"): (_deit3_load,          _deit3_cifar_preprocess,  _softmax_to_numpy,  64),
    ("cifar10", "clip"):        (_clip_load,           _clip_preprocess,         _clip_postprocess, 256),
    ("mnist",   "lenet5"):      (_lenet5_mnist_load,   _mnist_preprocess,        _softmax_to_numpy, 256),
    ("mnist",   "mlp"):         (_mlp_mnist_load,      _mlp_mnist_preprocess,    _softmax_to_numpy, 256),
    ("mnist",   "resnet20"):    (_resnet20_mnist_load, _mnist_preprocess,        _softmax_to_numpy, 256),
    ("mnist",   "deit3_small"): (_deit3_load,          _deit3_mnist_preprocess,  _softmax_to_numpy,  64),
}


def get_adapter(dataset, voter):
    key = (dataset, voter)
    if key not in REGISTRY:
        raise KeyError(
            f"No adapter for ({dataset!r}, {voter!r}). "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[key]
