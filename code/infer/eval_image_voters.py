"""Per-voter softmax inference for all image voters on clean + corrupted settings.

Loads each voter checkpoint, runs forward on:
  - the clean test set (saved once to clean/<dataset>/)
  - every corrupted setting under output_seed{S}/<dataset>/<corruption>/severity_{sev}/

For each setting we write:
  softmax_<voter>.npy   shape (N, 10) float32

Voters covered (excluding CLIP zero-shot, which the user wants deferred):
  MNIST:    lenet5, mlp, resnet20 (mnist), deit3_small (mnist)
  CIFAR-10: resnet20 (cifar), wrn28_10, deit3_small (cifar)

Each voter uses the same preprocessing transform that was applied during its
training's eval phase. Corrupted images are stored as uint8 (HWC for CIFAR,
HW for MNIST); we convert to PIL.Image so the torchvision transforms see the
same input type they saw during training.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
JOURNAL_ROOT = HERE.parent
sys.path.insert(0, str(JOURNAL_ROOT / "train"))

from models import LeNet5, ResNet20, WideResNet, DropoutMLP
from _deit_helpers import build_deit3_model, build_deit3_transforms, IMAGENET_MEAN, IMAGENET_STD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ---------- transform builders (must match the eval transform from training) ----------

def mnist_tensor_only():
    return transforms.ToTensor()


def cifar_normalize():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


# ---------- voter registry ----------
# Each entry: (voter_name, dataset, ckpt_filename, model_builder, transform_builder)

def _build_lenet5():     return LeNet5(num_classes=10)
def _build_resnet20_mnist(): return ResNet20(num_classes=10, in_channels=1)
def _build_resnet20_cifar(): return ResNet20(num_classes=10, in_channels=3)
def _build_wrn28_10():    return WideResNet(num_classes=10, in_channels=3)
def _build_mlp():        return DropoutMLP(num_classes=10, input_dim=784)
def _build_deit3():      return build_deit3_model(num_classes=10, drop_path_rate=0.0)


VOTERS = {
    "mnist": [
        ("lenet5",      "lenet5_mnist_best.pt",       _build_lenet5,         "mnist_plain"),
        ("mlp",         "mlp_mnist_best.pt",          _build_mlp,            "mnist_plain"),
        ("resnet20",    "resnet20_mnist_best.pt",     _build_resnet20_mnist, "mnist_plain"),
        ("deit3_small", "deit3_small_mnist_best.pt",  _build_deit3,          "deit3_mnist"),
    ],
    "cifar10": [
        ("resnet20",    "resnet20_cifar10_best.pt",   _build_resnet20_cifar, "cifar_norm"),
        ("wrn28_10",    "wrn28_10_cifar10_best.pt",   _build_wrn28_10,       "cifar_norm"),
        ("deit3_small", "deit3_small_cifar10_best.pt", _build_deit3,         "deit3_cifar"),
        # CLIP zero-shot. ckpt_name=None signals "no checkpoint to load",
        # transform kind="clip_raw" signals "raw uint8 images, CLIP handles its own preprocessing".
        ("clip",        None,                          None,                 "clip_raw"),
    ],
}

NOISY_SPLITS = ("noisy_label_train", "noisy_label_valid")


def get_transform(kind: str):
    if kind == "mnist_plain":
        return mnist_tensor_only()
    if kind == "cifar_norm":
        return cifar_normalize()
    if kind == "deit3_mnist":
        _, eval_tx = build_deit3_transforms(is_mnist=True)
        return eval_tx
    if kind == "deit3_cifar":
        _, eval_tx = build_deit3_transforms(is_mnist=False)
        return eval_tx
    if kind == "clip_raw":
        # CLIP handles its own preprocessing inside CLIPVoter.predict() (resize +
        # normalize on raw uint8 arrays); no torchvision transform needed.
        return None
    raise ValueError(kind)


# ---------- datasets ----------

class CorruptedImageDataset(Dataset):
    """Wraps a corrupted images.npy array (+ labels.npy) so we can apply a torchvision
    transform that expects a PIL.Image."""
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform, is_mnist: bool):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.is_mnist = is_mnist

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        arr = self.images[idx]
        if self.is_mnist:
            img = Image.fromarray(arr, mode="L")
        else:
            img = Image.fromarray(arr, mode="RGB")
        x = self.transform(img)
        return x, int(self.labels[idx])


def build_clean_test_dataset(dataset: str, transform):
    if dataset == "mnist":
        return datasets.MNIST(root=str(JOURNAL_ROOT / "data_cache"), train=False, download=False, transform=transform)
    if dataset == "cifar10":
        return datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=False, download=False, transform=transform)
    raise ValueError(dataset)


# ---------- inference ----------

@torch.no_grad()
def run_softmax(model: nn.Module, loader: DataLoader, n: int, num_classes: int = 10) -> np.ndarray:
    model.eval()
    out = np.empty((n, num_classes), dtype=np.float32)
    correct, total = 0, 0
    idx = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).float().cpu().numpy()
        b = probs.shape[0]
        out[idx:idx + b] = probs
        idx += b
        pred = probs.argmax(axis=1)
        correct += int((pred == y.numpy()).sum())
        total += b
    acc = correct / max(1, total)
    return out, acc


def load_voter(builder, ckpt_path: Path) -> nn.Module:
    model = builder()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(DEVICE)


def softmax_filename(voter_name: str) -> str:
    return f"softmax_{voter_name}.npy"


def is_image_setting_done(setting_path: Path, voter_names: list[str]) -> bool:
    return all((setting_path / softmax_filename(v)).exists() for v in voter_names)


def predict_on_arrays(model_or_voter, voter_name, transform_kind, transform,
                      images: np.ndarray, labels: np.ndarray,
                      is_mnist: bool, batch_size: int, num_workers: int):
    """Predict softmax probabilities for a batch of uint8 images.

    Routes to the appropriate inference path based on transform_kind:
      - 'clip_raw': call CLIPVoter.predict(images) directly (handles preprocessing).
      - everything else: wrap images+labels in CorruptedImageDataset + DataLoader.
    """
    if transform_kind == "clip_raw":
        # CLIP requires RGB uint8 (N, H, W, 3). For MNIST grayscale (N, 28, 28),
        # broadcast to 3-channel by repeating.
        if images.ndim == 3:
            images_rgb = np.repeat(images[..., None], 3, axis=-1)
        else:
            images_rgb = images
        probs = model_or_voter.predict(images_rgb, batch_size=batch_size)
        pred = probs.argmax(axis=1)
        acc = float((pred == labels).mean()) if len(labels) else 0.0
        return probs, acc
    # standard CNN/transformer path
    ds = CorruptedImageDataset(images, labels, transform, is_mnist=is_mnist)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    probs, acc = run_softmax(model_or_voter, loader, n=len(ds))
    return probs, acc


def eval_voter_on_setting(model_or_voter, voter_name, transform_kind, transform,
                          is_mnist, setting_path, batch_size, num_workers):
    images = np.load(setting_path / "images.npy")
    labels = np.load(setting_path / "labels.npy")
    probs, acc = predict_on_arrays(model_or_voter, voter_name, transform_kind, transform,
                                   images, labels, is_mnist, batch_size, num_workers)
    np.save(setting_path / softmax_filename(voter_name), probs)
    return acc, len(images)


def _load_clean_split_images(dataset: str, split: str):
    """Return (images_uint8, labels_int64) for a clean (no corruption) split.

    Splits we know about:
      - 'test':                full official test set (clean by definition)
      - 'cleanlabelvalid':     held-out clean val (in train), sliced by index
      - 'noisylabeltrain_clean': the pre-corruption images of NoisyLabelTrain
      - 'noisylabelvalid_clean': the pre-corruption images of NoisyLabelValid
    """
    splits_root = JOURNAL_ROOT / "output_seed0" / "splits" / dataset
    # Determine which source split (train or test) and which indices to take
    if split == "test":
        is_train = False
        idx = None
    elif split == "cleanlabelvalid":
        is_train = True
        idx = np.load(splits_root / "cleanlabelvalid_indices.npy")
    elif split == "noisylabeltrain_clean":
        is_train = True
        idx = np.load(splits_root / "noisylabeltrain_indices.npy")
    elif split == "noisylabelvalid_clean":
        is_train = True
        idx = np.load(splits_root / "noisylabelvalid_indices.npy")
    else:
        raise ValueError(split)

    if dataset == "mnist":
        ds = datasets.MNIST(root=str(JOURNAL_ROOT / "data_cache"), train=is_train, download=False)
        images = ds.data.numpy() if hasattr(ds.data, "numpy") else np.array(ds.data)
        labels = np.asarray(ds.targets, dtype=np.int64)
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(root=str(JOURNAL_ROOT / "data_cache"), train=is_train, download=False)
        images = np.array(ds.data)
        labels = np.asarray(ds.targets, dtype=np.int64)
    else:
        raise ValueError(dataset)

    if idx is None:
        return images, labels
    return images[idx], labels[idx]


def _load_clip_voter():
    """Lazy import + load CLIP voter on the active CUDA device."""
    from clip_voter import CLIPVoter
    return CLIPVoter.load(device="cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="master seed used in output_seed{seed}")
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--deit-batch-size", type=int, default=64,
                    help="smaller batch for DeiT3 (224x224 inputs)")
    ap.add_argument("--clip-batch-size", type=int, default=256,
                    help="batch size for CLIP voter (operates on uint8 then resizes to 224)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--clean-only", action="store_true",
                    help="only run on clean splits (cleanlabelvalid + test); skip corrupted settings")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out_root = JOURNAL_ROOT / f"output_seed{args.seed}"
    clean_root = out_root / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)

    summary = {"start": time.time(), "datasets": {}}

    for dataset in args.datasets:
        is_mnist = dataset == "mnist"
        voter_list = VOTERS[dataset]
        # Clean output dirs.
        # Four clean evaluations: clean test, clean cleanlabelvalid, plus the
        # pre-corruption images for noisylabeltrain and noisylabelvalid. The
        # latter two are needed downstream to compute the "all voters correct
        # on clean" mask for the noise-rate table filter (matches what the
        # original thesis did on test, now applied to the relevant train half).
        ds_clean_root = clean_root / dataset
        for cs in ("cleanlabelvalid", "test", "noisylabeltrain_clean", "noisylabelvalid_clean"):
            (ds_clean_root / cs).mkdir(parents=True, exist_ok=True)

        if not args.clean_only:
            corrs = sorted([p.name for p in (out_root / dataset).iterdir() if p.is_dir()])
        else:
            corrs = []
        print(f"\n=== dataset={dataset}  corruptions={len(corrs)}  voters={len(voter_list)} ===", flush=True)

        for voter_name, ckpt_name, builder, tkind in voter_list:
            transform = get_transform(tkind)
            batch_size = (args.clip_batch_size if tkind == "clip_raw"
                          else (args.deit_batch_size if tkind.startswith("deit3") else args.batch_size))

            t_voter = time.time()
            if tkind == "clip_raw":
                model_or_voter = _load_clip_voter()
                print(f"\n  --- voter={voter_name} ({tkind}, zero-shot) batch={batch_size} ---", flush=True)
            else:
                ckpt_path = JOURNAL_ROOT / "checkpoints" / ckpt_name
                if not ckpt_path.exists():
                    print(f"  [SKIP] missing checkpoint: {ckpt_path}", flush=True)
                    continue
                model_or_voter = load_voter(builder, ckpt_path)
                print(f"\n  --- voter={voter_name} ({tkind}) batch={batch_size} ---", flush=True)

            # 1) Clean splits: cleanlabelvalid + test + the pre-corruption views of
            # NoisyLabelTrain and NoisyLabelValid (needed for the "all voters correct
            # on clean" filter in the noise-rate table).
            for clean_split_name in ("cleanlabelvalid", "test", "noisylabeltrain_clean", "noisylabelvalid_clean"):
                clean_dir = ds_clean_root / clean_split_name
                clean_softmax_path = clean_dir / softmax_filename(voter_name)
                if args.resume and clean_softmax_path.exists():
                    continue
                images, labels = _load_clean_split_images(dataset, clean_split_name)
                probs, acc = predict_on_arrays(
                    model_or_voter, voter_name, tkind, transform,
                    images, labels, is_mnist, batch_size, args.num_workers,
                )
                np.save(clean_softmax_path, probs)
                labels_path = clean_dir / "labels.npy"
                if not labels_path.exists():
                    np.save(labels_path, labels.astype(np.int64))
                print(f"    [{voter_name}] clean/{clean_split_name}: acc={acc:.4f}  n={len(labels)}", flush=True)

            # 2) Corrupted settings: noisy_label_train + noisy_label_valid
            for corr in corrs:
                corr_dir = out_root / dataset / corr
                for sev_dir in sorted(corr_dir.iterdir()):
                    if not sev_dir.is_dir(): continue
                    for split_name in NOISY_SPLITS:
                        split_dir = sev_dir / split_name
                        if not (split_dir / "images.npy").exists():
                            continue
                        setting_softmax_path = split_dir / softmax_filename(voter_name)
                        if args.resume and setting_softmax_path.exists():
                            continue
                        acc, n = eval_voter_on_setting(
                            model_or_voter, voter_name, tkind, transform, is_mnist,
                            split_dir, batch_size=batch_size, num_workers=args.num_workers,
                        )
                        print(f"    [{voter_name}] {dataset}/{corr}/{sev_dir.name}/{split_name}: acc={acc:.4f}  n={n}", flush=True)

            print(f"  --- voter={voter_name} done in {time.time()-t_voter:.1f}s ---", flush=True)
            del model_or_voter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        summary["datasets"][dataset] = {"voters": [v[0] for v in voter_list], "n_corruptions": len(corrs)}

    summary["end"] = time.time()
    summary["total_sec"] = round(summary["end"] - summary["start"], 1)
    with open(out_root / "_eval_image_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== ALL DONE in {summary['total_sec']/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
