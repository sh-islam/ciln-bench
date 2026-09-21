"""Stage 2 part A: v2-pool inference on corrupted CIFAR-10 TEST images.

Reads:
  output_seed0/cifar10/<corruption>/severity_<S>/test/{images.npy, labels.npy}

Writes (per setting):
  output_seed0_TEST/cifar10/<corruption>/severity_<S>/v2/
    softmax_<voter>.npy x4   (resnet20, wrn28_10, deit3_small, clip)
    avg_softmax.npy
    argmax_majority.npy
    labels.npy (true CIFAR-10 test labels)
    meta.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

JOURNAL_ROOT = Path('/path/to/ciln-workspace/journal_edition')
sys.path.insert(0, str(JOURNAL_ROOT / 'train'))
sys.path.insert(0, str(JOURNAL_ROOT / 'eval'))

from models import LeNet5, ResNet20, WideResNet, DropoutMLP
from _deit_helpers import build_deit3_model, build_deit3_transforms, IMAGENET_MEAN, IMAGENET_STD

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

SRC_ROOT = JOURNAL_ROOT / 'output_seed0' / 'cifar10'
OUT_ROOT = JOURNAL_ROOT / 'output_seed0_TEST' / 'cifar10'

def cifar_normalize():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

VOTERS = [
    ("resnet20",    "resnet20_cifar10_best.pt",    lambda: ResNet20(num_classes=10),                 "cifar_norm"),
    ("wrn28_10",    "wrn28_10_cifar10_best.pt",    lambda: WideResNet(depth=28, widen_factor=10, dropout=0.3, num_classes=10), "cifar_norm"),
    ("deit3_small", "deit3_small_cifar10_best.pt", None, "deit3_cifar"),  # deit handled specially
    ("clip",        None,                          None, "clip_raw"),
]

def get_transform(kind):
    if kind == 'cifar_norm': return cifar_normalize()
    if kind == 'deit3_cifar': return build_deit3_transforms(is_mnist=False)[1]
    if kind == 'clip_raw': return None
    raise ValueError(kind)


class CorruptedDS(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = Image.fromarray(self.images[i], mode='RGB')
        return self.transform(img), int(self.labels[i])


@torch.no_grad()
def run_softmax(model, loader, n):
    model.eval()
    out = np.empty((n, 10), dtype=np.float32)
    idx = 0
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        p = torch.softmax(model(x), dim=1).float().cpu().numpy()
        b = p.shape[0]; out[idx:idx+b] = p; idx += b
    return out


def load_model(voter_name, ckpt_name, builder, tkind):
    if tkind == 'clip_raw':
        from clip_voter import CLIPVoter
        return CLIPVoter.load(device=DEVICE), None
    if tkind == 'deit3_cifar':
        model = build_deit3_model(num_classes=10, drop_path_rate=0.05).to(DEVICE)
    else:
        model = builder().to(DEVICE)
    ckpt = torch.load(JOURNAL_ROOT / 'checkpoints' / ckpt_name, map_location=DEVICE, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    return model.eval(), get_transform(tkind)


def main():
    print(f"[{time.strftime('%X')}] v2-pool TEST inference starting on {DEVICE}", flush=True)
    # Find all settings
    settings = []
    for c_dir in sorted(SRC_ROOT.iterdir()):
        if not c_dir.is_dir(): continue
        for s_dir in sorted(c_dir.iterdir()):
            if not s_dir.is_dir(): continue
            test_dir = s_dir / 'test'
            if not (test_dir / 'images.npy').exists(): continue
            name = f"{c_dir.name}_sev{s_dir.name.split('_')[1]}"
            settings.append((name, test_dir))
    print(f"  {len(settings)} settings to process", flush=True)

    # For each voter, load once, run on all settings
    for voter_name, ckpt_name, builder, tkind in VOTERS:
        t0 = time.time()
        try:
            model, transform = load_model(voter_name, ckpt_name, builder, tkind)
        except Exception as e:
            print(f"  [SKIP {voter_name}] load error: {e}", flush=True)
            continue
        print(f"\n  --- {voter_name} ({tkind}) ---", flush=True)
        bs = 64 if tkind == 'deit3_cifar' else 256

        for i, (name, src_dir) in enumerate(settings):
            out_dir = OUT_ROOT / src_dir.parent.parent.name / src_dir.parent.name / 'v2'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"softmax_{voter_name}.npy"
            if out_path.exists():
                continue
            images = np.load(src_dir / 'images.npy')
            labels = np.load(src_dir / 'labels.npy')
            # Save labels once
            lbl_path = out_dir / 'labels.npy'
            if not lbl_path.exists():
                np.save(lbl_path, labels.astype(np.int64))
            if tkind == 'clip_raw':
                probs = model.predict(images, batch_size=bs)
            else:
                ds = CorruptedDS(images, labels, transform)
                loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
                probs = run_softmax(model, loader, n=len(ds))
            np.save(out_path, probs)
            if (i+1) % 5 == 0 or i == len(settings)-1:
                print(f"    [{voter_name}] {i+1}/{len(settings)} done", flush=True)

        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"  --- {voter_name} done in {(time.time()-t0)/60:.1f} min ---", flush=True)

    # Aggregate avg_softmax + argmax_majority per setting
    print("\n  Aggregating avg_softmax + argmax_majority per setting", flush=True)
    for name, src_dir in settings:
        out_dir = OUT_ROOT / src_dir.parent.parent.name / src_dir.parent.name / 'v2'
        sm = []
        for v in [v[0] for v in VOTERS]:
            p = out_dir / f"softmax_{v}.npy"
            if p.exists(): sm.append(np.load(p))
        if not sm: continue
        sm = np.stack(sm, axis=0)  # (V, N, 10)
        avg = sm.mean(axis=0)
        # Argmax-majority per image
        argmaxes = sm.argmax(axis=2)  # (V, N)
        from scipy.stats import mode
        maj = mode(argmaxes, axis=0, keepdims=False).mode  # (N,)
        np.save(out_dir / 'avg_softmax.npy', avg.astype(np.float32))
        np.save(out_dir / 'argmax_majority.npy', maj.astype(np.int64))

    print(f"[{time.strftime('%X')}] v2-pool TEST inference DONE", flush=True)


if __name__ == '__main__':
    main()
