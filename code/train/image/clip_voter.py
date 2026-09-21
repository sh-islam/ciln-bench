"""CLIP ViT-B/32 zero-shot voter for CIFAR-10.

No training. The model is OpenAI's CLIP ViT-B/32 (downloaded via the `clip`
library or via open_clip with the openai pretrained tag). At inference:

  1. Text prompts "a photo of a <CIFAR class name>" are tokenized once.
  2. Each image is resized to 224x224, CLIP-normalized, encoded.
  3. Cosine similarity image_feats @ text_feats.T -> softmax with temperature
     100 (CLIP's standard) -> per-class probability distribution.

We use the openai weights via open_clip (already installed in venv). This
avoids depending on the older `clip` package and gets consistent preprocessing.

NOT used for MNIST: CLIP performs poorly on isolated digits (no relevant
training pairs in CLIP's web-scale data).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

CIFAR10_PROMPTS = [f"a photo of a {c}" for c in CIFAR10_CLASSES]

# CLIP normalization constants (same as the original CLIP repo)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPVoter:
    """Wraps an open_clip ViT-B/32 model + text features.

    Use:
        v = CLIPVoter.load(device='cuda')
        probs = v.predict(image_array)  # (N, H, W, 3) uint8 -> (N, 10) float32
    """

    def __init__(self, model, tokenizer, text_features, device):
        self.model = model
        self.tokenizer = tokenizer
        self.text_features = text_features  # (n_classes, d) float32, L2-normalized
        self.device = device
        self.mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(CLIP_STD).view(1, 3, 1, 1).to(device)

    @classmethod
    def load(cls, device: Optional[str] = None) -> "CLIPVoter":
        import open_clip
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # force_quick_gelu=True matches the OpenAI weights' original activation
        # (avoids a small accuracy drop from using standard GELU instead).
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", force_quick_gelu=True
        )
        model.eval().to(device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            text_tokens = tokenizer(CIFAR10_PROMPTS).to(device)
            text_features = model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return cls(model, tokenizer, text_features, device)

    @torch.no_grad()
    def predict(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Run zero-shot CLIP on a batch of uint8 RGB images of shape (N, H, W, 3).

        Returns softmax probabilities (N, 10) float32.
        """
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected (N, H, W, 3) uint8 RGB, got shape {images.shape}")
        n = len(images)
        out = np.empty((n, len(CIFAR10_CLASSES)), dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = images[start:end]
            # (B, H, W, 3) uint8 -> (B, 3, H, W) float in [0,1]
            t = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
            t = t.to(self.device)
            # Resize to 224x224 (CLIP's expected size)
            t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bicubic", align_corners=False)
            t = (t - self.mean) / self.std
            image_features = self.model.encode_image(t).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features.T)
            probs = similarity.softmax(dim=-1).cpu().numpy().astype(np.float32)
            out[start:end] = probs
        return out
