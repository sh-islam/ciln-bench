# voter_adapters.py

An adapter is the recipe for turning `(uint8 images, weights file)` into
`(softmax probabilities)` for a single voter. The benchmark builder calls it
on every setting.

This file ships adapters for the 8 image voters used in the benchmark, plus a
registry the builder reads. Each adapter was verified to produce
bit-identical softmaxes to the ones on HuggingFace.

| Dataset | Voter | Batch size |
| --- | --- | --- |
| cifar10 | resnet20 | 256 |
| cifar10 | wrn28_10 | 256 |
| cifar10 | deit3_small | 64 |
| cifar10 | clip | 256 |
| mnist | lenet5 | 256 |
| mnist | mlp | 256 |
| mnist | resnet20 | 256 |
| mnist | deit3_small | 64 |

Batch size matters: cuDNN picks different conv/attention algorithms at
different batch sizes, so a mismatch shows up as small but non-zero
drift in the softmaxes. The values above are the ones used at release time.

## The interface

Three functions per voter, plus a batch size:

```python
def load_model(ckpt_path):
    # Build the architecture, load weights, return a model in eval mode.
    ...

def preprocess(images_uint8):
    # images_uint8: (N, H, W) for grayscale or (N, H, W, 3) for RGB.
    # Return a torch tensor ready to feed into the model.
    ...

def postprocess(logits):
    # logits: model output, typically (N, K).
    # Return softmax probabilities as a (N, K) float32 numpy array.
    ...
```

CLIP is the one exception: its `preprocess` returns the uint8 array as-is and
its wrapper class does the resize, normalize, encode, and zero-shot dot
product internally.

## Bringing your own image voter

Copy the adapter closest to your architecture in `voter_adapters.py`, edit
the three functions, then either:

1. Add an entry to `REGISTRY` in the same file, or
2. Drop your three functions into your own module and pass its path to
   `build_noisy_benchmark.py --adapter ./my_adapter.py`.

## Why preprocessing goes through PIL.Image

`transforms.ToTensor()` on a `PIL.Image` produces tensors that differ
slightly from `arr.astype(float32) / 255` on the raw numpy array, and that
1e-7-scale difference snowballs to ~1e-3 after the first conv. To stay
bit-identical with the training-time preprocessing, all image adapters route
each image through `Image.fromarray(...)` first. Custom voters that don't
need bit-identity can skip PIL.

## What about the Adult (tabular) voters?

No adapters are shipped for `xgboost`, `catboost`, `mlp`, `ft_transformer`,
or `tabpfn`. Tabular preprocessing is genuinely dataset-specific (categorical
vocab union, quantile transformer fit on clean train, NaN handling per voter
family) and depends on having the clean Adult training data on disk. A user
with their own tabular dataset would not be able to copy our adapter
anyway, since their columns and categoricals differ.

The reference implementation lives in
[`pipeline/eval/eval_tabular_voters.py`](../eval/eval_tabular_voters.py). It
was the code that generated the released softmaxes for the Adult settings.
Anyone who needs to re-run inference on Adult can use it directly.
