"""Public image corruption interface.

Behaviour is identical to the canonical pipeline used to generate the released
CILN-Bench datasets, with cleaner names and an organised by-family table for
discovery.

Implementation note: the canonical implementations in
``code/corrupt/canonical/cifar_funcs.py`` and
``code/corrupt/canonical/mnist_funcs.py`` are based directly on
Hendrycks & Dietterich (2019) "Benchmarking Neural Network Robustness to
Common Corruptions and Perturbations". To guarantee bit-identical output
relative to the released benchmark we re-export those canonical functions
rather than re-implement them. The bit-identity is checked in
``tests/check_equivalence.py``.

Usage
-----

>>> from code.corrupt.public.image_corruptions import (
...     IMAGE_CORRUPTIONS, get_image_corruption
... )
>>> from code.corrupt.public.seeding import get_rng
>>> rng = get_rng(master_seed=0, dataset="cifar10",
...               corruption="gaussian_noise", severity=3, index=0)
>>> corrupted, params = get_image_corruption("cifar10", "gaussian_noise")(
...     clean_image, severity=3, rng=rng
... )

Each corruption returns ``(corrupted_image, params_dict)``. ``params_dict``
records the sampled parameters for reproducibility verification.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable, Dict

# Import the canonical implementations
_CANONICAL = Path(__file__).resolve().parent.parent / "canonical"
sys.path.insert(0, str(_CANONICAL))

from canonical import cifar_funcs as _cifar      # type: ignore
from canonical import mnist_funcs as _mnist      # type: ignore


# Family taxonomy. Every corruption is grouped by mechanism.
IMAGE_CORRUPTIONS: Dict[str, Dict[str, Dict[str, Callable]]] = {
    "cifar10": {
        "Noise": {
            "gaussian_noise":  _cifar.gaussian_noise,
            "shot_noise":      _cifar.shot_noise,
            "impulse_noise":   _cifar.impulse_noise,
        },
        "Blur": {
            "defocus_blur":    _cifar.defocus_blur,
            "glass_blur":      _cifar.glass_blur,
            "motion_blur":     _cifar.motion_blur,
            "zoom_blur":       _cifar.zoom_blur,
        },
        "Weather": {
            "fog":             _cifar.fog,
            "frost":           _cifar.frost,
            "snow":            _cifar.snow,
        },
        "Geometric": {
            "elastic_transform": _cifar.elastic_transform,
        },
        "Digital": {
            "brightness":      _cifar.brightness,
            "contrast":        _cifar.contrast,
            "jpeg_compression": _cifar.jpeg_compression,
            "pixelate":        _cifar.pixelate,
        },
    },
    "mnist": {
        "Noise": {
            "shot_noise":      _mnist.shot_noise,
            "impulse_noise":   _mnist.impulse_noise,
            "spatter":         _mnist.spatter,
        },
        "Blur": {
            "glass_blur":      _mnist.glass_blur,
            "motion_blur":     _mnist.motion_blur,
        },
        "Geometric": {
            "rotate":          _mnist.rotate,
            "shear":           _mnist.shear,
            "translate":       _mnist.translate,
            "scale":           _mnist.scale,
        },
        "Weather": {
            "fog":             _mnist.fog,
        },
        "Digital": {
            "brightness":      _mnist.brightness,
        },
        "Structural": {
            "canny_edges":     _mnist.canny_edges,
            "dotted_line":     _mnist.dotted_line,
            "stripe":          _mnist.stripe,
            "zigzag":          _mnist.zigzag,
        },
    },
}


def get_image_corruption(dataset: str, corruption: str) -> Callable:
    """Look up a (dataset, corruption) pair, returning the apply function.

    Raises a :class:`KeyError` with a helpful message if the pair is not
    defined.
    """
    if dataset not in IMAGE_CORRUPTIONS:
        raise KeyError(f"Unknown image dataset {dataset!r}. "
                       f"Known: {list(IMAGE_CORRUPTIONS)}")
    catalog = IMAGE_CORRUPTIONS[dataset]
    for family in catalog.values():
        if corruption in family:
            return family[corruption]
    families = {fam: list(types) for fam, types in catalog.items()}
    raise KeyError(f"Unknown {dataset} corruption {corruption!r}. "
                   f"Available families/types: {families}")


def list_image_corruptions(dataset: str) -> Dict[str, list]:
    """Return ``{family_name: [corruption_names]}`` for a dataset."""
    if dataset not in IMAGE_CORRUPTIONS:
        raise KeyError(f"Unknown dataset {dataset!r}")
    return {fam: list(types) for fam, types in IMAGE_CORRUPTIONS[dataset].items()}
