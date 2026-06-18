"""Corruption registry — single source of truth for which families/types are
available per modality. The CLI reads from here to build menus and validate
choices.

Each (modality, family, type) is a row. The CLI never asks the user for a
corruption that doesn't exist in this table.

Adding a new corruption later means appending a row here and writing an apply
function in cli/apply.py. Nothing else changes.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class CorruptionType:
    name: str                       # e.g. "gaussian_noise"
    description: str                # one-line for the menu
    severity_grid: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    needs_conditioning: bool = False        # MAR: condition column
    needs_target_columns: bool = False      # tabular value-level corruptions: which cols
    image_size_constraint: Optional[List[int]] = None   # e.g. [32, 28] (sides allowed)


@dataclass
class CorruptionFamily:
    name: str                       # e.g. "Noise"
    description: str
    types: List[CorruptionType]


@dataclass
class ModalitySpec:
    name: str                       # "image" / "tabular"
    families: List[CorruptionFamily]


# -------- Image catalog --------
IMAGE = ModalitySpec(
    name="image",
    families=[
        CorruptionFamily(
            name="Noise",
            description="Pixel-level random perturbations",
            types=[
                CorruptionType("gaussian_noise", "Additive Gaussian noise"),
                CorruptionType("shot_noise",     "Poisson (shot) noise"),
                CorruptionType("impulse_noise",  "Salt-and-pepper impulse noise"),
                CorruptionType("spatter",        "Random blotches"),
            ],
        ),
        CorruptionFamily(
            name="Blur",
            description="Reduce spatial detail and edges",
            types=[
                CorruptionType("defocus_blur", "Out-of-focus lens blur"),
                CorruptionType("glass_blur",   "Local pixel shuffling"),
                CorruptionType("motion_blur",  "Motion blur"),
                CorruptionType("zoom_blur",    "Zoom blur"),
            ],
        ),
        CorruptionFamily(
            name="Weather",
            description="Environmental effects",
            types=[
                CorruptionType("fog",   "Haze reducing visibility"),
                CorruptionType("frost", "Ice-crystal artifacts"),
                CorruptionType("snow",  "Snow-like occlusion"),
            ],
        ),
        CorruptionFamily(
            name="Digital",
            description="Acquisition and post-processing artifacts",
            types=[
                CorruptionType("brightness", "Global intensity shifts"),
                CorruptionType("contrast",   "Changes in intensity separation"),
                CorruptionType("jpeg",       "Lossy compression artifacts"),
                CorruptionType("pixelate",   "Blocky downsampling"),
            ],
        ),
        CorruptionFamily(
            name="Geometric",
            description="Spatial transformations",
            types=[
                CorruptionType("rotate",    "In-plane rotation"),
                CorruptionType("shear",     "Axis-slanted shear"),
                CorruptionType("translate", "Spatial shift"),
                CorruptionType("scale",     "Resize"),
                CorruptionType("elastic",   "Local elastic deformation"),
            ],
        ),
    ],
)


# -------- Tabular catalog --------
TABULAR = ModalitySpec(
    name="tabular",
    families=[
        CorruptionFamily(
            name="Missing",
            description="Replace observed values with NaN under different mechanisms",
            types=[
                CorruptionType(
                    "missing_mcar",
                    "Missing Completely At Random (uniform across cells)",
                    needs_target_columns=True,
                ),
                CorruptionType(
                    "missing_mar",
                    "Missing At Random (drop probability depends on a conditioning feature)",
                    needs_conditioning=True,
                    needs_target_columns=True,
                ),
                CorruptionType(
                    "missing_mnar",
                    "Missing Not At Random (drop probability depends on the cell's own value)",
                    needs_target_columns=True,
                ),
            ],
        ),
        CorruptionFamily(
            name="Value",
            description="Perturb observed values",
            types=[
                CorruptionType(
                    "gaussian_noise",
                    "Additive Gaussian noise on numeric cells",
                    needs_target_columns=True,
                ),
                CorruptionType(
                    "scaling",
                    "Multiplicative scaling on numeric cells",
                    needs_target_columns=True,
                ),
            ],
        ),
    ],
)


# -------- Registry --------
REGISTRY: Dict[str, ModalitySpec] = {
    "image":   IMAGE,
    "tabular": TABULAR,
}


def list_modalities() -> List[str]:
    return list(REGISTRY.keys())


def get_modality(name: str) -> ModalitySpec:
    if name not in REGISTRY:
        raise ValueError(f"Unknown modality {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name]


def find_type(modality: str, type_name: str) -> CorruptionType:
    """Return the CorruptionType object given (modality, type_name)."""
    spec = get_modality(modality)
    for fam in spec.families:
        for t in fam.types:
            if t.name == type_name:
                return t
    raise ValueError(f"{type_name!r} not found under modality {modality!r}")
