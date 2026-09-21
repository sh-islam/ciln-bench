"""Enumerate all 180 MNIST runs into a grid manifest CSV."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "manifest" / "mnist_grid.csv"
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

# Hardest per family (computed from noise rates, ≥ 2.5% cutoff)
SETTINGS = [
    "impulse_noise_sev5",   # Noise
    "glass_blur_sev5",      # Blur
    "rotate_sev5",          # Geometric
    "fog_sev5",             # Weather/Digital
    "canny_edges_sev1",     # Structural (binary)
]

VARIANTS = ["S", "C"]
METHODS = ["erm", "coteaching", "dividemix"]
SCENARIOS = ["clean", "noisy"]
SEEDS = [0, 1, 2]


def main():
    with MANIFEST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["status", "dataset", "setting", "variant", "method", "image_source", "seed"])
        for setting in SETTINGS:
            for variant in VARIANTS:
                for method in METHODS:
                    for scenario in SCENARIOS:
                        for seed in SEEDS:
                            w.writerow(["pending", "mnist", setting, variant, method, scenario, seed])
    n = sum(1 for _ in MANIFEST.open()) - 1
    print(f"wrote {n} rows to {MANIFEST}")


if __name__ == "__main__":
    main()
