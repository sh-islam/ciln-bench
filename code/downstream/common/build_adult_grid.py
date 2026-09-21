"""Enumerate all 360 Adult runs into a grid manifest CSV.

Schema:
  status,dataset,setting,variant,method,image_source,seed
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "manifest" / "adult_grid.csv"
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

# 15 Adult settings
CORRUPTIONS = ["gaussian_noise", "scaling", "missing_mcar", "missing_mar", "missing_mnar"]
SEVERITIES = [1, 3, 5]
SETTINGS = [f"{c}_sev{s}" for c in CORRUPTIONS for s in SEVERITIES]

VARIANTS = ["S", "C"]
METHODS = ["erm", "coteaching"]
SCENARIOS = ["clean", "noisy"]
SEEDS = [0, 1, 2]

# Missing has different corruption folder names; align with actual on-disk names
CORR_ALIAS = {
    "missing_mcar": "missing_mcar",
    "missing_mar": "missing_mar",
    "missing_mnar": "missing_mnar",
}


def main():
    with MANIFEST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["status", "dataset", "setting", "variant", "method", "image_source", "seed"])
        for setting in SETTINGS:
            for variant in VARIANTS:
                for method in METHODS:
                    for scenario in SCENARIOS:
                        for seed in SEEDS:
                            w.writerow(["pending", "adult", setting, variant, method, scenario, seed])
    n = sum(1 for _ in MANIFEST.open()) - 1
    print(f"wrote {n} rows to {MANIFEST}")


if __name__ == "__main__":
    main()
