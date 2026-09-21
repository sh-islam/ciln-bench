"""AG-News grid: 6 settings x 2 methods x 2 scenarios x 2 variants x 3 seeds = 144."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "manifest" / "agnews_grid.csv"
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

SETTINGS = [f"{c}_sev{s}" for c in ["butter_fingers", "front_truncation"] for s in [1, 3, 5]]
VARIANTS = ["S", "C"]
METHODS = ["erm", "coteaching"]
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
                            w.writerow(["pending", "agnews", setting, variant, method, scenario, seed])
    n = sum(1 for _ in MANIFEST.open()) - 1
    print(f"wrote {n} rows to {MANIFEST}")


if __name__ == "__main__":
    main()
