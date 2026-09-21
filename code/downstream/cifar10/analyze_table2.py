"""Aggregate v4 downstream result JSONs into Table 2 (PL-IDN vs CILN) cells.

Reads every JSON under results_v4/{erm,coteaching,dividemix}/, groups by
(band, variant, corruption, sev, method, image_source), averages best_test_acc
across seeds, and prints the 11 rows the paper Table 2 covers with all 6
Accuracy columns filled in.

Usage:
    python analyze_table2.py               # prints just the 11 rows we're filling
    python analyze_table2.py --all         # prints every condition on disk
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_v4"

# Table 2's 11-row layout, ordered as in experiments.tex lines 32-114
# (band, variant, corruption_display, sev, expected_noise_rate_pct)
ROWS = [
    # Band L (~10.8%)
    ("L", "CILN-S", "frost",          1, 10.1),
    ("L", "CILN-S", "pixelate",       1, 11.7),
    ("L", "CILN-C", "frost",          1,  2.8),
    ("L", "CILN-C", "pixelate",       1,  4.1),
    # Band M (~19.4%)
    ("M", "CILN-S", "elastic",        3, 17.4),
    ("M", "CILN-S", "snow",           3, 19.4),
    ("M", "CILN-S", "gaussian",       1, 21.5),
    ("M", "CILN-C", "elastic",        3, 10.1),
    ("M", "CILN-C", "snow",           3, 12.5),
    ("M", "CILN-C", "gaussian",       1, 14.4),
    # Band H (~47.8%)
    ("H", "CILN-S", "contrast",       5, 47.3),
    ("H", "CILN-S", "gaussian",       3, 49.2),
    ("H", "CILN-S", "shot",           5, 54.5),
    ("H", "CILN-S", "pixelate",       5, 58.2),
    ("H", "CILN-S", "glass-blur",     3, 64.6),
    ("H", "CILN-C", "contrast",       5, 43.7),
    ("H", "CILN-C", "gaussian",       3, 44.7),
    ("H", "CILN-C", "shot",           5, 51.0),
    ("H", "CILN-C", "pixelate",       5, 52.7),
    ("H", "CILN-C", "glass-blur",     3, 61.8),
]

# Display -> folder mapping used by prepare_pairs_v4*.py
CORR_FOLDER = {
    "frost":      "frost",
    "pixelate":   "pixelate",
    "elastic":    "elastic_transform",
    "snow":       "snow",
    "gaussian":   "gaussian_noise",
    "shot":       "shot_noise",
    "contrast":   "contrast",
    "glass-blur": "glass_blur",
}

# (method, image_source) -> Table 2 column key
COLUMN = {
    ("erm",        "clean"): "ERM-C",
    ("erm",        "noisy"): "ERM-N",
    ("coteaching", "clean"): "CoT-C",
    ("coteaching", "noisy"): "CoT-N",
    ("dividemix",  "clean"): "DM-C",
    ("dividemix",  "noisy"): "DM-N",
}


def load_all():
    """Return dict[(variant, corr_folder, sev, method, image_source)] -> list of best_test_acc"""
    acc = defaultdict(list)
    seeds = defaultdict(list)
    for method in ["erm", "coteaching", "dividemix"]:
        for jf in sorted(glob.glob(str(RESULTS / method / "*.json"))):
            if jf.endswith("__persample.npz"):
                continue
            try:
                d = json.loads(Path(jf).read_text())
            except Exception:
                continue
            cond = d.get("condition", "")
            m = re.match(r"^(ours_v2(?:_ccp)?)_(clean|noisy)__(.+)_sev(\d+)$", cond)
            if not m:
                continue
            pipeline, split, corr_folder, sev = m.group(1), m.group(2), m.group(3), int(m.group(4))
            variant = "CILN-C" if pipeline == "ours_v2_ccp" else "CILN-S"
            key = (variant, corr_folder, sev, method, split)
            acc[key].append(d["best_test_acc"] * 100.0)
            seeds[key].append(d.get("seed", -1))
    return acc, seeds


def mean_pm_std(values):
    if not values:
        return None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, None
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, var ** 0.5


def fmt_cell(values):
    ms = mean_pm_std(values)
    if ms is None:
        return "  --  "
    m, s = ms
    if s is None:
        return f"{m:5.1f}~ "
    return f"{m:5.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Also print rows not in Table 2 (debug)")
    parser.add_argument("--variance", action="store_true",
                        help="Flag cells with seed variance > 3 pp")
    args = parser.parse_args()

    acc, seeds = load_all()

    print(f"\n{'Row':<28} {'noise':>6} {'ERM-C':>6} {'ERM-N':>6} {'CoT-C':>6} {'CoT-N':>6} {'DM-C':>6} {'DM-N':>6}   {'n_seeds':<20}")
    print("-" * 118)

    for band, variant, corr_display, sev, exp_nr in ROWS:
        corr_folder = CORR_FOLDER[corr_display]
        row_label = f"{band} {variant} {corr_display}({sev})"
        cells = []
        n_seeds_summary = []
        for (method, split), col in COLUMN.items():
            key = (variant, corr_folder, sev, method, split)
            v = acc.get(key, [])
            cells.append(fmt_cell(v))
            if v:
                n_seeds_summary.append(f"{col}={len(v)}")

        n_str = ", ".join(n_seeds_summary) if n_seeds_summary else "(no data)"
        print(f"{row_label:<28} {exp_nr:>5.1f}% {cells[0]:>6} {cells[1]:>6} {cells[2]:>6} {cells[3]:>6} {cells[4]:>6} {cells[5]:>6}   {n_str}")

        if args.variance:
            for (method, split), col in COLUMN.items():
                key = (variant, corr_folder, sev, method, split)
                v = acc.get(key, [])
                if len(v) >= 2:
                    ms = mean_pm_std(v)
                    m, s = ms
                    if s > 3.0:
                        print(f"    *** VARIANCE ALERT: {col} = {v} (std={s:.1f})")

    if args.all:
        print("\n=== Every condition present on disk (raw) ===")
        for key in sorted(acc.keys()):
            v = acc[key]
            variant, corr_folder, sev, method, split = key
            ms = mean_pm_std(v)
            m_str = f"{ms[0]:5.1f}" if ms else "  --"
            std_str = f"±{ms[1]:.1f}" if (ms and ms[1] is not None) else ""
            print(f"  {variant:<7} {corr_folder:<20} sev{sev} {method:<10} {split:<6} n={len(v)} best={m_str} {std_str}")


if __name__ == "__main__":
    main()
