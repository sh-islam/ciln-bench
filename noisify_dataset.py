"""CILN-Bench: interactive dataset noisification starter.

Lets a user point at their own dataset, pick a corruption, and write out a
CILN-style corrupted dataset (plus reproducibility logs).

Usage:
    python noisify_dataset.py

Hard rule: this script never modifies anything in the user's input path. All
outputs go to a user-specified folder. We refuse to overwrite an existing
output folder unless the user says yes.
"""
from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

# Allow running from anywhere
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cli import registry, validators, apply, prompts


def banner():
    print("=" * 60)
    print("CILN-Bench: corruption-induced label noise (interactive)")
    print("=" * 60)


def pick_modality_path():
    modality = prompts.ask_choice("What kind of dataset?", registry.list_modalities())
    while True:
        path = prompts.ask_text("Path to dataset", default=None)
        try:
            if modality == "image":
                data = validators.load_image_dataset(path)
                print(f"  -> {validators.summarize_image_dataset(data)}")
                labels = _ask_image_labels(len(data))
                return modality, path, data, labels
            else:
                data = validators.load_tabular_dataset(path)
                guess = validators.detect_label_column(data)
                if guess is not None:
                    print(f"  guessed label column: {guess!r}")
                    lbl = prompts.ask_text(
                        f"Press Enter to keep, or type the column name to use",
                        default=guess,
                    )
                    if lbl not in data.columns:
                        print(f"  '{lbl}' is not a column; pick from the list")
                        lbl = prompts.ask_choice(
                            "Which column is the target label?",
                            list(data.columns),
                        )
                else:
                    print("  (no label column auto-detected)")
                    lbl = prompts.ask_choice(
                        "Which column is the target label?",
                        list(data.columns),
                    )
                print(f"  -> {validators.summarize_tabular_dataset(data, lbl)}")
                data.attrs["__label_col__"] = lbl
                return modality, path, data, None  # tabular: labels live in the dataframe column
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  Try again, or Ctrl-C to quit.")


def _ask_image_labels(n_images_expected):
    """Prompt for a labels .npy and validate it lines up with the images."""
    while True:
        labels_path = prompts.ask_text(
            "Path to labels .npy (must line up with the images by index)",
            default=None,
        )
        try:
            labels = validators.load_image_labels(labels_path, n_images_expected)
            print(f"  -> {validators.summarize_image_labels(labels)}")
            return labels
        except Exception as e:
            print(f"  ERROR: {e}")
            print("  Try again, or Ctrl-C to quit.")


def pick_corruption(modality_spec):
    fam_name = prompts.ask_choice(
        "Choose corruption family:",
        [f.name for f in modality_spec.families],
    )
    fam = next(f for f in modality_spec.families if f.name == fam_name)
    type_name = prompts.ask_choice(
        f"Choose corruption type in family '{fam_name}':",
        [t.name for t in fam.types],
    )
    type_obj = next(t for t in fam.types if t.name == type_name)
    sev = prompts.ask_severity(type_obj.severity_grid)
    return fam, type_obj, sev


def pick_tabular_options(type_obj, df):
    """Ask for target columns and (if needed) conditioning column."""
    label_col = df.attrs.get("__label_col__")
    candidate_cols = [c for c in df.columns if c != label_col]
    target_columns = None
    conditioning_column = None
    if type_obj.needs_target_columns:
        # For value-level corruptions we typically want numeric only
        if type_obj.name in ("gaussian_noise", "scaling"):
            numeric = validators.numeric_columns(df, exclude=[label_col] if label_col else [])
            if not numeric:
                raise ValueError("No numeric columns found for this corruption.")
            target_columns = prompts.ask_columns(
                "Which numeric columns to corrupt?", numeric, allow_all=True)
        else:
            target_columns = prompts.ask_columns(
                "Which columns to corrupt?", candidate_cols, allow_all=True)
    if type_obj.needs_conditioning:
        cats = validators.categorical_columns(df, exclude=[label_col] if label_col else [])
        if not cats:
            print("  (no categorical column to condition on, falling back to numeric)")
            cats = validators.numeric_columns(df, exclude=[label_col] if label_col else [])
        conditioning_column = prompts.ask_columns(
            "Pick the conditioning column:", cats, allow_all=False, multi=False)[0]
    return target_columns, conditioning_column


def write_output(out_dir: Path, manifest: dict, params_log: list,
                 corrupted_payload, labels=None, original=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    # manifest
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # params log (one line per row/image)
    with (out_dir / "params.jsonl").open("w") as f:
        for rec in params_log:
            f.write(json.dumps(rec) + "\n")
    # corrupted payload
    if manifest["modality"] == "image":
        np.save(out_dir / "images.npy", corrupted_payload)
        if labels is not None: np.save(out_dir / "labels.npy", labels)
        if original is not None: np.save(out_dir / "original.npy", original)
    else:
        # tabular
        corrupted_payload.to_parquet(out_dir / "features_corrupted.parquet")
        if original is not None:
            original.to_parquet(out_dir / "features_original.parquet")
    print(f"\n  Output written to {out_dir}/")
    for p in sorted(out_dir.iterdir()):
        print(f"    {p.name}")


def main():
    banner()
    try:
        modality, in_path, data, image_labels = pick_modality_path()
        modality_spec = registry.get_modality(modality)

        fam, type_obj, severity = pick_corruption(modality_spec)
        setting_key = f"{type_obj.name}_sev{severity}"

        target_columns = None
        conditioning_column = None
        if modality == "tabular":
            target_columns, conditioning_column = pick_tabular_options(type_obj, data)

        out_path_str = prompts.ask_text("Output folder", default=f"./out__{setting_key}")
        allow_overwrite = False
        while True:
            try:
                out_dir = validators.validate_output_path(out_path_str, allow_overwrite=allow_overwrite)
                break
            except FileExistsError as e:
                print(f"  {e}")
                allow_overwrite = prompts.ask_yes_no("Overwrite existing folder?", default=False)
                if not allow_overwrite:
                    out_path_str = prompts.ask_text("Output folder (new)", default=None)

        # Run
        t0 = time.time()
        print(f"\n  Running {type_obj.name} sev={severity} ...")
        if modality == "image":
            corrupted, params_log = apply.corrupt_image_dataset(
                data, type_obj.name, severity, setting_key)
            manifest = {
                "modality": "image",
                "family": fam.name,
                "type": type_obj.name,
                "severity": severity,
                "n": int(len(data)),
                "shape": list(data.shape[1:]),
                "input_path": str(in_path),
                "timestamp": int(t0),
            }
            write_output(out_dir, manifest, params_log,
                         corrupted_payload=corrupted,
                         labels=image_labels,
                         original=data)
        else:
            corrupted, params_log = apply.corrupt_tabular_dataset(
                data, type_obj.name, severity,
                target_columns=target_columns,
                conditioning_column=conditioning_column,
                setting_key=setting_key)
            manifest = {
                "modality": "tabular",
                "family": fam.name,
                "type": type_obj.name,
                "severity": severity,
                "n": int(len(data)),
                "columns": list(data.columns),
                "target_columns": target_columns,
                "conditioning_column": conditioning_column,
                "input_path": str(in_path),
                "timestamp": int(t0),
            }
            write_output(out_dir, manifest, params_log,
                         corrupted_payload=corrupted, original=data)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s.")
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
