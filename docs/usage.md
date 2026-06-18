# Run-your-own pipeline (interactive starter)

`run_pipeline.py` is an interactive entry point that lets you point at your own dataset, pick a corruption from the CILN-Bench catalog, and generate a CILN-style corrupted output. It is a thin orchestrator over the corruption code in `code/corrupt/` — your existing released datasets and code paths are untouched.

## What it does

1. Asks whether your data is **image** or **tabular**.
2. Asks for the path to your data and validates it (shape, dtype, columns).
3. Shows only the corruption families that exist for that modality.
4. Asks for type, severity, and (for tabular) target columns / conditioning column.
5. Applies the corruption and writes:
   - the corrupted dataset (`images.npy` or `features_corrupted.parquet`)
   - a copy of the original (`original.npy` or `features_original.parquet`)
   - a manifest describing every choice (`manifest.json`)
   - a per-image/row log with RNG seed, applied parameters, and output checksum (`params.jsonl`)

## Quick start

```bash
python run_pipeline.py
```

Then follow the prompts. Example session:

```
What kind of dataset?
  [1] image
  [2] tabular
> 1
Path to dataset: ./my_cifar_subset.npy
  -> 5000 RGB images, 32x32, dtype=uint8

Choose corruption family:
  [1] Noise
  [2] Blur
  [3] Weather
  [4] Digital
  [5] Geometric
> 1

Choose corruption type in family 'Noise':
  [1] gaussian_noise
  [2] shot_noise
  [3] impulse_noise
  [4] spatter
> gaussian_noise

Choose severity (1-5)
> 3

Output folder [./out__gaussian_noise_sev3]: ./my_noisy
  Running gaussian_noise sev=3 ...
  Output written to ./my_noisy/
```

## What kinds of input are accepted

### Image
- A `.npy` file with shape `(N, H, W)` (grayscale) or `(N, H, W, C)` with C ∈ {1, 3}.
- Square images only (H must equal W).
- Side length must be one of `{28, 32, 64, 224}` (extend `SUPPORTED_IMAGE_SIDES` in `cli/validators.py` to add more).
- Dtype `uint8` preferred; floats in `[0, 1]` or `[0, 255]` are auto-rescaled.

### Tabular
- A `.csv` or `.parquet` file readable by pandas.
- We try to auto-detect a label column among `label`, `target`, `y`, `class`, `income` (case-insensitive); if none match, you'll be asked.
- Numeric vs categorical columns are determined by pandas dtypes (`int*`/`float*` → numeric, everything else → categorical).

## Tabular conditioning options

| Corruption | Target columns? | Conditioning column? |
|---|---|---|
| `missing_mcar` | yes (which cols to drop) | no |
| `missing_mar`  | yes (which cols to drop) | yes (categorical preferred) |
| `missing_mnar` | yes (which cols to drop) | no (uses the cell's own value) |
| `gaussian_noise` | numeric cols only | no |
| `scaling`        | numeric cols only | no |

The CLI will *only* offer columns of the right kind. You cannot, for example, pick a categorical column as the noise target for `gaussian_noise`.

## Output structure

Every run produces:

```
<output_folder>/
├── manifest.json            modality, family, type, severity, choices, timestamp
├── params.jsonl             one line per image/row: seed, applied params, sha256
├── images.npy               (image only) the corrupted dataset
├── original.npy             (image only) a copy of your input, for reference
├── features_corrupted.parquet  (tabular only) the corrupted dataset
└── features_original.parquet   (tabular only) a copy of your input
```

`params.jsonl` is the per-image / per-row reproducibility log. Each entry includes:
- the per-image/row RNG seed used
- the corruption parameters actually applied
- a sha256 of the corrupted output (so anyone re-running can verify a match)

## Safety guarantees

- The script never modifies anything in your input path.
- The script refuses to overwrite an existing output folder unless you confirm.
- Invalid input causes a clear error and a re-prompt — nothing partial is written.

## Extending to a new corruption

1. Add a `CorruptionType` entry to the relevant family in `cli/registry.py`.
2. Add a branch in `cli/apply.py` that implements the corruption.
3. The CLI menus pick up the new type automatically.
