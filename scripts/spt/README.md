# SPT Scripts

This directory contains small utilities for SPT preprocessing, training, and plotting.

## Training

`train_tranad.py` runs the minimal TranAD training pipeline in this repository.

### Training Requirements

Set up the repo environment first:

```bash
bash setup.sh
source venv/bin/activate
```

The training code requires `torch` in addition to the repo's base requirements. If it is not already installed in your environment:

```bash
pip install torch
```

### Training Input

The current local SPT path uses two preprocessing `.npz` files:
- `train_processed.npz`
- `test_processed.npz`

Each file is a standalone preprocessing artifact containing a `data` array with shape `(T, C, D)` plus selected metadata when present.

The training code also accepts two compatibility formats:
- a legacy split-aware artifact with `train_data`, `val_data`, and `test_data`
- a legacy single-artifact file with only `data`

The current SPT preprocessing path does not write precomputed train/val/test splits into one file.

The default config file is:

```bash
src/training/configs/spt_tranad_no_trim.yaml
```

Use `src/training/configs/spt_tranad_trim.yaml` for the trimmed workflow. Update `input.train_npz_path` and `input.test_npz_path` in the selected YAML to point to your real preprocessing outputs.

### Minimal Training Command

```bash
python scripts/spt/train_tranad.py --config src/training/configs/spt_tranad_no_trim.yaml
```

### Minimal Example Config

```yaml
input:
  root: null
  experiment: spt
  data_tag: no_trim
  train_npz_path: train_processed.npz
  test_npz_path: test_processed.npz

split:
  train_ratio: 0.6
  val_ratio: 0.2

loader:
  batch_size: 16
  num_workers: 0

model:
  type: tranad
  params:
    window_size: 10
    d_model: 64
    nhead: 8
    num_layers: 2
    dim_feedforward: 128
    dropout: 0.1

training:
  scaling: robust
  epochs: 3
  learning_rate: 0.0001
  device: cpu

output:
  root: null
  data_tag: null
  checkpoint_path: /tmp/tranad.pt
  test_errors_path: /tmp/test_errors.npy
```

The script prints per-epoch train/validation losses, emits a final metrics JSON, and can optionally save:
- a checkpoint via `output.checkpoint_path`
- test-set reconstruction errors via `output.test_errors_path`

If you point it at an older artifact with a single `data` array, the loader still falls back to config-driven chronological splitting. The current supported SPT path, though, uses separate train and test preprocessing artifacts and creates the validation split during training.

## Job Submission

`submit_spt_jobs.sh` is a thin wrapper around `qsub` for the current SPT preprocessing and training variants.

It submits `scripts/spt/swing_spt_example.pbs` with the matching preprocessing and training config pair:
- `no_trim` -> `src/preprocessing/configs/spt_pipeline_no_trim.yaml` and `src/training/configs/spt_tranad_no_trim.yaml`
- `trim` -> `src/preprocessing/configs/spt_pipeline_trim.yaml` and `src/training/configs/spt_tranad_trim.yaml`
- `both` -> submits both jobs

Usage:

```bash
bash scripts/spt/submit_spt_jobs.sh no_trim
bash scripts/spt/submit_spt_jobs.sh trim
bash scripts/spt/submit_spt_jobs.sh both
```

The underlying PBS job reads these environment variables:
- `PREPROCESS_CONFIG_PATH`
- `TRAIN_CONFIG_PATH`

If you want to submit manually without the helper, the equivalent commands are:

```bash
qsub -v PREPROCESS_CONFIG_PATH=src/preprocessing/configs/spt_pipeline_no_trim.yaml,TRAIN_CONFIG_PATH=src/training/configs/spt_tranad_no_trim.yaml scripts/spt/swing_spt_example.pbs
qsub -v PREPROCESS_CONFIG_PATH=src/preprocessing/configs/spt_pipeline_trim.yaml,TRAIN_CONFIG_PATH=src/training/configs/spt_tranad_trim.yaml scripts/spt/swing_spt_example.pbs
```

## Histogram Plot

`plot_spt_histogram.py` plots test reconstruction errors against the saved raw SNR values from the preprocessing test artifact.

It writes two figures:
- a 1D histogram comparing detectors above and below the SNR threshold
- a 2D histogram of reconstruction error versus raw SNR

### Requirements

Set up the repo environment first:

```bash
bash setup.sh
source venv/bin/activate
```

The script needs at least:
- `numpy`
- `matplotlib`

### Required Input

The plotting path is config-driven. By default it resolves paths from the training config:

```bash
python scripts/spt/plot_spt_histogram.py --config src/training/configs/spt_tranad_no_trim.yaml
```

It expects:
- a saved test errors file, usually from `output.test_errors_path`
- a saved SPT test preprocessing artifact, usually from `input.test_npz_path`

You can override either path explicitly:

```bash
python scripts/spt/plot_spt_histogram.py \
  --config src/training/configs/spt_tranad_no_trim.yaml \
  --errors /path/to/test_errors.npy \
  --data-npz /path/to/test_processed.npz
```

### Saved Test Artifact Requirement

The script reads raw SNR values from the saved test preprocessing artifact. The expected input is the SPT test artifact written by `scripts/spt/run_preprocessing.py`, typically `test_processed.npz`.

The reconstruction errors and saved SNR values must describe the same retained detector set. If preprocessing changed the detector set and the plotting input does not include the matching filtered test artifact, the script raises a shape-mismatch error.

### Threshold And Window Size

The SNR threshold controls the 1D split between low-SNR and high-SNR detectors.

Defaults:

```bash
--threshold 20.0
```

The script also needs the training window size so it can align per-window errors with the saved test SNR series. By default it uses `model.params.window_size` from the training config, but you can override it:

```bash
--window-size 10
```

### Output

The current CLI writes two files:
- `--output-1d` for the thresholded 1D histogram
- `--output-2d` for the error-versus-SNR 2D histogram

If not provided, defaults are resolved under the training output directory:
- `test_error_histogram_1D.png`
- `test_error_histogram_2D.png`

Example:

```bash
python scripts/spt/plot_spt_histogram.py \
  --config src/training/configs/spt_tranad_no_trim.yaml \
  --errors /path/to/test_errors.npy \
  --data-npz /path/to/test_processed.npz \
  --output-1d output/test_error_histogram_1D.png \
  --output-2d output/test_error_histogram_2D.png
```

## Notes

- `--labels`, `--use-spt-loader`, `--spt-root`, and `--output` are not part of the current CLI.
- The plotting code expects positive finite reconstruction errors.
- This is a lightweight local plotting utility and does not depend on the `anldq` package.
