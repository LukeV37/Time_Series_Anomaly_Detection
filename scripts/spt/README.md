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

The training script expects a preprocessing `.npz` file containing at least:
- `data` with shape `(T, C, D)`

The default config file is:

```bash
src/training/configs/spt_tranad.yaml
```

Update `input.npz_path` in that YAML to point to your real preprocessing output.

### Minimal Training Command

```bash
python scripts/spt/train_tranad.py --config src/training/configs/spt_tranad.yaml
```

### Minimal Example Config

```yaml
input:
  npz_path: /absolute/path/to/processed.npz

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
  epochs: 3
  learning_rate: 0.0001
  device: cpu

output:
  checkpoint_path: /tmp/tranad.pt
```

The script prints a small metrics JSON at the end and optionally saves a checkpoint if `output.checkpoint_path` is set.

## Histogram Plot

`plot_spt_histogram.py` plots reconstruction-error histograms for:
- all detectors
- the high-SNR detector subset

It also draws a ratio panel comparing the two histograms.

### Requirements

Set up the repo environment first:

```bash
bash setup.sh
source venv/bin/activate
```

The script needs at least:
- `numpy`
- `matplotlib`
- `h5py` if you use the local SPT loader mode

### Required Input

The script always needs an errors file:

```bash
--errors /path/to/errors.npy
```

Supported shapes:
- `(T, C)`
- `(T, C, 1)`

### Label Sources

The script needs per-channel labels to decide which detectors are high-SNR vs low-SNR.

You can provide labels in one of three ways.

#### 1. Explicit Labels

Provide a `.npy` file with shape `(C,)`.

Convention:
- `0` = high-SNR detector
- `1` = low-SNR detector

Example:

```bash
python scripts/spt/plot_spt_histogram.py \
  --errors /path/to/errors.npy \
  --labels /path/to/channel_labels.npy \
  --output output/spt_validation_error_histogram.png
```

#### 2. Derive Labels From A Preprocessing `.npz`

If you have a preprocessing output file with `data` shaped `(T, C, D)`, the script can derive channel labels from the detector median response.

Example:

```bash
python scripts/spt/plot_spt_histogram.py \
  --errors /path/to/errors.npy \
  --data-npz /path/to/processed.npz \
  --threshold 20.0 \
  --output output/spt_validation_error_histogram.png
```

#### 3. Derive Labels With The Local SPT Loader

This uses `src/preprocessing/data_loader/spt.py` to load the benchmark SPT calibrator-response data and derive labels from detector medians.

Example:

```bash
python scripts/spt/plot_spt_histogram.py \
  --errors /path/to/errors.npy \
  --use-spt-loader \
  --spt-root /path/to/calibrator_responses \
  --threshold 20.0 \
  --output output/spt_validation_error_histogram.png
```

If `--spt-root` is omitted, the loader uses the repo default logic from `src/preprocessing/data_loader/spt.py`.

### Threshold Meaning

When labels are derived instead of loaded explicitly:
- channels with median response `< threshold` are labeled low-SNR
- channels with median response `>= threshold` are labeled high-SNR

Default:

```bash
--threshold 20.0
```

### Output

The script writes a PNG plot to the path given by `--output`.

Default:

```bash
output/spt_validation_error_histogram.png
```

## Notes

- The number of channels in `errors.npy` must match the number of channel labels.
- The plotting code expects positive finite reconstruction errors.
- This is a lightweight local port of the CrossExperimental histogram utility. It does not depend on the `anldq` package.
