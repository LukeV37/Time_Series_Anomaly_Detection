# SPT

## Scope

The current SPT codepath in this repository covers two pieces:

- preprocessing benchmark calibrator-response HDF5 files into NumPy arrays
- training a minimal TranAD model from the saved preprocessing output

This is not a full port of the external `anldq` framework. The code here is a smaller local implementation focused on the benchmark HDF5 dataset and a minimal training path.

## Code Locations

- `src/preprocessing/data_loader/spt.py`: SPT benchmark HDF5 loader and metadata handling
- `src/preprocessing/pipeline.py`: config-driven preprocessing pipeline runner
- `src/preprocessing/configs/spt_pipeline.yaml`: benchmark preprocessing config
- `src/training/data.py`: `.npz` loading, chronological splitting, and sliding-window preparation
- `src/training/models/tranad.py`: minimal standalone TranAD model
- `src/training/train.py`: config-driven training entrypoint logic
- `src/training/configs/spt_tranad.yaml`: training config
- `scripts/spt/train_tranad.py`: CLI wrapper for training

## Data Assumptions

The verified preprocessing flow targets benchmark calibrator-response HDF5 files under `SPT_DATA_DIR_BENCHMARK`.

The current loader expects benchmark files with detector data arranged by year and observation. It produces a NumPy array with shape `(T, C, D)` plus metadata saved into a compressed `.npz` file.

The current training flow expects that `.npz` file to include a `data` array. Training flattens each time step from `(C, D)` to a single feature dimension `F = C * D`, applies sliding windows, and trains on tensors shaped like `(B, W, F)`.

## Preprocessing

The preprocessing config lives at `src/preprocessing/configs/spt_pipeline.yaml`.

From the repo root, a typical programmatic entrypoint is:

```python
import sys
sys.path.insert(0, "src")

from preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline.from_config_file("configs/spt_pipeline.yaml")
result = pipeline.load_and_run()
```

With the current config, the pipeline:

- loads benchmark HDF5 data for the configured years
- drops channels with too many NaNs
- drops time steps that are mostly NaN
- fills remaining NaNs with zero
- clips values to a configured range
- saves the result to `$OUTPUT_DIR/spt/<data_tag>/processed.npz` when output saving is enabled

## Training

The training config lives at `src/training/configs/spt_tranad.yaml`.

From the repo root, the main CLI is:

```bash
python scripts/spt/train_tranad.py --config src/training/configs/spt_tranad.yaml
```

The current training path is intentionally small:

- load preprocessing output from `.npz`
- split the series chronologically into train, validation, and test segments
- build sliding windows
- train a minimal TranAD model with PyTorch
- print metrics to stdout

## Current Limits

The current SPT implementation does not provide full parity with the external `CrossExperimentalAIDQM` codebase.

Notable gaps include:

- no full `anldq` package or registry system
- no verified inference or scoring pipeline in the local minimal path
- no richer checkpoint or experiment-output management yet
- some existing SPT scripts still depend on external code not present in this repository

In particular, `scripts/spt/infer_spt_v2.py` imports `anldq.*` and should be treated as an external or legacy path unless verified separately.
