# SPT

## Scope

The current SPT codepath in this repository covers two pieces:

- preprocessing benchmark calibrator-response HDF5 files into NumPy arrays
- training a minimal TranAD model from the saved preprocessing output

This is not a full port of the external `anldq` framework. The code here is a smaller local implementation focused on the benchmark HDF5 dataset and a minimal training path.

## Code Locations

- `src/preprocessing/data_loader/spt.py`: SPT benchmark HDF5 loader and metadata handling
- `src/preprocessing/pipeline.py`: config-driven preprocessing pipeline runner
- `src/preprocessing/configs/spt_pipeline_no_trim.yaml`: benchmark preprocessing config without quality trimming
- `src/preprocessing/configs/spt_pipeline_trim.yaml`: benchmark preprocessing config with stable-channel and timestep trimming
- `src/preprocessing/configs/spt_pipeline.yaml`: compatibility alias for the no-trim config
- `src/training/data.py`: `.npz` loading, chronological splitting, and sliding-window preparation
- `src/training/models/tranad.py`: minimal standalone TranAD model
- `src/training/train.py`: config-driven training entrypoint logic
- `src/training/configs/spt_tranad.yaml`: training config
- `scripts/spt/train_tranad.py`: CLI wrapper for training

## Data Assumptions

The verified preprocessing flow targets benchmark calibrator-response HDF5 files under `SPT_DATA_DIR_BENCHMARK`.

The current loader expects yearly benchmark HDF5 files with an `Observation ID` dataset plus one dataset per channel. It produces a NumPy array with shape `(T, C, D)` plus selected metadata embedded in a compressed `.npz` file.

The current supported SPT preprocessing path is the serial wrapper `scripts/spt/run_preprocessing.py`. It reads `src/preprocessing/configs/spt_pipeline_no_trim.yaml` by default, uses `loader.params.train_years` and `loader.params.test_years`, and writes two standalone artifacts:

- `train_processed.npz`
- `test_processed.npz`

Each artifact currently contains a single `data` array plus selected metadata such as timestamps, channel names, feature names, and year information.

The current training flow expects `input.train_npz_path` and `input.test_npz_path`. Training flattens each time step from `(C, D)` to a single feature dimension `F = C * D`, applies sliding windows, and trains on tensors shaped like `(B, W, F)`.

## Preprocessing

The default preprocessing config lives at `src/preprocessing/configs/spt_pipeline_no_trim.yaml`.

From the repo root, the current CLI entrypoint is:

```bash
python scripts/spt/run_preprocessing.py --config src/preprocessing/configs/spt_pipeline_no_trim.yaml --mode both
```

That wrapper runs the generic pipeline twice, once for train years and once for test years.

With the default no-trim config, the preprocessing path:

- loads benchmark HDF5 data for the configured train or test years
- uses `data_variant: snr`
- runs the single active step `interpolate_nan_per_channel`
- does not load response-reference arrays
- saves the result to `$OUTPUT_DIR/spt/<data_tag>/train_processed.npz` and `$OUTPUT_DIR/spt/<data_tag>/test_processed.npz` when output saving is enabled

Use `src/preprocessing/configs/spt_pipeline_trim.yaml` when you want stable-channel selection and timestep trimming based on response-reference metadata.

You can still construct `PreprocessingPipeline` directly, but the checked-in SPT config is written for the wrapper script because it uses `train_years` and `test_years` rather than a single `years` field.

## Training

The training config lives at `src/training/configs/spt_tranad.yaml`.

From the repo root, the main CLI is:

```bash
python scripts/spt/train_tranad.py --config src/training/configs/spt_tranad.yaml
```

The current training path is intentionally small:

- load `train_processed.npz` and `test_processed.npz`
- split the train artifact chronologically into train and validation segments
- scale train, validation, and test data using train-only statistics
- build sliding windows
- train a minimal TranAD model with PyTorch
- optionally save a checkpoint, loss curve, and test reconstruction errors

`src/training/data.py` still supports two compatibility cases:

- a legacy single-artifact input with one `data` array
- a split-aware artifact with `train_data`, `val_data`, and `test_data`

Those formats are accepted by the code, but they are not what the current local SPT preprocessing path produces.

## Current Limits

The current SPT implementation does not provide full parity with the external `CrossExperimentalAIDQM` codebase.

Notable gaps include:

- no full `anldq` package or registry system
- no verified inference or scoring pipeline in the local minimal path
- no richer checkpoint or experiment-output management yet
- some existing SPT scripts still depend on external code not present in this repository

In particular, `scripts/spt/infer_spt_v2.py` imports `anldq.*` and should be treated as an external or legacy path unless verified separately.
