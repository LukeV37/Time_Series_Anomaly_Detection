# `src/preprocessing`

`src/preprocessing` is a small NumPy-based preprocessing package for loading raw inputs into canonical `(T, C, D)` arrays and then applying an ordered sequence of registered transforms.

The current package is intentionally simple:
- YAML configs are loaded as plain Python dictionaries.
- Loaders return `(data, metadata)`.
- An explicit registry maps YAML loader and step names to plain Python functions.
- A pipeline runner applies those functions in order and can optionally save a `.npz` artifact.

This package sits after data fetching and alignment. In the ATLAS workflow, `atlas.pbeast_fetcher` is responsible for producing a merged CSV; `preprocessing` is responsible for loading that CSV into arrays and cleaning or reshaping those arrays for downstream models. In the SPT workflow, `preprocessing` loads benchmark HDF5 files directly.

## Public API

The package currently exports two symbols from `src/preprocessing/__init__.py`:

- `load_config`
- `PreprocessingPipeline`

Typical usage looks like this:

```python
from preprocessing import load_config, PreprocessingPipeline

cfg = load_config("configs/atlas_pipeline.yaml")
pipeline = PreprocessingPipeline(cfg)
result, metadata = pipeline.load_and_run()
```

## Data Model

All registered preprocessing steps operate on NumPy arrays with shape `(T, C, D)`:

- `T`: number of time steps
- `C`: number of channels
- `D`: number of per-channel features

For the current ATLAS adapter, `D = 2`:

- feature `0`: raw aligned signal value
- feature `1`: aligned `deltaT` value for that signal

This means a merged ATLAS table is converted into an array of shape `(time, channels, 2)` before the configurable pipeline runs.

## Loaders

The data-loading boundary lives under `src/preprocessing/data_loader/`.

- `atlas.py` loads merged ATLAS CSV exports into `(T, C, 2)` arrays with `value` and `deltaT` features.
- `spt.py` loads yearly benchmark SPT HDF5 files into `(T, C, 1)` arrays.

The ATLAS loader expects an aligned merged table where:
- each signal has one value column
- each signal also has a matching `*_deltaT` column
- a `timestamp` column is present

For each channel column `X`, it expects a matching `X_deltaT` column and builds a NumPy array where:

- `array[:, channel_index, 0]` contains `X`
- `array[:, channel_index, 1]` contains `X_deltaT`

If a value column is missing its matching `*_deltaT` column, the loader raises `ValueError`.

## `config_loader.py`

`config_loader.py` provides one helper:

- `load_config(path: str | Path) -> dict`

This loads a YAML file with `yaml.safe_load` and returns a plain dictionary.

Relative paths are resolved from the current working directory first. If no file exists there, `src/preprocessing/` and `src/training/` are checked so this still works from the repo root:

```python
cfg = load_config("configs/atlas_pipeline.yaml")
```

There is no Pydantic schema or custom config object in the current implementation. The caller is expected to work with plain dictionaries.

## `pipeline.py`

`pipeline.py` defines `PreprocessingPipeline`, the ordered runner for registered preprocessing steps.

### Construction

```python
pipeline = PreprocessingPipeline(config)
```

The expected config shape is:

```yaml
loader:
  type: atlas
  params:
    root: null
    run_number: 123456

steps:
  - name: drop_nan_channels
    params:
      threshold: 0.2

output:
  save: true
  root: null
  experiment: atlas
  data_tag: merged_csv
  file_name: processed.npz
```

Each step must define:
- `name`: the registered function name
- `params`: optional keyword arguments for that function

### Execution

```python
processed = pipeline.run(data, metadata={"run_number": 123456})
```

`run()` applies every configured step in order.

Transforms that accept a `metadata` parameter receive the loader metadata automatically when you call `load_and_run()`, or manually when you pass `metadata=` to `run()`. This is currently used by `trim_edges`, which can apply per-run overrides from `metadata["run_number"]`.

`PreprocessingPipeline.__repr__()` returns a compact summary of configured step labels, which is useful for debugging.

## `registry.py`

`registry.py` contains the explicit step lookup used by the pipeline.

It provides:
- `STEP_MAP`: mapping from YAML `name` values to callables
- `resolve_step(name)`: lookup used by the pipeline runner

Example shape:

```python
STEP_MAP = {
    "clip_values": clip_values,
}
```

If a configured step cannot be resolved, `resolve_step()` raises `ValueError` and includes the available step names.

## `transforms/`

The subpackage is split by transform category. `preprocessing.registry` imports the concrete functions it exposes through `STEP_MAP`.

### `transforms/filters.py`

Available functions:
- `drop_nan_channels(threshold=0.2)`
  - Drops channels whose NaN fraction across time and features exceeds the threshold.
  - Shape change: `(T, C, D) -> (T, C', D)`
- `drop_nan_timesteps(threshold=0.005)`
  - Drops time steps whose NaN fraction across channels and features exceeds the threshold.
  - Shape change: `(T, C, D) -> (T', C, D)`
- `select_stable_channels(...)`
  - Builds a percentile-based channel mask from the input or a metadata reference array.
- `keep_channel_mask(keep, metadata=None)`
  - Applies a precomputed boolean channel mask.
- `keep_named_channels(channel_names, metadata=None)`
  - Keeps only channels whose names appear in the provided list.
- `filter_quality_timesteps(...)`
  - Filters time steps based on a metadata reference array and quantile rules.
- `trim_edges(remove_first=0, remove_last=0, run_specific=None, metadata=None)`
  - Removes fixed numbers of time steps from the beginning or end.
  - Supports per-run overrides through the `run_specific` mapping.

### `transforms/imputer.py`

Available functions:
- `fill_channel_median()`
  - Replaces NaNs in each `(channel, feature)` slice with the median over time.
- `fill_channel_mean()`
  - Replaces NaNs in each `(channel, feature)` slice with the mean over time.

Both preserve the input shape.

### `transforms/normalizer.py`

Available functions:
- `clip_values(low=None, high=None)`
  - Clips values into a fixed range.
- `subtract_mean(axis=0)`
  - Helper function present in the module but not currently registered for YAML use.
- `apply_scale(mean, std)`
  - Helper function present in the module but not currently registered for YAML use.

Notes:
- only `clip_values` is currently registered in `src/preprocessing/registry.py`
- `subtract_mean` and `apply_scale` are not available by name in YAML configs unless they are added to `STEP_MAP`

### `transforms/reducer.py`

Available functions:
- `subsample_time(stride)`
  - Keeps every `stride`-th time step
  - Raises `ValueError` if `stride < 1`

### `transforms/transforms.py`

Available functions:
- `fill_nan(value=0.0)`
  - Replaces every NaN with a constant value.
- `interpolate_nan_per_channel()`
  - Interpolates NaN values independently along the time axis for each channel-feature slice.
- `drop_features(indices)`
  - Removes feature indices from the `D` axis.
- `keep_features(indices)`
  - Retains only selected feature indices from the `D` axis.

These are useful when downstream models should ignore the `deltaT` feature, interpolate sparse gaps, or keep only a subset of features.

## Config Examples

The package ships with example YAML configs under `src/preprocessing/configs/`:

- `atlas_pipeline.yaml`
- `spt_pipeline_no_trim.yaml`
- `spt_pipeline_trim.yaml`

For example, `spt_pipeline_no_trim.yaml` currently does the following:
1. loads train/test year groups for the benchmark HDF5 dataset
2. uses `interpolate_nan_per_channel` as its only active preprocessing step
3. does not load response-reference arrays
4. saves separate train and test standalone `.npz` artifacts through `scripts/spt/run_preprocessing.py`

## Minimal End-to-End Example

```python
from preprocessing import load_config, PreprocessingPipeline

cfg = load_config("configs/atlas_pipeline.yaml")
pipeline = PreprocessingPipeline(cfg)
processed, metadata = pipeline.load_and_run()
```

For the built-in ATLAS config, the loaded array has shape `(T, C, 2)` and `metadata` includes timestamps, channel names, feature names, and the run number when available.

## Adding a New Step

To add a new preprocessing step:

1. implement a function that accepts a NumPy array as its first argument
2. return a NumPy array in `(T, C, D)` form
3. add that function to `STEP_MAP` in `registry.py`
4. reference it from YAML using the same `name`

Example:

```python
import numpy as np


def square_values(data: np.ndarray) -> np.ndarray:
    return data ** 2
```

Then add it to `STEP_MAP`:

```python
STEP_MAP = {
    "fill_nan": fill_nan,
    "square_values": square_values,
}
```

Then in YAML:

```yaml
steps:
  - name: square_values
```

## Current Scope

This package is intentionally not doing a few things yet:
- no fit/transform training lifecycle
- no serialization of learned preprocessing state
- no metadata object returned alongside arrays
- no schema validation beyond normal Python errors
- no separate channel-selection abstraction beyond plain array transforms

That keeps the package close to the current repo needs: convert aligned tables to arrays, run a small ordered set of NumPy transforms, and hand the result to downstream anomaly-detection code.
