# `src/preprocessing`

`src/preprocessing` is a small NumPy-based preprocessing package for turning aligned time-series tables into model-ready arrays and then applying an ordered sequence of transforms.

The current package is intentionally simple:
- YAML configs are loaded as plain Python dictionaries.
- A minimal adapter converts aligned `pandas.DataFrame` tables into `(T, C, D)` arrays.
- An explicit step map connects YAML step names to plain Python functions.
- A pipeline runner applies those functions in order.

This package sits after data fetching and alignment. In the ATLAS workflow, `atlas.pbeast_fetcher` is responsible for producing a merged, aligned table; `preprocessing` is responsible for converting that table into arrays and cleaning or reshaping those arrays for downstream models.

## Public API

The package currently exports three symbols from `src/preprocessing/__init__.py`:

- `dataframe_to_array`
- `load_config`
- `PreprocessingPipeline`

Typical usage looks like this:

```python
from preprocessing import dataframe_to_array, load_config, PreprocessingPipeline

cfg = load_config("configs/atlas_pipeline.yaml")
pipeline = PreprocessingPipeline(cfg)

array = dataframe_to_array(merged_dataframe)
processed = pipeline.run(array, metadata={"run_number": run_number})
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

## `adapters.py`

`adapters.py` contains the DataFrame-to-NumPy boundary:

- `dataframe_to_array(dataframe: pd.DataFrame) -> np.ndarray`

This function expects an aligned ATLAS-style merged table where:
- each signal has one value column
- each signal also has a matching `*_deltaT` column
- a `timestamp` column may be present

The adapter treats all columns except `timestamp` and `*_deltaT` as channels. For each channel column `X`, it expects a matching `X_deltaT` column. It then builds a NumPy array where:

- `array[:, channel_index, 0]` contains `X`
- `array[:, channel_index, 1]` contains `X_deltaT`

If a value column is missing its matching `*_deltaT` column, the adapter raises `ValueError`.

Current limitations:
- it drops DataFrame metadata instead of preserving it alongside the array
- it does not return channel names
- it does not preserve the `timestamp` column in a separate structure
- it assumes the ATLAS merged-column naming convention already exists

That is deliberate for now: the adapter is only meant to create the model input tensor, not to be a full reversible table representation.

## `config_loader.py`

`config_loader.py` provides one helper:

- `load_config(path: str | Path) -> dict`

This loads a YAML file with `yaml.safe_load` and returns a plain dictionary.

Relative paths are resolved from the package directory first, so this works from anywhere:

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
  - Subtracts the mean computed on the current input array.
- `apply_scale(mean, std)`
  - Applies precomputed per-channel standardization values.

Notes:
- these are currently stateless runtime transforms
- `apply_scale` expects `mean` and `std` lists with length equal to the number of channels
- zero standard deviations are replaced with `1.0` during division to avoid divide-by-zero errors

### `transforms/reducer.py`

Available functions:
- `subsample_time(stride)`
  - Keeps every `stride`-th time step
  - Raises `ValueError` if `stride < 1`

### `transforms/transforms.py`

Available functions:
- `fill_nan(value=0.0)`
  - Replaces every NaN with a constant value.
- `drop_features(indices)`
  - Removes feature indices from the `D` axis.
- `keep_features(indices)`
  - Retains only selected feature indices from the `D` axis.

These are useful when downstream models should ignore the `deltaT` feature or keep only a subset of features.

## Config Examples

The package ships with example YAML configs under `src/preprocessing/configs/`:

- `atlas_pipeline.yaml`
- `spt_pipeline.yaml`

For example, `spt_pipeline.yaml` currently does the following:
1. drops channels with too many NaNs
2. drops heavily missing time steps
3. fills remaining NaNs with `0.0`
4. clips values to a fixed range

## Minimal End-to-End Example

```python
import pandas as pd
from preprocessing import dataframe_to_array, load_config, PreprocessingPipeline

merged = pd.DataFrame(
    {
        "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:00:05"],
        "signal_a": [1.0, 2.0],
        "signal_a_deltaT": [0.0, 5.0],
        "signal_b": [3.0, 4.0],
        "signal_b_deltaT": [0.0, 1.0],
    }
)

cfg = load_config("configs/atlas_pipeline.yaml")
pipeline = PreprocessingPipeline(cfg)

array = dataframe_to_array(merged)
processed = pipeline.run(array)
```

At the adapter boundary, `array.shape` will be `(2, 2, 2)`.

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
