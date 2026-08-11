# Config Reference

This page describes the YAML structure consumed by `PreprocessingPipeline`.

## Top-Level Schema

A preprocessing config is a YAML mapping with these top-level keys:

- `loader`: required
- `steps`: required as a top-level key, but may be an empty list
- `output`: optional

Minimal shape:

```yaml
loader:
  type: atlas
  params: {}

steps:
  - name: fill_nan
    params:
      value: 0.0

output:
  save: false
```

If `steps` is missing entirely, pipeline construction raises an error.

## `loader`

The `loader` section selects the function used by `PreprocessingPipeline.load()`.

```yaml
loader:
  type: atlas
  params:
    root: null
    run_number: 123456
```

Fields:

- `type`: required string
- `params`: optional mapping of keyword arguments passed to the selected loader

Supported `type` values in `src/preprocessing/pipeline.py`:

- `atlas`
- `spt`

Unknown loader types raise a `ValueError`.

## `steps`

The `steps` section is an ordered list. Each entry selects one preprocessing function by name.

```yaml
steps:
  - name: drop_nan_channels
    params:
      threshold: 0.02

  - name: fill_channel_median

  - name: clip_values
    params:
      low: 0.0
      high: 100.0
```

Fields per item:

- `name`: required string resolved through `src/preprocessing/registry.py`
- `params`: optional mapping of keyword arguments passed to the step function

Only registered step names are valid in YAML. Functions present in `src/preprocessing/transforms/` are not automatically available unless they are added to `STEP_MAP`.

## `output`

The `output` section controls save behavior used by `load_and_run()`.

```yaml
output:
  save: true
  root: null
  experiment: atlas
  data_tag: merged_csv
  file_name: processed.npz
```

Fields:

- `save`: optional boolean, defaults to `false`
- `root`: optional output root directory
- `experiment`: required when `save: true`
- `data_tag`: optional subdirectory name, defaults to `default`
- `file_name`: optional file name, defaults to `processed.npz`

When saving is enabled, the output path is:

```text
<root>/<experiment>/<data_tag>/<file_name>
```

The saved file is a compressed NumPy archive created with `np.savez_compressed()` and currently contains only `data=data`.

## Environment Variable Fallbacks

The current code uses these path fallbacks:

- `OUTPUT_DIR`: used when `output.root` is not set
- `ATLAS_DATA_DIR`: used by the ATLAS loader when `root` is not set and `csv_path` is not passed
- `SPT_DATA_DIR_BENCHMARK`: used by the SPT loader when `root` is not set; otherwise the loader falls back to its built-in benchmark path

## Complete Example Config

ATLAS example:

```yaml
loader:
  type: atlas
  params:
    root: null
    run_number: 123456

steps:
  - name: drop_nan_channels
    params:
      threshold: 0.02

  - name: fill_channel_median

  - name: clip_values
    params:
      low: 0.0
      high: 100.0

output:
  save: true
  root: null
  experiment: atlas
  data_tag: merged_csv
  file_name: processed.npz
```

SPT example:

```yaml
loader:
  type: spt
  params:
    root: null
    years: [2019]

steps:
  - name: drop_nan_channels
    params:
      threshold: 0.1

  - name: drop_nan_timesteps
    params:
      threshold: 0.005

  - name: fill_nan
    params:
      value: 0.0

  - name: clip_values
    params:
      low: -5.0
      high: 5.0

output:
  save: true
  root: null
  experiment: spt
  data_tag: default
  file_name: processed.npz
```

See also: [Loaders](./loaders.md), [Steps](./steps.md), [Troubleshooting](./troubleshooting.md)
