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

Supported `type` values in `src/preprocessing/registry.py`:

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

The saved file is a compressed NumPy archive created with `np.savez_compressed()` containing `data`. When present, selected loader metadata is embedded in the same archive under keys such as `timestamps`, `channel_names`, `feature_names`, `years`, `sample_years`, and `run_number`.

## Environment Variable Fallbacks

The current code uses these path fallbacks:

- `OUTPUT_DIR`: used when `output.root` is not set
- `ATLAS_DATA_DIR`: used by the ATLAS loader when `root` is not set and `csv_path` is not passed
- `SPT_DATA_DIR_BENCHMARK`: used by the SPT loader when `root` is not set

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

SPT example for the current wrapper-driven path:

```yaml
loader:
  type: spt
  params:
    train_years: [2019, 2020, 2021]
    test_years: [2022, 2023]
    data_variant: snr
    response_template: calibrator_responses_095ghz_{year}.hdf5
    snr_template: calibrator_response_snrs_095ghz_{year}.hdf5
    observation_id_key: Observation ID
    require_monotonic_timestamps: true
    load_response_reference: true
    feature_names:
      response: response
      snr: SNR

steps:
  - name: interpolate_nan_per_channel

output:
  save: true
  root: null
  experiment: spt
  data_tag: no_trim
  file_name: train_processed.npz
```

When this config is used through `scripts/spt/run_preprocessing.py`, the wrapper swaps `train_years` and `test_years` into the loader's `years` argument and writes `train_processed.npz` and `test_processed.npz` separately.

See also: [Loaders](./loaders.md), [Steps](./steps.md), [Troubleshooting](./troubleshooting.md)
