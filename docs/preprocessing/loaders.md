# Loaders

Loaders are responsible for reading raw input data and returning:

- `data`: a NumPy array shaped `(T, C, F)`
- `metadata`: a dictionary with loader-specific context

The built-in loaders are selected by `loader.type` in the YAML config.

## Loader Overview

Current built-in loader types:

- `atlas`
- `spt`

They have different source formats, environment assumptions, and output feature dimensions.

## ATLAS Loader

Implementation: `src/preprocessing/data_loader/atlas.py`

The ATLAS loader reads a merged CSV export.

### Expected Input

The CSV must contain:

- a `timestamp` column
- one or more signal columns
- a matching `*_deltaT` column for each signal column

For a base signal column such as `my_signal`, the loader expects a companion column named `my_signal_deltaT`.

### Path Resolution

The loader supports three ways to identify the input:

1. Pass `csv_path` directly.
2. Pass `root` and `run_number`.
3. Rely on `ATLAS_DATA_DIR` and pass `run_number`.

When `csv_path` is not provided, the loader resolves the file as:

```text
<root>/<run_number>/merged.csv
```

### Output Shape

The loader returns shape `(T, C, 2)`.

The final feature axis contains:

- feature 0: value
- feature 1: deltaT

### Metadata Returned

The loader currently returns metadata including:

- `timestamps`
- `channel_names`
- `detector_names`
- `feature_names`
- `source_path`
- `run_number` when available

## SPT Loader

Implementation: `src/preprocessing/data_loader/spt.py`

The SPT loader reads benchmark calibrator-response HDF5 data across one or more seasons.

### Expected Input

The loader expects yearly HDF5 files whose names follow the built-in template:

```text
calibrator_responses_095ghz_{year}.hdf5
```

Each file must contain:

- an `Observation ID` dataset used as timestamps
- detector datasets keyed by detector name

### Path Resolution

The loader resolves its data root in this order:

1. explicit `root`
2. `SPT_DATA_DIR_BENCHMARK`
3. the built-in benchmark root in the loader module

The `years` parameter selects which yearly files to open.

### Built-In Filtering And Trimming

The SPT loader is not a thin file reader. It performs several domain-specific filtering steps before returning data:

- keeps only detector keys common across all selected yearly files
- filters detectors to a configured wafer ID using `BolometerProperties`
- trims detectors by stability using percentile-based thresholds
- trims timestamps using per-detector quantile bounds
- requires finite values and, by default, positive values
- sorts the final result by timestamp

### Runtime Dependencies

Wafer filtering requires the `spt3g` runtime to read `BolometerProperties` from the configured calibration archive path.

If the environment does not provide `spt3g` or the required calibration types, the loader will fail at runtime.

### Output Shape

The loader returns shape `(T, C, 1)`.

The final feature axis has a single response value per detector.

### Metadata Returned

The loader currently returns metadata including:

- `timestamps`
- `detector_names`
- `wafer_id`
- `boloproperties_path`
- `years`
- `data_paths`
- `observation_id_key`

## Metadata Returned By Loaders

Pipeline steps may optionally accept a `metadata` keyword argument. When a registered step function includes `metadata` in its signature, `PreprocessingPipeline.run()` passes loader metadata automatically.

This makes metadata useful for steps that depend on run-specific or loader-specific context.

## Loader Comparison

ATLAS:

- file format: CSV
- path model: explicit path or root plus run number
- output shape: `(T, C, 2)`
- main assumptions: timestamp and matching `*_deltaT` columns

SPT:

- file format: HDF5
- path model: root plus year set
- output shape: `(T, C, 1)`
- main assumptions: benchmark file naming, `Observation ID`, wafer filtering, `spt3g`

See also: [Config Reference](./config-reference.md), [Troubleshooting](./troubleshooting.md)
