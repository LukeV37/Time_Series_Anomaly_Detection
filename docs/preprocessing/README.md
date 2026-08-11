# Preprocessing

This package provides a YAML-driven preprocessing pipeline for time-series anomaly detection workflows.

The main entry point is `preprocessing.PreprocessingPipeline` in `src/preprocessing/pipeline.py`. A pipeline config selects a loader, applies an ordered list of steps, and can optionally save the processed array to disk.

## What This Package Does

The preprocessing package is built around a simple flow:

1. Load raw input data and metadata.
2. Apply zero or more NumPy-based preprocessing steps.
3. Optionally save the final array as a compressed `.npz` file.

The package currently includes two built-in loaders:

- `atlas` for merged CSV exports
- `spt` for benchmark HDF5 calibrator-response data

## Core Concepts

A pipeline is configured with three top-level sections:

- `loader`: selects how raw input data is read
- `steps`: ordered preprocessing operations resolved by name
- `output`: optional save settings for the final result

The pipeline class supports three main workflows:

- `load()` to load data and metadata only
- `run(data, metadata=...)` to apply steps to an existing array
- `load_and_run()` to do both and optionally save output

## Quick Start

```python
from preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline.from_config_file("configs/atlas_pipeline.yaml")
data, metadata = pipeline.load_and_run()
```

`load_and_run()` returns a tuple of `(data, metadata)`.

## Built-In Pipeline Configs

The package ships with example configs under `src/preprocessing/configs/`:

- `atlas_pipeline.yaml`
- `spt_pipeline.yaml`

These are good starting points for copying and adjusting loader parameters, step parameters, and output settings.

## Data Shape Convention

Preprocessing steps operate on arrays with shape `(T, C, F)`:

- `T`: time steps
- `C`: channels or detectors
- `F`: per-channel input features

Examples:

- The ATLAS loader returns `(T, C, 2)` for `(value, deltaT)`.
- The SPT loader returns `(T, C, 1)`.

## How Config Resolution Works

Config files are loaded through `utils.load_config()` in `src/utils/config_loader.py`.

Relative paths are first resolved from the current working directory. If that does not work, the loader also checks package-relative paths under:

- `src/preprocessing/`
- `src/training/`

This allows calls such as `PreprocessingPipeline.from_config_file("configs/atlas_pipeline.yaml")` to work without depending on the caller's current directory in the usual way.

## Where To Go Next

- See [Config Reference](./config-reference.md) for the YAML schema.
- See [Loaders](./loaders.md) for the two built-in loader implementations.
- See [Steps](./steps.md) for pipeline steps that can be used in YAML.
- See [Extending](./extending.md) for adding custom steps or loaders.
- See [Troubleshooting](./troubleshooting.md) for common errors and environment issues.
