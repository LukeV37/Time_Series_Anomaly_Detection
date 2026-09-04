# Troubleshooting

This page collects common preprocessing failures based on the current package behavior.

## Config Errors

### `This pipeline config does not define a loader.`

Cause:

- the top-level `loader` key is missing

Check:

- confirm the YAML has a `loader:` section
- confirm `PreprocessingPipeline` is receiving the config you expect

### `This pipeline config does not define top-level 'steps'.`

Cause:

- the top-level `steps` key is missing entirely

Check:

- add a `steps:` section even if you intend to leave it empty

### `Unsupported loader type ...`

Cause:

- `loader.type` does not match any key in `LOADER_MAP`

Check:

- use one of the currently supported types: `atlas` or `spt`

## Loader Errors

### ATLAS: missing `timestamp`

Cause:

- the merged CSV does not contain a `timestamp` column

Check:

- verify the input file is the expected merged export format

### ATLAS: missing `*_deltaT` columns

Cause:

- one or more signal columns do not have matching deltaT companions

Check:

- for each base signal column, confirm a corresponding `<signal>_deltaT` column exists

### ATLAS: missing path configuration

Cause:

- the loader was not given `csv_path`
- and `root` is missing
- and `ATLAS_DATA_DIR` is not set
- or `run_number` is missing when resolving from `root`

Check:

- pass `csv_path`
- or pass both `root` and `run_number`
- or set `ATLAS_DATA_DIR` and pass `run_number`

### SPT: benchmark root is missing or invalid

Cause:

- `root` is invalid
- `SPT_DATA_DIR_BENCHMARK` is invalid
- or neither is set

Check:

- point `loader.params.root` to the correct benchmark directory
- or set `SPT_DATA_DIR_BENCHMARK`

### SPT: missing HDF5 files

Cause:

- one or more expected yearly files are absent for the selected `years`

Check:

- confirm the requested `years` match files present under the benchmark root

### SPT: missing `Observation ID`

Cause:

- an HDF5 file does not contain the required timestamp dataset

Check:

- verify file contents and dataset names

## Step Resolution Errors

### `No preprocessing step ...`

Cause:

- `steps[].name` is not registered in `src/preprocessing/registry.py`

Check:

- use only currently registered step names from [Steps](./steps.md)
- if the function exists in a module but is not registered, add it to `STEP_MAP`

## Output Saving Errors

### `Output saving requested but no output root was configured.`

Cause:

- `output.save` is true
- but `output.root` is unset and `OUTPUT_DIR` is not defined

Check:

- set `output.root`
- or define `OUTPUT_DIR`

### `Output saving requested but no experiment was configured.`

Cause:

- `output.save` is true but `output.experiment` is missing or empty

Check:

- set `output.experiment`

### `Expected final preprocessing output shape (T, C, F), got ...`

Cause:

- save behavior only accepts a 3D final array

Check:

- ensure the pipeline still returns shape `(T, C, F)` before saving

## Environment And Dependency Issues

### SPT: non-monotonic timestamps

Cause:

- a yearly HDF5 file has timestamps that decrease
- and `require_monotonic_timestamps` is enabled

Check:

- inspect the configured file for timestamp ordering issues
- disable `require_monotonic_timestamps` only if out-of-order samples are expected and safe for your downstream use

### SPT: channel schema mismatch across years

Cause:

- yearly HDF5 inputs do not expose the same detector dataset keys

Check:

- verify the same channel names exist in every selected yearly file
- preprocess a narrower year set if the benchmark files are not schema-compatible

### Config file not found when using a relative path

Cause:

- the relative path is not valid from the current working directory
- and it is not found under the package-relative fallback locations

Check:

- pass an explicit path
- or use a path relative to `src/preprocessing/` such as `configs/atlas_pipeline.yaml`

See also: [Config Reference](./config-reference.md), [Loaders](./loaders.md), [Steps](./steps.md)
