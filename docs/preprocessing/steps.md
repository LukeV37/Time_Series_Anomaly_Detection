# Steps

Pipeline steps are NumPy-based functions applied in order after loading data.

Step names in YAML are resolved through `STEP_MAP` in `src/preprocessing/registry.py`.

## How Steps Work

Each configured step has this shape in YAML:

```yaml
- name: fill_nan
  params:
    value: 0.0
```

At runtime, the pipeline:

1. Looks up the step function by `name`.
2. Copies `params` into keyword arguments.
3. Passes loader `metadata` automatically if the function signature accepts `metadata`.
4. Replaces the current array with the returned array.

## Registered Step Names

The following step names are currently available in YAML configs:

- `drop_nan_channels`
- `drop_nan_timesteps`
- `fill_channel_median`
- `clip_values`
- `fill_nan`

These are the only names currently registered in `src/preprocessing/registry.py`.

## Step Reference

### `drop_nan_channels`

Implementation: `src/preprocessing/transforms/filters.py`

Drops channels whose NaN fraction across time and feature axes exceeds a threshold.

Parameters:

- `threshold`: float in `[0, 1]`, default `0.2`

Shape behavior:

- input: `(T, C, D)`
- output: `(T, C', D)`

Example:

```yaml
- name: drop_nan_channels
  params:
    threshold: 0.02
```

### `drop_nan_timesteps`

Implementation: `src/preprocessing/transforms/filters.py`

Drops time steps whose NaN fraction across channels and features exceeds a threshold.

Parameters:

- `threshold`: float in `[0, 1]`, default `0.005`

Shape behavior:

- input: `(T, C, D)`
- output: `(T', C, D)`

Example:

```yaml
- name: drop_nan_timesteps
  params:
    threshold: 0.005
```

### `fill_channel_median`

Implementation: `src/preprocessing/transforms/imputer.py`

Fills NaN values in each channel-feature slice using the median over the time axis.

Parameters:

- none

Shape behavior:

- input: `(T, C, D)`
- output: `(T, C, D)`

Example:

```yaml
- name: fill_channel_median
```

### `clip_values`

Implementation: `src/preprocessing/transforms/normalizer.py`

Clips array values to the closed interval `[low, high]`.

Parameters:

- `low`: optional lower bound
- `high`: optional upper bound

Shape behavior:

- input: `(T, C, D)`
- output: `(T, C, D)`

Example:

```yaml
- name: clip_values
  params:
    low: -5.0
    high: 5.0
```

### `fill_nan`

Implementation: `src/preprocessing/transforms/transforms.py`

Replaces every NaN value with a constant.

Parameters:

- `value`: replacement value, default `0.0`

Shape behavior:

- input: `(T, C, D)`
- output: `(T, C, D)`

Example:

```yaml
- name: fill_nan
  params:
    value: 0.0
```

## Metadata-Aware Steps

The pipeline checks whether a registered step function accepts a `metadata` keyword parameter. If it does, loader metadata is passed automatically during `run()` and `load_and_run()`.

This supports steps that depend on input-specific context such as run numbers or timestamp metadata.

## Notes On Unregistered Helper Functions

There are additional functions in `src/preprocessing/transforms/` that are not currently registered in `STEP_MAP`.

That means:

- they may be useful as implementation building blocks
- they are not available by name in YAML configs unless they are added to the registry

When documenting or using steps, treat `src/preprocessing/registry.py` as the source of truth for what is supported in config files.

See also: [Config Reference](./config-reference.md), [Extending](./extending.md)
