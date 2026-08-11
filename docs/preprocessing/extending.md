# Extending

This page outlines the current extension points for adding custom preprocessing behavior.

## Add A New Step

Step functions are plain Python callables that operate on a NumPy array.

Basic pattern:

1. Add a function under `src/preprocessing/transforms/`.
2. Accept `data` as the first argument.
3. Accept additional keyword-only parameters for YAML-configurable options.
4. Return the transformed array.
5. Register the function name in `STEP_MAP` in `src/preprocessing/registry.py`.

Example shape:

```python
def my_step(data, *, scale: float = 1.0):
    return data * scale
```

Register it in `src/preprocessing/registry.py`:

```python
STEP_MAP = {
    "my_step": my_step,
}
```

Then it can be used in YAML:

```yaml
steps:
  - name: my_step
    params:
      scale: 2.0
```

## Add A New Loader

Loaders are selected by `loader.type` in `PreprocessingPipeline.load()`.

Basic pattern:

1. Implement a function that reads raw input data.
2. Return a tuple of `(data, metadata)`.
3. Add the loader to `LOADER_MAP` in `src/preprocessing/registry.py`.
4. Reference the new loader by name in the YAML config.

Expected return shape:

- `data`: NumPy array, usually shaped `(T, C, F)`
- `metadata`: dictionary with any useful context for downstream processing

## Shape And Metadata Conventions

Current code assumes preprocessing steps work on arrays shaped `(T, C, F)`.

Conventions to preserve where possible:

- `T`: time axis
- `C`: channel or detector axis
- `F`: input feature axis

A step may reduce `T`, `C`, or `F`, but save behavior still expects the final result to remain 3-dimensional.

If a step depends on loader metadata, add a `metadata` keyword parameter to the function signature. `PreprocessingPipeline.run()` will pass metadata automatically when it sees that parameter.

## Recommended Validation

There is no verified repo-wide test command for this package in the current checkout.

For focused validation, prefer small smoke checks:

- import `PreprocessingPipeline`
- load a known config with `from_config_file()`
- construct an in-memory config dictionary
- run a small synthetic array through a new step
- verify the loader or step can be resolved successfully

See also: [Steps](./steps.md), [Config Reference](./config-reference.md)
